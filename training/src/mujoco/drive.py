"""drive.py -- tap-to-walk in the browser: the policy runs in MuJoCo here, the page shows it live.

    python -m src.mujoco.drive --bundle exported/chhaya_final            # open http://localhost:8000
    python -m src.mujoco.drive --bundle exported/chhaya_final --host 0.0.0.0   # phone on the same Wi-Fi

Tap the floor and the robot turns, walks there and stops. Arrow keys or the buttons drive it directly.
Every input becomes the one command the policy knows (vx, vy, wz), limited to the training range
(utils/teleop.py). Headless machines: MUJOCO_GL=egl (or osmesa).
"""
import argparse
import io
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import mujoco
import numpy as np
from PIL import Image

from envs.mujoco.biped_env_v2 import STEP_DT, BipedEnvV2
from src.mujoco.policy_gate import Policy, _run_config_path
from src.mujoco.replay_video import XML, _styled_xml
from utils import Config
from utils.teleop import axes_to_command, go_to_axes

KEYS = {"up": (1, 0, 0), "down": (-1, 0, 0), "left": (0, 0, 1), "right": (0, 0, -1),
        "sleft": (0, 1, 0), "sright": (0, -1, 0), "stop": (0, 0, 0)}
LEVEL = 0.75        # fraction of the training command range used by the keys and the go-to
W, H = 960, 600

PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Chhaya</title><style>
:root{--bg:#f5f5f7;--ink:#1d1d1f;--mute:#6e6e73;--card:#fff;--acc:#0071e3}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.4 -apple-system,system-ui,sans-serif;display:flex;flex-direction:column;align-items:center}
h1{font-size:22px;font-weight:600;margin:24px 0 2px}p{margin:0 0 14px;color:var(--mute)}
.stage{position:relative;width:min(960px,calc(100vw - 32px));border-radius:18px;overflow:hidden;background:#ddd;box-shadow:0 8px 30px #0001;cursor:crosshair}
.stage img{display:block;width:100%}.ring{position:absolute;width:28px;height:28px;margin:-14px;border:2px solid var(--acc);border-radius:50%;animation:p .6s ease-out forwards;pointer-events:none}
@keyframes p{from{transform:scale(.3);opacity:1}to{transform:scale(1.6);opacity:0}}
.pad{display:grid;grid-template-columns:repeat(5,52px);gap:8px;margin:16px 0 8px}
button{height:44px;border:0;border-radius:12px;background:var(--card);color:var(--ink);font-size:17px;box-shadow:0 1px 3px #0002;cursor:pointer}
button:active{background:#e8e8ed}#s{font-variant-numeric:tabular-nums;color:var(--mute);margin-bottom:24px}
</style></head><body><h1>Chhaya</h1><p>Tap the floor to walk there. Arrow keys or the buttons drive it directly.</p>
<div class="stage" id="st"><img src="/stream" id="v" alt="live view"></div>
<div class="pad"><button data-k="sleft">&#8676;</button><button data-k="left">&#8634;</button><button data-k="up">&#9650;</button>
<button data-k="right">&#8635;</button><button data-k="sright">&#8677;</button><span></span><span></span>
<button data-k="down">&#9660;</button><button data-k="stop">&#9632;</button></div><div id="s"></div>
<script>
const go=(u)=>fetch(u).then(r=>r.text()).then(t=>s.textContent=t);
st.onclick=e=>{const r=v.getBoundingClientRect(),x=(e.clientX-r.left)/r.width,y=(e.clientY-r.top)/r.height;
 const d=document.createElement('div');d.className='ring';d.style.left=(x*100)+'%';d.style.top=(y*100)+'%';st.append(d);setTimeout(()=>d.remove(),700);
 go('/tap?x='+x+'&y='+y)};
document.querySelectorAll('button').forEach(b=>b.onclick=()=>go('/key?k='+b.dataset.k));
const K={ArrowUp:'up',ArrowDown:'down',ArrowLeft:'left',ArrowRight:'right',',':'sleft','.':'sright',' ':'stop'};
onkeydown=e=>{if(K[e.key]){e.preventDefault();go('/key?k='+K[e.key])}};
setInterval(()=>go('/status'),500);
</script></body></html>"""


class Sim:
    """Policy + MuJoCo at 50 Hz in a thread; renders the styled robot for the stream."""

    def __init__(self, bundle):
        self.policy = Policy(bundle)
        self.env = BipedEnvV2(history_size=self.policy.hist, use_history=True, critic_has_privileged_info=True,
                              env_config=Config(_run_config_path(bundle)).train.env_config)
        self.ranges, self.shape = self.env.command_limit()
        self.m = mujoco.MjModel.from_xml_path(_styled_xml(XML))
        self.m.vis.global_.offwidth, self.m.vis.global_.offheight = W, H
        self.m.tex_data[:] = 228 + self.m.tex_data // 10        # pale floor and sky, faint grid (SuperMimic look)
        self.d = mujoco.MjData(self.m)
        em = self.env.model         # copy the env's free joint and joints into the styled model by name
        names = {self.m.joint(j).name for j in range(self.m.njnt)}
        js = [j for j in range(em.njnt) if em.joint(j).name in names]   # (the env's backlash joints are not drawn)
        self.src = [em.jnt_qposadr[j] for j in js]
        self.dst = [self.m.jnt_qposadr[self.m.joint(em.joint(j).name).id] for j in js]
        self.n = [7 if em.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 1 for j in js]
        self.r = mujoco.Renderer(self.m, H, W)
        self.cam = mujoco.MjvCamera()
        self.cam.distance, self.cam.elevation, self.cam.azimuth = 1.8, -35.0, 135.0
        self.axes, self.target, self.tap, self.jpg = (0, 0, 0), None, None, b""
        self.lock = threading.Lock()

    def pose(self):
        w, x, y, z = self.env.data.qpos[3:7]
        return self.env.data.qpos[0], self.env.data.qpos[1], math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def run(self):
        obs, _ = self.env.reset()
        look, i = np.array([0.0, 0.0, 0.2]), 0
        while True:
            t0 = time.time()
            if self.target is not None:
                self.axes = go_to_axes(*self.pose(), *self.target)
            self.cmd = self.env.set_command(*axes_to_command(*self.axes, self.ranges, self.shape, LEVEL))
            obs, _, term, _, _ = self.env.step(self.policy(obs["actor"]))
            if term:
                obs, _ = self.env.reset()
                self.target, self.axes = None, (0, 0, 0)
            i += 1
            if i % 2 == 0:                                   # 25 fps stream
                for s, d_, n in zip(self.src, self.dst, self.n):
                    self.d.qpos[d_:d_ + n] = self.env.data.qpos[s:s + n]
                mujoco.mj_kinematics(self.m, self.d)
                look += 0.05 * (np.r_[self.d.qpos[:2], 0.2] - look)
                self.cam.lookat[:] = look
                self.r.update_scene(self.d, self.cam)
                if self.tap is not None:                     # tap -> the floor point under it
                    pnt, gid, fid, sid = np.zeros(3), np.zeros(1, np.int32), np.zeros(1, np.int32), np.zeros(1, np.int32)
                    mujoco.mjv_select(self.m, self.d, mujoco.MjvOption(), W / H, self.tap[0], 1 - self.tap[1],
                                      self.r.scene, pnt, gid, fid, sid)
                    if gid[0] >= 0:
                        self.target = pnt[:2].copy()
                    self.tap = None
                if self.target is not None:
                    g = self.r.scene.geoms[self.r.scene.ngeom]
                    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CYLINDER, [0.03, 0.002, 0],
                                        [*self.target, 0.002], np.eye(3).ravel(), [0.0, 0.44, 0.89, 1])
                    self.r.scene.ngeom += 1
                buf = io.BytesIO()
                Image.fromarray(self.r.render()).save(buf, "JPEG", quality=80)
                with self.lock:
                    self.jpg = buf.getvalue()
            time.sleep(max(0.0, STEP_DT - (time.time() - t0)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True, help="exported policy bundle (train_supermimic.py --export)")
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to reach it from a phone on the same network")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    sim = Sim(a.bundle)
    threading.Thread(target=sim.run, daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, body, ctype="text/plain"):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.end_headers()
            self.wfile.write(body.encode() if isinstance(body, str) else body)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path == "/":
                return self.reply(PAGE, "text/html")
            if u.path == "/stream":                          # MJPEG: the browser's <img> plays it natively
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
                self.end_headers()
                try:
                    while True:
                        with sim.lock:
                            jpg = sim.jpg
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                        time.sleep(0.04)
                except (BrokenPipeError, ConnectionResetError):
                    return
            if u.path == "/tap":
                sim.tap = (min(max(float(q.get("x", 0.5)), 0.0), 1.0), min(max(float(q.get("y", 0.5)), 0.0), 1.0))
            elif u.path == "/key" and q.get("k") in KEYS:
                sim.axes, sim.target = KEYS[q["k"]], None
            vx, vy, wz = getattr(sim, "cmd", (0.0, 0.0, 0.0))
            goal = "" if sim.target is None else f"  going to ({sim.target[0]:.2f}, {sim.target[1]:.2f}) m"
            self.reply(f"vx {vx:+.2f} m/s   vy {vy:+.2f} m/s   wz {wz:+.2f} rad/s{goal}")

    print(f"[drive] open http://{'localhost' if a.host == '127.0.0.1' else a.host}:{a.port}")
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
