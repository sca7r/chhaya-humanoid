import re
from pathlib import Path

import torch

from supermimic.eth import GETUP_KEYS, getup_keys

URDF = Path(__file__).resolve().parents[1] / "robot_hardware/robot_description/urdf/chhaya.urdf"


def test_keyframes_cover_every_joint_and_score_progress():
    names = re.findall(r'<joint name="([^"]+)" type="revolute"', URDF.read_text())
    assert len(names) == 20
    q, up, h = getup_keys(names)
    assert q.shape == (len(GETUP_KEYS), 20) and (up[1:] >= up[:-1] - 0.2).all()

    class M:                                   # key_match without building the full MDP (needs the URDF FK)
        from supermimic.eth import GetUpCfg, GetUpMDP
        gcfg, key_q, key_up, key_h = GetUpCfg(), q, up, h
        key_w = torch.arange(1, len(GETUP_KEYS) + 1) / len(GETUP_KEYS)
        key_match = GetUpMDP.key_match
    s = M.key_match(M, q, up, h)               # each key matched exactly
    assert torch.allclose(s, M.key_w)
