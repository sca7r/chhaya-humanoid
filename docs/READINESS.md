# Ready-to-train review (2026-11-13)

## 1. Can the STS3215 servos carry the robot? Yes, for walking

`python -m chhaya.servo_load` (from `training/`), results in `physics/SERVO_LOAD.md`. Chhaya is 2.57 kg;
the STS3215 at 12 V stalls at 2.94 N·m.

| Load case | Hardest joint | Torque | Share of stall | Legs-only rig, same case |
|---|---|---|---|---|
| Standing on both feet (simulated) | ankle pitch | 0.30 N·m | 10 % | — |
| Whole weight on one foot, body not shifted over it (worst static case in a step) | hip roll | 1.34 N·m | 46 % | 0.87 N·m (30 %) |
| same | ankle roll | 1.26 N·m | 43 % | 0.82 N·m (28 %) |
| One-leg squat, knee bent 90° | knee | 1.17 N·m | 40 % | — |
| Arm held straight out | shoulder | 0.20 N·m | 7 % | — |

- Every static case stays under half of stall, so the servos have at least 2× headroom for the dynamic
  peaks of walking.
- The loads are 1.55× the legs-only rig's, which is the mass ratio. Walking will be less agile and
  the hip-roll and ankle-roll servos will run warmer.
- Not enough for: carrying more than a few hundred grams, deep one-leg squats held for long, or
  standing on one leg for minutes (above the servo's continuous rating of roughly a third of stall).
- The arms are far from their limits.

**Stronger servos.** Rhoban's BAM has identified models for the Dynamixel MX-64 and MX-106 and the
Waveshare ST3025. The MX servos have a different size and mounting, so every leg part and the model
would have to be redesigned. The ST3025 (40 kg·cm at 12 V, about a third stronger) is the same servo
family as the STS3215, but its size must be checked against a datasheet before assuming it drops in,
and our servo-model code would need its parameters and control law ported. Recommendation: train with
the STS3215; if the trained policy shows the hip-roll or ankle-roll servos saturating (the MuJoCo gate
reports this), move those four joints to a stronger servo then.

## 2. Frame review

Fixed in this pass:

| Flaw | Fix |
|---|---|
| The hands could pass through the torso and chest in simulation: those collision pairs were switched off, and the policy now commands the arms | Forearm and elbow links collide with the torso and chest in the MuJoCo and Isaac models. Only the shoulder and upper-arm links (which sit against the chest) stay excluded |
| At the arm rest pose the claws hung 12 mm from the hip servos | Rest pose moved out (shoulder roll 0.12 → 0.20 rad): 23 mm clear |

Fixed earlier (2026-11-12): the chest's bolts went through empty space in the torso shelf and its legs
cut the tray rim; the chest now stands on the electronics rack.

Known and accepted:

| Property | Value | Consequence |
|---|---|---|
| Centre of mass height when standing | 265 mm (legs-only rig: 206 mm) | Tips over at 8° fore/aft and 15° sideways (rig: 11° and 19°). The robot is top-heavy for its 90 mm feet; the policy has to work harder to balance |
| Centre of mass 8 mm ahead of the foot centre | 37 mm margin to the toes, 53 mm to the heels | The policy leans back slightly. Could be trimmed by sliding the battery back on the rack |
| Hands reach the hips and thighs | 2–3 % of random arm poses in the policy's range touch a leg | These contacts are simulated, so the policy learns to avoid them |
| Battery and the computer's ports stick out front and back | ~26 mm and ~22 mm | They take the hit in a fall. Add a printed bumper before fall tests on the real robot |
| Torso collision mesh has 15,000 vertices | | Isaac Lab 3.0 collides its convex hull; CPU MuJoCo uses the mesh. Replace with boxes if contact cost shows up in profiling |

Checks that pass on every rebuild (`hardware/regenerate.sh`): no overlap in 153 poses, moving gaps ≥ 1 mm,
torso fit, all printed parts at safety factor ≥ 3, model consistency (URDF = MuJoCo = USD), and from
`training/`: the physics check and the 19 Chhaya tests.

## 3. Before you press start

Not verified, and only a GPU can verify it:

1. **The training has not run on a GPU.** The Isaac Lab 3.0 environment, a short training run and
   playback work on a CPU (`ISAACLAB3.md`); speed, memory use and learning quality are unknown. The
   10-minute smoke test in `GPU_RUNBOOK.md` is step one. The reward weights were tuned on PhysX.
2. **The arm reward weights are first guesses.** Expect a second run.
3. **Export, the policy gate and the on-robot runtime still assume 12 joints.** They are needed after
   training, not for it.
4. Battery, computer and torso masses are estimates until the real parts are weighed.

Start: `GPU_RUNBOOK.md`.
