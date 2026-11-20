# Changelog

## Chhaya v0.1.0 (2026-11-20)

- Whole robot: 20 servo joints (12 legs, 8 arms), one-servo claws, halo head with a sensor plate.
- Electronics rack under the chest: battery and Raspberry Pi 5; the chest stands on the rack.
- Parametric CAD for the chest, head, arms, claws and rack, with automatic clearance, fit and
  strength checks (`regenerate.sh`).
- URDF, MuJoCo and USD models generated from the CAD.

## Leg platform v2.0 (2026-10-30)

The legs, hips, torso and power electronics, as taken from their source (see `../NOTICE.md`).

- 12 Waveshare ST3215 servos, BNO055 IMU, 3S LiPo.
- Custom power distribution PCB.
- URDF with an `imu_link` at the BNO055 mounting point.
