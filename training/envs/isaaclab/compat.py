"""Boundary between this code base and Isaac Lab 3.0 (Newton backend, no Isaac Sim).

Isaac Lab 3.0 changed two things that touch nearly every line of the environment:

  * data properties return a ``ProxyArray`` (Warp array with a ``.torch`` view), not a tensor;
  * quaternions are (x, y, z, w). Isaac Lab 2.x, MuJoCo, the rewards, the gait tables and the
    on-robot runtime all use (w, x, y, z).

Rather than rewrite every reward for a new quaternion order (and then disagree with the MuJoCo
env), the conversion happens ONCE, here:

  * ``TorchData`` wraps ``articulation.data`` / ``sensor.data``: every read is a torch tensor,
    and every quaternion (or pose) read is converted to (w, x, y, z);
  * ``write_root_pose`` / ``write_root_velocity`` / ``write_joint_state`` / ``set_joint_targets``
    take (w, x, y, z) and call the 3.0 writers;
  * ``envs.isaaclab.quat`` holds the few quaternion functions the env needs, in (w, x, y, z).

Inside the environment a quaternion is therefore always (w, x, y, z), as before.
"""
from __future__ import annotations

import torch

# fields that hold a pose (pos 3 + quat 4 [+ vel 6]); the quaternion sits in columns 3:7
_POSE_FIELDS = (
    "root_pose_w", "root_link_pose_w", "root_com_pose_w", "default_root_pose", "default_root_state",
    "root_state_w", "root_link_state_w", "root_com_state_w",
    "body_pose_w", "body_link_pose_w", "body_com_pose_w", "body_com_pose_b",
    "body_state_w", "body_link_state_w", "body_com_state_w", "pose_w",
)


def xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., 3:4], q[..., 0:3]], dim=-1)


def wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., 1:4], q[..., 0:1]], dim=-1)


def as_torch(x):
    """ProxyArray / warp array -> torch tensor (zero copy); tensors and None pass through."""
    if x is None or isinstance(x, torch.Tensor):
        return x
    t = getattr(x, "torch", None)
    if t is not None:
        return t
    import warp as wp
    return wp.to_torch(x)


class TorchData:
    """Read-only view of an Isaac Lab data object: tensors out, quaternions in (w, x, y, z)."""

    __slots__ = ("_d", "_owner")

    def __init__(self, data=None, owner=None):
        """data: the data object, or owner: the asset / sensor whose ``.data`` is read on every access
        (it does not exist until the simulation has started)."""
        object.__setattr__(self, "_d", data)
        object.__setattr__(self, "_owner", owner)

    def __getattr__(self, name: str):
        d = self._d if self._owner is None else self._owner.data
        v = as_torch(getattr(d, name))
        if not isinstance(v, torch.Tensor):
            return v
        if name in _POSE_FIELDS:
            return torch.cat([v[..., :3], xyzw_to_wxyz(v[..., 3:7]), v[..., 7:]], dim=-1)
        if "quat" in name:
            return xyzw_to_wxyz(v)
        return v


# ---------------------------------------------------------------------------------- writers
def _ids(env_ids):
    return None if env_ids is None or isinstance(env_ids, slice) else env_ids


def write_root_pose(robot, pose_wxyz: torch.Tensor, env_ids=None) -> None:
    """pose (n, 7) = pos + quat (w, x, y, z)."""
    pose = torch.cat([pose_wxyz[..., :3], wxyz_to_xyzw(pose_wxyz[..., 3:7])], dim=-1)
    robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=_ids(env_ids))


def write_root_velocity(robot, vel: torch.Tensor, env_ids=None) -> None:
    robot.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=_ids(env_ids))


def write_joint_state(robot, pos: torch.Tensor, vel: torch.Tensor, joint_ids=None, env_ids=None) -> None:
    robot.write_joint_position_to_sim_index(position=pos, joint_ids=joint_ids, env_ids=_ids(env_ids))
    robot.write_joint_velocity_to_sim_index(velocity=vel, joint_ids=joint_ids, env_ids=_ids(env_ids))


def set_joint_targets(robot, target: torch.Tensor, joint_ids=None, env_ids=None) -> None:
    robot.actuators.target_command.set_position_index(
        value=target, joint_ids=joint_ids, env_ids=_ids(env_ids), full_data=False)
