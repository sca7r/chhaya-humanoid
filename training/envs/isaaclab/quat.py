"""Quaternion helpers in (w, x, y, z), the order used everywhere inside this code base.

Pure torch, no simulator import: the same functions run in the unit tests. Isaac Lab 3.0's own
``isaaclab.utils.math`` uses (x, y, z, w); see compat.py for why the env does not.
"""
from __future__ import annotations

import math

import torch


def wrap_to_pi(a: torch.Tensor) -> torch.Tensor:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate v by q (body -> world for a body orientation q)."""
    w = q[..., 0:1]
    u = q[..., 1:4]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate v by the inverse of q (world -> body)."""
    w = q[..., 0:1]
    u = q[..., 1:4]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v - w * t + torch.cross(u, t, dim=-1)


def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    return torch.stack([
        cy * cr * cp + sy * sr * sp,
        cy * sr * cp - sy * cr * sp,
        cy * cr * sp + sy * sr * cp,
        sy * cr * cp - cy * sr * sp,
    ], dim=-1)


def euler_xyz_from_quat(q: torch.Tensor):
    """(roll, pitch, yaw), each wrapped to [0, 2*pi) as Isaac Lab 2.x did."""
    w, x, y, z = q.unbind(-1)
    roll = torch.atan2(2.0 * (w * x + y * z), w * w - x * x - y * y + z * z)
    sinp = 2.0 * (w * y - z * x)
    pitch = torch.where(sinp.abs() >= 1.0, torch.copysign(torch.full_like(sinp, math.pi / 2.0), sinp),
                        torch.asin(sinp.clamp(-1.0, 1.0)))
    yaw = torch.atan2(2.0 * (w * z + x * y), w * w + x * x - y * y - z * z)
    two_pi = 2.0 * math.pi
    return roll % two_pi, pitch % two_pi, yaw % two_pi


def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    h = angle.unsqueeze(-1) * 0.5
    q = torch.cat([torch.cos(h), axis * torch.sin(h)], dim=-1)
    return q / q.norm(dim=-1, keepdim=True)


def matrix_from_quat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = (q / q.norm(dim=-1, keepdim=True)).unbind(-1)
    m = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], dim=-1)
    return m.reshape(q.shape[:-1] + (3, 3))
