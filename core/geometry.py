"""
Module isolate.core.geometry
============================
Opérations d'algèbre 3D purement basées sur NumPy:
- Quaternions avec convention [w, x, y, z]
- Conversions matrices de rotation (3x3) <-> quaternions
- Rotation de vecteurs
"""
from __future__ import annotations

import numpy as np
from typing import Union, Sequence


def quat_normalize(q: Union[np.ndarray, Sequence[float]], eps: float = 1e-12) -> np.ndarray:
    """Normalise un quaternion [w, x, y, z]."""
    q_arr = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q_arr)
    if norm < eps or not np.isfinite(norm):
        raise ValueError(f"Quaternion invalide ou norme proche de zéro: {norm}")
    return q_arr / norm


def quat_identity() -> np.ndarray:
    """Retourne le quaternion identité [1.0, 0.0, 0.0, 0.0]."""
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def quat_inv(q: np.ndarray) -> np.ndarray:
    """Calcule le conjugué/inverse d'un quaternion unitaire [w, x, y, z]."""
    q_norm = quat_normalize(q)
    return np.array([q_norm[0], -q_norm[1], -q_norm[2], -q_norm[3]], dtype=np.float64)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiplie deux quaternions a et b (convention [w, x, y, z])."""
    aw, ax, ay, az = quat_normalize(a)
    bw, bx, by, bz = quat_normalize(b)
    return quat_normalize(np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dtype=np.float64))


def rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Fait tourner un vecteur 3D v par le quaternion q [w, x, y, z]."""
    q = quat_normalize(q)
    v = np.asarray(v, dtype=np.float64)
    q_v = np.array([0.0, v[0], v[1], v[2]], dtype=np.float64)
    q_res = quat_mul(quat_mul(q, q_v), quat_inv(q))
    return q_res[1:4]


def quat_from_two_vectors(u: np.ndarray, v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Calcule le quaternion de rotation minimale alignant le vecteur u sur v."""
    u_arr = np.asarray(u, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    norm_u = np.linalg.norm(u_arr)
    norm_v = np.linalg.norm(v_arr)
    if norm_u < eps or norm_v < eps:
        return quat_identity()
    u_norm = u_arr / norm_u
    v_norm = v_arr / norm_v

    dot = float(np.dot(u_norm, v_norm))
    if dot >= 1.0 - eps:
        return quat_identity()
    if dot <= -1.0 + eps:
        # 180 degrés : trouver un axe orthogonal
        ortho = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(u_norm[0]) > 0.9:
            ortho = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = np.cross(u_norm, ortho)
        axis /= np.linalg.norm(axis)
        return np.array([0.0, axis[0], axis[1], axis[2]], dtype=np.float64)

    axis = np.cross(u_norm, v_norm)
    w = 1.0 + dot
    return quat_normalize(np.array([w, axis[0], axis[1], axis[2]], dtype=np.float64))


def matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convertit une matrice de rotation 3x3 en quaternion [w, x, y, z]."""
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"R doit avoir la forme (3,3), reçu {R.shape}")
    if not np.isfinite(R).all():
        raise ValueError("R contient des valeurs non finies (NaN / Inf)")

    # Anti-réflexion si nécessaire
    det = np.linalg.det(R)
    if det < 0.0:
        raise ValueError(f"R contient une réflexion (det={det:.4f})")

    t = np.trace(R)
    if t > 0.0:
        s = 2.0 * np.sqrt(t + 1.0)
        q = np.array([
            0.25 * s,
            (R[2, 1] - R[1, 2]) / s,
            (R[0, 2] - R[2, 0]) / s,
            (R[1, 0] - R[0, 1]) / s,
        ])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 1e-12))
        q = np.array([
            (R[2, 1] - R[1, 2]) / s,
            0.25 * s,
            (R[0, 1] + R[1, 0]) / s,
            (R[0, 2] + R[2, 0]) / s,
        ])
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 1e-12))
        q = np.array([
            (R[0, 2] - R[2, 0]) / s,
            (R[0, 1] + R[1, 0]) / s,
            0.25 * s,
            (R[1, 2] + R[2, 1]) / s,
        ])
    else:
        s = 2.0 * np.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 1e-12))
        q = np.array([
            (R[1, 0] - R[0, 1]) / s,
            (R[0, 2] + R[2, 0]) / s,
            (R[1, 2] + R[2, 1]) / s,
            0.25 * s,
        ])
    return quat_normalize(q)


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convertit un quaternion [w, x, y, z] en matrice de rotation 3x3."""
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1.0 - 2.0*(y*y + z*z), 2.0*(x*y - z*w),       2.0*(x*z + y*w)],
        [2.0*(x*y + z*w),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z - x*w)],
        [2.0*(x*z - y*w),       2.0*(y*z + x*w),       1.0 - 2.0*(x*x + y*y)],
    ], dtype=np.float64)
