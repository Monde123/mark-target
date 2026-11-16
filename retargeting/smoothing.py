"""Module retargeting.smoothing
======================================
Lissage temporel et anti-jitter pour le retargeting :
- Algorithme de De Casteljau pour trajectoires euclidiennes R^3 (racine / Hips translation)
- Courbes de Bézier Sphériques SQUAD (Spherical and Quadrangle Splines) sur la variété S^3
- Interpolation SLERP et opérations d'algèbre de Lie (log/exp)
"""

from __future__ import annotations
import numpy as np
from typing import List, Sequence
from core.geometry import quat_normalize, quat_mul, quat_inv


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Interpolation linéaire sphérique (SLERP) entre deux quaternions unitaires [w, x, y, z]."""
    q0_n = quat_normalize(q0)
    q1_n = quat_normalize(q1)

    dot = float(np.dot(q0_n, q1_n))

    # Plus court chemin sur la sphère S^3 (antipodal identification)
    if dot < 0.0:
        q1_n = -q1_n
        dot = -dot

    if dot > 0.9995:
        # Interpolation linéaire normalisée (NLERP) pour petits angles (évite division par zéro)
        res = (1.0 - t) * q0_n + t * q1_n
        return quat_normalize(res)

    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)

    theta = theta_0 * t
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return quat_normalize(s0 * q0_n + s1 * q1_n)


def quat_log(q: np.ndarray) -> np.ndarray:
    """Logarithme d'un quaternion unitaire vers l'espace tangent R^3."""
    q_n = quat_normalize(q)
    w = np.clip(q_n[0], -1.0, 1.0)
    v = q_n[1:4]
    v_norm = np.linalg.norm(v)

    if v_norm < 1e-8:
        return np.zeros(3, dtype=np.float64)

    theta = np.arccos(w)
    return (theta / v_norm) * v


def quat_exp(v: np.ndarray) -> np.ndarray:
    """Exponentielle d'un vecteur tangent R^3 vers la 3-sphère des quaternions unitaires S^3."""
    theta = float(np.linalg.norm(v))
    if theta < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    s = np.sin(theta) / theta
    return np.array([np.cos(theta), s * v[0], s * v[1], s * v[2]], dtype=np.float64)


def compute_squad_control_point(q_prev: np.ndarray, q_curr: np.ndarray, q_next: np.ndarray) -> np.ndarray:
    """Calcule le point de contrôle Bézier intermédiaire s_i sur S^3."""
    q_curr_inv = quat_inv(q_curr)
    d1 = quat_log(quat_mul(q_curr_inv, q_prev))
    d2 = quat_log(quat_mul(q_curr_inv, q_next))
    w = -0.25 * (d1 + d2)
    return quat_normalize(quat_mul(q_curr, quat_exp(w)))


def quat_squad(q0: np.ndarray, q1: np.ndarray, s0: np.ndarray, s1: np.ndarray, t: float) -> np.ndarray:
    """Évaluation d'une spline Bézier sphérique SQUAD (Shoemake 1985) sur S^3."""
    slerp_q = quat_slerp(q0, q1, t)
    slerp_s = quat_slerp(s0, s1, t)
    return quat_slerp(slerp_q, slerp_s, 2.0 * t * (1.0 - t))


def de_casteljau_bezier_3d(control_points: Sequence[np.ndarray], t: float) -> np.ndarray:
    """Algorithme classique de De Casteljau pour évaluer une courbe de Bézier en t dans R^3."""
    pts = [np.asarray(p, dtype=np.float64) for p in control_points]
    n = len(pts)
    if n == 0:
        raise ValueError("Aucun point de contrôle fourni")
    if n == 1:
        return pts[0]

    for r in range(1, n):
        pts = [(1.0 - t) * pts[i] + t * pts[i + 1] for i in range(n - r)]

    return pts[0]


def smooth_quaternion_trajectory(
    quaternions: Sequence[np.ndarray],
    smoothing_factor: float = 0.35,
    use_squad: bool = True,
) -> List[np.ndarray]:
    """
    Filtre temporel adaptatif préservant les orientations :
    Applique une interpolation spline SQUAD (C2) ou SLERP pour supprimer le jitter haute fréquence.
    
    Paramètres:
        quaternions: liste de N quaternions [w, x, y, z]
        smoothing_factor: intensité du lissage (0.0 = aucun, 0.5 = modéré, 0.9 = fort)
        use_squad: si True, utilise la spline sphérique Bézier SQUAD (Shoemake 1985)
    """
    n = len(quaternions)
    if n <= 2 or smoothing_factor <= 0.0:
        return [quat_normalize(q) for q in quaternions]

    # Alignement d'hémisphère préalable pour éliminer les sauts de signe S³
    aligned = [quat_normalize(quaternions[0])]
    for i in range(1, n):
        q_curr = quat_normalize(quaternions[i])
        if float(np.dot(q_curr, aligned[-1])) < 0.0:
            q_curr = -q_curr
        aligned.append(q_curr)

    alpha = np.clip(smoothing_factor, 0.0, 0.85)

    if use_squad and n >= 4:
        # Calcul des points de contrôle SQUAD intermédiaires s_i sur la variété S³
        s = [aligned[0]] * n
        for i in range(1, n - 1):
            s[i] = compute_squad_control_point(aligned[i - 1], aligned[i], aligned[i + 1])
        s[0] = compute_squad_control_point(aligned[0], aligned[0], aligned[1])
        s[-1] = compute_squad_control_point(aligned[-2], aligned[-1], aligned[-1])

        smoothed = [aligned[0]]
        for i in range(1, n - 1):
            # Évaluation SQUAD centrée à partir des voisins (i-1 et i+1)
            q_squad_pred = quat_squad(aligned[i - 1], aligned[i + 1], s[i - 1], s[i + 1], 0.5)
            # Pondération adaptative entre la frame courante et la prédiction SQUAD C²
            q_filtered = quat_slerp(aligned[i], q_squad_pred, alpha)
            smoothed.append(q_filtered)
        smoothed.append(aligned[-1])
        return smoothed
    else:
        # Fallback SLERP récursif (si n < 4)
        smoothed = [aligned[0]]
        for i in range(1, n):
            q_smooth = quat_slerp(smoothed[-1], aligned[i], 1.0 - alpha)
            smoothed.append(q_smooth)
        return smoothed


def smooth_translation_trajectory(
    translations: Sequence[np.ndarray],
    window_size: int = 3,
) -> List[np.ndarray]:
    """Lisse les positions de translation (racine / bassin) par moyenne mobile centrée."""
    arr = np.asarray(translations, dtype=np.float64)
    n = len(arr)
    if n < window_size or window_size <= 1:
        return [arr[i] for i in range(n)]

    half = window_size // 2
    res = []
    for i in range(n):
        i_start = max(0, i - half)
        i_end = min(n, i + half + 1)
        res.append(np.mean(arr[i_start:i_end], axis=0))

    return res
