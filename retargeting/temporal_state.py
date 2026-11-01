"""Module retargeting.temporal_state
==================================
Moteur d'état temporel récursif (t-1 -> t) pour le retargeting géométrique.
Permet d'éliminer les sauts d'hémisphère de quaternions, de stabiliser le roulis (twist)
sur les os à un seul enfant, de maintenir une échelle morphologique constante,
et d'extrapoler par inertie cinématique en cas d'occlusion (au lieu de sauter en T-Pose).
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple, Any
import numpy as np

from core.geometry import (
    quat_identity,
    quat_normalize,
    quat_mul,
    quat_conjugate,
)


def align_quaternion_hemisphere(q_curr: np.ndarray, q_prev: np.ndarray) -> np.ndarray:
    """
    Garantit que q_curr est dans le même hémisphère de S³ que q_prev (<q_curr, q_prev> >= 0).
    Empêche les rotations de 360° et les sauts brusques d'interpolation causés par le double-cover SO(3).
    """
    dot = float(np.dot(q_curr, q_prev))
    if dot < 0.0:
        return -q_curr
    return q_curr


def quaternion_log_map(q: np.ndarray) -> np.ndarray:
    """
    Application logarithmique de S³ vers l'algèbre de Lie so(3) (vecteur rotation 3D).
    q = [w, x, y, z] -> w = cos(theta/2), v = sin(theta/2) * u
    """
    q_norm = quat_normalize(q)
    w = np.clip(q_norm[0], -1.0, 1.0)
    v = q_norm[1:4]
    v_norm = float(np.linalg.norm(v))
    if v_norm < 1e-8:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * np.arccos(w)
    return (angle / v_norm) * v


def quaternion_exp_map(omega_dt: np.ndarray) -> np.ndarray:
    """
    Application exponentielle de so(3) vers S³ (quaternion unitaire).
    omega_dt: vecteur de rotation infinitésimale (3,)
    """
    angle = float(np.linalg.norm(omega_dt))
    if angle < 1e-8:
        return quat_identity()
    axis = omega_dt / angle
    w = np.cos(angle * 0.5)
    xyz = axis * np.sin(angle * 0.5)
    return np.array([w, xyz[0], xyz[1], xyz[2]], dtype=np.float64)


class TemporalRetargetState:
    """
    Gestionnaire d'état temporel persistant d'une frame à l'autre (t-1 -> t).
    
    Attributs:
        rotations_prev: {bone_name: quat_world} à l'instant t-1
        angular_velocities: {bone_name: omega (rad/s)} estimée entre t-2 et t-1
        scale_prev: facteur d'échelle Umeyama lissé
        twist_poles_prev: {bone_name: vector3} repère de torsion résolu à t-1
        foot_contact_prev: {bone_name: bool} statut de contact au sol
        scale_alpha: coefficient de lissage exponentiel de l'échelle (0.0 à 1.0)
        fps: fréquence d'échantillonnage de la séquence
    """

    def __init__(self, fps: float = 30.0, scale_alpha: float = 0.85):
        self.fps = max(fps, 1.0)
        self.dt = 1.0 / self.fps
        self.scale_alpha = np.clip(scale_alpha, 0.0, 1.0)
        
        self.rotations_prev: Dict[str, np.ndarray] = {}
        self.angular_velocities: Dict[str, np.ndarray] = {}
        self.scale_prev: Optional[float] = None
        self.twist_poles_prev: Dict[str, np.ndarray] = {}
        self.foot_contact_prev: Dict[str, bool] = {}
        self.frame_index: int = 0

    def has_history(self) -> bool:
        """Indique si l'état dispose d'au moins une frame d'historique (t-1)."""
        return len(self.rotations_prev) > 0

    def predict_rotation(self, bone: str, fallback_rest: np.ndarray) -> np.ndarray:
        """
        Extrapole l'orientation du membre par inertie cinématique (t-1 + omega * dt).
        Évite le snap brutal vers la T-pose si les marqueurs sont masqués à l'instant t.
        """
        if bone not in self.rotations_prev:
            return fallback_rest.copy()
        
        q_prev = self.rotations_prev[bone]
        if bone in self.angular_velocities:
            omega = self.angular_velocities[bone]
            # delta_q = exp(0.5 * omega * dt)
            delta_q = quaternion_exp_map(omega * self.dt)
            q_pred = quat_normalize(quat_mul(delta_q, q_prev))
            return align_quaternion_hemisphere(q_pred, q_prev)
        
        return q_prev.copy()

    def filter_scale(self, raw_scale: float) -> float:
        """
        Lisse le facteur d'échelle Umeyama pour stabiliser la corpulence du personnage.
        """
        if self.scale_prev is None:
            self.scale_prev = float(raw_scale)
            return self.scale_prev
        
        smoothed = self.scale_alpha * self.scale_prev + (1.0 - self.scale_alpha) * float(raw_scale)
        self.scale_prev = smoothed
        return smoothed

    def regularize_quaternion(self, bone: str, q_raw: np.ndarray) -> np.ndarray:
        """
        Projette le quaternion résolu dans l'hémisphère de t-1 pour éliminer les sauts de signe.
        """
        q_norm = quat_normalize(q_raw)
        if bone in self.rotations_prev:
            return align_quaternion_hemisphere(q_norm, self.rotations_prev[bone])
        return q_norm

    def update_frame(
        self,
        rotations_t: Dict[str, np.ndarray],
        scale_t: Optional[float] = None,
        twist_poles_t: Optional[Dict[str, np.ndarray]] = None,
        contacts_t: Optional[Dict[str, bool]] = None,
    ) -> None:
        """
        Met à jour l'état persistant avec les résultats validés de la frame t.
        Calcule les vitesses angulaires omega = log(q_t * q_{t-1}^*) / dt.
        """
        for bone, q_t in rotations_t.items():
            q_norm = quat_normalize(q_t)
            if bone in self.rotations_prev:
                q_prev = self.rotations_prev[bone]
                q_aligned = align_quaternion_hemisphere(q_norm, q_prev)
                # q_rel = q_aligned * q_prev^*
                q_rel = quat_mul(q_aligned, quat_conjugate(q_prev))
                omega = quaternion_log_map(q_rel) / self.dt
                self.angular_velocities[bone] = omega
                self.rotations_prev[bone] = q_aligned
            else:
                self.rotations_prev[bone] = q_norm
                self.angular_velocities[bone] = np.zeros(3, dtype=np.float64)

        if scale_t is not None:
            self.scale_prev = float(scale_t)

        if twist_poles_t is not None:
            self.twist_poles_prev.update(twist_poles_t)

        if contacts_t is not None:
            self.foot_contact_prev.update(contacts_t)

        self.frame_index += 1
