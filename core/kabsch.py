"""
Module isolate.core.kabsch
==========================
Solveurs d'alignement rigide optimal par l'algorithme de Kabsch (SVD):
- Kabsch non-pondéré classique
- Kabsch pondéré par importance de marqueur (LBS skinning weight)
- Protection anti-réflexion (garantie det(R) = +1)
- Mesures de rang, conditionnement et RMS error
"""
from __future__ import annotations

import numpy as np
from typing import Tuple, Optional


def kabsch_rotation(
    P_rest: np.ndarray,
    P_current: np.ndarray,
    min_rank_ratio: float = 1e-3,
    max_rms_ratio: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Estime la rotation optimale R entre deux nuages de points correspondants.
    
    P_current_centered ~= (R @ P_rest_centered.T).T
    
    Paramètres:
        P_rest: ndarray de forme (N, 3), N >= 3
        P_current: ndarray de forme (N, 3), N >= 3
        min_rank_ratio: Seuil de détection de points colinéaires
        max_rms_ratio: Seuil max d'erreur RMS acceptable avant de lever une alerte
        
    Retourne:
        R: ndarray (3, 3) matrice de rotation propre (det = +1)
        centroid_rest: ndarray (3,)
        centroid_current: ndarray (3,)
        rms_error: float
        singular_values: ndarray (3,)
    """
    A = np.asarray(P_rest, dtype=np.float64)
    B = np.asarray(P_current, dtype=np.float64)

    if A.shape != B.shape or A.ndim != 2 or A.shape[1] != 3:
        raise ValueError(f"Nuages incompatibles: A={A.shape}, B={B.shape}")
    if A.shape[0] < 3:
        raise ValueError(f"Kabsch requiert >= 3 points, reçu {A.shape[0]}")
    if not np.isfinite(A).all() or not np.isfinite(B).all():
        raise ValueError("Les nuages contiennent des valeurs NaN ou Inf")

    centroid_rest = A.mean(axis=0)
    centroid_current = B.mean(axis=0)

    A_c = A - centroid_rest
    B_c = B - centroid_current

    scale_A = np.linalg.norm(A_c)
    scale_B = np.linalg.norm(B_c)
    if scale_A < 1e-8 or scale_B < 1e-8:
        raise ValueError("Nuage de points quasi dégénéré (dispersion nulle)")

    # Matrice de covariance croisée H
    H = A_c.T @ B_c
    U, S, Vt = np.linalg.svd(H)

    # Vérification du rang / colinéarité
    if S[0] > 0 and (S[2] / S[0]) < min_rank_ratio:
        raise ValueError(
            f"Points quasi colinéaires ou coplanaires dégradés (S2/S0 = {S[2]/S[0]:.2e} < {min_rank_ratio})"
        )

    # Correction anti-réflexion
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    if d == 0:
        d = 1.0
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T

    # Calcul de l'erreur RMS résiduelle
    B_pred_c = (R @ A_c.T).T
    diff = B_c - B_pred_c
    rms = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))

    if max_rms_ratio is not None:
        typical_size = max(scale_A, scale_B) / np.sqrt(len(A))
        if typical_size > 1e-8 and (rms / typical_size) > max_rms_ratio:
            raise ValueError(
                f"Déformation trop importante pour un alignement rigide (RMS ratio = {rms/typical_size:.2f})"
            )

    return R, centroid_rest, centroid_current, rms, S


def weighted_kabsch_rotation(
    P_rest: np.ndarray,
    P_current: np.ndarray,
    weights: np.ndarray,
    min_points: int = 3,
    min_rank_ratio: float = 1e-3,
    max_rms_ratio: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Algorithme de Kabsch pondéré.
    Chaque point i contribue proportionnellement à weights[i].
    """
    A = np.asarray(P_rest, dtype=np.float64)
    B = np.asarray(P_current, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)

    if A.shape != B.shape or A.shape[0] != w.shape[0]:
        raise ValueError(f"Dimensions incompatibles: A={A.shape}, B={B.shape}, w={w.shape}")
    if A.shape[0] < min_points:
        raise ValueError(f"Nombre de points insuffisant ({A.shape[0]} < {min_points})")

    w = np.clip(w, 0.0, None)
    w_sum = float(w.sum())
    if w_sum <= 1e-12:
        w = np.ones(len(A), dtype=np.float64)
        w_sum = float(len(A))

    w_norm = (w / w_sum)[:, None]  # (N, 1)

    centroid_rest = np.sum(A * w_norm, axis=0)
    centroid_current = np.sum(B * w_norm, axis=0)

    A_c = A - centroid_rest
    B_c = B - centroid_current

    # H pondéré : (A_c * w).T @ B_c
    H = (A_c * w[:, None]).T @ B_c
    U, S, Vt = np.linalg.svd(H)

    if S[0] > 0 and (S[2] / S[0]) < min_rank_ratio:
        raise ValueError(
            f"Configuration de marqueurs dégénérée (S2/S0 = {S[2]/S[0]:.2e} < {min_rank_ratio})"
        )

    d = np.sign(np.linalg.det(Vt.T @ U.T))
    if d == 0:
        d = 1.0
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T

    B_pred_c = (R @ A_c.T).T
    diff = B_c - B_pred_c
    weighted_sq_err = np.sum(diff * diff, axis=1) * (w / w_sum)
    rms = float(np.sqrt(np.sum(weighted_sq_err)))

    return R, centroid_rest, centroid_current, rms, S
