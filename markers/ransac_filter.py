"""Module isolate.markers.ransac_filter
=======================================
Filtre algébrique robuste RANSAC et métrique de déformation élastique Green-Lagrange :
- Élimination des marqueurs aberrants (occlusions, glissements, bruit optique) par consensus
- Tenseur de déformation de Green-Lagrange E = 0.5 * (F^T @ F - I_3) pour quantifier la non-rigidité de la peau
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, List, Optional
from core.kabsch import kabsch_rotation


def ransac_kabsch_alignment(
    P_rest: np.ndarray,
    P_current: np.ndarray,
    inlier_threshold: float = 0.05,
    max_iterations: int = 40,
    min_inlier_ratio: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Estime la rotation optimale via consensus robuste RANSAC (RANdom SAmple Consensus).
    Rejette les marqueurs aberrants (outliers) causés par les occlusions ou glissements de peau.
    
    Paramètres:
        P_rest: ndarray (N, 3), N >= 3
        P_current: ndarray (N, 3), N >= 3
        inlier_threshold: distance euclidienne maximale pour être considéré inlier
        max_iterations: nombre d'échantillonnages aléatoires
        min_inlier_ratio: proportion minimale requise d'inliers
        
    Retourne:
        R_best: ndarray (3, 3) matrice de rotation optimale
        inlier_mask: ndarray (N,) booléen indiquant les points inliers
        rms_inliers: float, erreur résiduelle RMS sur les inliers
        n_inliers: int, nombre d'inliers validés
    """
    A = np.asarray(P_rest, dtype=np.float64)
    B = np.asarray(P_current, dtype=np.float64)
    n = A.shape[0]

    if n < 3:
        raise ValueError(f"RANSAC requiert au moins 3 points, reçu {n}")

    # Si 3 ou 4 points seulement, Kabsch standard direct
    if n <= 3:
        R, _, _, rms, _ = kabsch_rotation(A, B)
        return R, np.ones(n, dtype=bool), rms, n

    best_inlier_count = 0
    best_inlier_mask = np.ones(n, dtype=bool)
    best_R = np.eye(3, dtype=np.float64)
    best_rms = float("inf")

    for _ in range(max_iterations):
        # 1. Échantillonnage minimal de 3 points
        sample_indices = np.random.choice(n, size=3, replace=False)
        pts_A_sample = A[sample_indices]
        pts_B_sample = B[sample_indices]

        try:
            R_cand, cent_A_s, cent_B_s, _, _ = kabsch_rotation(pts_A_sample, pts_B_sample)
        except ValueError:
            continue

        # 2. Translation candidate issue uniquement de l'échantillon sain
        t_cand = cent_B_s - cent_A_s @ R_cand.T

        # 3. Prédiction globale sur l'ensemble des N points
        B_pred = A @ R_cand.T + t_cand
        residuals = np.linalg.norm(B - B_pred, axis=1)

        # 4. Décompte du consensus
        inliers = residuals < inlier_threshold
        inlier_count = int(np.sum(inliers))

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inlier_mask = inliers
            best_R = R_cand
            best_rms = float(np.sqrt(np.mean(residuals[inliers] ** 2))) if inlier_count > 0 else float("inf")

            if inlier_count == n:
                break

    # 5. Raffinement final : ré-estimation par Kabsch sur tous les inliers trouvés
    if best_inlier_count >= 3:
        try:
            A_inliers = A[best_inlier_mask]
            B_inliers = B[best_inlier_mask]
            best_R, cent_A_in, cent_B_in, _, _ = kabsch_rotation(A_inliers, B_inliers)
            t_final = cent_B_in - cent_A_in @ best_R.T
            diff = B_inliers - (A_inliers @ best_R.T + t_final)
            best_rms = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
        except ValueError:
            pass

    return best_R, best_inlier_mask, best_rms, best_inlier_count


def compute_green_lagrange_strain(
    P_rest: np.ndarray,
    P_current: np.ndarray,
    regularization: float = 1e-10,
) -> Tuple[np.ndarray, float]:
    """
    Calcule le tenseur de déformation finie de Green-Lagrange :
    E = 0.5 * (F^T @ F - I_3)
    
    Où F est le tenseur gradient de déformation affine entre P_rest et P_current.
    
    Retourne:
        E: ndarray (3, 3) tenseur symétrique de Green-Lagrange
        strain_energy: float, norme de Frobenius ||E||_F quantifiant l'étirement/compression non rigide
    """
    A = np.asarray(P_rest, dtype=np.float64)
    B = np.asarray(P_current, dtype=np.float64)

    if A.shape[0] < 3:
        return np.zeros((3, 3), dtype=np.float64), 0.0

    A_c = A - A.mean(axis=0)
    B_c = B - B.mean(axis=0)

    cov_AA = A_c.T @ A_c + regularization * np.eye(3)
    cov_AB = A_c.T @ B_c

    try:
        Ft = np.linalg.solve(cov_AA, cov_AB)
        F = Ft.T
    except np.linalg.LinAlgError:
        return np.zeros((3, 3), dtype=np.float64), 0.0

    C = F.T @ F
    E = 0.5 * (C - np.eye(3, dtype=np.float64))
    strain_energy = float(np.linalg.norm(E, ord="fro"))

    return E, strain_energy
