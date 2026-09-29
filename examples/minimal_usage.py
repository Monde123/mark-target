"""
Exemple d'utilisation minimal et autonome du module `isolate`.
Ne nécessite que Python et NumPy. Aucune donnée externe requise.
"""
import numpy as np

from core.geometry import quat_identity, quaternion_to_matrix
from retargeting.solver import MarkerRetargetSolver
from retargeting.kinematics import global_to_local_hierarchy


def run_minimal_example():
    print("=== Démonstration autonome du solveur isolate ===")

    # 1. Définition d'un rig squelettique simple (Hips -> Chest -> Arm -> ForeArm)
    parents = {
        "Hips": None,
        "Chest": "Hips",
        "Arm": "Chest",
        "ForeArm": "Arm",
    }
    rest_rotations = {bone: quat_identity() for bone in parents}

    # 2. Instanciation du solveur
    solver = MarkerRetargetSolver(rest_rotations, parents)

    # 3. Définition synthétique de marqueurs 3D en pose de repos (4 marqueurs par segment)
    #    (Un tétraèdre non colinéaire par os)
    tetrahedron_template = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.0, 0.0, 0.1],
    ], dtype=np.float64)

    positions_rest = {
        "Hips": tetrahedron_template + np.array([0.0, 1.0, 0.0]),
        "Chest": tetrahedron_template + np.array([0.0, 1.3, 0.0]),
        "Arm": tetrahedron_template + np.array([0.2, 1.4, 0.0]),
        "ForeArm": tetrahedron_template + np.array([0.5, 1.4, 0.0]),
    }

    # 4. Simulation d'un mouvement à l'instant t :
    #    Rotation de 45° sur l'Arm et 30° sur le ForeArm
    theta_arm = np.radians(45.0)
    R_arm = np.array([
        [np.cos(theta_arm), -np.sin(theta_arm), 0.0],
        [np.sin(theta_arm),  np.cos(theta_arm), 0.0],
        [0.0,                0.0,               1.0],
    ])

    positions_t = {
        "Hips": positions_rest["Hips"].copy(),
        "Chest": positions_rest["Chest"].copy(),
        "Arm": (R_arm @ positions_rest["Arm"].T).T,
        "ForeArm": (R_arm @ positions_rest["ForeArm"].T).T,
    }

    # 5. Résolution par Kabsch
    Q_global, diagnostics = solver.solve_frame_multi_kabsch(positions_t, positions_rest)
    print("\n[Statut de résolution]")
    for bone, diag in diagnostics.items():
        print(f"  - {bone:8s} : {diag.get('status')}")

    # 6. Extraction des rotations locales prêtes pour le moteur 3D
    Q_local = global_to_local_hierarchy(Q_global, parents)
    print("\n[Rotations locales calculées (quaternions [w, x, y, z])]")
    for bone, q in Q_local.items():
        print(f"  - {bone:8s} : [{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}]")

    print("\n=== Fin de la démonstration avec succès ===")


if __name__ == "__main__":
    run_minimal_example()
