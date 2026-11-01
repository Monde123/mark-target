"""
Tests pour markers.observer
===========================
Vérifie le calcul des métriques cinématiques (vitesses, accélérations),
l'analyse de rigidité et la génération du payload d'observation WebGL.
"""

import os
import json
import numpy as np

from markers.observer import MarkerObserver


class TestMarkerObserver:

    def test_rigidity_analysis(self):
        """Vérifie que l'observateur détecte correctement les cages rigides vs déformées."""
        parents = {"boneA": None}
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([0.0, 1.0, 0.0])
        p2 = np.array([0.2, 0.5, 0.0])
        p3 = np.array([0.0, 0.5, 0.15])
        rest = {"boneA": np.array([p0, p1, p2, p3])}

        # Frame 1: translation pure (rigide)
        f1 = {"boneA": rest["boneA"] + np.array([1.0, 0.0, 0.0])}
        # Frame 2: rotation pure 90 deg autour de Z (rigide)
        R_z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        f2 = {"boneA": (R_z @ rest["boneA"].T).T}

        observer = MarkerObserver(parents, rest, [f1, f2], fps=30.0)
        rigidity = observer.analyze_rigidity()

        assert "boneA" in rigidity
        assert rigidity["boneA"]["is_rigid"] is True
        assert rigidity["boneA"]["max_rigidity_deviation"] < 1e-12

    def test_kinematics_computation(self):
        """Vérifie le calcul des vitesses et accélérations scalaires."""
        parents = {"boneA": None}
        rest = {"boneA": np.zeros((4, 3))}

        # Déplacement à vitesse constante de 3 m/s le long de X à 30 fps
        # dx par frame = 3.0 / 30 = 0.1 m
        fps = 30.0
        frames = []
        for i in range(10):
            pos = np.zeros((4, 3))
            pos[0] = np.array([i * 0.1, 0.0, 0.0])
            frames.append({"boneA": pos})

        observer = MarkerObserver(parents, rest, frames, fps=fps)
        kin = observer.compute_kinematics()

        assert "boneA" in kin
        # Vitesse théorique = 3.0 m/s
        np.testing.assert_allclose(kin["boneA"]["mean_speed"], 3.0, atol=1e-5)
        np.testing.assert_allclose(kin["boneA"]["max_speed"], 3.0, atol=1e-5)
        # Accélération à vitesse constante = 0.0 m/s²
        np.testing.assert_allclose(kin["boneA"]["max_accel"], 0.0, atol=1e-5)

    def test_compute_frames_axes(self):
        """Vérifie que les axes extraits forment bien une base orthonormée (X, Y, Z)."""
        parents = {"boneA": None}
        rest = {
            "boneA": np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.2, 0.0],
                [0.5, 0.0, 0.15],
            ])
        }
        frames = [rest]

        observer = MarkerObserver(parents, rest, frames, fps=30.0)
        axes = observer.compute_frames_axes()

        assert len(axes) == 1
        ax = axes[0]["boneA"]
        ux = np.array(ax["axis_x"])
        uy = np.array(ax["axis_y"])
        uz = np.array(ax["axis_z"])

        # Vérification d'orthonormalité
        np.testing.assert_allclose(np.linalg.norm(ux), 1.0, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(uy), 1.0, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(uz), 1.0, atol=1e-6)
        np.testing.assert_allclose(np.dot(ux, uy), 0.0, atol=1e-6)
        np.testing.assert_allclose(np.dot(ux, uz), 0.0, atol=1e-6)
        np.testing.assert_allclose(np.dot(uy, uz), 0.0, atol=1e-6)

    def test_export_observation_json(self, tmp_path):
        """Vérifie l'exportation et la structure JSON du payload de diagnostic."""
        parents = {"root": None, "child": "root"}
        rest = {
            "root": np.zeros((4, 3)),
            "child": np.ones((4, 3)),
        }
        frames = [rest, rest]

        observer = MarkerObserver(parents, rest, frames, fps=30.0)
        out_json = str(tmp_path / "test_diag.json")
        observer.export_observation_json(out_json, source_name="TestRig", target_name="TargetRig")

        assert os.path.exists(out_json)
        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["metadata"]["source_name"] == "TestRig"
        assert data["metadata"]["frame_count"] == 2
        assert "bones_hierarchy" in data
        assert "frames" in data
        assert "axes" in data
        assert "diagnostics" in data
