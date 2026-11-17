"""Exemple prêt à l'emploi : Retargeting BVH avec mark-target
=============================================================
Démontre comment lancer mark-target sur un fichier BVH.
Usage :
    python examples/run_bvh_example.py
"""

import os
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
SAMPLE_BVH = os.path.join(SCRIPT_DIR, "sample_walk.bvh")
OUTPUT_BVH = os.path.join(SCRIPT_DIR, "output_retargeted.bvh")

def main():
    print("=" * 60)
    print("🎬 LANCEMENT DE LA DÉMO BVH MARK-TARGET")
    print(f"Fichier source : {SAMPLE_BVH}")
    print(f"Fichier cible  : {OUTPUT_BVH}")
    print("=" * 60)

    cmd = [
        sys.executable,
        os.path.join(ROOT_DIR, "run_marker_retarget.py"),
        "--source", SAMPLE_BVH,
        "--source-type", "bvh",
        "--target", SAMPLE_BVH,
        "--target-type", "bvh",
        "--output", OUTPUT_BVH,
        "--use-umeyama",
        "--smoothing-factor", "0.3",
        "--use-ransac",
    ]

    print("Commande exécutée :\n" + " ".join(cmd) + "\n")
    ret = subprocess.run(cmd)
    if ret.returncode == 0:
        print("\n✅ Animation BVH retargetée avec succès !")
        print(f"Fichier généré : {OUTPUT_BVH}")
    else:
        print(f"\n❌ Erreur lors de l'exécution (code {ret.returncode}).")

if __name__ == "__main__":
    main()
