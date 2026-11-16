"""CLI Simplifiée et Moderne pour mark-target
===========================================
Interface universelle en ligne de commande :
- target <source> <target> : Retargeting direct avec tous les filtres optimaux activés par défaut
- target demo              : Démonstration instantanée 1-clic (a_out_precise.bvh -> clara.glb)
- target view              : Ouvre l'observateur 3D interactif WebGL dans le navigateur
- target bench             : Lance le benchmark scientifique réel (CPU SVD, FPS, RANSAC, SQUAD)
- target test              : Lance la suite de tests automatisés (pytest)
"""

from __future__ import annotations

import argparse
import os
import sys
import webbrowser
import subprocess
from pathlib import Path


def infer_format(path_str: str, is_source: bool = False) -> str:
    """Déduit automatiquement le type de format à partir de l'extension de fichier."""
    ext = Path(path_str).suffix.lower()
    if ext == ".bvh":
        return "bvh"
    elif ext == ".pk":
        return "pk"
    elif ext in [".glb", ".gltf"]:
        if is_source:
            raise ValueError(
                f"Le format '{ext}' n'est actuellement supporté qu'en cible (avatar 3D), pas en source d'animation. "
                "Formats sources d'animation supportés : .bvh (MoCap), .pk (Vision 3D / HybrIK)."
            )
        return "mixamo"
    elif ext == ".vrm":
        if is_source:
            raise ValueError(
                f"Le format '{ext}' n'est actuellement supporté qu'en cible (avatar VRoid), pas en source d'animation. "
                "Formats sources d'animation supportés : .bvh (MoCap), .pk (Vision 3D / HybrIK)."
            )
        return "vrm"
    raise ValueError(f"Extension inconnue '{ext}' pour '{path_str}'. Formats supportés : .bvh, .pk (source) / .glb, .vrm, .bvh (cible).")


def cmd_retarget(args, extra_args: list[str] | None = None):
    """Exécute le retargeting en inférant les formats et en appliquant les meilleurs paramètres par défaut."""
    source_path = args.source
    target_path = args.target

    if not os.path.exists(source_path):
        print(f"❌ Erreur : Fichier source introuvable : '{source_path}'")
        sys.exit(1)
    if not os.path.exists(target_path):
        print(f"❌ Erreur : Fichier cible introuvable : '{target_path}'")
        sys.exit(1)

    # 1. Déduction automatique des formats
    try:
        source_type = args.source_type or infer_format(source_path, is_source=True)
        target_type = args.target_type or infer_format(target_path, is_source=False)
    except ValueError as err:
        print(f"❌ Erreur : {err}")
        sys.exit(1)

    # 2. Déduction automatique du fichier de sortie (dans le même dossier que la cible)
    if args.output:
        output_path = args.output
    else:
        src_stem = Path(source_path).stem
        tgt_stem = Path(target_path).stem
        tgt_ext = Path(target_path).suffix
        tgt_parent = Path(target_path).parent
        output_path = str(tgt_parent / f"{tgt_stem}_{src_stem}_retargeted{tgt_ext}")

    print("=" * 70)
    print("🚀 MARK-TARGET : RETARGETING UNIVERSEL EN COURS")
    print("=" * 70)
    print(f"📁 Source       : {source_path} (format détecté: {source_type})")
    print(f"📁 Cible        : {target_path} (format détecté: {target_type})")
    print(f"🎯 Sortie       : {output_path}")
    print(f"⚡ Paramètres   : auto-map={not args.no_automap}, cages={not args.no_cages}, ransac={not args.no_ransac}, temporal={not args.no_temporal}, smoothing={args.smoothing}")
    if extra_args:
        print(f"➕ Options supp : {' '.join(extra_args)}")
    print("-" * 70)
    sys.stdout.flush()

    # Construction des arguments pour run_marker_retarget
    script_path = os.path.join(os.path.dirname(__file__), "run_marker_retarget.py")
    cmd = [
        sys.executable,
        script_path,
        "--source", source_path,
        "--source-type", source_type,
        "--target", target_path,
        "--target-type", target_type,
        "--output", output_path,
    ]

    if not args.no_automap:
        cmd.append("--auto-map")
    if not args.no_cages:
        cmd.append("--use-virtual-cages")
    if not args.no_temporal:
        cmd.append("--use-temporal-prior")
    if not args.no_ransac:
        cmd.append("--use-ransac")
    if args.smoothing > 0.0:
        cmd.extend(["--smoothing-factor", str(args.smoothing)])

    # Transmission des arguments supplémentaires éventuels
    if extra_args:
        cmd.extend(extra_args)

    try:
        ret = subprocess.run(cmd)
        if ret.returncode == 0:
            print("=" * 70)
            print(f"✅ SUCCÈS ! Animation générée dans : {output_path}")
            print("=" * 70)
            if args.view:
                cmd_view(args)
        else:
            sys.exit(ret.returncode)
    except KeyboardInterrupt:
        print("\n🛑 Opération interrompue par l'utilisateur.")
        sys.exit(130)


def cmd_demo(args):
    """Lance la démonstration rapide 1-clic."""
    print("🚀 Lancement de la démonstration 1-clic...")
    quickstart_path = os.path.join(os.path.dirname(__file__), "quickstart.py")
    try:
        ret = subprocess.run([sys.executable, quickstart_path])
        if ret.returncode == 0:
            if not args.no_view:
                cmd_view(args)
        else:
            sys.exit(ret.returncode)
    except KeyboardInterrupt:
        print("\n🛑 Démonstration interrompue.")
        sys.exit(130)


def cmd_view(args=None):
    """Ouvre le visualiseur 3D WebGL dans le navigateur."""
    viewer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "visualizer", "marker_viewer.html"))
    if not os.path.exists(viewer_path):
        print(f"❌ Visualiseur introuvable : {viewer_path}")
        return
    url = f"file:///{viewer_path.replace(os.sep, '/')}"
    print(f"🔭 Ouverture du visualiseur 3D : {url}")
    webbrowser.open(url)


def cmd_bench(args=None):
    """Lance le banc d'évaluation comparatif réel."""
    bench_script = os.path.join(os.path.dirname(__file__), "benchmark", "run_comparative_benchmark.py")
    try:
        ret = subprocess.run([sys.executable, bench_script])
        sys.exit(ret.returncode)
    except KeyboardInterrupt:
        print("\n🛑 Benchmark interrompu.")
        sys.exit(130)


def cmd_test(args=None):
    """Lance la suite complète de 52 tests."""
    print("🧪 Exécution des 52 tests de non-régression...")
    try:
        ret = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])
        sys.exit(ret.returncode)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrompus.")
        sys.exit(130)


def main():
    valid_subcommands = {"retarget", "demo", "view", "bench", "test"}
    is_help_only = len(sys.argv) == 2 and sys.argv[1] in ("-h", "--help")
    has_subcmd = any(arg in valid_subcommands for arg in sys.argv[1:])

    # Si aucun nom de sous-commande réservée n'est spécifié et que ce n'est pas --help seul,
    # on insère automatiquement 'retarget' à l'indice 1, même si des options (-v, -o) le précèdent.
    if len(sys.argv) > 1 and not has_subcmd and not is_help_only:
        sys.argv.insert(1, "retarget")

    parser = argparse.ArgumentParser(
        prog="target",
        description="Moteur Universel de Retargeting d'Animation 3D (100% CPU en forme fermée SVD).",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commandes disponibles")

    # 1. retarget
    p_retarget = subparsers.add_parser("retarget", help="Transfère une animation source vers un avatar 3D")
    p_retarget.add_argument("source", help="Fichier source d'animation (.bvh, .pk)")
    p_retarget.add_argument("target", help="Avatar 3D cible (.glb, .vrm, .bvh)")
    p_retarget.add_argument("-o", "--output", default=None, help="Chemin du fichier de sortie (déduit automatiquement si omis)")
    p_retarget.add_argument("--source-type", choices=["bvh", "pk"], default=None, help="Type source (déduit automatiquement de l'extension)")
    p_retarget.add_argument("--target-type", choices=["mixamo", "vrm", "bvh"], default=None, help="Type cible (déduit automatiquement de l'extension)")
    p_retarget.add_argument("--smoothing", type=float, default=0.35, help="Facteur de lissage temporel SQUAD (défaut: 0.35)")
    p_retarget.add_argument("--no-automap", action="store_true", help="Désactive l'auto-mapping et l'alignement A-Pose/T-Pose")
    p_retarget.add_argument("--no-cages", action="store_true", help="Désactive la génération de cages 4 marqueurs")
    p_retarget.add_argument("--no-temporal", action="store_true", help="Désactive le verrouillage temporel S3")
    p_retarget.add_argument("--no-ransac", action="store_true", help="Désactive le filtre robuste RANSAC")
    p_retarget.add_argument("-v", "--view", action="store_true", help="Ouvre le visualiseur 3D dès la fin du retargeting")

    # 2. demo / quickstart
    p_demo = subparsers.add_parser("demo", help="Lance la démonstration en 1 clic sur les assets locaux")
    p_demo.add_argument("--no-view", action="store_true", help="Ne pas ouvrir le navigateur à la fin")
    p_demo.set_defaults(func=cmd_demo)

    # 3. view
    p_view = subparsers.add_parser("view", help="Ouvre le visualiseur WebGL 3D dans le navigateur")
    p_view.set_defaults(func=cmd_view)

    # 4. bench
    p_bench = subparsers.add_parser("bench", help="Lance le benchmark scientifique mesuré sur CPU")
    p_bench.set_defaults(func=cmd_bench)

    # 5. test
    p_test = subparsers.add_parser("test", help="Lance la suite de 52 tests unitaires (pytest)")
    p_test.set_defaults(func=cmd_test)

    # On utilise parse_known_args pour accepter les arguments supplémentaires passés au moteur
    args, unknown = parser.parse_known_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "retarget":
        cmd_retarget(args, extra_args=unknown)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
