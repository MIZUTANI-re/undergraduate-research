# Run all four variants sequentially with the same CLI options.
# Examples:
#   python run_all_ablation_optuna.py --mode search --n-trials 30
#   python run_all_ablation_optuna.py --mode final --seeds 42 123 2026
#   python run_all_ablation_optuna.py --mode search_and_final --n-trials 30

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "01_Full_BiFiLM_HQ_Residual_Optuna.py",
    "02_NoFiLM_SimpleConcat_HQ_Residual_Optuna.py",
    "03_BiFiLM_NoHQ_Residual_Optuna.py",
    "04_BiFiLM_HQ_NoResidual_Optuna.py",
]


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", default="search")
    known, remaining = parser.parse_known_args()

    folder = Path(__file__).resolve().parent
    forwarded = ["--mode", known.mode, *remaining]
    for script in SCRIPTS:
        command = [sys.executable, str(folder / script), *forwarded]
        print("\n" + "#" * 100)
        print("RUN:", " ".join(command))
        print("#" * 100)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
