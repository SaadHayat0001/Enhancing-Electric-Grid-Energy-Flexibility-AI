"""Master end-to-end reproducibility runner for the Three-Layer AI-Physics Framework.

Executes:
  1. pandapower/sim1_layer1_forecasting.py
  2. pandapower/sim2_layer2_bess_rainflow.py
  3. pandapower/sim3_layer3_pigat_fvsi.py
  4. pandapower/sim4_closed_loop_acopf.py

Usage:
  python run_all.py --fast    # Quick verification run (2 seeds, reduced epochs/LHS states)
  python run_all.py           # Full dissertation 10-seed reproduction (GPU recommended)
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all four dissertation simulation scripts sequentially.")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--fast", action="store_true", help="Run reduced-budget verification mode.")
    args = parser.parse_args()

    scripts = [
        ("Layer 1: Multi-Horizon Forecasting & TreeSHAP", REPO_ROOT / "pandapower/sim1_layer1_forecasting.py", True),
        ("Layer 2: LSTM BESS, Rainflow & IEEE 33-Bus", REPO_ROOT / "pandapower/sim2_layer2_bess_rainflow.py", True),
        ("Layer 3a: Leak-Free PI-GAT & TGCN Screening", REPO_ROOT / "pandapower/sim3_layer3_pigat_fvsi.py", False),
        ("Layer 3b: Closed-Loop AC OPF & Multi-Bus Scaling", REPO_ROOT / "pandapower/sim4_closed_loop_acopf.py", True),
    ]

    for title, script_path, takes_data_dir in scripts:
        print(f"\n{'=' * 72}\nRunning {title}\nScript: {script_path.relative_to(REPO_ROOT)}\n{'=' * 72}")
        cmd = [sys.executable, str(script_path), "--out-dir", str(args.out_dir)]
        if takes_data_dir:
            cmd.extend(["--data-dir", str(args.data_dir)])
        if args.fast:
            cmd.append("--fast")
        subprocess.run(cmd, check=True)

    print(f"\nAll 4 simulation scripts completed cleanly. Outputs saved to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
