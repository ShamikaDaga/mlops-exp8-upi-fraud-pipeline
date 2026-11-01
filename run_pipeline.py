"""Run the whole ML pipeline (or a single stage) from the project root.

    python run_pipeline.py                 # etl -> eda -> train -> evaluate -> shap
    python run_pipeline.py --stage train   # one stage only
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

STAGES = {
    "etl": [sys.executable, "-m", "src.pipeline"],
    "eda": [sys.executable, "-m", "eda.eda_plots"],
    "train": [sys.executable, "-m", "models.train_models"],
    "evaluate": [sys.executable, "-m", "evaluation.evaluate"],
    "shap": [sys.executable, str(Path("shap") / "shap_explain.py")],
}


def main():
    parser = argparse.ArgumentParser(description="UPI fraud ML pipeline")
    parser.add_argument("--stage", choices=["all", *STAGES], default="all")
    args = parser.parse_args()
    selected = STAGES if args.stage == "all" else {args.stage: STAGES[args.stage]}
    for name, command in selected.items():
        print(f"\n{'=' * 70}\n STAGE: {name}\n{'=' * 70}", flush=True)
        subprocess.run(command, cwd=ROOT_DIR, check=True)
    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
