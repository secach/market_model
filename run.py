from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PipelineStep:
    name: str
    script: Path


PIPELINE_STEPS = [
    PipelineStep("Download dual-listed US returns", BASE_DIR / "src" / "download_dual_yahoo.py"),
    PipelineStep("Merge Yahoo returns with dual weights", BASE_DIR / "src" / "merge_dual_data.py"),
    PipelineStep("Build dual daily features", BASE_DIR / "src" / "dual_daily_features.py"),
    PipelineStep("Build TA-35 model dataset", BASE_DIR / "src" / "build_ta35_model_data.py"),
]


def run_script(step: PipelineStep) -> None:
    if not step.script.exists():
        raise FileNotFoundError(f"Missing script for step '{step.name}': {step.script}")

    print(f"\n=== {step.name} ===", flush=True)
    subprocess.run([sys.executable, str(step.script)], cwd=BASE_DIR, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the TA-35 dual-listed data pipeline."
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use the existing data/yahoo_dual_returns.csv instead of downloading from Yahoo.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    steps = PIPELINE_STEPS
    if args.skip_download:
        steps = [step for step in steps if step.script.name != "download_dual_yahoo.py"]

    for step in steps:
        run_script(step)

    print("\nPipeline completed.", flush=True)


if __name__ == "__main__":
    main()
