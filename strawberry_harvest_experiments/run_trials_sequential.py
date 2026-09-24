"""Launch strawberry trials strictly one at a time.

Run this file with the same Isaac Sim Python command used for
harvest_pipeline.py.  subprocess.run() is intentionally blocking, so trial N+1
cannot start until trial N has fully closed its SimulationApp process.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from trial_configs import available_trial_ids


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Isaac Sim harvesting trials sequentially, never concurrently."
    )
    parser.add_argument(
        "--trials",
        type=int,
        nargs="+",
        choices=available_trial_ids(),
        default=available_trial_ids(),
        help="Trial IDs to run in order. Default: all ten trials.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory shared by all trial result files.",
    )
    parser.add_argument(
        "--show-ui",
        action="store_true",
        help="Show the Isaac Sim window. Batch mode is headless by default.",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=3.0,
        help="Pause after each process exits before starting the next trial.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to later trials if one Isaac Sim process crashes.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    pipeline_path = project_dir / "harvest_pipeline.py"
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = (project_dir / results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    statuses = []
    print("Trials will run strictly sequentially.")
    print(f"Trial order: {args.trials}")
    print(f"Python executable: {sys.executable}")

    for sequence_index, trial_id in enumerate(args.trials, start=1):
        print("=" * 80)
        print(
            f"Starting trial {trial_id:02d} "
            f"({sequence_index}/{len(args.trials)})"
        )

        command = [
            sys.executable,
            str(pipeline_path),
            "--trial-id",
            str(trial_id),
            "--auto-close",
            "--results-dir",
            str(results_dir),
        ]
        if not args.show_ui:
            command.append("--headless")

        start_time = time.perf_counter()
        completed = subprocess.run(command, cwd=project_dir, check=False)
        elapsed = time.perf_counter() - start_time
        status = {
            "trial_id": trial_id,
            "return_code": completed.returncode,
            "launcher_wall_time_s": round(elapsed, 6),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        statuses.append(status)

        print(
            f"Trial {trial_id:02d} process exited with code "
            f"{completed.returncode} after {elapsed:.1f}s."
        )

        status_path = results_dir / "batch_status.json"
        status_path.write_text(json.dumps(statuses, indent=2), encoding="utf-8")

        if completed.returncode != 0 and not args.continue_on_error:
            print("Stopping batch because a trial process crashed.")
            print("Use --continue-on-error only if later trials should still run.")
            return completed.returncode

        if sequence_index < len(args.trials) and args.cooldown_seconds > 0:
            print(
                f"Cooling down for {args.cooldown_seconds:.1f}s before "
                "the next process."
            )
            time.sleep(args.cooldown_seconds)

    print("=" * 80)
    print("All requested trials finished. No trials were run concurrently.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
