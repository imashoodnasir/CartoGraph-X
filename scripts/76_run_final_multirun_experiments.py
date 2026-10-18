#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
76_run_final_multirun_experiments.py

CartoGen — Final Multi-Run Experiment Orchestrator

PURPOSE
-------
Run the final repeated experimental pipeline without changing the individual
modelling scripts.

The orchestrator:
    * executes existing experiment scripts as subprocesses;
    * streams output to the terminal;
    * saves complete logs;
    * creates checkpoints only after successful completion;
    * resumes automatically after interruption;
    * skips previously completed jobs;
    * records failures without falsely marking them complete;
    * optionally continues after a failed stage;
    * validates expected output files;
    * writes machine-readable status and final reports.

IMPORTANT
---------
This script DOES NOT contain model-training methodology.
It only orchestrates existing scripts.

Default final pipeline:
    66 — classical baselines
    67 — domain-shift / transfer analysis
    69 — graph-aware baselines
    71 — CartoGraph-X

Scripts 64, 65, 68 and 70 are deterministic preparation stages and are not
repeated here.

CHECKPOINT PRINCIPLE
--------------------
A checkpoint is written only when:

    1. subprocess exits with return code 0;
    2. expected output(s) exist;
    3. expected output(s) are non-empty.

If the computer shuts down during a job, no checkpoint is created.
The job is therefore retried on the next execution.

USAGE
-----
Normal final run:

    python scripts\\76_run_final_multirun_experiments.py

Force rerun everything:

    python scripts\\76_run_final_multirun_experiments.py --force

Run selected stages only:

    python scripts\\76_run_final_multirun_experiments.py --stages 66 69 71

Show planned jobs without executing:

    python scripts\\76_run_final_multirun_experiments.py --dry-run

Stop immediately if a stage fails:

    python scripts\\76_run_final_multirun_experiments.py --stop-on-failure

NOTE ABOUT SEEDS
----------------
This orchestrator intentionally does not alter the individual scripts.

Therefore, repeated seeds are controlled by each training script's existing
"--mode full" implementation.

The checkpoint unit here is one complete script invocation.

If an individual script itself already implements per-seed resume/checkpoints,
those internal checkpoints remain effective. Script 76 adds an outer
script-level checkpoint as an additional safety layer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import traceback

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence


# ======================================================================================================================
# PATHS
# ======================================================================================================================

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

RUN_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "cross_country"
    / "final_multirun"
)

CHECKPOINT_ROOT = RUN_ROOT / "checkpoints"
LOG_ROOT = RUN_ROOT / "logs"
STATUS_ROOT = RUN_ROOT / "status"

STATUS_CSV = RUN_ROOT / "final_run_status.csv"
REPORT_JSON = RUN_ROOT / "final_run_report.json"


# ======================================================================================================================
# PIPELINE CONFIGURATION
# ======================================================================================================================

FINAL_SEEDS = list(range(42, 52))


@dataclass(frozen=True)
class Stage:
    number: str
    name: str
    script: str
    arguments: Sequence[str]
    expected_outputs: Sequence[Path]
    depends_on: Sequence[str]


STAGES: Dict[str, Stage] = {

    "66": Stage(
        number="66",
        name="Classical baselines — 10 runs",
        script="66_train_evaluate_cross_country_baselines.py",
        arguments=("--mode", "full"),
        expected_outputs=(
            ROOT
            / "data"
            / "processed"
            / "cross_country"
            / "baseline_models"
            / "all_runs_metrics.csv",

            ROOT
            / "data"
            / "processed"
            / "cross_country"
            / "baseline_models"
            / "summary_10_runs.csv",

            ROOT
            / "data"
            / "processed"
            / "cross_country"
            / "baseline_models"
            / "baseline_evaluation_report.json",
        ),
        depends_on=(),
    ),

    "67": Stage(
        number="67",
        name="Domain-shift and transfer analysis",
        script="67_analyze_cross_country_domain_shift.py",
        arguments=("--mode", "full"),
        expected_outputs=(
            ROOT
            / "data"
            / "processed"
            / "cross_country"
            / "domain_shift"
            / "domain_shift_analysis_report.json",
        ),
        depends_on=("66",),
    ),

    "69": Stage(
        number="69",
        name="Graph-aware baselines — 10 runs",
        script="69_train_evaluate_graph_aware_models.py",
        arguments=("--mode", "full"),
        expected_outputs=(
            ROOT
            / "data"
            / "processed"
            / "cross_country"
            / "graph_aware_models"
            / "all_runs_test_results.csv",
        ),
        depends_on=(),
    ),

    "71": Stage(
        number="71",
        name="CartoGraph-X — 10 runs",
        script="71_train_cartographx.py",
        arguments=("--mode", "full"),
        expected_outputs=(
            ROOT
            / "data"
            / "processed"
            / "cross_country"
            / "cartographx_models"
            / "full_all_runs.csv",

            ROOT
            / "data"
            / "processed"
            / "cross_country"
            / "cartographx_models"
            / "full_aggregate_metrics.csv",

            ROOT
            / "data"
            / "processed"
            / "cross_country"
            / "cartographx_models"
            / "full_experiment_report.json",
        ),
        depends_on=(),
    ),
}


# ======================================================================================================================
# TERMINAL HELPERS
# ======================================================================================================================

def banner(text: str) -> None:

    print()
    print("=" * 120)
    print(text)
    print("=" * 120)


def section(text: str) -> None:

    print()
    print("-" * 120)
    print(text)
    print("-" * 120)


def now_iso() -> str:

    return datetime.now().astimezone().isoformat(
        timespec="seconds"
    )


def elapsed_text(seconds: float) -> str:

    seconds = max(
        0,
        int(seconds),
    )

    hours, remainder = divmod(
        seconds,
        3600,
    )

    minutes, seconds = divmod(
        remainder,
        60,
    )

    if hours:

        return (
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{seconds:02d}"
        )

    return (
        f"{minutes:02d}:"
        f"{seconds:02d}"
    )


# ======================================================================================================================
# ARGUMENTS
# ======================================================================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "CartoGen final resumable "
            "multi-run experiment orchestrator."
        )
    )

    parser.add_argument(
        "--stages",
        nargs="+",
        choices=sorted(STAGES.keys()),
        default=None,
        help=(
            "Stages to execute. "
            "Default: 66 67 69 71."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Ignore Script-76 checkpoints "
            "and rerun selected stages."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Show the execution plan "
            "without running anything."
        ),
    )

    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help=(
            "Stop immediately when a stage fails. "
            "Default behavior records the failure "
            "and continues where dependencies allow."
        ),
    )

    parser.add_argument(
        "--python",
        default=sys.executable,
        help=(
            "Python executable used for child scripts. "
            "Default: the Python running Script 76."
        ),
    )

    return parser.parse_args()


# ======================================================================================================================
# FILE / HASH HELPERS
# ======================================================================================================================

def ensure_directories() -> None:

    for path in [
        RUN_ROOT,
        CHECKPOINT_ROOT,
        LOG_ROOT,
        STATUS_ROOT,
    ]:

        path.mkdir(
            parents=True,
            exist_ok=True,
        )


def file_sha256(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def valid_file(path: Path) -> bool:

    return (
        path.exists()
        and path.is_file()
        and path.stat().st_size > 0
    )


# ======================================================================================================================
# CHECKPOINTS
# ======================================================================================================================

def checkpoint_path(stage: Stage) -> Path:

    return (
        CHECKPOINT_ROOT
        / f"script_{stage.number}.done.json"
    )


def failure_path(stage: Stage) -> Path:

    return (
        CHECKPOINT_ROOT
        / f"script_{stage.number}.failed.json"
    )


def running_path(stage: Stage) -> Path:

    return (
        CHECKPOINT_ROOT
        / f"script_{stage.number}.running.json"
    )


def read_json(path: Path) -> Optional[dict]:

    if not path.exists():
        return None

    try:

        with path.open(
            "r",
            encoding="utf-8",
        ) as f:

            return json.load(f)

    except Exception:
        return None


def write_json_atomic(
    path: Path,
    payload: dict,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    with tmp.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False,
        )

    os.replace(
        tmp,
        path,
    )


def expected_outputs_valid(
    stage: Stage,
) -> bool:

    return all(
        valid_file(path)
        for path in stage.expected_outputs
    )


def checkpoint_valid(
    stage: Stage,
) -> bool:

    path = checkpoint_path(
        stage
    )

    checkpoint = read_json(
        path
    )

    if checkpoint is None:
        return False

    if checkpoint.get(
        "status"
    ) != "COMPLETE":

        return False

    script_path = (
        SCRIPTS
        / stage.script
    )

    if not valid_file(
        script_path
    ):

        return False

    current_hash = file_sha256(
        script_path
    )

    saved_hash = checkpoint.get(
        "script_sha256"
    )

    if saved_hash != current_hash:

        print(
            f"[STALE] Script {stage.number} "
            "changed since its checkpoint."
        )

        return False

    if not expected_outputs_valid(
        stage
    ):

        print(
            f"[STALE] Script {stage.number} "
            "checkpoint exists but one or more "
            "expected outputs are missing/empty."
        )

        return False

    return True


def write_running_marker(
    stage: Stage,
    command: Sequence[str],
    log_path: Path,
) -> None:

    payload = {
        "stage":
            stage.number,

        "name":
            stage.name,

        "status":
            "RUNNING",

        "started_at":
            now_iso(),

        "command":
            list(command),

        "log":
            str(log_path),
    }

    write_json_atomic(
        running_path(stage),
        payload,
    )


def write_success_checkpoint(
    stage: Stage,
    command: Sequence[str],
    log_path: Path,
    started_at: str,
    runtime_seconds: float,
) -> None:

    script_path = (
        SCRIPTS
        / stage.script
    )

    outputs = []

    for path in stage.expected_outputs:

        outputs.append(
            {
                "path":
                    str(path),

                "exists":
                    path.exists(),

                "size_bytes":
                    (
                        path.stat().st_size
                        if path.exists()
                        else 0
                    ),
            }
        )

    payload = {
        "stage":
            stage.number,

        "name":
            stage.name,

        "status":
            "COMPLETE",

        "started_at":
            started_at,

        "completed_at":
            now_iso(),

        "runtime_seconds":
            runtime_seconds,

        "command":
            list(command),

        "python":
            command[0],

        "script":
            str(script_path),

        "script_sha256":
            file_sha256(
                script_path
            ),

        "expected_outputs":
            outputs,

        "log":
            str(log_path),

        "final_seeds":
            FINAL_SEEDS,
    }

    write_json_atomic(
        checkpoint_path(stage),
        payload,
    )

    failure_path(
        stage
    ).unlink(
        missing_ok=True
    )

    running_path(
        stage
    ).unlink(
        missing_ok=True
    )


def write_failure_checkpoint(
    stage: Stage,
    command: Sequence[str],
    log_path: Path,
    started_at: str,
    runtime_seconds: float,
    return_code: Optional[int],
    reason: str,
) -> None:

    payload = {
        "stage":
            stage.number,

        "name":
            stage.name,

        "status":
            "FAILED",

        "started_at":
            started_at,

        "failed_at":
            now_iso(),

        "runtime_seconds":
            runtime_seconds,

        "return_code":
            return_code,

        "reason":
            reason,

        "command":
            list(command),

        "log":
            str(log_path),
    }

    write_json_atomic(
        failure_path(stage),
        payload,
    )

    running_path(
        stage
    ).unlink(
        missing_ok=True
    )


# ======================================================================================================================
# LOGGING
# ======================================================================================================================

def make_log_path(
    stage: Stage,
) -> Path:

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    return (
        LOG_ROOT
        / (
            f"script_{stage.number}_"
            f"{timestamp}.log"
        )
    )


def stream_subprocess(
    command: Sequence[str],
    cwd: Path,
    log_path: Path,
) -> int:

    env = os.environ.copy()

    # Unbuffered Python output from child processes.
    env["PYTHONUNBUFFERED"] = "1"

    creationflags = 0

    if (
        os.name == "nt"
        and hasattr(
            subprocess,
            "CREATE_NEW_PROCESS_GROUP",
        )
    ):

        creationflags = (
            subprocess.CREATE_NEW_PROCESS_GROUP
        )

    process = subprocess.Popen(
        list(command),
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
        creationflags=creationflags,
    )

    try:

        with log_path.open(
            "w",
            encoding="utf-8",
            buffering=1,
        ) as log:

            header = (
                f"CartoGen Script 76 subprocess log\n"
                f"Started: {now_iso()}\n"
                f"Working directory: {cwd}\n"
                f"Command: {' '.join(command)}\n"
                + "=" * 120
                + "\n"
            )

            log.write(
                header
            )

            assert (
                process.stdout
                is not None
            )

            for line in process.stdout:

                print(
                    line,
                    end="",
                    flush=True,
                )

                log.write(
                    line
                )

            return_code = (
                process.wait()
            )

            log.write(
                "\n"
                + "=" * 120
                + "\n"
            )

            log.write(
                f"Finished: {now_iso()}\n"
            )

            log.write(
                f"Return code: {return_code}\n"
            )

            return return_code

    except KeyboardInterrupt:

        print(
            "\n[INTERRUPT] Ctrl+C received."
        )

        try:

            if os.name == "nt":

                process.send_signal(
                    signal.CTRL_BREAK_EVENT
                )

                time.sleep(
                    1
                )

            if (
                process.poll()
                is None
            ):

                process.terminate()

                try:

                    process.wait(
                        timeout=10
                    )

                except subprocess.TimeoutExpired:

                    process.kill()

        except Exception:

            try:
                process.kill()
            except Exception:
                pass

        raise


# ======================================================================================================================
# STATUS TABLE
# ======================================================================================================================

STATUS_COLUMNS = [
    "stage",
    "name",
    "status",
    "started_at",
    "finished_at",
    "runtime_seconds",
    "return_code",
    "checkpoint",
    "log",
    "message",
]


def read_status_rows() -> List[dict]:

    if not STATUS_CSV.exists():
        return []

    try:

        return pd_read_csv_without_pandas(
            STATUS_CSV
        )

    except Exception:
        return []


def pd_read_csv_without_pandas(
    path: Path,
) -> List[dict]:

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        return list(
            csv.DictReader(f)
        )


def write_status_rows(
    rows: List[dict],
) -> None:

    tmp = STATUS_CSV.with_suffix(
        ".csv.tmp"
    )

    with tmp.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=STATUS_COLUMNS,
        )

        writer.writeheader()

        for row in rows:

            writer.writerow(
                {
                    key:
                        row.get(
                            key,
                            "",
                        )
                    for key
                    in STATUS_COLUMNS
                }
            )

    os.replace(
        tmp,
        STATUS_CSV,
    )


def update_status(
    stage: Stage,
    status: str,
    started_at: str = "",
    finished_at: str = "",
    runtime_seconds: object = "",
    return_code: object = "",
    log: object = "",
    message: str = "",
) -> None:

    rows = read_status_rows()

    rows = [
        row
        for row in rows
        if row.get("stage")
        != stage.number
    ]

    rows.append(
        {
            "stage":
                stage.number,

            "name":
                stage.name,

            "status":
                status,

            "started_at":
                started_at,

            "finished_at":
                finished_at,

            "runtime_seconds":
                runtime_seconds,

            "return_code":
                return_code,

            "checkpoint":
                str(
                    checkpoint_path(
                        stage
                    )
                ),

            "log":
                str(log),

            "message":
                message,
        }
    )

    rows.sort(
        key=lambda x:
            int(
                x["stage"]
            )
    )

    write_status_rows(
        rows
    )


# ======================================================================================================================
# DEPENDENCIES
# ======================================================================================================================

def dependency_satisfied(
    dependency_number: str,
    selected_stage_numbers: Sequence[str],
) -> bool:

    dependency = STAGES[
        dependency_number
    ]

    # If dependency was selected for this invocation,
    # require its checkpoint.
    if (
        dependency_number
        in selected_stage_numbers
    ):

        return checkpoint_valid(
            dependency
        )

    # If not selected, accept either:
    #   * valid Script-76 checkpoint, or
    #   * expected outputs already existing.
    return (
        checkpoint_valid(
            dependency
        )
        or expected_outputs_valid(
            dependency
        )
    )


# ======================================================================================================================
# STAGE EXECUTION
# ======================================================================================================================

def run_stage(
    stage: Stage,
    python_executable: str,
    force: bool,
    dry_run: bool,
) -> dict:

    script_path = (
        SCRIPTS
        / stage.script
    )

    if not script_path.exists():

        return {
            "status":
                "FAILED",

            "return_code":
                None,

            "message":
                (
                    f"Script not found: "
                    f"{script_path}"
                ),
        }

    command = [
        python_executable,
        str(script_path),
        *stage.arguments,
    ]
    # Recover a successfully completed stage whose child process
    # finished correctly but Script 76 previously failed only because
    # its expected-output specification was incorrect.
    if (
        not force
        and not checkpoint_path(stage).exists()
        and expected_outputs_valid(stage)
    ):
        script_path = SCRIPTS / stage.script

        recovery_payload = {
            "stage": stage.number,
            "name": stage.name,
            "status": "COMPLETE",
            "started_at": None,
            "completed_at": now_iso(),
            "runtime_seconds": None,
            "command": [
                python_executable,
                str(script_path),
                *stage.arguments,
            ],
            "python": python_executable,
            "script": str(script_path),
            "script_sha256": file_sha256(script_path),
            "expected_outputs": [
                {
                    "path": str(path),
                    "exists": path.exists(),
                    "size_bytes": path.stat().st_size if path.exists() else 0,
                }
                for path in stage.expected_outputs
            ],
            "recovered_from_existing_outputs": True,
            "final_seeds": FINAL_SEEDS,
        }

        write_json_atomic(
            checkpoint_path(stage),
            recovery_payload,
        )

        failure_path(stage).unlink(missing_ok=True)
        running_path(stage).unlink(missing_ok=True)

        print(
            f"[RECOVERED] Script {stage.number}: "
            "validated existing outputs and created completion checkpoint."
        )

    if (
        not force
        and checkpoint_valid(
            stage
        )
    ):

        print(
            f"[SKIP] Script {stage.number}: "
            "valid completion checkpoint found."
        )

        update_status(
            stage,
            status="SKIPPED_COMPLETE",
            finished_at=now_iso(),
            return_code=0,
            message=(
                "Valid checkpoint and "
                "expected outputs found."
            ),
        )

        return {
            "status":
                "SKIPPED_COMPLETE",

            "return_code":
                0,

            "message":
                "Already complete.",
        }

    if dry_run:

        print(
            "[DRY RUN] "
            + " ".join(command)
        )

        return {
            "status":
                "DRY_RUN",

            "return_code":
                None,

            "message":
                "Not executed.",
        }

    # A previous incomplete running marker means the prior
    # process did not finish successfully.
    stale_running = read_json(
        running_path(stage)
    )

    if stale_running:

        print(
            f"[RECOVERY] Script {stage.number} has an "
            "unfinished RUNNING marker from a previous attempt."
        )

        print(
            "[RECOVERY] No success checkpoint exists, "
            "so the stage will be retried."
        )

    checkpoint_path(
        stage
    ).unlink(
        missing_ok=True
    )

    failure_path(
        stage
    ).unlink(
        missing_ok=True
    )

    log_path = make_log_path(
        stage
    )

    started_at = now_iso()
    start = time.perf_counter()

    write_running_marker(
        stage,
        command,
        log_path,
    )

    update_status(
        stage,
        status="RUNNING",
        started_at=started_at,
        log=log_path,
        message="Process started.",
    )

    print()
    print(
        f"[RUN] Script {stage.number}: "
        f"{stage.name}"
    )

    print(
        f"[LOG] {log_path}"
    )

    print(
        "[CMD] "
        + " ".join(command)
    )

    try:

        return_code = stream_subprocess(
            command=command,
            cwd=ROOT,
            log_path=log_path,
        )

    except KeyboardInterrupt:

        runtime = (
            time.perf_counter()
            - start
        )

        reason = (
            "Interrupted by user. "
            "No completion checkpoint created."
        )

        write_failure_checkpoint(
            stage=stage,
            command=command,
            log_path=log_path,
            started_at=started_at,
            runtime_seconds=runtime,
            return_code=None,
            reason=reason,
        )

        update_status(
            stage,
            status="INTERRUPTED",
            started_at=started_at,
            finished_at=now_iso(),
            runtime_seconds=round(
                runtime,
                3,
            ),
            log=log_path,
            message=reason,
        )

        raise

    except Exception as exc:

        runtime = (
            time.perf_counter()
            - start
        )

        reason = (
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        write_failure_checkpoint(
            stage=stage,
            command=command,
            log_path=log_path,
            started_at=started_at,
            runtime_seconds=runtime,
            return_code=None,
            reason=reason,
        )

        update_status(
            stage,
            status="FAILED",
            started_at=started_at,
            finished_at=now_iso(),
            runtime_seconds=round(
                runtime,
                3,
            ),
            log=log_path,
            message=reason,
        )

        return {
            "status":
                "FAILED",

            "return_code":
                None,

            "message":
                reason,
        }

    runtime = (
        time.perf_counter()
        - start
    )

    if return_code != 0:

        reason = (
            f"Child process exited with "
            f"return code {return_code}."
        )

        write_failure_checkpoint(
            stage=stage,
            command=command,
            log_path=log_path,
            started_at=started_at,
            runtime_seconds=runtime,
            return_code=return_code,
            reason=reason,
        )

        update_status(
            stage,
            status="FAILED",
            started_at=started_at,
            finished_at=now_iso(),
            runtime_seconds=round(
                runtime,
                3,
            ),
            return_code=return_code,
            log=log_path,
            message=reason,
        )

        print(
            f"[FAILED] Script {stage.number}: "
            f"{reason}"
        )

        return {
            "status":
                "FAILED",

            "return_code":
                return_code,

            "message":
                reason,
        }

    missing_outputs = [
        path
        for path in stage.expected_outputs
        if not valid_file(path)
    ]

    if missing_outputs:

        reason = (
            "Process returned success but expected "
            "output validation failed:\n"
            + "\n".join(
                str(path)
                for path
                in missing_outputs
            )
        )

        write_failure_checkpoint(
            stage=stage,
            command=command,
            log_path=log_path,
            started_at=started_at,
            runtime_seconds=runtime,
            return_code=return_code,
            reason=reason,
        )

        update_status(
            stage,
            status="FAILED_OUTPUT_VALIDATION",
            started_at=started_at,
            finished_at=now_iso(),
            runtime_seconds=round(
                runtime,
                3,
            ),
            return_code=return_code,
            log=log_path,
            message=reason,
        )

        print(
            f"[FAILED] Script {stage.number}: "
            "expected outputs were not validated."
        )

        for path in missing_outputs:

            print(
                f"  missing/empty: {path}"
            )

        return {
            "status":
                "FAILED_OUTPUT_VALIDATION",

            "return_code":
                return_code,

            "message":
                reason,
        }

    write_success_checkpoint(
        stage=stage,
        command=command,
        log_path=log_path,
        started_at=started_at,
        runtime_seconds=runtime,
    )

    update_status(
        stage,
        status="COMPLETE",
        started_at=started_at,
        finished_at=now_iso(),
        runtime_seconds=round(
            runtime,
            3,
        ),
        return_code=0,
        log=log_path,
        message="Stage completed and validated.",
    )

    print(
        f"[DONE] Script {stage.number} "
        f"completed in {elapsed_text(runtime)}."
    )

    print(
        f"[CHECKPOINT] "
        f"{checkpoint_path(stage)}"
    )

    return {
        "status":
            "COMPLETE",

        "return_code":
            0,

        "runtime_seconds":
            runtime,

        "message":
            "Stage completed.",
    }


# ======================================================================================================================
# FINAL REPORT
# ======================================================================================================================

def create_final_report(
    selected: Sequence[str],
    results: Dict[str, dict],
    started_at: str,
    runtime_seconds: float,
    python_executable: str,
) -> dict:

    stage_reports = []

    for number in selected:

        stage = STAGES[number]

        result = results.get(
            number,
            {}
        )

        stage_reports.append(
            {
                "stage":
                    number,

                "name":
                    stage.name,

                "script":
                    str(
                        SCRIPTS
                        / stage.script
                    ),

                "arguments":
                    list(
                        stage.arguments
                    ),

                "status":
                    result.get(
                        "status",
                        "UNKNOWN",
                    ),

                "return_code":
                    result.get(
                        "return_code"
                    ),

                "checkpoint_valid":
                    checkpoint_valid(
                        stage
                    ),

                "checkpoint":
                    str(
                        checkpoint_path(
                            stage
                        )
                    ),

                "expected_outputs": [
                    {
                        "path":
                            str(path),

                        "valid":
                            valid_file(
                                path
                            ),

                        "size_bytes":
                            (
                                path.stat().st_size
                                if path.exists()
                                else 0
                            ),
                    }
                    for path
                    in stage.expected_outputs
                ],
            }
        )

    successful_states = {
        "COMPLETE",
        "SKIPPED_COMPLETE",
    }

    overall_pass = all(
        result.get("status")
        in successful_states
        for result
        in results.values()
    )

    report = {
        "script":
            "76_run_final_multirun_experiments.py",

        "purpose":
            (
                "Resumable orchestration of final "
                "CartoGen repeated experiments."
            ),

        "started_at":
            started_at,

        "finished_at":
            now_iso(),

        "runtime_seconds":
            runtime_seconds,

        "status":
            (
                "PASS"
                if overall_pass
                else "INCOMPLETE"
            ),

        "python_executable":
            python_executable,

        "python_version":
            sys.version,

        "platform":
            platform.platform(),

        "project_root":
            str(ROOT),

        "final_seed_design":
            FINAL_SEEDS,

        "selected_stages":
            list(selected),

        "checkpoint_policy": {
            "checkpoint_after_successful_exit":
                True,

            "validate_expected_outputs":
                True,

            "atomic_checkpoint_write":
                True,

            "skip_valid_completed_stages":
                True,

            "retry_incomplete_stage":
                True,

            "individual_training_code_modified":
                False,
        },

        "stages":
            stage_reports,
    }

    write_json_atomic(
        REPORT_JSON,
        report,
    )

    return report


# ======================================================================================================================
# MAIN
# ======================================================================================================================

def main():

    args = parse_args()

    ensure_directories()

    selected = (
        args.stages
        if args.stages
        else [
            "66",
            "67",
            "69",
            "71",
        ]
    )

    # Preserve pipeline order regardless of argument order.
    pipeline_order = [
        "66",
        "67",
        "69",
        "71",
    ]

    selected = [
        stage
        for stage in pipeline_order
        if stage in selected
    ]

    banner(
        "CARTOGEN — FINAL RESUMABLE "
        "MULTI-RUN EXPERIMENT ORCHESTRATOR"
    )

    print(
        f"Project root: {ROOT}"
    )

    print(
        f"Python:       {args.python}"
    )

    print(
        f"Stages:       {selected}"
    )

    print(
        f"Final seeds:  {FINAL_SEEDS}"
    )

    print(
        f"Checkpoints:  {CHECKPOINT_ROOT}"
    )

    print(
        f"Logs:         {LOG_ROOT}"
    )

    print(
        f"Force:        {args.force}"
    )

    print(
        f"Dry run:      {args.dry_run}"
    )

    print(
        f"Stop failure: {args.stop_on_failure}"
    )

    section(
        "1. VERIFY PYTHON AND SCRIPTS"
    )

    python_path = shutil.which(
        args.python
    )

    if (
        python_path is None
        and not Path(
            args.python
        ).exists()
    ):

        raise FileNotFoundError(
            f"Python executable not found: "
            f"{args.python}"
        )

    for number in selected:

        stage = STAGES[number]

        script_path = (
            SCRIPTS
            / stage.script
        )

        status = (
            "OK"
            if script_path.exists()
            else "MISSING"
        )

        print(
            f"Script {number:<3} "
            f"{status:<8} "
            f"{script_path}"
        )

        if not script_path.exists():

            raise FileNotFoundError(
                script_path
            )

    section(
        "2. CURRENT CHECKPOINT STATE"
    )

    for number in selected:

        stage = STAGES[number]

        if checkpoint_valid(
            stage
        ):

            state = "COMPLETE"

        elif running_path(
            stage
        ).exists():

            state = (
                "INCOMPLETE/RUNNING-MARKER"
            )

        elif failure_path(
            stage
        ).exists():

            state = "FAILED/PENDING RETRY"

        else:

            state = "PENDING"

        print(
            f"Script {number:<3} "
            f"{state}"
        )

    section(
        "3. EXECUTION PLAN"
    )

    for number in selected:

        stage = STAGES[number]

        command = [
            args.python,
            str(
                SCRIPTS
                / stage.script
            ),
            *stage.arguments,
        ]

        if (
            not args.force
            and checkpoint_valid(
                stage
            )
        ):

            action = "SKIP"

        else:

            action = "RUN"

        print(
            f"{action:<5} "
            f"Script {number}: "
            f"{stage.name}"
        )

        print(
            "      "
            + " ".join(command)
        )

    if args.dry_run:

        banner(
            "DRY RUN COMPLETE — NOTHING EXECUTED"
        )

        return

    section(
        "4. EXECUTE FINAL EXPERIMENTS"
    )

    overall_start = (
        time.perf_counter()
    )

    started_at = now_iso()

    results: Dict[
        str,
        dict
    ] = {}

    try:

        for number in selected:

            stage = STAGES[number]

            dependency_problem = None

            for dependency in stage.depends_on:

                if not dependency_satisfied(
                    dependency,
                    selected,
                ):

                    dependency_problem = (
                        dependency
                    )

                    break

            if dependency_problem:

                message = (
                    f"Dependency Script "
                    f"{dependency_problem} "
                    f"is not complete."
                )

                print(
                    f"[BLOCKED] Script {number}: "
                    f"{message}"
                )

                result = {
                    "status":
                        "BLOCKED",

                    "return_code":
                        None,

                    "message":
                        message,
                }

                results[
                    number
                ] = result

                update_status(
                    stage,
                    status="BLOCKED",
                    finished_at=now_iso(),
                    message=message,
                )

                if args.stop_on_failure:
                    break

                continue

            result = run_stage(
                stage=stage,
                python_executable=args.python,
                force=args.force,
                dry_run=False,
            )

            results[
                number
            ] = result

            if (
                result[
                    "status"
                ]
                not in {
                    "COMPLETE",
                    "SKIPPED_COMPLETE",
                }
                and args.stop_on_failure
            ):

                print(
                    "\n[STOP] Failure encountered "
                    "and --stop-on-failure is active."
                )

                break

    except KeyboardInterrupt:

        runtime = (
            time.perf_counter()
            - overall_start
        )

        print()
        print(
            "=" * 120
        )

        print(
            "FINAL RUN INTERRUPTED"
        )

        print(
            "=" * 120
        )

        print(
            "Completed stages retain their checkpoints."
        )

        print(
            "The interrupted stage has no success "
            "checkpoint and will be retried next time."
        )

        print()
        print(
            "Resume with:"
        )

        print(
            "python scripts\\"
            "76_run_final_multirun_experiments.py"
        )

        create_final_report(
            selected=selected,
            results=results,
            started_at=started_at,
            runtime_seconds=runtime,
            python_executable=args.python,
        )

        raise SystemExit(
            130
        )

    overall_runtime = (
        time.perf_counter()
        - overall_start
    )

    section(
        "5. FINAL STATUS"
    )

    for number in selected:

        result = results.get(
            number
        )

        if result is None:

            state = "NOT RUN"

        else:

            state = result[
                "status"
            ]

        print(
            f"Script {number:<3} "
            f"{state}"
        )

    report = create_final_report(
        selected=selected,
        results=results,
        started_at=started_at,
        runtime_seconds=overall_runtime,
        python_executable=args.python,
    )

    banner(
        "SCRIPT 76 COMPLETE"
    )

    print(
        f"Status:      {report['status']}"
    )

    print(
        f"Runtime:     "
        f"{elapsed_text(overall_runtime)}"
    )

    print(
        f"Status CSV:  {STATUS_CSV}"
    )

    print(
        f"Report:      {REPORT_JSON}"
    )

    print(
        f"Checkpoints: {CHECKPOINT_ROOT}"
    )

    print(
        f"Logs:        {LOG_ROOT}"
    )

    if report[
        "status"
    ] == "PASS":

        print()
        print(
            "[PASS] ALL SELECTED FINAL "
            "MULTI-RUN EXPERIMENTS COMPLETE."
        )

    else:

        print()
        print(
            "[INCOMPLETE] One or more stages "
            "did not complete."
        )

        print(
            "Fix the failed stage if necessary "
            "and run the same command again."
        )

        print(
            "Valid completed checkpoints will "
            "be skipped automatically."
        )


if __name__ == "__main__":
    main()