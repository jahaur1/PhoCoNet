"""Prepare or run the version-isolated PhoCoNet experiment suite.

Default: print the plan and validate the environment. Pass --execute to train.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("main", "noise", "missingness", "shap", "information")
MODULES = dict(zip(STAGES, ("run_main", "run_sensor_noise", "run_history_tp_missingness", "run_shap", "run_information_sets")))

def utc():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)

def source_files():
    files = []
    for directory in ("experiments", "ts_benchmark", "scripts", "config", "tests", "experiment_records"):
        files.extend(p for p in (ROOT / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix in {".py", ".json", ".md", ".csv", ".sh", ".ps1"})
    files.extend(p for p in ROOT.iterdir() if p.is_file() and (p.suffix in {".py", ".md", ".txt", ".ps1"} or p.name in {"LICENSE", ".gitignore"}))
    return sorted(set(files))

def commands(args):
    prefix = [sys.executable, "-B", "-u", "-m"]
    result = {stage: prefix + ["experiments." + MODULES[stage]] for stage in STAGES}
    result["main"] += ["--gpu", str(args.gpu)]
    result["noise"] += ["--gpus", str(args.gpu)]
    result["information"] += ["--device", "cuda:0"]
    if args.profile == "smoke":
        result["main"] += ["--horizons", "24", "--seeds", "2021", "--epochs", "1"]
        result["noise"] += ["--snr-db", "100", "--num-epochs", "1"]
        result["missingness"] += ["--rates", "0", "0.3", "--mask-repetitions", "1"]
        result["shap"] += ["--num-windows", "2", "--background-count", "2", "--integration-steps", "2", "--num-epochs", "1"]
        result["information"] += ["--seeds", "2021", "--epochs", "1"]
    return result

def require_dependencies(stage, selected, state, output):
    dependencies = {"noise": ("main",), "missingness": ("main", "noise"), "shap": ("main", "noise")}
    for dependency in dependencies.get(stage, ()):
        if state.get(dependency, {}).get("state") != "complete":
            raise RuntimeError(f"{stage} requires completed {dependency}; include it in --stages or run it first")
    if stage in {"missingness", "shap"}:
        for station in ("juzizhou", "sanjiaozhou", "laodaohe"):
            main = output / "phoconet_main" / station / "h24" / "seed_2021"
            if not (main / "predictions.npz").is_file() or not (main / "checkpoint.pt").is_file():
                raise RuntimeError(f"Missing main prediction/checkpoint for {station}")
            caches = list((output / "phoconet_sensor_noise" / "model_cache" / "PhoCoNet" / station).glob("h24_seed2021_*.pt"))
            if len(caches) != 1 or sha(caches[0]) != sha(main / "checkpoint.pt"):
                raise RuntimeError(f"Noise checkpoint differs from main for {station}")

def verify_stage(stage, output, profile):
    import csv
    def count(path):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return len(list(csv.DictReader(handle)))
    expected_main = 75 if profile == "formal" else 3
    if stage == "main":
        if count(output / "phoconet_main" / "metrics_by_seed.csv") != expected_main:
            raise RuntimeError("Incomplete main matrix")
        if len(list((output / "phoconet_main").glob("*/h*/seed_*/checkpoint.pt"))) != expected_main:
            raise RuntimeError("Missing main checkpoints")
        if len(list((output / "phoconet_main").glob("*/h*/seed_*/predictions.npz"))) != expected_main:
            raise RuntimeError("Missing main prediction archives")
    elif stage == "noise":
        expected = 21 if profile == "formal" else 3
        if count(output / "phoconet_sensor_noise" / "sensor_noise_metrics.csv") != expected:
            raise RuntimeError("Incomplete noise matrix")
        if len(list(output.glob("*/phoconet_sensor_noise/PhoCoNet/*/predictions.npz"))) != expected:
            raise RuntimeError("Missing noise prediction archives")
    elif stage == "missingness":
        verification = json.loads((output / "phoconet_history_tp_missingness" / "verification.json").read_text(encoding="utf-8"))
        if not verification["complete"] or not verification["all_outputs_finite"]:
            raise RuntimeError("Missingness verification failed")
    elif stage == "shap":
        for station in ("juzizhou", "sanjiaozhou", "laodaohe"):
            metadata = json.loads((output / "phoconet_shap_seed2021" / station / "metadata.json").read_text(encoding="utf-8"))
            if not metadata["checkpoint_verification"]["allclose_rtol_1e-6_atol_1e-6"]:
                raise RuntimeError("SHAP checkpoint verification failed")
            if metadata["equivalent_forward_max_abs_difference"] > 1e-6:
                raise RuntimeError("SHAP graph differs from production")
    elif stage == "information":
        expected = 75 if profile == "formal" else 15
        if count(output / "phoconet_information_sets" / "metrics_by_run.csv") != expected:
            raise RuntimeError("Incomplete information matrix")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile", choices=("formal", "smoke"), default="formal")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id):
        parser.error("run-id must be a safe directory name")
    if args.execute and args.prepare_only:
        parser.error("Choose --execute or --prepare-only")
    if len(set(args.stages)) != len(args.stages):
        parser.error("Duplicate stages")
    os.environ.update(PHOCONET_RUN_ID=args.run_id, PYTHONHASHSEED="2021",
                      PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1",
                      OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                      CUBLAS_WORKSPACE_CONFIG=":4096:8", CUDA_VISIBLE_DEVICES=str(args.gpu))
    from experiments.preflight import check
    environment = check()
    if not environment["cuda_available"]:
        raise RuntimeError("CUDA is required for this suite; select a CUDA Python environment")
    output = ROOT / "result" / args.run_id
    plan = commands(args)
    # CUDA_VISIBLE_DEVICES remaps the physical GPU for all child processes.
    plan["main"][plan["main"].index("--gpu") + 1] = "0"
    plan["noise"][plan["noise"].index("--gpus") + 1] = "0"
    for stage in STAGES:
        if stage in args.stages:
            print(stage + ": " + subprocess.list2cmdline(plan[stage]), flush=True)
    print("Output: " + str(output), flush=True)
    files = source_files()
    signature = {"profile": args.profile, "model": "phoconet.PhoCoNet",
                 "sources": {p.relative_to(ROOT).as_posix(): sha(p) for p in files},
                 "data": {p.name: sha(p) for p in sorted((ROOT / "dataset" / "forecasting").glob("*.csv")) if p.stem in ("juzizhou", "sanjiaozhou", "laodaohe", "FORECAST_META")}}
    manifest_path = output / "suite_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["signature"] != signature:
            raise RuntimeError("Source/data/profile changed; choose a new --run-id")
    elif output.exists() and any(output.iterdir()):
        raise RuntimeError("Run directory exists without a suite manifest; choose a new run-id")
    if not args.execute and not args.prepare_only:
        print("Preflight passed. No experiment started; add --prepare-only or --execute.", flush=True)
        return 0
    output.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        write_json(manifest_path, {"created_utc": utc(), "signature": signature, "environment": environment})
        with zipfile.ZipFile(output / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for file in files:
                archive.write(file, file.relative_to(ROOT).as_posix())
        frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        (output / "environment_freeze.txt").write_text(frozen, encoding="utf-8")
    write_json(output / "execution_plan.json", {s: plan[s] for s in STAGES})
    if args.prepare_only:
        print("Prepared and archived. No training started.", flush=True)
        return 0
    state_path = output / "suite_status.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    log_root = output / "logs"
    log_root.mkdir(exist_ok=True)
    for stage in STAGES:
        if stage not in args.stages:
            continue
        if state.get(stage, {}).get("state") == "complete":
            verify_stage(stage, output, args.profile)
            print("[skip] completed " + stage, flush=True)
            continue
        require_dependencies(stage, args.stages, state, output)
        # These two scripts have no per-station resume protocol. Preserve an
        # interrupted attempt in full before starting a fresh evaluation.
        family = {"missingness": "phoconet_history_tp_missingness", "shap": "phoconet_shap_seed2021"}.get(stage)
        if family and (output / family).exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            (output / family).rename(output / (family + "_previous_" + stamp))
        log = log_root / (stage + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".log")
        state[stage] = {"state": "running", "started_utc": utc(), "command": plan[stage], "log": log.relative_to(ROOT).as_posix()}
        write_json(state_path, state)
        print("[run] " + stage + " -> " + str(log), flush=True)
        try:
            with log.open("w", encoding="utf-8") as handle:
                code = subprocess.call(plan[stage], cwd=ROOT, env=os.environ.copy(), stdout=handle, stderr=subprocess.STDOUT)
            if code:
                raise RuntimeError(f"{stage} exited {code}; see {log}")
            verify_stage(stage, output, args.profile)
        except BaseException as error:
            state[stage].update(state="failed", finished_utc=utc(), error=str(error))
            write_json(state_path, state)
            raise
        state[stage].update(state="complete", finished_utc=utc())
        write_json(state_path, state)
        print("[done] " + stage, flush=True)
    print("Requested stages complete: " + str(state_path), flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
