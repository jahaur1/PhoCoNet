"""Verify all five completed stages without training or GPU initialization."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_suite import STAGES, verify_stage

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile", choices=("formal", "smoke"), default="formal")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    output = (ROOT / "result" / args.run_id).resolve()
    output.relative_to((ROOT / "result").resolve())
    states = json.loads((output / "suite_status.json").read_text(encoding="utf-8"))
    for stage in STAGES:
        if states.get(stage, {}).get("state") != "complete":
            raise RuntimeError(f"Incomplete stage: {stage}")
        verify_stage(stage, output, args.profile)
    stations = {}
    for station in ("juzizhou", "sanjiaozhou", "laodaohe"):
        metadata = json.loads((output / "phoconet_shap_seed2021" / station / "metadata.json").read_text(encoding="utf-8"))
        checkpoint = output / "phoconet_main" / station / "h24" / "seed_2021" / "checkpoint.pt"
        cache = list((output / "phoconet_sensor_noise" / "model_cache" / "PhoCoNet" / station).glob("h24_seed2021_*.pt"))
        digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        if len(cache) != 1 or digest(cache[0]) != digest(checkpoint):
            raise RuntimeError(f"Checkpoint mismatch for {station}")
        stations[station] = {
            "checkpoint_matches_main": True,
            "main_prediction_max_abs_difference": metadata["checkpoint_verification"]["max_abs_prediction_difference"],
            "equivalent_forward_max_abs_difference": metadata["equivalent_forward_max_abs_difference"],
            "explained_windows": len(metadata["selected_test_origins"]),
            "attribution_diagnostic": metadata["attribution_diagnostic"]}
    upstream = json.loads((ROOT / "provenance" / "upstream_manifest.json").read_text())
    missing = [f["path"] for f in upstream["files"] if not (ROOT / f["path"]).is_file()]
    if missing:
        raise RuntimeError(f"Missing original source files: {missing}")
    result = {"run_id": args.run_id, "profile": args.profile, "all_stages_complete": True,
              "upstream_files_retained": len(upstream["files"]), "stations": stations,
              "note": "Smoke results validate execution only, not scientific performance."}
    if args.report:
        if args.report.exists():
            raise ValueError("Report exists; choose a new file")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
