"""Copy the three station CSV files and generate benchmark metadata."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import shutil
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    destination = ROOT / "dataset" / "forecasting"
    destination.mkdir(parents=True, exist_ok=True)
    from ts_benchmark.data.utils import read_data
    rows = []
    for station in ("juzizhou", "sanjiaozhou", "laodaohe"):
        source = args.source / (station + ".csv")
        target = destination / source.name
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError(f"Existing data differs: {target}")
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        data = read_data(str(target))
        rows.append({"file_name": target.name, "freq": "hourly", "if_univariate": False,
                     "size": "user", "length": len(data)})
    fields = ["file_name", "freq", "if_univariate", "size", "length", "trend", "seasonal",
              "stationary", "transition", "shifting", "correlation"]
    metadata = destination / "FORECAST_META.csv"
    import io
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    content = buffer.getvalue()
    if metadata.exists() and metadata.read_text(encoding="utf-8").splitlines() != content.splitlines():
        raise ValueError("Metadata already exists with different content")
    metadata.write_text(content, encoding="utf-8")
    print(f"Prepared {len(rows)} datasets and {metadata}")

if __name__ == "__main__":
    main()
