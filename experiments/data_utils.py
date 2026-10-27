from pathlib import Path

import numpy as np
import pandas as pd


AUXILIARIES = ("WT", "pH", "DO", "Cond", "Turb", "PI", "AN", "TN")


def load_frame(path: Path, protocol: str = "full8") -> pd.DataFrame:
    if protocol != "full8":
        raise ValueError("PhoCoNet requires the full8 data configuration")
    frame = pd.read_csv(path)
    if "date" not in frame:
        raise ValueError(f"{path}: missing date column")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if {"data", "cols"}.issubset(frame.columns):
        frame = frame.pivot(index="date", columns="cols", values="data")
    else:
        frame = frame.set_index("date")
    frame = frame.loc[:, ["TP", *AUXILIARIES]].sort_index().astype(float)
    if frame.index.has_duplicates or frame.index.hasnans:
        raise ValueError(f"{path}: duplicate or missing timestamps")
    if len(frame) < 2 or not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"{path}: insufficient data or non-finite measurements")
    if not (np.diff(frame.index.asi8) == pd.Timedelta(hours=4).value).all():
        raise ValueError(f"{path}: expected a regular four-hour grid")
    return frame
