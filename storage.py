"""Portable, pickle-free result files, provenance and tidy CSV tables."""
import csv
import hashlib
import json
import platform
import shutil
from pathlib import Path
import numpy as np
import scipy


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def provenance():
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(Path(__file__).parent.glob("*.py"))}
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "platform": platform.platform(), "source_sha256": hashes}


def prepare_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"output directory is not empty: {path}; choose a new directory")
    path.mkdir(parents=True, exist_ok=True)
    write_json(path/"provenance.json", provenance())
    snapshot = path/"source_snapshot"/"self_healing"
    snapshot.mkdir(parents=True)
    for source in Path(__file__).parent.glob("*.py"):
        shutil.copy2(source, snapshot/source.name)
    shutil.copy2(Path(__file__).parent/"requirements.txt", snapshot/"requirements.txt")
    return path


def snapshot_analysis(directory):
    """Keep analysis code separately from the immutable simulation snapshot."""
    root = Path(directory)/"analysis_source"
    package = root/"self_healing"
    package.mkdir(parents=True, exist_ok=True)
    for source in Path(__file__).parent.glob("*.py"):
        shutil.copy2(source, package/source.name)
    shutil.copy2(Path(__file__).parent/"requirements.txt", package/"requirements.txt")
    write_json(root/"provenance.json", provenance())


def save_result(result, directory):
    summary = result.summary
    stem = f"{summary['scenario']}__{summary['controller']}__seed{summary['seed']}"
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {"config": result.config, "summary": summary, "events": result.events}
    write_json(directory/f"{stem}.json", metadata)
    np.savez_compressed(directory/f"{stem}.npz", **result.series,
                        positions=result.positions, headings=result.headings,
                        snapshot_time=result.snapshot_time, groups=result.groups,
                        obstacles=result.obstacles, metadata=np.array(json.dumps(metadata, allow_nan=False)))


def load_result(path):
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files if key != "metadata"}
        metadata = json.loads(str(saved["metadata"]))
    return arrays, metadata


def digest(path):
    """SHA-256 of a configuration or protocol file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
