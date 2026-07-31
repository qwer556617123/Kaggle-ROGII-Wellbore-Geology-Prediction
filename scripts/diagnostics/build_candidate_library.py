"""Build an auditable submission candidate library.

The Public LB fusion workflow needs vectors, not notebook names. This script
scans local/internal outputs and downloaded public notebook outputs, keeps only
sample-aligned ``id,tvt`` submissions, and writes a manifest plus distance
tables for downstream LB tomography.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
SAMPLE_PATH = ROOT_DIR / "sample_submission.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "docs"


KNOWN_SCORES = {
    # Internal notebook scores supplied by the running experiment log/user.
    "artifact_blend_output_v38/submission.csv": 8.279,
    "artifact_blend_output_v39/submission.csv": 8.314,
    "artifact_blend_output_v40/submission.csv": 8.336,
    "artifact_blend_output_v41/submission.csv": 8.136,
    "artifact_blend_output_v48/submission.csv": 8.271,
    "artifact_blend_output_v49/submission.csv": 8.271,
    "artifact_blend_output_v50/submission.csv": 8.268,
    "artifact_blend_output_v51/submission.csv": 8.239,
    "artifact_blend_output_v52/submission.csv": 9.365,
    "artifact_blend_output_v53/submission.csv": 8.162,
    "artifact_blend_output_v54/submission.csv": 8.160,
    "artifact_blend_output_v55/submission.csv": 8.155,
    "artifact_blend_output_v57/submission.csv": 8.452,
    "artifact_blend_output_v58/submission.csv": 8.294,
    "kernel_outputs/rogii-lightning-rebuild-v1/submission.csv": 7.215,
    "kernel_outputs/rogii-pilkwang-rebuild-v2/submission.csv": 7.609,
    # Public notebook title/description scores. Treat as claimed until we
    # confirm by submitting the exact downloaded vector ourselves.
    "external_outputs/bernubritz_rogii_lb7295_public_rebuild/submission.csv": 7.295,
    "external_outputs/lightningv08_rogii_lb_7_168/submission.csv": 7.168,
    "external_outputs/degnonguidi_public_score_rogii_lb_7_159/submission.csv": 7.159,
    "external_outputs/baidalinadilzhan_rogii_lb_7_201/submission.csv": 7.201,
}


EXTERNAL_SCORE_TRUST = {
    "external_outputs/bernubritz_rogii_lb7295_public_rebuild/submission.csv": "claimed_title",
    "external_outputs/lightningv08_rogii_lb_7_168/submission.csv": "claimed_title",
    "external_outputs/degnonguidi_public_score_rogii_lb_7_159/submission.csv": "claimed_title",
    "external_outputs/baidalinadilzhan_rogii_lb_7_201/submission.csv": "claimed_title",
}


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    rel_path: str
    path: Path
    source: str
    family: str
    sha256: str
    rows: int
    tvt_mean: float
    tvt_std: float
    tvt_min: float
    tvt_max: float
    public_lb: float | None
    score_trust: str
    notes: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_rel(path: Path) -> str:
    rel = path.relative_to(ROOT_DIR).as_posix()
    if rel.startswith("kaggle/"):
        rel = rel[len("kaggle/") :]
    return rel


def infer_source_and_family(rel_path: str) -> tuple[str, str]:
    parts = rel_path.split("/")
    filename = parts[-1].lower()
    source = parts[0] if len(parts) == 1 else parts[1] if parts[0] == "external_outputs" else parts[0]

    if rel_path.startswith("external_outputs/"):
        source = parts[1]
    elif rel_path.startswith("artifact_blend_output"):
        source = parts[0]
    elif rel_path.startswith("datasets/"):
        source = "local_dataset_" + (parts[1] if len(parts) > 1 else "unknown")
    elif rel_path.startswith("submissions/"):
        source = "local_submissions"

    if "sp45" in filename and "fleongg" in filename:
        family = "external_sp45_fleongg_blend"
    elif "sp45" in filename or "projection" in filename or "ridge_pf" in filename:
        family = "external_physical_pf"
    elif "fleongg" in filename or "pretrained" in filename or "lgbm" in filename:
        family = "external_learned_model"
    elif "gold_prefix" in filename or "public_self" in filename or "contact" in filename:
        family = "external_gold_prefix_calibration"
    elif rel_path.startswith("artifact_blend_output"):
        family = "internal_pf_artifact_geo"
    elif "v22" in rel_path or "base" in filename:
        family = "internal_base_archive"
    else:
        family = "unknown_submission"
    return source, family


def candidate_id_from_path(rel_path: str) -> str:
    stem = rel_path.replace("/", "__").replace("\\", "__")
    stem = stem.replace(".csv", "")
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)
    return safe[:180]


def read_submission(path: Path, sample: pd.DataFrame) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, usecols=["id", "tvt"])
    except Exception:
        return None
    if len(df) != len(sample):
        return None
    if not df["id"].astype(str).equals(sample["id"].astype(str)):
        return None
    tvt = pd.to_numeric(df["tvt"], errors="coerce")
    if not np.isfinite(tvt.to_numpy(dtype=float)).all():
        return None
    return pd.DataFrame({"id": df["id"].astype(str), "tvt": tvt.astype(float)})


def scan_paths(extra_roots: list[Path]) -> list[Path]:
    roots = [
        ROOT_DIR / "kaggle" / "external_outputs",
        ROOT_DIR / "kaggle",
        ROOT_DIR / "submissions",
    ] + extra_roots
    paths: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.csv"):
            if "__pycache__" in path.parts:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(path)
    return sorted(paths)


def build_manifest(paths: list[Path], sample: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows: list[dict[str, object]] = []
    vectors: dict[str, np.ndarray] = {}
    sample_well = sample["id"].astype(str).str.rsplit("_", n=1, expand=True)[0]

    for path in paths:
        if path.resolve() == SAMPLE_PATH.resolve():
            continue
        sub = read_submission(path, sample)
        if sub is None:
            continue
        rel = normalize_rel(path)
        source, family = infer_source_and_family(rel)
        candidate_id = candidate_id_from_path(rel)
        tvt = sub["tvt"].to_numpy(dtype=float)
        sha = sha256_file(path)
        score = KNOWN_SCORES.get(rel)
        trust = EXTERNAL_SCORE_TRUST.get(rel, "verified_own_submission" if score is not None else "")
        notes = ""
        if rel.startswith("external_outputs/") and score is None:
            notes = "downloaded_public_output_unscored"
        elif rel.startswith("external_outputs/") and trust == "claimed_title":
            notes = "score_from_public_notebook_title_needs_exact_anchor_submit"

        rows.append(
            {
                "candidate_id": candidate_id,
                "rel_path": rel,
                "source": source,
                "family": family,
                "sha256": sha,
                "rows": int(len(sub)),
                "well_count": int(sample_well.nunique()),
                "tvt_mean": float(np.mean(tvt)),
                "tvt_std": float(np.std(tvt)),
                "tvt_min": float(np.min(tvt)),
                "tvt_max": float(np.max(tvt)),
                "public_lb": score,
                "score_trust": trust,
                "notes": notes,
            }
        )
        vectors[candidate_id] = tvt

    manifest = pd.DataFrame(rows).sort_values(
        ["public_lb", "family", "source", "rel_path"], na_position="last"
    )
    return manifest, vectors


def build_pairwise(manifest: pd.DataFrame, vectors: dict[str, np.ndarray]) -> pd.DataFrame:
    ids = manifest["candidate_id"].tolist()
    rows: list[dict[str, object]] = []
    for i, left in enumerate(ids):
        lv = vectors[left]
        for right in ids[i + 1 :]:
            rv = vectors[right]
            diff = lv - rv
            rows.append(
                {
                    "left_id": left,
                    "right_id": right,
                    "rmse": float(np.sqrt(np.mean(diff * diff))),
                    "mean_delta": float(np.mean(diff)),
                    "max_abs_delta": float(np.max(np.abs(diff))),
                }
            )
    return pd.DataFrame(rows).sort_values("rmse") if rows else pd.DataFrame()


def build_well_stats(
    manifest: pd.DataFrame,
    vectors: dict[str, np.ndarray],
    sample: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    work = sample[["id"]].copy()
    split = work["id"].astype(str).str.rsplit("_", n=1, expand=True)
    work["well_id"] = split[0]
    rows: list[pd.DataFrame] = []
    for candidate_id in manifest["candidate_id"].head(top_n):
        df = work.copy()
        df["tvt"] = vectors[candidate_id]
        stats = (
            df.groupby("well_id", sort=True)["tvt"]
            .agg(["count", "mean", "std", "min", "max"])
            .reset_index()
        )
        stats.insert(0, "candidate_id", candidate_id)
        rows.append(stats)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def write_vectors(manifest: pd.DataFrame, vectors: dict[str, np.ndarray], out_path: Path) -> None:
    # Store a compact vector index for scripts that need exact arrays without
    # reparsing every CSV. Values are float32 to keep the cache modest.
    payload = {
        "candidate_ids": manifest["candidate_id"].tolist(),
        "vectors": np.vstack([vectors[cid].astype(np.float32) for cid in manifest["candidate_id"]]),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--extra-root", action="append", type=Path, default=[])
    parser.add_argument("--well-stats-top-n", type=int, default=60)
    args = parser.parse_args()

    sample = pd.read_csv(SAMPLE_PATH, usecols=["id", "tvt"])
    paths = scan_paths(args.extra_root)
    manifest, vectors = build_manifest(paths, sample)
    if manifest.empty:
        raise SystemExit("No valid sample-aligned candidate submissions found.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairwise = build_pairwise(manifest, vectors)
    well_stats = build_well_stats(manifest, vectors, sample, args.well_stats_top_n)

    manifest_path = args.out_dir / "candidate_manifest.csv"
    pairwise_path = args.out_dir / "candidate_pairwise_rmse.csv"
    well_stats_path = args.out_dir / "candidate_well_stats.csv"
    vector_path = args.out_dir / "candidate_vectors.npz"
    summary_path = args.out_dir / "candidate_manifest_summary.json"

    manifest.to_csv(manifest_path, index=False)
    pairwise.to_csv(pairwise_path, index=False)
    well_stats.to_csv(well_stats_path, index=False)
    write_vectors(manifest, vectors, vector_path)

    hash_counts = manifest.groupby("sha256")["candidate_id"].nunique().sort_values(ascending=False)
    summary = {
        "candidate_count": int(len(manifest)),
        "unique_hash_count": int(manifest["sha256"].nunique()),
        "scored_count": int(manifest["public_lb"].notna().sum()),
        "duplicate_hash_groups": int((hash_counts > 1).sum()),
        "best_scored": manifest[manifest["public_lb"].notna()]
        .sort_values("public_lb")
        .head(10)[["candidate_id", "rel_path", "sha256", "public_lb", "score_trust"]]
        .to_dict(orient="records"),
        "top_duplicate_hashes": hash_counts[hash_counts > 1].head(10).to_dict(),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
