"""Estimate Public LB-aware fusion weights from scored submission vectors.

For an anchor prediction ``a`` with score ``A`` and another scored candidate
``a + d_i`` with score ``S_i``, the squared-score identity gives:

    S_i^2 = A^2 + 2 <e, d_i> / N + ||d_i||^2 / N

where ``e`` is the anchor residual on Public LB rows. This lets us estimate the
residual projection along scored candidate directions, then search conservative
non-negative blends around the anchor.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT_DIR / "docs"
FUSION_DIR = ROOT_DIR / "kaggle" / "fusion_candidates"
SAMPLE_PATH = ROOT_DIR / "sample_submission.csv"


def load_vectors(vector_path: Path) -> dict[str, np.ndarray]:
    data = np.load(vector_path)
    ids = [str(x) for x in data["candidate_ids"]]
    vectors = data["vectors"].astype(float)
    return {cid: vectors[i] for i, cid in enumerate(ids)}


def dedupe_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, group in manifest.groupby("sha256", sort=False):
        scored = group[group["public_lb"].notna()].sort_values("public_lb")
        verified = scored[scored["score_trust"].astype(str) == "verified_own_submission"].sort_values("public_lb")
        if not scored.empty:
            row = (verified if not verified.empty else scored).iloc[0].copy()
            aliases = group["candidate_id"].tolist()
            row["alias_count"] = len(aliases)
            row["aliases"] = "|".join(aliases[:20])
            if len(scored["public_lb"].dropna().unique()) > 1:
                row["notes"] = str(row.get("notes", "")) + "; duplicate_hash_conflicting_scores"
            rows.append(row)
        else:
            row = group.iloc[0].copy()
            row["alias_count"] = len(group)
            row["aliases"] = "|".join(group["candidate_id"].tolist()[:20])
            rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["public_lb", "family", "source", "rel_path"], na_position="last").reset_index(drop=True)


def duplicate_score_conflicts(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scored = manifest[manifest["public_lb"].notna()].copy()
    for sha, group in scored.groupby("sha256", sort=False):
        scores = sorted(float(x) for x in group["public_lb"].dropna().unique())
        if len(scores) <= 1:
            continue
        rows.append(
            {
                "sha256": sha,
                "score_min": min(scores),
                "score_max": max(scores),
                "score_range": max(scores) - min(scores),
                "has_verified": bool((group["score_trust"].astype(str) == "verified_own_submission").any()),
                "candidate_count": int(len(group)),
                "candidates": "|".join(group["candidate_id"].tolist()[:20]),
            }
        )
    return pd.DataFrame(rows).sort_values("score_range", ascending=False) if rows else pd.DataFrame()


def cap_for_row(row: pd.Series) -> float:
    family = str(row.get("family", ""))
    trust = str(row.get("score_trust", ""))
    if family.startswith("internal_"):
        return 0.30
    if trust == "claimed_title":
        return 0.35
    return 0.45


def estimate_basis(
    manifest: pd.DataFrame,
    vectors: dict[str, np.ndarray],
    anchor: pd.Series,
) -> pd.DataFrame:
    anchor_id = str(anchor["candidate_id"])
    anchor_vec = vectors[anchor_id]
    anchor_score = float(anchor["public_lb"])
    rows = []
    scored = manifest[manifest["public_lb"].notna()].copy()
    for _, row in scored.iterrows():
        cid = str(row["candidate_id"])
        if cid == anchor_id:
            continue
        d = vectors[cid] - anchor_vec
        q = float(np.mean(d * d))
        projection = (float(row["public_lb"]) ** 2 - anchor_score**2 - q) / 2.0
        rows.append(
            {
                "candidate_id": cid,
                "rel_path": row["rel_path"],
                "source": row["source"],
                "family": row["family"],
                "sha256": row["sha256"],
                "public_lb": row["public_lb"],
                "score_trust": row["score_trust"],
                "alias_count": row.get("alias_count", 1),
                "rmse_to_anchor": float(np.sqrt(q)),
                "basis_q": q,
                "residual_projection": projection,
                "alpha_cap": cap_for_row(row),
                "notes": row.get("notes", ""),
            }
        )
    return pd.DataFrame(rows).sort_values("residual_projection")


def predict_score(anchor_score: float, weights: np.ndarray, p: np.ndarray, gram: np.ndarray) -> float:
    square = anchor_score**2 + 2.0 * float(weights @ p) + float(weights @ gram @ weights)
    return float(np.sqrt(max(square, 0.0)))


def search_weights(anchor_score: float, basis: pd.DataFrame, deltas: np.ndarray) -> pd.DataFrame:
    if basis.empty:
        return pd.DataFrame()
    p = basis["residual_projection"].to_numpy(dtype=float)
    caps = basis["alpha_cap"].to_numpy(dtype=float)
    gram = (deltas @ deltas.T) / deltas.shape[1]

    rows: list[dict[str, object]] = []

    # Single-direction exact grid. It is more stable than trusting a tiny,
    # underdetermined multi-direction solve.
    grid = np.linspace(0.0, 1.0, 101)
    for i, row in basis.reset_index(drop=True).iterrows():
        best_alpha = 0.0
        best_score = anchor_score
        for frac in grid:
            alpha = float(frac * caps[i])
            w = np.zeros(len(basis), dtype=float)
            w[i] = alpha
            score = predict_score(anchor_score, w, p, gram)
            if score < best_score:
                best_score = score
                best_alpha = alpha
        rows.append(
            {
                "fusion_id": f"single_{i:03d}",
                "predicted_lb": best_score,
                "anchor_weight": 1.0 - best_alpha,
                "non_anchor_weight": best_alpha,
                "weights": json.dumps({str(row["candidate_id"]): best_alpha}),
                "members": str(row["candidate_id"]),
                "risk_flags": "single_direction_grid",
            }
        )

    # Small pair search among the most promising directions. Pair weights are
    # capped and total non-anchor weight is capped at 0.65 to keep the anchor
    # dominant until we have exact confirmed 7.x scores.
    top = basis.reset_index(drop=True).head(min(12, len(basis)))
    top_indices = top.index.to_list()
    pair_grid = np.linspace(0.0, 1.0, 41)
    for a_pos, i in enumerate(top_indices):
        for j in top_indices[a_pos + 1 :]:
            best_score = anchor_score
            best = (0.0, 0.0)
            for fa in pair_grid:
                wa = float(fa * caps[i])
                for fb in pair_grid:
                    wb = float(fb * caps[j])
                    if wa + wb > 0.65:
                        continue
                    w = np.zeros(len(basis), dtype=float)
                    w[i] = wa
                    w[j] = wb
                    score = predict_score(anchor_score, w, p, gram)
                    if score < best_score:
                        best_score = score
                        best = (wa, wb)
            if best_score < anchor_score:
                rows.append(
                    {
                        "fusion_id": f"pair_{i:03d}_{j:03d}",
                        "predicted_lb": best_score,
                        "anchor_weight": 1.0 - sum(best),
                        "non_anchor_weight": sum(best),
                        "weights": json.dumps(
                            {
                                str(basis.iloc[i]["candidate_id"]): best[0],
                                str(basis.iloc[j]["candidate_id"]): best[1],
                            }
                        ),
                        "members": "|".join(
                            [str(basis.iloc[i]["candidate_id"]), str(basis.iloc[j]["candidate_id"])]
                        ),
                        "risk_flags": "pair_grid_under_rank_limited",
                    }
                )

    return pd.DataFrame(rows).sort_values("predicted_lb").reset_index(drop=True)


def make_submission(
    anchor: pd.Series,
    top: pd.Series,
    vectors: dict[str, np.ndarray],
    out_path: Path,
) -> dict[str, object]:
    sample = pd.read_csv(SAMPLE_PATH, usecols=["id"])
    anchor_id = str(anchor["candidate_id"])
    pred = vectors[anchor_id].copy()
    weights = {str(k): float(v) for k, v in json.loads(str(top["weights"])).items()}
    for cid, weight in weights.items():
        pred += weight * (vectors[cid] - vectors[anchor_id])
    out = pd.DataFrame({"id": sample["id"].astype(str), "tvt": pred})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return {
        "out_path": str(out_path.relative_to(ROOT_DIR)),
        "anchor_id": anchor_id,
        "anchor_rel_path": anchor["rel_path"],
        "predicted_lb": float(top["predicted_lb"]),
        "weights": weights,
    }


def copy_anchor(anchor: pd.Series, out_path: Path) -> dict[str, object]:
    src = ROOT_DIR / "kaggle" / str(anchor["rel_path"])
    if not src.exists():
        src = ROOT_DIR / str(anchor["rel_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, out_path)
    return {
        "out_path": str(out_path.relative_to(ROOT_DIR)),
        "anchor_id": anchor["candidate_id"],
        "anchor_rel_path": anchor["rel_path"],
        "anchor_score": float(anchor["public_lb"]),
        "anchor_score_trust": anchor.get("score_trust", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DOCS_DIR / "candidate_manifest.csv")
    parser.add_argument("--vectors", type=Path, default=DOCS_DIR / "candidate_vectors.npz")
    parser.add_argument("--out-dir", type=Path, default=DOCS_DIR)
    parser.add_argument("--fusion-dir", type=Path, default=FUSION_DIR)
    parser.add_argument("--min-improvement", type=float, default=0.03)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    vectors = load_vectors(args.vectors)
    conflicts = duplicate_score_conflicts(manifest)
    unique = dedupe_manifest(manifest)
    scored = unique[unique["public_lb"].notna()].copy()
    if scored.empty:
        raise SystemExit("No scored candidate vectors available for LB tomography.")

    anchor = scored.sort_values("public_lb").iloc[0]
    basis = estimate_basis(unique, vectors, anchor)
    deltas = np.vstack([vectors[cid] - vectors[str(anchor["candidate_id"])] for cid in basis["candidate_id"]])
    fusions = search_weights(float(anchor["public_lb"]), basis, deltas)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    unique.to_csv(args.out_dir / "candidate_manifest_unique_hash.csv", index=False)
    conflicts.to_csv(args.out_dir / "candidate_duplicate_score_conflicts.csv", index=False)
    basis.to_csv(args.out_dir / "lb_tomography_candidates.csv", index=False)
    fusions.to_csv(args.out_dir / "lb_tomography_top_ensembles.csv", index=False)

    anchor_payload = copy_anchor(anchor, args.fusion_dir / "submission_external_anchor.csv")
    fusion_payload = None
    submit_recommendation = "anchor_only"
    blocking_reasons: list[str] = []
    if str(anchor.get("score_trust", "")) != "verified_own_submission":
        blocking_reasons.append("anchor_score_not_yet_confirmed_by_our_submission")
    unresolved_conflicts = (
        conflicts[(conflicts["has_verified"] == False) & (conflicts["score_range"] > 0.05)]
        if not conflicts.empty
        else conflicts
    )
    if not unresolved_conflicts.empty:
        blocking_reasons.append("same_submission_hash_has_conflicting_claimed_scores")
    if not fusions.empty:
        top = fusions.iloc[0]
        improvement = float(anchor["public_lb"]) - float(top["predicted_lb"])
        if improvement > 0.75:
            blocking_reasons.append("predicted_fusion_gain_is_too_large_for_current_low_rank_tomography")
        if improvement >= args.min_improvement and not blocking_reasons:
            fusion_payload = make_submission(
                anchor,
                top,
                vectors,
                args.fusion_dir / "submission_lb_tomography_best.csv",
            )
            submit_recommendation = "anchor_then_fusion"
        elif blocking_reasons:
            submit_recommendation = "anchor_only_fusion_blocked_until_anchor_confirmed"
        else:
            submit_recommendation = "anchor_only_predicted_fusion_gain_below_gate"

    summary = {
        "anchor": anchor_payload,
        "scored_unique_count": int(len(scored)),
        "basis_count": int(len(basis)),
        "best_fusion": None if fusions.empty else fusions.iloc[0].to_dict(),
        "fusion_payload": fusion_payload,
        "submit_recommendation": submit_recommendation,
        "blocking_reasons": blocking_reasons,
        "duplicate_score_conflicts": conflicts.head(10).to_dict(orient="records"),
        "risk_notes": [
            "external title scores are claimed until exact anchor submit confirms the downloaded vector",
            "duplicate hashes with conflicting claimed scores are deduped using the best score but flagged",
            "unscored branches are audited but excluded from residual-projection optimization",
        ],
    }
    (args.out_dir / "lb_tomography_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
