"""Build timestamp-valid AV2 latency pairs from downloaded real files.

This stage deliberately stores timestamp/image pairs, not labels.  Future
annotations remain evaluation-only and are not used to construct inputs.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(r"C:\work\auto")
COHORT = ROOT / "results" / "av2_confirmation"
DATA = ROOT / "data" / "av2" / "sensor"
DELTA_MS = [100, 200, 300, 400, 500]
CAM = "ring_front_center"

def nearest(ts, values):
    i = int(np.searchsorted(values, ts))
    choices = [j for j in (i - 1, i) if 0 <= j < len(values)]
    if not choices: return None, None
    j = min(choices, key=lambda k: abs(int(values[k]) - int(ts)))
    return int(values[j]), abs(int(values[j]) - int(ts))

def main():
    roles = {}
    for role, fn in [("development", "av2_development_logs.txt"), ("model_selection", "av2_model_selection_logs.txt"), ("heldout", "av2_heldout_logs.txt")]:
        roles.update({x.strip(): role for x in (ROOT / "configs" / fn).read_text().splitlines() if x.strip()})
    meta = pd.read_csv(COHORT / "cohort_source_metadata.csv")
    meta = meta[meta.log_id.astype(str).isin(roles)].copy()
    pair_rows, audit_rows = [], []
    for _, log in meta.iterrows():
        source_split, log_id = log["split"], str(log["log_id"])
        root = DATA / source_split / log_id
        ann = pd.read_feather(root / "annotations.feather", columns=["timestamp_ns", "track_uuid"]).sort_values("timestamp_ns")
        ann_ts = np.sort(ann.timestamp_ns.unique().astype(np.int64))
        ann_counts = ann.groupby("timestamp_ns").size().to_dict()
        image_ts = np.array(sorted(int(p.stem) for p in (root / "sensors" / "cameras" / CAM).glob("*.jpg")), dtype=np.int64)
        out = {"split": roles[log_id], "source_split": source_split, "log_id": log_id, "city": log.get("city", "UNKNOWN")}
        for delta in DELTA_MS:
            timestamp_pairs, track_pairs = 0, 0
            for t in ann_ts:
                target, target_error = nearest(int(t + delta * 1_000_000), ann_ts)
                if target is None or target_error > 10_000_000: continue
                src_img, src_err = nearest(int(t), image_ts); dst_img, dst_err = nearest(target, image_ts)
                if src_img is None or dst_img is None or src_err > 30_000_000 or dst_err > 30_000_000: continue
                timestamp_pairs += 1
                src_tracks = set(ann.loc[ann.timestamp_ns == t, "track_uuid"].astype(str))
                dst_tracks = set(ann.loc[ann.timestamp_ns == target, "track_uuid"].astype(str))
                track_pairs += len(src_tracks & dst_tracks)
                pair_rows.append({"split": roles[log_id], "source_split": source_split, "log_id": log_id, "city": log.get("city", "UNKNOWN"), "delta_ms": delta, "source_timestamp_ns": int(t), "target_timestamp_ns": int(target), "source_image_path": str(root / "sensors" / "cameras" / CAM / f"{src_img}.jpg"), "target_image_path": str(root / "sensors" / "cameras" / CAM / f"{dst_img}.jpg"), "source_image_error_ns": int(src_err), "target_image_error_ns": int(dst_err), "target_annotation_geometry_eval_only": True, "future_annotation_geometry_input": False})
            out[f"{delta}ms_annotation_pairs"] = timestamp_pairs
            out[f"{delta}ms_track_pairs"] = track_pairs
        audit_rows.append(out)
    pairs = pd.DataFrame(pair_rows)
    pairs.to_parquet(COHORT / "latency_pairs.parquet", index=False)
    pd.DataFrame(audit_rows).to_csv(COHORT / "timing_audit.csv", index=False)
    cats = []
    for _, log in meta.iterrows():
        cats.extend(pd.read_feather(DATA / log["split"] / str(log.log_id) / "annotations.feather", columns=["category"]).category.unique().tolist())
    cats = sorted(set(cats))
    pd.DataFrame({"av2_category": cats, "analysis_group": ["vehicle" if ("VEHICLE" in c or c in {"TRUCK", "LARGE_VEHICLE", "VEHICULAR_TRAILER", "TRUCK_CAB"}) else ("pedestrian_cyclist" if c in {"PEDESTRIAN", "WHEELED_DEVICE", "STROLLER"} else "other") for c in cats]}).to_csv(COHORT / "class_mapping.csv", index=False)
    (COHORT / "causality_audit.md").write_text("# AV2 causality audit\n\n- Pair construction used only real image and annotation timestamps.\n- Annotation timestamps were matched to the nearest real timestamp within 10 ms of the requested offset; no interpolation or pseudo-labels were created.\n- Future annotation geometry is evaluation-only and explicitly marked; it is not present as a propagation input.\n- Image timestamps were matched to real files within 30 ms.\n- Detector inference and propagation outcomes are not yet run.\n", encoding="utf-8")
    (COHORT / "data_presence_audit.md").open("a", encoding="utf-8").write("\n- Timing/pair construction status: COMPLETE; no interpolation; future geometry marked evaluation-only.\n")
    print(json.dumps({"timestamp_pairs": len(pairs), "scenes": pairs.log_id.nunique(), "by_delta": pairs.groupby("delta_ms").size().to_dict()}, indent=2, default=int))

if __name__ == "__main__": main()
