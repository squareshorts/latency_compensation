"""Check every value added to the manuscript against the analysis outputs."""
import re
from pathlib import Path

import pandas as pd

ROOT = Path("/sessions/beautiful-eager-ramanujan/mnt/latency_compensation")
OUT = ROOT / "results/rtdetr_extension_20260721"
TEX = Path("/tmp/b2/main.tex").read_text() + Path("/tmp/b2/supplement.tex").read_text()

cls = pd.read_csv(OUT / "class_breakdown_all_detectors.csv")
ext = pd.read_csv(OUT / "extension_replication.csv")
dirn = pd.read_csv(OUT / "direction_agreement_image_space.csv")
dens = pd.read_csv(OUT / "detection_density.csv")

checks = []


def check(label, value, fmt="{:.2f}"):
    s = fmt.format(value)
    checks.append((label, s, s in TEX))


for _, r in cls[cls.detector == "rtdetr_l"].iterrows():
    tag = r["class"][:3]
    check(f"rtdetr {tag} rel", r.median_paired_relative_diff * 100)
    check(f"rtdetr {tag} ci_lo", r.ci_lower * 100)
    check(f"rtdetr {tag} ci_hi", r.ci_upper * 100)
    check(f"rtdetr {tag} objects", r.objects, "{:,.0f}")
    check(f"rtdetr {tag} logs", r.eligible_logs, "{:.0f}")

for _, r in ext.iterrows():
    check(f"ext {r.detector} rel", r.median_relative_diff * 100)
    check(f"ext {r.detector} ci_lo", r.ci_lower * 100)
    check(f"ext {r.detector} ci_hi", r.ci_upper * 100)
    check(f"ext {r.detector} B3", r.B3_error, "{:.6f}")
    check(f"ext {r.detector} B5", r.B5_error, "{:.6f}")

for _, r in dirn.iterrows():
    check(f"dir {r.detector} toward", r.same_halfplane * 100)
    check(f"dir {r.detector} eligible", r.eligible, "{:,.0f}")

for _, r in dens.iterrows():
    check(f"dens {r.detector} per_frame", r["detections_per_frame_mean"], "{:.1f}")
    check(f"dens {r.detector} conf", r.conf_median, "{:.3f}")

bad = [c for c in checks if not c[2]]
for label, val, ok in checks:
    print(f"{'OK ' if ok else 'MISS'} {label:28s} {val}")
print(f"\n{len(checks) - len(bad)}/{len(checks)} values found verbatim in manuscript")
if bad:
    print("MISSING:", [b[0] for b in bad])
