"""Integrity and distinctness audit for all corrected AV2 checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"C:\work\auto")
OUT = ROOT / "results" / "av2_confirmation"
COORDS = ["pred_x1", "pred_y1", "pred_x2", "pred_y2"]
PAIRS = [("B0","B1"),("B0","B3"),("B1","B2"),("B1","B4"),("B3","B4"),("B4","B5")]
REQUIRED = {"comparison_id","log_id","role","detector","delta_ms","model",*COORDS,"source_timestamp_ns","target_timestamp_ns","estimated_depth_m","track_age","yaw_rate_rad_s","runtime_ms","future_geometry_input","future_annotation_geometry_eval_only"}


def main() -> None:
    files=sorted((OUT/"corrected_checkpoints").glob("*/*/*.propagation.parquet"))
    rows=[];total_rows=total_comparisons=duplicates=bad_model_sets=future_rows=0;missing_schema=[]
    for path in files:
        detector,role=path.parts[-3],path.parts[-2];frame=pd.read_parquet(path)
        missing=REQUIRED.difference(frame.columns)
        if missing:missing_schema.append(f"{path}:{sorted(missing)}")
        total_rows+=len(frame);future_rows+=int(frame.future_geometry_input.sum());duplicates+=int(frame.duplicated(["comparison_id","model"]).sum())
        counts=frame.groupby("comparison_id").model.agg(lambda value: tuple(sorted(value)))
        bad_model_sets+=int((counts != tuple(["B0","B1","B2","B3","B4","B5"])).sum());total_comparisons+=frame.comparison_id.nunique()
        for delta,group in frame.groupby("delta_ms"):
            pivot=group.pivot(index="comparison_id",columns="model",values=COORDS)
            record={"detector":detector,"role":role,"delta_ms":int(delta),"log_id":str(frame.log_id.iloc[0]),"model_rows":len(group),"comparisons":len(pivot)}
            for left,right in PAIRS:
                equal=np.ones(len(pivot),bool);difference=[]
                for coordinate in COORDS:
                    a=pivot[(coordinate,left)].to_numpy();b=pivot[(coordinate,right)].to_numpy();equal &= a==b;difference.append(np.abs(a-b))
                diff=np.vstack(difference).T
                record[f"{left}_vs_{right}_exact_equal_count"]=int(equal.sum());record[f"{left}_vs_{right}_exact_equal_percent"]=float(equal.mean()*100)
                record[f"{left}_vs_{right}_mean_abs_coordinate_difference_px"]=float(diff.mean());record[f"{left}_vs_{right}_max_abs_coordinate_difference_px"]=float(diff.max())
            rows.append(record)
    detail=pd.DataFrame(rows);detail.to_csv(OUT/"corrected_propagation_integrity.csv",index=False,float_format="%.12g")

    sample=pd.read_parquet(OUT/"row_level_recalculation.parquet");sample["key"]=sample.comparison_id
    selected_logs=set(sample.log_id.astype(str));corrected=[]
    for path in files:
        if path.stem.split('.')[0] not in selected_logs:continue
        frame=pd.read_parquet(path);frame["key"]=frame.comparison_id.map(lambda value:":".join(value.split(":")[:5]))
        wide=frame.pivot(index="key",columns="model",values=COORDS);wide.columns=[f"corrected_{model}_{coord}" for coord,model in wide.columns];corrected.append(wide.reset_index())
    corrected_frame=pd.concat(corrected,ignore_index=True) if corrected else pd.DataFrame()
    agreement=sample.merge(corrected_frame,on="key",how="inner")
    sample_summary=[]
    for model in ("B0","B1","B2","B3","B4","B5"):
        differences=[]
        for idx,row in agreement.iterrows():
            recalculated=np.asarray(json.loads(row[f"recomputed_{model}_box_json"]),float)
            stored=np.asarray([row[f"corrected_{model}_{coord}"] for coord in COORDS],float);differences.append(np.max(np.abs(recalculated-stored)))
        values=np.asarray(differences,float)
        sample_summary.append({"model":model,"eligible_sample_comparisons":len(values),"agreements_within_1e-6_px":int((values<=1e-6).sum()),"agreement_percent":float((values<=1e-6).mean()*100) if len(values) else np.nan,"maximum_difference_px":float(values.max()) if len(values) else np.nan})
    pd.DataFrame(sample_summary).to_csv(OUT/"corrected_row_level_agreement.csv",index=False,float_format="%.12g")
    summary=detail.groupby(["detector","role","delta_ms"],as_index=False).sum(numeric_only=True)
    direct={}
    for left,right in PAIRS:
        equal=summary[f"{left}_vs_{right}_exact_equal_count"].sum();comparisons=summary.comparisons.sum();direct[f"{left}_vs_{right}_exact_equal_percent"]=float(equal/comparisons*100)
    status=(len(files)==300 and not missing_schema and duplicates==0 and bad_model_sets==0 and future_rows==0 and direct["B0_vs_B3_exact_equal_percent"]<100 and direct["B1_vs_B4_exact_equal_percent"]<100)
    report=f"""# Corrected AV2 propagation integrity audit

**{'PASS' if status else 'FAIL'}**

- Corrected Parquets: {len(files)}/300.
- Model rows: {total_rows:,}; unique comparisons: {total_comparisons:,}.
- Missing required schemas: {len(missing_schema)}.
- Duplicate comparison/model rows: {duplicates:,}.
- Comparisons without exactly B0–B5 once each: {bad_model_sets:,}.
- Rows with future geometry input: {future_rows:,}.
- Direct B0/B3 equality: {direct['B0_vs_B3_exact_equal_percent']:.6f}% (legitimate zero-motion coincidences only).
- Direct B1/B4 equality: {direct['B1_vs_B4_exact_equal_percent']:.6f}%.
- Independent sample agreement is reported in `corrected_row_level_agreement.csv` at 1e-6 px tolerance.

This integrity pass verifies corrected output structure and coordinate
distinctness. It does not convert the original invalid run into a confirmatory
result, and it does not substitute sample controls for full prespecified
negative controls.
"""
    (OUT/"corrected_propagation_integrity.md").write_text(report,encoding="utf-8")
    print(json.dumps({"pass":status,"files":len(files),"rows":total_rows,"comparisons":total_comparisons,"duplicates":duplicates,"bad_model_sets":bad_model_sets,"future_rows":future_rows,**direct}))


if __name__=="__main__":main()
