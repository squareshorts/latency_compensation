"""Create the six bounded publication figures and their source CSVs."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(r"C:\work\auto")
OUT = ROOT / "results" / "object_motion_harm"
FIG = OUT / "figures"


def save(fig, name: str, source: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    source.to_csv(FIG / f"{name}.csv", index=False, float_format="%.12g")
    for extension in ("png", "svg", "pdf"):
        fig.savefig(FIG / f"{name}.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    bootstrap = pd.read_csv(OUT / "bootstrap_reversal.csv")
    primary = bootstrap[(bootstrap.role == "heldout") & (bootstrap.delta_ms == 300) & (bootstrap.yaw_group == "high_yaw") &
                        (bootstrap.metric == "normalized_center_error") &
                        (bootstrap.contrast == "median_paired_log_relative_B5_minus_B3")].copy()

    # 1. Expected versus observed hierarchy.
    source1 = pd.DataFrame({"panel": ["Prior expectation"]*2 + ["Observed YOLO11n"]*2 + ["Observed YOLO11s"]*2,
                            "model": ["B3", "B5"]*3,
                            "relative_error_index": [1.0, .85, 1.0, 1+primary.set_index("detector").loc["yolo11n","estimate"],
                                                     1.0, 1+primary.set_index("detector").loc["yolo11s","estimate"]]})
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.7), sharey=True)
    for axis, (panel, group) in zip(axes, source1.groupby("panel", sort=False)):
        axis.bar(group.model, group.relative_error_index, color=["#4C78A8", "#E45756"])
        axis.axhline(1, color="0.5", lw=.8); axis.set_title(panel); axis.set_ylim(0, 1.35)
        for index, value in enumerate(group.relative_error_index): axis.text(index, value+.025, f"{value:.2f}", ha="center")
    axes[0].set_ylabel("Error relative to B3")
    fig.suptitle("Expected hierarchy versus observed hierarchy")
    save(fig, "figure_1_expected_vs_observed", source1)

    # 2. Paired held-out logs.
    logs = pd.read_csv(OUT / "per_log_reversal.csv")
    source2 = logs[(logs.role == "heldout") & (logs.delta_ms == 300) & (logs.yaw_group == "high_yaw")][
        ["detector", "log_id", "B3_normalized_center_error", "B5_normalized_center_error"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), sharex=False, sharey=False)
    for axis, (detector, group) in zip(axes, source2.groupby("detector")):
        limit = max(group.B3_normalized_center_error.max(), group.B5_normalized_center_error.max()) * 1.05
        axis.scatter(group.B3_normalized_center_error, group.B5_normalized_center_error, s=18, alpha=.8, color="#4C78A8")
        axis.plot([0,limit],[0,limit], color="0.3", lw=.8); axis.set_xlim(0,limit); axis.set_ylim(0,limit)
        axis.set_title(detector); axis.set_xlabel("B3 error"); axis.set_ylabel("B5 error")
    fig.suptitle("Paired held-out log medians (300 ms, high yaw)")
    save(fig, "figure_2_paired_logs", source2)

    # 3. Component substitution.
    component = pd.read_csv(OUT / "component_substitution.csv")
    source3 = component[(component.role == "heldout") & (component.delta_ms == 300) &
                        (component.high_yaw.astype(str).str.lower() == "true")].copy()
    source3 = source3[["detector","model","median_of_log_medians_normalized_center_error"]]
    fig, axes = plt.subplots(1,2,figsize=(8.5,3.2),sharey=True)
    for axis,(detector,group) in zip(axes,source3.groupby("detector")):
        group=group.sort_values("model"); axis.bar(group.model,group.median_of_log_medians_normalized_center_error,color="#72B7B2")
        axis.set_title(detector); axis.tick_params(axis="x",rotation=45); axis.set_ylabel("Normalized center error")
    fig.suptitle("Component substitution and oracle diagnostics")
    save(fig,"figure_3_component_substitution",source3)

    # 4. Development harm probability strata.
    envelope = pd.read_csv(OUT / "operating_envelope.csv")
    wanted = ["track_age","velocity_dispersion_m_s","object_distance_m","delta_ms"]
    source4 = envelope[envelope.variable.isin(wanted)].copy()
    fig,axes=plt.subplots(2,2,figsize=(8.5,5.2))
    for axis,variable in zip(axes.ravel(),wanted):
        group=source4[source4.variable==variable]
        for detector,cell in group.groupby("detector"):
            cell = cell.reset_index(drop=True)
            axis.plot(np.arange(len(cell)),cell.benefit_probability,marker="o",label=detector)
        count = group.groupby("detector").size().max()
        axis.set_xticks(np.arange(count)); axis.set_xticklabels([str(index+1) for index in range(count)])
        axis.set_title(variable); axis.set_ylim(0,1); axis.set_ylabel("P(B5 improves)")
        axis.set_xlabel("Stratum (low → high)")
    axes[0,0].legend(frameon=False)
    fig.suptitle("Development operating envelope")
    save(fig,"figure_4_harm_probability",source4)

    # 5. nuScenes-to-AV2 reversal.
    cross=pd.read_csv(OUT/"cross_dataset_harmonization.csv")
    source5=cross[cross.analysis.isin(["direct_common_definition","nuScenes_real_data_feasibility_result"])].copy()
    source5["label"] = source5.apply(lambda row: (
        "nuScenes feasibility implementation" if row.analysis == "nuScenes_real_data_feasibility_result"
        else ("nuScenes common B3/B5" if row.dataset == "nuScenes" else "AV2 common B3/B5")
    ), axis=1)
    fig,axis=plt.subplots(figsize=(7.8,3.2))
    colors=np.where(source5.B5_relative_difference<0,"#54A24B","#E45756")
    axis.barh(source5.label,source5.B5_relative_difference*100,color=colors); axis.axvline(0,color="0.3",lw=.8)
    axis.set_xlabel("B5 relative error difference vs B3 (%)"); axis.set_title("nuScenes real-data feasibility result and AV2 confirmatory reversal")
    save(fig,"figure_5_cross_dataset_reversal",source5)

    # 6. Frozen gate recommendation.
    gate=pd.read_csv(OUT/"gate_heldout_results.csv")
    source6=gate[gate.record_type=="endpoint"].copy()
    fig,axes=plt.subplots(1,2,figsize=(7.5,3.2),sharey=True)
    for axis,(detector,group) in zip(axes,source6.groupby("detector")):
        axis.bar(group.gate,group.median_of_log_median_center_error,color=["#4C78A8","#E45756","#72B7B2"])
        axis.set_title(detector); axis.set_ylabel("Normalized center error")
    fig.suptitle("Frozen uncertainty gate versus universal B3/B5")
    save(fig,"figure_6_gate_recommendation",source6)


if __name__=="__main__":
    main()
