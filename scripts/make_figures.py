"""Regenerate all six manuscript figures from source tables.

Conventions (uniform across all figures):
- No figure or axes titles; panel tags "(a)"/"(b)" inside the axes.
- No hatch patterns.
- Greyscale criterion: B3 = black (reference), B5 = mid grey (comparator),
  other causal baselines = light grey, oracle substitutions = open (white) bars.
- Detector series use three evenly spaced greys (YOLO11n dark -> RT-DETR-L light).
- In sign-coded panels, dark = favors B5, light = favors B3.
"""
from pathlib import Path
import io
import zipfile
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'source_data'
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size': 9.0,
    'axes.labelsize': 9.0,
    'legend.fontsize': 9.0,
    'xtick.labelsize': 9.0,
    'ytick.labelsize': 9.0,
    'figure.dpi': 180,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'Nimbus Sans', 'Liberation Sans'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

C_B3, C_B5, C_BASE = '0.15', '0.50', '0.82'
DET_GREYS = ['0.25', '0.55', '0.82']
det_order = ['yolo11n', 'yolo11s', 'rtdetr_l']
det_labels = ['YOLO11n', 'YOLO11s', 'RT-DETR-L']


def load_table(name: str) -> pd.DataFrame:
    if (DATA / name).exists():
        return pd.read_csv(DATA / name)
    mapping = {
        'tracker_baselines.csv': ROOT / 'results' / 'sivp_strengthening' / 'tracker_baselines.csv',
        'tracker_bootstrap_intervals.csv': ROOT / 'results' / 'sivp_strengthening' / 'tracker_bootstrap_intervals.csv',
        'sensitivity_recalculated.csv': ROOT / 'results' / 'analysis_closure' / 'sensitivity_recalculated.csv',
        'component_substitution.csv': ROOT / 'results' / 'object_motion_harm' / 'component_substitution.csv',
        'failure_mechanism_summary.csv': ROOT / 'results' / 'object_motion_harm' / 'failure_mechanism_summary.csv',
        'full_frame_metrics_recalculated.csv': ROOT / 'results' / 'analysis_closure' / 'full_frame_metrics_recalculated.csv',
        'cross_dataset_harmonization.csv': ROOT / 'results' / 'object_motion_harm' / 'cross_dataset_harmonization.csv',
    }
    if name in mapping and mapping[name].exists():
        return pd.read_csv(mapping[name])
    zip_path = ROOT / 'submission_polish' / 'latency_compensation_overleaf_v3.zip'
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as z:
            return pd.read_csv(io.BytesIO(z.read(f'source_data/{name}')))
    raise FileNotFoundError(f"Could not find source data for {name}")


def tag(ax, text):
    ax.text(0.02, 0.98, text, transform=ax.transAxes, ha='left', va='top',
            fontsize=9.5, fontweight='bold')


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f'{name}.pdf')
    fig.savefig(FIG / f'{name}.png', dpi=240)
    plt.close(fig)


def main():
    # ------------------------------------------------ figure1: tracker baselines
    base = load_table('tracker_baselines.csv')
    ci = load_table('tracker_bootstrap_intervals.csv')
    sel = base[(base.delta_ms == 300) & (base.yaw_stratum == 'high_yaw')].copy()
    model_order = ['T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6']
    model_labels = ['Stale', 'Const. vel.', 'SORT', 'ByteTrack', 'OC-SORT', 'B3', 'B5']
    colors = [C_BASE] * 5 + [C_B3, C_B5]

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.4), sharey=True)
    for ax, det, dlab, letter in zip(axes, det_order, det_labels, 'abc'):
        d = sel[sel.detector == det].set_index('model').reindex(model_order)
        x = np.arange(len(model_order))
        ax.bar(x, d['median_of_log_center_error'], color=colors,
               edgecolor='black', linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=40, ha='right')
        tag(ax, f'({letter}) {dlab}')
        if letter == 'a':
            ax.set_ylabel('Normalized center error')
        ax.grid(axis='y', alpha=0.25, linewidth=0.5)

        rel = ci[(ci.detector == det) & (ci.delta_ms == 300)
                 & (ci.yaw_stratum == 'high_yaw') & (ci.reference == 'T5')
                 & (ci.candidate == 'T6')]['paired_relative_difference']
        rel_pct = 100 * float(rel.iloc[0])
        b3 = float(d.loc['T5', 'median_of_log_center_error'])
        b5 = float(d.loc['T6', 'median_of_log_center_error'])
        top = max(b3, b5)
        y1 = top * 1.14
        ax.plot([5, 5, 6, 6], [b3 * 1.04, y1, y1, b5 * 1.04],
                color='black', linewidth=0.7)
        ax.text(5.5, y1 * 1.01, f'+{rel_pct:.1f}%', ha='center', va='bottom',
                fontsize=9.0)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.05)
    save(fig, 'figure1_tracker_architecture')

    # ------------------------------------------------ figure2: reversal + latency trend
    pri = ci[(ci.delta_ms == 300) & (ci.yaw_stratum == 'high_yaw')
             & (ci.candidate == 'T6') & (ci.reference == 'T5')].copy()
    pri = pri.set_index('detector').reindex(det_order).reset_index()
    sens = load_table('sensitivity_recalculated.csv')
    lat = sens[sens.dimension == 'latency_ms'].copy()

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    ax = axes[0]
    x = np.arange(len(pri))
    y = 100 * pri['paired_relative_difference'].to_numpy()
    lo = 100 * pri['bootstrap_ci_low'].to_numpy()
    hi = 100 * pri['bootstrap_ci_high'].to_numpy()
    ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt='o', color='black', capsize=4,
                linewidth=1.2)
    ax.axhline(0, color='0.4', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(det_labels)
    ax.set_xlim(-0.5, len(pri) - 0.5)
    ax.set_ylabel('B5 disadvantage relative to B3 (%)')
    tag(ax, '(a)')
    ax.grid(axis='y', alpha=0.25, linewidth=0.5)

    ax = axes[1]
    ax.plot(lat['stratum'].astype(int), 100 * lat['median_relative_difference'],
            marker='o', color='black', linewidth=1.4)
    ax.axhline(0, color='0.4', linewidth=0.8)
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Median paired relative difference (%)')
    tag(ax, '(b)')
    ax.grid(alpha=0.25, linewidth=0.5)
    save(fig, 'figure2_reversal_latency')

    # ------------------------------------------------ figure3: component substitution
    comp = load_table('component_substitution.csv')
    compsel = comp[(comp.role == 'heldout') & (comp.delta_ms == 300)
                   & (comp.high_yaw.astype(str) == 'True')].copy()
    m_order = ['M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7']
    m_labels = ['M0 (B3)', 'M1 (B5)', 'M2 assoc.', 'M3 hist. boxes',
                'M4 hist. 3-D', 'M5 depth', 'M6 future', 'M7 full oracle']
    m_colors = [C_B3, C_B5] + ['white'] * 6   # filled = deployed, open = oracle

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.6), sharey=False)
    for ax, det, dlab, letter in zip(axes, ['yolo11n', 'yolo11s'],
                                     ['YOLO11n', 'YOLO11s'], 'ab'):
        d = compsel[compsel.detector == det].set_index('model').reindex(m_order)
        x = np.arange(len(m_order))
        ax.bar(x, d['median_of_log_medians_normalized_center_error'],
               color=m_colors, edgecolor='black', linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(m_labels, rotation=40, ha='right')
        tag(ax, f'({letter}) {dlab}')
        if letter == 'a':
            ax.set_ylabel('Normalized center error')
        ax.grid(axis='y', alpha=0.25, linewidth=0.5)
    save(fig, 'figure3_component_substitution')

    # ------------------------------------------------ figure4: mechanisms + strata
    mech = load_table('failure_mechanism_summary.csv')
    mech = mech[mech.role == 'heldout']
    mechanisms = ['velocity_sign_error', 'incorrect_association',
                  'velocity_magnitude_error', 'velocity_direction_error',
                  'detector_box_jitter_interpreted_as_motion']
    mech_labels = ['Velocity\nsign', 'Association', 'Velocity\nmagnitude',
                   'Velocity\ndirection', 'Box\njitter']

    strata = [('distance', '0_20m', 'Distance 0-20 m'),
              ('distance', '20_40m', 'Distance 20-40 m'),
              ('distance', '40m_plus', 'Distance >40 m'),
              ('detector_confidence', 'high', 'Confidence high'),
              ('detector_confidence', 'medium', 'Confidence medium'),
              ('detector_confidence', 'low', 'Confidence low')]
    sub = sens[sens.dimension.isin(['distance', 'detector_confidence'])].copy()
    labels, vals = [], []
    for dim, s, lab in strata:
        row = sub[(sub.dimension == dim) & (sub.stratum.astype(str) == s)]
        if len(row):
            labels.append(lab)
            vals.append(100 * float(row.iloc[0]['median_relative_difference']))

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6))
    ax = axes[0]
    width = 0.36
    x = np.arange(len(mechanisms))
    for i, (det, dlab) in enumerate([('yolo11n', 'YOLO11n'), ('yolo11s', 'YOLO11s')]):
        dd = mech[mech.detector == det].set_index('primary_failure_mechanism')
        vv = [float(dd.loc[m, 'percent_of_harmed']) for m in mechanisms]
        ax.bar(x + (i - 0.5) * width, vv, width, label=dlab,
               color=DET_GREYS[i], edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(mech_labels)
    ax.set_ylabel('Harmed objects (%)')
    ax.legend(frameon=False)
    tag(ax, '(a)')
    ax.grid(axis='y', alpha=0.25, linewidth=0.5)

    ax = axes[1]
    y = np.arange(len(labels))
    colors = ['0.25' if v < 0 else '0.80' for v in vals]  # dark favors B5, light favors B3
    ax.barh(y, vals, color=colors, edgecolor='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('B5 relative change in center error (%)')
    tag(ax, '(b)')
    ax.grid(axis='x', alpha=0.25, linewidth=0.5)
    save(fig, 'figure4_failure_sensitivity')

    # ------------------------------------------------ figure5: full-frame metrics
    ff = load_table('full_frame_metrics_recalculated.csv')
    ff = ff[(ff.delta_ms == 300) & (ff.yaw_stratum == 'high_yaw')
            & (ff.scope == 'full_frame') & (ff.model.isin(['T5', 'T6']))]
    metrics = ['ap_50_95', 'ap75', 'recall', 'false_positives_per_frame']
    metric_labels = ['AP50:95', 'AP75', 'Recall', 'FP/frame']
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    x = np.arange(len(metrics))
    width = 0.23
    for i, det in enumerate(det_order):
        d = ff[ff.detector == det].set_index('model')
        changes = [100 * (float(d.loc['T6', m]) / float(d.loc['T5', m]) - 1)
                   for m in metrics]
        ax.bar(x + (i - 1) * width, changes, width, label=det_labels[i],
               color=DET_GREYS[i], edgecolor='black', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel('B5 relative change from B3 (%)')
    ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis='y', alpha=0.25, linewidth=0.5)
    save(fig, 'figure5_full_frame')

    # ------------------------------------------------ figure6: cross-dataset
    cross = load_table('cross_dataset_harmonization.csv')
    sel6 = cross[cross.analysis.isin(['direct_common_definition',
                                      'nuScenes_real_data_feasibility_result'])].copy()

    def _lab(row):
        if row.analysis == 'nuScenes_real_data_feasibility_result':
            return 'nuScenes original\nfeasibility'
        return ('nuScenes common\ndefinition' if row.dataset == 'nuScenes'
                else 'AV2 common\ndefinition')

    sel6['label'] = sel6.apply(_lab, axis=1)
    order6 = ['nuScenes original\nfeasibility', 'nuScenes common\ndefinition',
              'AV2 common\ndefinition']
    sel6 = sel6.set_index('label').reindex(order6).reset_index()
    vals6 = 100 * sel6['B5_relative_difference'].to_numpy()

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    x = np.arange(len(sel6))
    ax.bar(x, vals6, width=0.55, color='0.65', edgecolor='black', linewidth=0.7)
    ax.axhline(0, color='black', linewidth=0.8)
    for xi, v in zip(x, vals6):
        if v >= 0:
            ax.text(xi, v + 1.2, f'+{v:.1f}%', ha='center', va='bottom', fontsize=9.0)
        else:
            ax.text(xi, v - 1.2, f'{v:.1f}%', ha='center', va='top', fontsize=9.0)
    ax.set_xticks(x)
    ax.set_xticklabels(order6)
    ax.set_ylabel('B5 relative error difference vs B3 (%)')
    lo6, hi6 = min(vals6.min(), 0), max(vals6.max(), 0)
    ax.set_ylim(lo6 - 7, hi6 + 6)
    ax.grid(axis='y', alpha=0.25, linewidth=0.5)
    save(fig, 'figure6_cross_dataset_reversal')

    print('Figures written to', FIG)


if __name__ == '__main__':
    main()
