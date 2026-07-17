import numpy as np
import pandas as pd


def paired_scene_bootstrap(scene_metrics: pd.DataFrame, model: str, reference: str, metric: str, replicates: int = 10_000, seed: int = 20260717) -> dict:
    pivot = scene_metrics.pivot(index="scene_name", columns="model", values=metric).dropna(subset=[model, reference])
    differences = (pivot[reference] - pivot[model]).to_numpy(float)
    relative = differences / np.maximum(np.abs(pivot[reference].to_numpy(float)), 1e-12)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(replicates, len(differences)))
    boot_difference = np.median(differences[indices], axis=1)
    boot_relative = np.median(relative[indices], axis=1)
    return {
        "model": model, "reference": reference, "metric": metric, "scenes": len(differences),
        "paired_median_difference": float(np.median(differences)),
        "paired_relative_improvement": float(np.median(relative)),
        "ci95_difference_low": float(np.quantile(boot_difference, 0.025)),
        "ci95_difference_high": float(np.quantile(boot_difference, 0.975)),
        "ci95_relative_low": float(np.quantile(boot_relative, 0.025)),
        "ci95_relative_high": float(np.quantile(boot_relative, 0.975)),
        "replicates": replicates, "seed": seed,
    }
