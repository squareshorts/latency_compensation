"""Append frozen benefit-model coefficients without reopening held-out data."""

from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(r"C:\work\auto")
OUT = ROOT / "results" / "object_motion_harm"


def main() -> None:
    pipeline = joblib.load(OUT / "frozen_benefit_probability_model.joblib")
    names = pipeline.named_steps["features"].get_feature_names_out()
    model = pipeline.named_steps["model"]
    coefficients = pd.DataFrame({"record_type": "coefficient", "model": "spline_logistic_gam", "split": "frozen_development_fit",
                                 "feature": names, "coefficient": model.coef_[0]})
    intercept = pd.DataFrame({"record_type": ["intercept"], "model": ["spline_logistic_gam"],
                              "split": ["frozen_development_fit"], "feature": ["intercept"], "coefficient": [model.intercept_[0]]})
    existing = pd.read_csv(OUT / "benefit_probability_model.csv")
    existing = existing[~existing.record_type.isin(["coefficient", "intercept"])]
    pd.concat([existing, coefficients, intercept], ignore_index=True, sort=False).to_csv(
        OUT / "benefit_probability_model.csv", index=False, float_format="%.12g"
    )


if __name__ == "__main__":
    main()
