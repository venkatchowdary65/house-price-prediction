from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from src.features import HousingFeatureEngineer

RANDOM_STATE = 42
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    nonzero = y_true != 0
    mape = float(
        np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100.0
    )
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "MAPE (%)": mape}


def build_preprocessor(numeric_cols: list[str], use_poly: bool = False, poly_degree: int = 2):
    if use_poly:
        num_pipe = Pipeline(
            [
                ("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        num_pipe = Pipeline([("scaler", StandardScaler())])

    return ColumnTransformer(
        transformers=[("num", num_pipe, numeric_cols)],
        remainder="passthrough",
    )


def main():
    data = fetch_california_housing(as_frame=True)
    df = data.frame.drop_duplicates().copy()
    print(f"Dataset shape after deduplication: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    fe_preview = HousingFeatureEngineer(n_clusters=8, random_state=RANDOM_STATE).fit_transform(
        X_train
    )
    all_feature_cols = list(fe_preview.columns)

    experiments = {
        "Linear Regression": (
            Pipeline(
                [
                    ("fe", HousingFeatureEngineer(random_state=RANDOM_STATE)),
                    ("prep", build_preprocessor(all_feature_cols)),
                    ("model", LinearRegression()),
                ]
            ),
            None,
            "grid",
        ),
        "Ridge": (
            Pipeline(
                [
                    ("fe", HousingFeatureEngineer(random_state=RANDOM_STATE)),
                    ("prep", build_preprocessor(all_feature_cols)),
                    ("model", Ridge(random_state=RANDOM_STATE)),
                ]
            ),
            {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
            "grid",
        ),
        "Lasso": (
            Pipeline(
                [
                    ("fe", HousingFeatureEngineer(random_state=RANDOM_STATE)),
                    ("prep", build_preprocessor(all_feature_cols)),
                    ("model", Lasso(max_iter=10000, random_state=RANDOM_STATE)),
                ]
            ),
            {"model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0]},
            "grid",
        ),
        "Elastic Net": (
            Pipeline(
                [
                    ("fe", HousingFeatureEngineer(random_state=RANDOM_STATE)),
                    ("prep", build_preprocessor(all_feature_cols)),
                    ("model", ElasticNet(max_iter=10000, random_state=RANDOM_STATE)),
                ]
            ),
            {
                "model__alpha": [0.001, 0.01, 0.1, 1.0],
                "model__l1_ratio": [0.1, 0.5, 0.9],
            },
            "grid",
        ),
        "Polynomial Ridge": (
            Pipeline(
                [
                    ("fe", HousingFeatureEngineer(random_state=RANDOM_STATE)),
                    ("prep", build_preprocessor(all_feature_cols, use_poly=True, poly_degree=2)),
                    ("model", Ridge(random_state=RANDOM_STATE)),
                ]
            ),
            {
                "prep__num__poly__degree": [2],
                "model__alpha": [0.1, 1.0, 10.0, 100.0],
            },
            "grid",
        ),
        "SVR (RBF)": (
            Pipeline(
                [
                    ("fe", HousingFeatureEngineer(random_state=RANDOM_STATE)),
                    ("prep", build_preprocessor(all_feature_cols)),
                    ("model", SVR(kernel="rbf")),
                ]
            ),
            {
                "model__C": [1.0, 10.0, 100.0],
                "model__epsilon": [0.01, 0.1],
                "model__gamma": ["scale", 0.01, 0.1],
            },
            "random",
        ),
        "Random Forest": (
            Pipeline(
                [
                    ("fe", HousingFeatureEngineer(random_state=RANDOM_STATE)),
                    ("model", RandomForestRegressor(n_jobs=-1, random_state=RANDOM_STATE)),
                ]
            ),
            {
                "model__n_estimators": [100, 300],
                "model__max_depth": [15, 25, None],
                "model__min_samples_leaf": [1, 3, 5],
            },
            "random",
        ),
        "XGBoost": (
            Pipeline(
                [
                    ("fe", HousingFeatureEngineer(random_state=RANDOM_STATE)),
                    (
                        "model",
                        XGBRegressor(
                            objective="reg:squarederror",
                            n_jobs=-1,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            {
                "model__n_estimators": [300, 500],
                "model__max_depth": [4, 6, 8],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__subsample": [0.8, 1.0],
            },
            "random",
        ),
    }

    results = []
    fitted_models = {}

    for name, (pipe, param_grid, search_type) in experiments.items():
        print(f"\n--- Training & tuning: {name} ---")
        if param_grid is None:
            estimator = pipe.fit(X_train, y_train)
            best_params = "Default baseline"
        elif search_type == "grid":
            search = GridSearchCV(
                pipe,
                param_grid=param_grid,
                scoring="neg_root_mean_squared_error",
                cv=cv,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            estimator = search.best_estimator_
            best_params = search.best_params_
        else:
            search = RandomizedSearchCV(
                pipe,
                param_distributions=param_grid,
                n_iter=12,
                scoring="neg_root_mean_squared_error",
                cv=cv,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
            search.fit(X_train, y_train)
            estimator = search.best_estimator_
            best_params = search.best_params_

        y_pred = estimator.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        metrics["Model"] = name
        metrics["Best Params"] = str(best_params)
        results.append(metrics)
        fitted_models[name] = (estimator, y_pred)
        print(
            f"{name}: R2={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}, "
            f"MAE={metrics['MAE']:.4f}, MAPE={metrics['MAPE (%)']:.2f}%"
        )

    results_df = pd.DataFrame(results).sort_values(by="RMSE", ascending=True)
    results_df.to_csv(ARTIFACTS_DIR / "model_comparison.csv", index=False)
    print("\n=== Final Test Set Ranking (by RMSE) ===")
    print(results_df[["Model", "R2", "RMSE", "MAE", "MAPE (%)"]].to_string(index=False))

    best_model_name = results_df.iloc[0]["Model"]
    best_pipeline, best_preds = fitted_models[best_model_name]
    joblib.dump(best_pipeline, ARTIFACTS_DIR / "best_model_pipeline.joblib")
    joblib.dump(best_model_name, ARTIFACTS_DIR / "best_model_name.joblib")
    print(f"\nSaved best model ({best_model_name}) to artifacts/best_model_pipeline.joblib")

    residuals = y_test - best_preds
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_test, best_preds, alpha=0.35, s=12)
    min_val, max_val = min(y_test.min(), best_preds.min()), max(y_test.max(), best_preds.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5)
    axes[0].set_xlabel("Actual MedHouseVal ($100k)")
    axes[0].set_ylabel("Predicted MedHouseVal ($100k)")
    axes[0].set_title(f"{best_model_name}: Actual vs. Predicted")

    axes[1].scatter(best_preds, residuals, alpha=0.35, s=12)
    axes[1].axhline(0, color="r", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Predicted MedHouseVal ($100k)")
    axes[1].set_ylabel("Residual (Actual - Predicted)")
    axes[1].set_title(f"{best_model_name}: Residual Plot")

    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "diagnostics.png", dpi=150)
    plt.close()

    final_model = best_pipeline.named_steps["model"]
    if isinstance(final_model, (RandomForestRegressor, XGBRegressor)):
        X_test_fe = best_pipeline.named_steps["fe"].transform(X_test)
        sample_X = X_test_fe.sample(n=min(500, len(X_test_fe)), random_state=RANDOM_STATE)
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(sample_X)

        plt.figure()
        shap.summary_plot(shap_values, sample_X, show=False)
        plt.tight_layout()
        plt.savefig(ARTIFACTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved SHAP summary plot to artifacts/shap_summary.png")


if __name__ == "__main__":
    main()