import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def load_data():
    """
    Load all project datasets.
    """
    visser = pd.read_csv("Visser_final_cleaned_filled.csv")
    bissell = pd.read_csv("Bissell_inverters_production.csv")
    weather = pd.read_csv("edmonton_weather.csv")
    solar = pd.read_csv("solar_radiation.csv")
    price = pd.read_csv("final_pool_price.csv")

    return {
        "Visser": visser,
        "Bissell": bissell,
        "Weather": weather,
        "Solar Radiation": solar,
        "Pool Price": price
    }


def prepare_model_data(df, target_col):
    """
    Keep only numeric columns and prepare X and y.
    """
    numeric_df = df.select_dtypes(include="number").copy()

    if target_col not in numeric_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in numeric columns.")

    X = numeric_df.drop(columns=[target_col])
    y = numeric_df[target_col]

    # Drop missing rows
    combined = pd.concat([X, y], axis=1).dropna()
    X = combined.drop(columns=[target_col])
    y = combined[target_col]

    if X.shape[1] == 0:
        raise ValueError("No numeric feature columns available after removing target.")

    if len(X) < 10:
        raise ValueError("Not enough rows to train a model.")

    return X, y


def train_model(df, target_col):
    """
    Train Random Forest model and return results.
    """
    X, y = prepare_model_data(df, target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    results = {
        "model": model,
        "X": X,
        "y": y,
        "X_test": X_test,
        "y_test": y_test,
        "preds": preds,
        "r2": r2_score(y_test, preds),
        "mae": mean_absolute_error(y_test, preds),
        "rmse": mean_squared_error(y_test, preds) ** 0.5
    }

    return results


def plot_feature_importance(model, feature_names, output_path="feature_importance.png"):
    """
    Save feature importance plot.
    """
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["Feature"], importance_df["Importance"])
    plt.title("Feature Importance")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_actual_vs_predicted(y_test, preds, output_path="actual_vs_predicted.png"):
    """
    Save actual vs predicted scatter plot.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, preds, alpha=0.7)
    plt.title("Actual vs Predicted")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_residuals(y_test, preds, output_path="residual_plot.png"):
    """
    Save residual plot.
    """
    residuals = y_test - preds

    plt.figure(figsize=(8, 6))
    plt.scatter(preds, residuals, alpha=0.7)
    plt.axhline(y=0, linestyle="--")
    plt.title("Residual Plot")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_xai_for_dataset(dataset_name, target_col):
    """
    Train model for selected dataset and save XAI plots.
    """
    datasets = load_data()

    if dataset_name not in datasets:
        raise ValueError(f"Dataset '{dataset_name}' not found.")

    df = datasets[dataset_name]
    results = train_model(df, target_col)

    model = results["model"]
    X = results["X"]
    y_test = results["y_test"]
    preds = results["preds"]

    safe_name = dataset_name.replace(" ", "_")
    plot_feature_importance(model, X.columns, f"{safe_name}_feature_importance.png")
    plot_actual_vs_predicted(y_test, preds, f"{safe_name}_actual_vs_predicted.png")
    plot_residuals(y_test, preds, f"{safe_name}_residual_plot.png")

    print(f"\nXAI plots saved for dataset: {dataset_name}")
    print(f"R² Score: {results['r2']:.3f}")
    print(f"MAE: {results['mae']:.3f}")
    print(f"RMSE: {results['rmse']:.3f}")


if __name__ == "__main__":
    dataset_name = "Bissell"
    target_col = "Bissell_total_filled"

    run_xai_for_dataset(dataset_name, target_col)
  