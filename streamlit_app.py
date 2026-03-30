import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="SPICE Solar Dashboard", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background-color: #f7fbff;
}
h1, h2, h3 {
    color: #17324d;
}
section[data-testid="stSidebar"] {
    background-color: #17324d;
}
section[data-testid="stSidebar"] * {
    color: white !important;
}
div[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #d9e2ec;
    padding: 14px;
    border-radius: 14px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "EDA Dashboard", "Prediction", "Explainable AI", "Conclusion"]
)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    visser = pd.read_csv("Visser_final_cleaned_filled.csv")
    bissell = pd.read_csv("Bissell_inverters_production.csv")
    weather = pd.read_csv("edmonton_weather.csv")
    solar = pd.read_csv("solar_radiation.csv")
    price = pd.read_csv("final_pool_price.csv")
    return visser, bissell, weather, solar, price

visser, bissell, weather, solar, price = load_data()

# ---------------- APP TITLE ----------------
st.title("☀️ Solar Energy Production Analysis and Prediction")
st.markdown("A professional Streamlit dashboard for the SPICE project with EDA, prediction, and explainable AI.")

# ---------------- HELPER FUNCTIONS ----------------
def get_dataset(name):
    if name == "Visser":
        return visser.copy()
    elif name == "Bissell":
        return bissell.copy()
    elif name == "Weather":
        return weather.copy()
    elif name == "Solar Radiation":
        return solar.copy()
    else:
        return price.copy()

def prepare_model_data(df, target_col):
    numeric_df = df.select_dtypes(include="number").copy()
    if target_col not in numeric_df.columns:
        return None, None
    X = numeric_df.drop(columns=[target_col])
    y = numeric_df[target_col]

    # Remove rows with missing values
    full_df = pd.concat([X, y], axis=1).dropna()
    X = full_df.drop(columns=[target_col])
    y = full_df[target_col]

    if X.shape[1] == 0 or len(X) < 10:
        return None, None

    return X, y

def train_random_forest(df, target_col):
    X, y = prepare_model_data(df, target_col)
    if X is None or y is None:
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = mean_squared_error(y_test, preds) ** 0.5

    return {
        "model": model,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preds": preds,
        "r2": r2_score(y_test, preds),
        "mae": mean_absolute_error(y_test, preds),
        "rmse": rmse,
    }

# ---------------- OVERVIEW PAGE ----------------
if page == "Overview":
    st.header("Project Overview")
    st.write("""
    This project analyzes solar energy production and predicts future output
    using weather, solar radiation, and electricity market data.
    """)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Datasets Used", "5")
    with c2:
        st.metric("Project Type", "Solar ML")
    with c3:
        st.metric("Main Focus", "EDA + XAI")

    st.subheader("Project Scope")
    st.write("""
    - Analyze solar production patterns
    - Study weather and solar radiation effects
    - Explore electricity price and revenue trends
    - Build a machine learning model for prediction
    - Explain model results using XAI visuals
    """)

    st.subheader("Datasets Used")
    st.write("- Visser solar production data")
    st.write("- Bissell inverter production data")
    st.write("- Edmonton weather data")
    st.write("- Solar radiation data")
    st.write("- Electricity pool price data")

# ---------------- EDA PAGE ----------------
elif page == "EDA Dashboard":
    st.header("EDA Dashboard")
    st.write("This section explores the main patterns in solar, weather, radiation, and market data.")

    dataset_name = st.selectbox(
        "Choose dataset",
        ["Visser", "Bissell", "Weather", "Solar Radiation", "Pool Price"]
    )

    chosen_df = get_dataset(dataset_name)
    numeric_cols = chosen_df.select_dtypes(include="number").columns.tolist()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Rows", chosen_df.shape[0])
    with c2:
        st.metric("Columns", chosen_df.shape[1])
    with c3:
        st.metric("Numeric Columns", len(numeric_cols))

    st.markdown("---")

    left, right = st.columns(2)

    with left:
        st.subheader("Distribution Chart")
        if len(numeric_cols) >= 1:
            hist_col = st.selectbox("Choose column for distribution", numeric_cols, key="hist_col")

            fig, ax = plt.subplots()
            ax.hist(chosen_df[hist_col].dropna(), bins=20)
            ax.set_title(f"Distribution of {hist_col}")
            ax.set_xlabel(hist_col)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

            st.caption(f"This chart shows how values of {hist_col} are distributed.")
        else:
            st.info("No numeric columns available.")

    with right:
        st.subheader("Relationship Chart")
        if len(numeric_cols) >= 2:
            x_col = st.selectbox("Choose X-axis", numeric_cols, key="eda_x")
            y_col = st.selectbox("Choose Y-axis", numeric_cols, index=1, key="eda_y")

            fig2, ax2 = plt.subplots()
            ax2.scatter(chosen_df[x_col], chosen_df[y_col], alpha=0.7)
            ax2.set_title(f"{y_col} vs {x_col}")
            ax2.set_xlabel(x_col)
            ax2.set_ylabel(y_col)
            st.pyplot(fig2)

            st.caption(f"This chart helps compare {x_col} and {y_col}.")
        else:
            st.info("Need at least 2 numeric columns.")

    st.markdown("---")

    st.subheader("Trend Chart")
    if len(numeric_cols) >= 1:
        trend_col = st.selectbox("Choose column for trend", numeric_cols, key="trend_col")

        fig3, ax3 = plt.subplots()
        ax3.plot(chosen_df.index, chosen_df[trend_col])
        ax3.set_title(f"Trend of {trend_col}")
        ax3.set_xlabel("Index")
        ax3.set_ylabel(trend_col)
        st.pyplot(fig3)

        st.caption(f"This line chart shows how {trend_col} changes across the dataset.")

    with st.expander("View raw data"):
        st.dataframe(chosen_df.head(20))

# ---------------- PREDICTION PAGE ----------------
elif page == "Prediction":
    st.header("Prediction Model")
    st.write("""
    This section trains a Random Forest model using numeric features from the selected dataset
    and predicts the selected target variable.
    """)

    dataset_name = st.selectbox(
        "Choose dataset for modeling",
        ["Visser", "Bissell", "Weather", "Solar Radiation", "Pool Price"],
        key="pred_dataset"
    )

    model_df = get_dataset(dataset_name)
    numeric_cols = model_df.select_dtypes(include="number").columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("This dataset does not have enough numeric columns for modeling.")
    else:
        target_col = st.selectbox("Choose target column", numeric_cols, key="target_col")

        results = train_random_forest(model_df, target_col)

        if results is None:
            st.error("Unable to train model. Please choose another target column or dataset.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("R² Score", f"{results['r2']:.3f}")
            with c2:
                st.metric("MAE", f"{results['mae']:.3f}")
            with c3:
                st.metric("RMSE", f"{results['rmse']:.3f}")

            st.subheader("Features Used")
            st.write(list(results["X"].columns))

            st.subheader("Actual vs Predicted")
            fig4, ax4 = plt.subplots()
            ax4.scatter(results["y_test"], results["preds"], alpha=0.7)
            ax4.set_xlabel("Actual Values")
            ax4.set_ylabel("Predicted Values")
            ax4.set_title("Actual vs Predicted")
            st.pyplot(fig4)

            st.caption("A strong model will show points closer to the diagonal pattern.")

            comparison_df = pd.DataFrame({
                "Actual": results["y_test"].values,
                "Predicted": results["preds"]
            })

            with st.expander("View prediction results table"):
                st.dataframe(comparison_df.head(20))

# ---------------- XAI PAGE ----------------
elif page == "Explainable AI":
    st.header("Explainable AI")
    st.write("""
    This section explains how the model makes predictions using feature importance
    and residual analysis.
    """)

    dataset_name = st.selectbox(
        "Choose dataset for XAI",
        ["Visser", "Bissell", "Weather", "Solar Radiation", "Pool Price"],
        key="xai_dataset"
    )

    xai_df = get_dataset(dataset_name)
    numeric_cols = xai_df.select_dtypes(include="number").columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("This dataset does not have enough numeric columns for XAI.")
    else:
        target_col = st.selectbox("Choose target column for XAI", numeric_cols, key="xai_target")

        results = train_random_forest(xai_df, target_col)

        if results is None:
            st.error("Unable to generate XAI plots for this selection.")
        else:
            model = results["model"]
            X = results["X"]
            y_test = results["y_test"]
            preds = results["preds"]

            st.subheader("Feature Importance")
            importance_df = pd.DataFrame({
                "Feature": X.columns,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False)

            fig5, ax5 = plt.subplots()
            ax5.barh(importance_df["Feature"], importance_df["Importance"])
            ax5.set_title("Feature Importance")
            ax5.set_xlabel("Importance Score")
            ax5.set_ylabel("Feature")
            st.pyplot(fig5)

            st.caption("Features with higher importance have more influence on the prediction.")

            st.subheader("Residual Plot")
            residuals = y_test - preds

            fig6, ax6 = plt.subplots()
            ax6.scatter(preds, residuals, alpha=0.7)
            ax6.axhline(y=0, linestyle="--")
            ax6.set_title("Residual Plot")
            ax6.set_xlabel("Predicted Values")
            ax6.set_ylabel("Residuals")
            st.pyplot(fig6)

            st.caption("Residuals show the difference between actual and predicted values.")

            st.subheader("Top Important Features Table")
            st.dataframe(importance_df.reset_index(drop=True))

# ---------------- CONCLUSION PAGE ----------------
elif page == "Conclusion":
    st.header("Key Insights and Conclusion")
    st.write("""
    - Solar production is influenced by environmental and seasonal variables.
    - Radiation and weather data can help explain production changes.
    - Machine learning can estimate future output using numeric project features.
    - Explainable AI improves transparency by showing feature importance and model errors.
    - This dashboard supports data exploration, model evaluation, and client understanding.
    """)