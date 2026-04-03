from rag_chatbot import get_answer

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="SPICE Solar Dashboard", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background-color: #f6f9fc;
}
h1, h2, h3 {
    color: #17324d;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #17324d 0%, #0f2236 100%);
}
section[data-testid="stSidebar"] * {
    color: white !important;
}
div[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #d9e2ec;
    padding: 16px;
    border-radius: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
[data-testid="stExpander"] {
    background-color: white;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}
.stButton > button {
    background-color: #f4b942;
    color: black;
    border-radius: 10px;
    border: none;
    padding: 0.5rem 1rem;
    font-weight: 600;
}
.stButton > button:hover {
    background-color: #e3a826;
    color: black;
}
.custom-card {
    background-color: white;
    padding: 18px;
    border-radius: 16px;
    border: 1px solid #d9e2ec;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "EDA Dashboard", "Prediction", "Explainable AI", "RAG Chatbot", "Conclusion"]
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
st.markdown("A professional interactive dashboard for the SPICE project with EDA, prediction, and explainable AI.")

# ---------------- HELPERS ----------------
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

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = mean_squared_error(y_test, preds) ** 0.5

    return {
        "model": model,
        "X": X,
        "y": y,
        "X_test": X_test,
        "y_test": y_test,
        "preds": preds,
        "r2": r2_score(y_test, preds),
        "mae": mean_absolute_error(y_test, preds),
        "rmse": rmse,
    }

def styled_histogram(df, column_name, title):
    fig = px.histogram(
        df,
        x=column_name,
        nbins=20,
        color_discrete_sequence=["#f4b942"],
        title=title,
        template="plotly_white"
    )

    fig.update_traces(
        marker=dict(
            color="#f4b942",
            line=dict(color="#17324d", width=1.5)
        ),
        opacity=0.9
    )

    fig.update_layout(
        title_x=0.1,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#17324d", size=14),
        xaxis=dict(
            title=column_name,
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            title="Count",
            showgrid=True,
            gridcolor="#d9e2ec",
            zeroline=False
        )
    )

    return fig

def styled_scatter(df, x_col, y_col, title):
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=y_col,
        color_continuous_scale="Viridis",
        title=title,
        template="plotly_white",
        hover_data=df.columns.tolist()[:8]
    )

    fig.update_traces(
        marker=dict(
            size=9,
            line=dict(color="#17324d", width=1)
        )
    )

    fig.update_layout(
        title_x=0.1,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#17324d", size=14),
        xaxis=dict(showgrid=True, gridcolor="#e5e7eb"),
        yaxis=dict(showgrid=True, gridcolor="#e5e7eb")
    )

    return fig

def styled_line(df, x_col, y_col, title):
    fig = px.line(
        df,
        x=x_col,
        y=y_col,
        title=title,
        template="plotly_white"
    )
    fig.update_traces(line=dict(color="#17324d", width=3))
    fig.update_layout(title_x=0.1)
    return fig

# ---------------- OVERVIEW ----------------
if page == "Overview":
    st.header("Project Overview")
    st.write("""
    This project analyzes solar energy production and predicts future solar output
    using weather, solar radiation, and electricity market data.
    """)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Datasets Used", "5")
    with c2:
        st.metric("Project Type", "Solar ML")
    with c3:
        st.metric("Main Focus", "EDA + Prediction + XAI")

    st.markdown("""
    <div class='custom-card'>
    <h3>Project Scope</h3>
    <ul>
    <li>Analyze solar production patterns</li>
    <li>Study weather and solar radiation effects</li>
    <li>Explore electricity price and revenue trends</li>
    <li>Build a machine learning model for prediction</li>
    <li>Explain model results using XAI visuals</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='custom-card'>
    <h3>Datasets Used</h3>
    <ul>
    <li>Visser solar production data</li>
    <li>Bissell inverter production data</li>
    <li>Edmonton weather data</li>
    <li>Solar radiation data</li>
    <li>Electricity pool price data</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ---------------- EDA ----------------
elif page == "EDA Dashboard":
    st.header("EDA Dashboard")
    st.write("This section lets users interactively explore solar, weather, radiation, and market data.")

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

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribution View")
        if len(numeric_cols) >= 1:
            hist_col = st.selectbox("Choose column", numeric_cols, key="hist_col")
            fig = styled_histogram(chosen_df, hist_col, f"Distribution of {hist_col}")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Hover on the bars to view values.")
        else:
            st.info("No numeric columns available.")

    with col2:
        st.subheader("Relationship View")
        if len(numeric_cols) >= 2:
            x_col = st.selectbox("Choose X-axis", numeric_cols, key="eda_x")
            y_col = st.selectbox("Choose Y-axis", numeric_cols, index=1, key="eda_y")
            fig2 = styled_scatter(chosen_df, x_col, y_col, f"{y_col} vs {x_col}")
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("Hover over the dots to inspect values. You can also zoom and pan.")
        else:
            st.info("Need at least 2 numeric columns.")

    st.markdown("---")

    st.subheader("Trend View")
    if len(numeric_cols) >= 1:
        trend_col = st.selectbox("Choose trend column", numeric_cols, key="trend_col")
        trend_df = chosen_df.reset_index()
        fig3 = styled_line(trend_df, "index", trend_col, f"Trend of {trend_col}")
        st.plotly_chart(fig3, use_container_width=True)
        st.caption(f"This interactive line chart shows the trend of {trend_col}.")

    with st.expander("View raw data"):
        st.dataframe(chosen_df.head(20), use_container_width=True)

# ---------------- PREDICTION ----------------
elif page == "Prediction":
    st.header("Prediction Model")
    st.write("""
    This section trains a Random Forest model using numeric features from the selected dataset.
    Users can adjust the values below to estimate the target output.
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

            st.subheader("Actual vs Predicted")
            comparison_df = pd.DataFrame({
                "Actual": results["y_test"].values,
                "Predicted": results["preds"]
            })

            fig4 = px.scatter(
                comparison_df,
                x="Actual",
                y="Predicted",
                color="Predicted",
                color_continuous_scale="Turbo",
                title="Actual vs Predicted",
                template="plotly_white"
            )
            fig4.update_layout(title_x=0.1)
            st.plotly_chart(fig4, use_container_width=True)
            st.caption("Hover over the dots to compare actual and predicted values.")

            st.markdown("---")
            st.subheader("Interactive Output Estimator")
            st.write("Adjust the values below and click the button to estimate the output.")

            input_data = {}
            feature_cols = results["X"].columns.tolist()

            for feature in feature_cols:
                feature_min = float(results["X"][feature].min())
                feature_max = float(results["X"][feature].max())
                feature_mean = float(results["X"][feature].mean())

                if feature_min == feature_max:
                    input_data[feature] = st.number_input(
                        feature,
                        value=feature_mean,
                        key=f"const_{feature}"
                    )
                else:
                    input_data[feature] = st.slider(
                        feature,
                        min_value=float(feature_min),
                        max_value=float(feature_max),
                        value=float(feature_mean),
                        key=f"slider_{feature}"
                    )

            input_df = pd.DataFrame([input_data])

            if st.button("Predict Output"):
                predicted_value = results["model"].predict(input_df)[0]
                st.metric("Estimated Output", f"{predicted_value:.2f}")

            with st.expander("View prediction results table"):
                st.dataframe(comparison_df.head(20), use_container_width=True)

# ---------------- XAI ----------------
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

            fig5 = px.bar(
                importance_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="YlOrBr",
                title="Feature Importance",
                template="plotly_white"
            )
            fig5.update_layout(title_x=0.1, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig5, use_container_width=True)
            st.caption("Hover on the bars to see exact importance values.")

            st.subheader("Residual Plot")
            residuals_df = pd.DataFrame({
                "Predicted": preds,
                "Residuals": y_test - preds
            })

            fig6 = px.scatter(
                residuals_df,
                x="Predicted",
                y="Residuals",
                color="Residuals",
                color_continuous_scale="RdYlGn",
                title="Residual Plot",
                template="plotly_white"
            )
            fig6.add_hline(y=0, line_dash="dash")
            fig6.update_layout(title_x=0.1)
            st.plotly_chart(fig6, use_container_width=True)
            st.caption("Hover over the dots to inspect model errors.")

            with st.expander("View feature importance table"):
                st.dataframe(importance_df.reset_index(drop=True), use_container_width=True)

# ---------------- CONCLUSION ----------------
elif page == "Conclusion":
    st.header("Key Insights and Conclusion")
    st.write("""
    - Solar production is influenced by environmental and seasonal variables.
    - Radiation and weather data help explain production changes.
    - Electricity prices affect revenue trends.
    - Machine learning can estimate future output using project features.
    - Explainable AI improves transparency by showing feature importance and model errors.
    """)
    
elif page == "RAG Chatbot":
    st.title("RAG Chatbot")
    st.write("Ask a question about the solar project data.")

    user_question = st.text_input("Enter your question")

    if st.button("Ask"):
        if user_question:
            answer = get_answer(user_question)
            st.write("Answer:")
            st.write(answer)
            