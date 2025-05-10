# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from scipy import stats
from io import BytesIO
import base64

st.set_page_config(layout="wide")
st.title("Multiverse Analysis of Lead Exposure in Sparrows")

# Load data
data = pd.read_csv("dosing1.csv")
data.columns = data.columns.str.strip().str.lower()

# Preprocessing
data['treatment'] = data['treat'].map({'trt': 1, 'ctr': 0})
data['background_high'] = (data['background'] == 'contamhigh').astype(int)
data['background_low'] = (data['background'] == 'contamlow').astype(int)
data['town_binary'] = (data['town'] == 'Broken Hill').astype(int)
data['log_lead1'] = np.log1p(data['lead1'])
data['log_lead2'] = np.log1p(data['lead2'])

# Define model specs
specs = {
    "Treatment * Background": "treatment + background_high + background_low + treatment:background_high + treatment:background_low",
    "Treatment * Town": "treatment + town_binary + treatment:town_binary",
    "log(Lead2) * Background": "log_lead2 + background_high + background_low + log_lead2:background_high + log_lead2:background_low"
}

outlier_strategies = ["None", "Z-Score", "IQR"]
dependent_vars = [
    "routine", "leak", "oxphos", "ets", "fcr", "oxce", "alad", 
    "bone_lead", "lead2", "te", "tof", "totalmovement", "csa"
]

st.sidebar.header("Model Settings")
model_spec = st.sidebar.selectbox("Select Model Specification", list(specs.keys()))
outlier_method = st.sidebar.selectbox("Outlier Removal Method", outlier_strategies)
selected_dep_vars = st.sidebar.multiselect("Dependent Variables", dependent_vars, default=dependent_vars[:3])

# Handle outliers
def remove_outliers(df, col):
    df = df.dropna(subset=[col])
    if outlier_method == "Z-Score":
        z_scores = np.abs(stats.zscore(df[col]))
        return df[z_scores < 3]
    elif outlier_method == "IQR":
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        return df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
    else:
        return df

results = []

for dep in selected_dep_vars:
    if dep not in data.columns:
        results.append({"Variable": dep, "Error": f"'{dep}' not in dataset columns"})
        continue
    try:
        df_clean = remove_outliers(data, dep)
        formula = f"{dep} ~ {specs[model_spec]}"
        model = smf.ols(formula, data=df_clean).fit()
        results.append({
            "Variable": dep,
            "Adj R2": round(model.rsquared_adj, 3),
            "AIC": round(model.aic, 1),
            "BIC": round(model.bic, 1),
            "P(treatment)": round(model.pvalues.get("treatment", np.nan), 4)
        })
    except Exception as e:
        results.append({"Variable": dep, "Error": str(e)})

st.subheader("Model Results Summary")
results_df = pd.DataFrame(results)
st.dataframe(results_df)

# Plotting
st.subheader("Specification Curve")
if "Adj R2" in results_df.columns:
    fig = px.bar(results_df, x="Variable", y="Adj R2", color="P(treatment)",
                 title="Specification Curve: Adjusted R-Squared per Dependent Variable")
    st.plotly_chart(fig, use_container_width=True)

# Additional visualizations per dependent variable
st.subheader("Exploratory Data Visualizations")

for dep in selected_dep_vars:
    if dep in data.columns:
        st.markdown(f"### {dep} by Treatment and Background")
        fig = px.box(data, x="treat", y=dep, color="background", title=f"{dep} distribution by Treatment and Background")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"### {dep} by Town")
        fig2 = px.violin(data, x="town", y=dep, box=True, points="all", title=f"{dep} by Town")
        st.plotly_chart(fig2, use_container_width=True)

# Export function
def download_dataframe(df):
    towrite = BytesIO()
    df.to_csv(towrite, index=False)
    towrite.seek(0)
    b64 = base64.b64encode(towrite.read()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="multiverse_results.csv">Download Results CSV</a>'

st.markdown(download_dataframe(results_df), unsafe_allow_html=True)

st.info("This app includes full multiverse exploration, visualization, and export. Additional model types and report generation available on request.")
