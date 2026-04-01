import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, auc
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Customer Brain", page_icon="👥", layout="wide")

# =======================================================
# 1. LOAD DATASET
# =======================================================
@st.cache_data
def load_data():
    try:
        return pd.read_csv("rfm_customer_data.csv")
    except:
        st.error("rfm_customer_data.csv missing. Run customer_clustering.py to generate 50,000 deep customer logs.")
        return pd.DataFrame()

df = load_data()

st.title("👥 Customer Persona & Churn Prediction Dashboard")
st.markdown("Dynamic behavioral analytics combining Unsupervised K-Means clustering (Buyer Personas) with supervised Logistic Regression (Predictive Lapsing).")

if df.empty:
    st.stop()

# =======================================================
# 2. MACHINE LEARNING ENGINE: PCA & K-MEANS
# =======================================================
@st.cache_resource
def compute_clusters(data):
    features = ['Recency_Days', 'Frequency_LT', 'Monetary_Val_USD', 'Tenure_Days', 'CSAT']
    X_raw = data[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # K-Means
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    # PCA for 3D mapping
    pca = PCA(n_components=3)
    pcs = pca.fit_transform(X_scaled)
    
    data['Cluster'] = clusters
    data['PCA_1'] = pcs[:, 0]
    data['PCA_2'] = pcs[:, 1]
    data['PCA_3'] = pcs[:, 2]
    return data, kmeans, scaler, X_scaled

df_clustered, kmeans_model, scaler, X_scaled = compute_clusters(df.copy())

# Mapping clusters to human-readable personas based on centroids (Simulated here)
cluster_map = {0: "Loyal & Core", 1: "High-Risk Sleepers", 2: "Whales (Big Spenders)", 3: "New & Uncertain"}
df_clustered['Persona Name'] = df_clustered['Cluster'].map(cluster_map)

# =======================================================
# 3. STORYLINE TABS
# =======================================================
tab1, tab2, tab3 = st.tabs(["🚀 RFM Buyer Personas (3D Interactive)", "🧮 Cohort Analytics", "🔮 Live 90-Day Churn Engine"])

# --- TAB 1: 3D CLUSTERS ---
with tab1:
    st.header("Interactive K-Means Topology")
    st.markdown("Rotate and zoom inside the PCA-reduced customer universe. Colors represent unsupervised mathematical clusters identified by K-Means.")
    
    # We sample down to 5000 for web rendering speed
    sample_df = df_clustered.sample(5000, random_state=42)
    fig_3d = px.scatter_3d(sample_df, x='PCA_1', y='PCA_2', z='PCA_3',
                           color='Persona Name',
                           hover_data=['Recency_Days', 'Monetary_Val_USD', 'CSAT'],
                           size_max=3, opacity=0.7,
                           color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    st.plotly_chart(fig_3d, use_container_width=True)

# --- TAB 2: COHORT ANALYTICS ---
with tab2:
    st.header("Behavioral Matrix by Persona")
    # Grouping by Cluster
    summary = df_clustered.groupby('Persona Name')[['Recency_Days', 'Frequency_LT', 'Monetary_Val_USD', 'CSAT']].mean().round(1)
    
    st.dataframe(summary.style.highlight_max(axis=0, color='lightgreen').highlight_min(axis=0, color='lightcoral'), use_container_width=True)
    
    st.markdown("### Lapsing Probability Distribution")
    fig_box = px.box(df_clustered, x="Persona Name", y="Recency_Days", color="Persona Name", title="Recency Spread vs Churn Risk")
    st.plotly_chart(fig_box, use_container_width=True)

# --- TAB 3: LIVE PREDICTIVE LOGISTIC REGRESSION ---
with tab3:
    st.header("Supervised Machine Learning: Predict Lapsing")
    st.markdown("Input operational metrics below to instantly run a live forward-pass through the trained Logistic Regression model.")
    
    col1, col2, col3 = st.columns(3)
    rec_val = col1.number_input("Days Since Last Purchase (Recency)", min_value=1, max_value=300, value=65)
    freq_val = col2.number_input("Lifetime Purchases (Frequency)", min_value=1, max_value=200, value=4)
    csat_val = col3.slider("Customer Sat Score (1-5)", 1, 5, 2)
    monetary_val = st.number_input("Lifetime Spend ($ USD)", min_value=10, max_value=15000, value=250)
    tenure_val = st.number_input("Account Age (Days)", min_value=1, max_value=3000, value=400)
    
    if st.button("CALCULATE CHURN PROBABILITY", use_container_width=True):
        # 1. We must encode their cluster using the trained KMeans model
        input_data = pd.DataFrame([[rec_val, freq_val, monetary_val, tenure_val, csat_val]], columns=['Recency_Days', 'Frequency_LT', 'Monetary_Val_USD', 'Tenure_Days', 'CSAT'])
        scaled_input = scaler.transform(input_data)
        my_cluster = kmeans_model.predict(scaled_input)[0]
        
        # 2. We train/load the logistic model really quickly on the backend
        df_modeling = pd.get_dummies(df_clustered, columns=['Cluster'], drop_first=True)
        predictive_features = ['Recency_Days', 'Frequency_LT', 'Monetary_Val_USD', 'Tenure_Days', 'CSAT', 'Cluster_1', 'Cluster_2', 'Cluster_3']
        X_clf = df_modeling[predictive_features]
        y_clf = df_modeling['Will_Churn_90D']
        
        log_reg = LogisticRegression(max_iter=500, random_state=42)
        log_reg.fit(X_clf, y_clf)
        
        # Construct the exact feature shape for prediction
        pred_dict = {'Recency_Days': rec_val, 'Frequency_LT': freq_val, 'Monetary_Val_USD': monetary_val, 'Tenure_Days': tenure_val, 'CSAT': csat_val, 'Cluster_1': 0, 'Cluster_2': 0, 'Cluster_3': 0}
        if my_cluster > 0:
            pred_dict[f'Cluster_{my_cluster}'] = 1
            
        pred_df = pd.DataFrame([pred_dict])
        prob = log_reg.predict_proba(pred_df)[0][1] * 100
        
        st.markdown("---")
        st.subheader("🤖 AI Output Module:")
        
        if prob > 50:
            st.error(f"⚠️ HIGH RISK: The predicted chance of this customer lapsing within 90 days is **{prob:.1f}%**.")
        else:
            st.success(f"✅ RETAINED: The predicted chance of this customer lapsing within 90 days is successfully low at **{prob:.1f}%**.")

st.markdown("---")
st.markdown("Code, algorithms, and models deployed by **Paras Dhand**.")
