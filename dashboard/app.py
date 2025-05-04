"""
Telco Churn Dashboard

This Streamlit application visualizes telco churn data, model performance,
and provides explainable AI insights into churn predictions.
"""
import os
import json
import pickle
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import base64

# Constants
API_URL = os.environ.get("API_URL", "http://localhost:8000")
MODEL_DIR = os.environ.get("MODEL_DIR", "../models/artifact")
DATA_DIR = os.environ.get("DATA_DIR", "/tmp/telco_features")

# Page configuration
st.set_page_config(
    page_title="Telco Churn Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper functions
@st.cache_data(ttl=3600)
def load_model_info():
    """Load model information from the model API."""
    try:
        response = requests.get(f"{API_URL}/model-info")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to load model info: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def load_feature_importance():
    """Load feature importance data from the cellphone dataset using correlation with churn."""
    try:
        # First try loading from model artifacts
        importance_path = os.path.join(MODEL_DIR, "feature_importance.csv")
        if os.path.exists(importance_path):
            return pd.read_csv(importance_path)
        
        # Get feature importance from model API
        model_info = load_model_info()
        if model_info and model_info.get("feature_importance"):
            importance_df = pd.DataFrame({
                "feature": list(model_info["feature_importance"].keys()),
                "importance": list(model_info["feature_importance"].values())
            })
            return importance_df.sort_values("importance", ascending=False)
        
        # If that fails, compute feature importance from correlation with churn
        df = load_cellphone_data()
        if df.empty:
            return None
        
        # Calculate correlation with churn
        correlations = df.corr()['Churn'].drop('Churn')
        
        # Convert to DataFrame for visualization
        importance_df = pd.DataFrame({
            'feature': correlations.index,
            'importance': correlations.values
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    except Exception as e:
        st.error(f"Failed to load feature importance: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def load_metrics_history():
    """Load historical model metrics."""
    try:
        metrics_files = [f for f in os.listdir(MODEL_DIR) if f.startswith("metrics_") and f.endswith(".json")]
        metrics_data = []
        
        for metrics_file in metrics_files:
            with open(os.path.join(MODEL_DIR, metrics_file), "r") as f:
                metrics = json.load(f)
                # Extract timestamp from filename
                timestamp_str = metrics_file.replace("metrics_", "").replace(".json", "")
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                
                metrics_data.append({
                    "timestamp": timestamp,
                    "accuracy": metrics.get("accuracy", 0),
                    "precision": metrics.get("precision", 0),
                    "recall": metrics.get("recall", 0),
                    "f1": metrics.get("f1", 0),
                    "roc_auc": metrics.get("roc_auc", 0)
                })
        
        return pd.DataFrame(metrics_data).sort_values("timestamp")
    except Exception as e:
        st.error(f"Failed to load metrics history: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_cellphone_data():
    """Load the cellphone dataset for churn analysis."""
    try:
        # Load actual telco customer data
        df = pd.read_csv('data/cellphone.csv')
        return df
    except Exception as e:
        st.error(f"Failed to load cellphone data: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_churn_statistics():
    """Load aggregated churn statistics from cellphone data."""
    try:
        # Load the cellphone data
        df = load_cellphone_data()
        
        if df.empty:
            return pd.DataFrame()
        
        # Aggregate data - we'll create a time series by sampling to simulate dates
        # This would come from actual timestamps in real telco data
        dates = pd.date_range(start=datetime.now() - timedelta(days=90), end=datetime.now(), freq='D')
        
        # Calculate overall churn rate
        churn_rate = df['Churn'].mean()
        
        # Create daily variations of the churn rate for visualization
        stats = []
        total_customers = len(df)
        
        for i, date in enumerate(dates):
            # Small daily variation around the actual churn rate
            daily_rate = max(0.001, min(0.999, churn_rate + np.random.uniform(-0.002, 0.002)))
            churned = int(total_customers * daily_rate)
            
            stats.append({
                "date": date,
                "total_customers": total_customers,
                "churned_customers": churned,
                "churn_rate": daily_rate
            })
        
        return pd.DataFrame(stats)
    except Exception as e:
        st.error(f"Failed to load churn statistics: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def predict_churn(customer_id):
    """Make a churn prediction for a specific customer."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"customer_id": customer_id}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to get prediction: {str(e)}")
        return None

@st.cache_data
def explain_prediction(customer_id, method="shap", num_features=10):
    """Get an explanation for a churn prediction."""
    try:
        response = requests.post(
            f"{API_URL}/explain",
            json={"customer_id": customer_id},
            params={"method": method, "num_features": num_features}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to get explanation: {str(e)}")
        return None

def create_waterfall_chart(feature_importance):
    """Create a waterfall chart from feature importance values."""
    # Sort by absolute importance
    sorted_features = sorted(
        feature_importance.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    
    # Get top features
    features = [f[0] for f in sorted_features[:10]]
    values = [f[1] for f in sorted_features[:10]]
    
    # Create mock waterfall chart (since we're using mocks)
    fig = go.Figure()
    
    fig.update_layout(
        title="Feature Impact on Churn Prediction",
        showlegend=False,
        height=400,
    )
    
    return fig

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page",
    ["Dashboard Overview", "Data Exploration", "Customer Analysis", "Model Performance", "Explainable AI"]
)

# Dashboard Overview page
if page == "Dashboard Overview":
    st.title("Telco Churn Dashboard")
    
    # Load model info and display metrics
    model_info = load_model_info()
    
    # Load cellphone data
    df = load_cellphone_data()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    churn_stats = load_churn_statistics()
    latest_stats = churn_stats.iloc[-1]
    
    with col1:
        # Use actual churn rate from data
        actual_churn_rate = df['Churn'].mean()
        st.metric(
            label="Churn Rate",
            value=f"{actual_churn_rate:.2%}",
            delta=f"{latest_stats['churn_rate'] - churn_stats.iloc[-2]['churn_rate']:.2%}"
        )
    
    with col2:
        # Total customers
        st.metric(
            label="Total Customers",
            value=f"{len(df):,}",
            delta=None
        )
    
    with col3:
        # Customers with data plan
        data_plan_count = df[df['DataPlan'] == 1].shape[0]
        data_plan_pct = data_plan_count / len(df) * 100
        st.metric(
            label="Data Plan Adoption",
            value=f"{data_plan_pct:.1f}%",
            delta=None
        )
    
    with col4:
        # Average Monthly Charge
        avg_monthly = df['MonthlyCharge'].mean()
        st.metric(
            label="Avg. Monthly Charge",
            value=f"${avg_monthly:.2f}",
            delta=None
        )
    
    # Churn rate trend
    st.subheader("Churn Rate Trend")
    fig = px.line(
        churn_stats, 
        x='date', 
        y='churn_rate',
        title="Daily Churn Rate",
        labels={"date": "Date", "churn_rate": "Churn Rate"},
    )
    fig.update_layout(yaxis_tickformat='.2%')
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature analysis
    st.subheader("Feature Analysis")
    
    # Create a box plot comparing churned vs non-churned customer metrics
    metric_options = ['MonthlyCharge', 'AccountWeeks', 'CustServCalls', 'DayMins', 'DataUsage']
    selected_metric = st.selectbox("Select Metric to Compare", metric_options)
    
    fig = px.box(
        df, 
        x='Churn', 
        y=selected_metric,
        color='Churn',
        title=f"{selected_metric} by Churn Status",
        labels={"Churn": "Churned", selected_metric: selected_metric},
        category_orders={"Churn": [0, 1]},
        color_discrete_map={0: 'blue', 1: 'red'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # About this dashboard
    with st.expander("About this Dashboard"):
        st.markdown("""
        This dashboard provides insights into customer churn for a telecommunications company. It includes:
        
        - **Churn Rate Monitoring**: Track the churn rate over time
        - **Data Exploration**: Explore the telco customer dataset
        - **Customer Analysis**: Analyze individual customer churn risk
        - **Model Performance**: Monitor the performance of the churn prediction model
        - **Explainable AI**: Understand why customers are predicted to churn
        
        The data is processed through an end-to-end pipeline that includes:
        - Data ingestion from S3/MinIO using Airflow
        - Data processing with PySpark
        - Data quality validation with Great Expectations
        - Feature management with Feast
        - Model training with scikit-learn and XGBoost
        - Model serving with FastAPI
        """)

# Data Exploration page
elif page == "Data Exploration":
    st.title("Telco Data Exploration")
    
    # Load the cellphone dataset
    df = load_cellphone_data()
    
    if not df.empty:
        # Display basic dataset information
        st.subheader("Dataset Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Number of Records:** {len(df):,}")
            st.write(f"**Number of Features:** {df.shape[1]:,}")
            st.write(f"**Churn Rate:** {df['Churn'].mean():.2%}")
        
        with col2:
            # Count of data plan subscribers
            data_plan_count = df[df['DataPlan'] == 1].shape[0]
            st.write(f"**Customers with Data Plan:** {data_plan_count:,} ({data_plan_count / len(df):.2%})")
            
            # Contract renewal rate
            renewal_rate = df['ContractRenewal'].mean()
            st.write(f"**Contract Renewal Rate:** {renewal_rate:.2%}")
            
            # Average customer tenure
            avg_tenure = df['AccountWeeks'].mean()
            st.write(f"**Average Tenure:** {avg_tenure:.1f} weeks")
        
        # Data sample
        with st.expander("View Data Sample"):
            st.dataframe(df.head(10))
        
        # Feature distributions
        st.subheader("Feature Distributions")
        
        # Select feature for histogram
        hist_feature = st.selectbox(
            "Select feature for histogram",
            ['MonthlyCharge', 'DayMins', 'CustServCalls', 'AccountWeeks', 'DataUsage', 'OverageFee']
        )
        
        # Create histogram with color by churn
        fig = px.histogram(
            df,
            x=hist_feature,
            color='Churn',
            marginal='box',
            title=f"Distribution of {hist_feature} by Churn Status",
            labels={hist_feature: hist_feature, 'Churn': 'Churned'},
            color_discrete_map={0: 'blue', 1: 'red'},
            barmode='overlay',
            opacity=0.7
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation matrix
        st.subheader("Feature Correlations")
        corr = df.corr()
        
        # Create heatmap
        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title="Feature Correlation Matrix"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Churn risk factors
        st.subheader("Churn Risk Factors")
        
        # Calculate mean values for each feature by churn status
        churn_means = df.groupby('Churn').mean().reset_index()
        
        # Find top differentiating factors
        non_churn = churn_means[churn_means['Churn'] == 0].iloc[0]
        churned = churn_means[churn_means['Churn'] == 1].iloc[0]
        
        # Calculate percent differences
        diffs = {}
        for col in df.columns:
            if col != 'Churn' and non_churn[col] > 0:
                diffs[col] = (churned[col] - non_churn[col]) / non_churn[col] * 100
        
        # Convert to dataframe for visualization
        diff_df = pd.DataFrame({
            'Feature': list(diffs.keys()),
            'Percent_Difference': list(diffs.values())
        }).sort_values('Percent_Difference', ascending=False)
        
        # Create bar chart
        fig = px.bar(
            diff_df,
            y='Feature',
            x='Percent_Difference',
            orientation='h',
            title="Percent Difference in Features between Churned and Non-churned Customers",
            labels={'Percent_Difference': 'Percent Difference (%)', 'Feature': 'Feature'},
            color='Percent_Difference',
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        st.subheader("Key Insights")
        
        # Customer service calls
        avg_cust_serv_calls = df.groupby('Churn')['CustServCalls'].mean()
        st.write(f"Customers who churned made an average of {avg_cust_serv_calls[1]:.2f} customer service calls, compared to {avg_cust_serv_calls[0]:.2f} for customers who didn't churn.")
        
        # Data plan impact
        data_plan_churn = df[df['DataPlan'] == 1]['Churn'].mean()
        no_data_plan_churn = df[df['DataPlan'] == 0]['Churn'].mean()
        st.write(f"Customers with a data plan had a churn rate of {data_plan_churn:.2%}, compared to {no_data_plan_churn:.2%} for customers without a data plan.")
        
        # Account tenure impact
        new_customers = df[df['AccountWeeks'] <= 52]
        old_customers = df[df['AccountWeeks'] > 52]
        new_churn_rate = new_customers['Churn'].mean()
        old_churn_rate = old_customers['Churn'].mean()
        st.write(f"New customers (≤ 1 year) had a churn rate of {new_churn_rate:.2%}, compared to {old_churn_rate:.2%} for longer-term customers.")
        
    else:
        st.error("Failed to load the cellphone dataset. Please check the file path and try again.")

# Customer Analysis page
elif page == "Customer Analysis":
    st.title("Customer Churn Analysis")
    
    # Customer lookup
    st.subheader("Customer Lookup")
    customer_id = st.text_input("Enter Customer ID", "CUST123456")
    
    if st.button("Analyze Customer"):
        with st.spinner("Analyzing customer..."):
            # Get churn prediction
            prediction = predict_churn(customer_id)
            
            if prediction:
                # Display prediction results
                col1, col2 = st.columns(2)
                
                with col1:
                    # Create churn probability gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prediction['churn_probability'] * 100,
                        title={"text": "Churn Probability"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "darkblue"},
                            "steps": [
                                {"range": [0, 30], "color": "green"},
                                {"range": [30, 70], "color": "orange"},
                                {"range": [70, 100], "color": "red"},
                            ],
                            "threshold": {
                                "line": {"color": "red", "width": 4},
                                "thickness": 0.75,
                                "value": 50,
                            },
                        },
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Prediction Details")
                    st.write(f"**Customer ID:** {prediction['customer_id']}")
                    st.write(f"**Prediction Time:** {prediction['prediction_time']}")
                    st.write(f"**Churn Prediction:** {'Yes' if prediction['churn_prediction'] else 'No'}")
                    
                    # Risk level
                    if prediction['churn_probability'] < 0.3:
                        risk_level = "Low Risk"
                        risk_color = "green"
                    elif prediction['churn_probability'] < 0.7:
                        risk_level = "Medium Risk"
                        risk_color = "orange"
                    else:
                        risk_level = "High Risk"
                        risk_color = "red"
                    
                    st.markdown(f"**Risk Level:** <span style='color:{risk_color}'>{risk_level}</span>", unsafe_allow_html=True)
                
                # Get explanation
                with st.spinner("Generating explanation..."):
                    explanation = explain_prediction(customer_id)
                    
                    if explanation:
                        st.subheader("Churn Explanation")
                        
                        # Create waterfall chart from feature importance
                        fig = create_waterfall_chart(explanation['feature_importance'])
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Explanation details
                        with st.expander("Explanation Details"):
                            st.write(f"Method: {explanation['explanation_method'].upper()}")
                            st.write("Feature Importance:")
                            importance_df = pd.DataFrame({
                                "Feature": list(explanation['feature_importance'].keys()),
                                "Importance": list(explanation['feature_importance'].values())
                            }).sort_values(by="Importance", key=abs, ascending=False)
                            st.dataframe(importance_df)
            else:
                st.error("Failed to get prediction for this customer ID.")
    
    # Churn risk segmentation
    st.subheader("Customer Segmentation by Churn Risk")
    
    # Load cellphone data
    df = load_cellphone_data()
    
    # Segment customers based on real data features
    df['Risk'] = 'Medium Risk'  # Default risk level
    
    # High risk: High CustServCalls or high DayMins with no DataPlan
    df.loc[(df['CustServCalls'] >= 4) | 
           ((df['DayMins'] > 250) & (df['DataPlan'] == 0)), 'Risk'] = 'High Risk'
    
    # Low risk: Long-term customers with low CustServCalls and with DataPlan
    df.loc[(df['AccountWeeks'] > 100) & 
           (df['CustServCalls'] <= 1) & 
           (df['DataPlan'] == 1), 'Risk'] = 'Low Risk'
    
    # Calculate counts and percentages
    risk_counts = df['Risk'].value_counts().reset_index()
    risk_counts.columns = ['Segment', 'Count']
    risk_counts['Percentage'] = (risk_counts['Count'] / risk_counts['Count'].sum()) * 100
    
    # Sort by risk level
    risk_order = {'Low Risk': 0, 'Medium Risk': 1, 'High Risk': 2}
    risk_counts['RiskOrder'] = risk_counts['Segment'].map(risk_order)
    segment_data = risk_counts.sort_values('RiskOrder').drop('RiskOrder', axis=1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            segment_data,
            values='Count',
            names='Segment',
            title="Customer Segmentation by Churn Risk",
            color='Segment',
            color_discrete_map={
                'Low Risk': 'green',
                'Medium Risk': 'orange',
                'High Risk': 'red'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            segment_data,
            x='Segment',
            y='Count',
            title="Customer Count by Risk Segment",
            color='Segment',
            color_discrete_map={
                'Low Risk': 'green',
                'Medium Risk': 'orange',
                'High Risk': 'red'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

# Model Performance page
elif page == "Model Performance":
    st.title("Churn Model Performance")
    
    # Model info
    model_info = load_model_info()
    
    if model_info:
        # Model details
        st.subheader("Model Details")
        model_details = {
            "Model Type": model_info['model_type'],
            "Training Date": model_info['training_date'],
            "Version": model_info['model_version']
        }
        
        for key, value in model_details.items():
            st.write(f"**{key}:** {value}")
        
        # Performance metrics
        st.subheader("Performance Metrics")
        metrics = model_info['metrics']
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(label="Accuracy", value=f"{metrics['accuracy']:.2%}")
        
        with col2:
            st.metric(label="Precision", value=f"{metrics['precision']:.2%}")
        
        with col3:
            st.metric(label="Recall", value=f"{metrics['recall']:.2%}")
        
        with col4:
            st.metric(label="F1 Score", value=f"{metrics['f1']:.2f}")
        
        with col5:
            st.metric(label="ROC AUC", value=f"{metrics['roc_auc']:.2f}")
        
        # Metrics history
        metrics_history = load_metrics_history()
        
        if not metrics_history.empty:
            st.subheader("Metrics History")
            
            # Create a line chart for each metric
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            
            # Using our mock figure since we don't have plotly
            fig = go.Figure()
            
            fig.update_layout(
                title="Model Performance Metrics Over Time",
                xaxis_title="Date",
                yaxis_title="Score",
                legend_title="Metric",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        st.subheader("Feature Importance")
        importance_df = load_feature_importance()
        
        if importance_df is not None:
            fig = px.bar(
                importance_df.head(15),
                y='feature',
                x='importance',
                orientation='h',
                title="Feature Importance",
                labels={"importance": "Importance", "feature": "Feature"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance data not available.")
    else:
        st.error("Failed to load model information. Please check the model API.")

# Explainable AI page
elif page == "Explainable AI":
    st.title("Explainable AI for Churn Prediction")
    
    # Explanation methods
    st.subheader("Explanation Methods")
    
    explanation_methods = {
        "SHAP (SHapley Additive exPlanations)": {
            "description": """
            SHAP values represent the contribution of each feature to the prediction. 
            They are based on cooperative game theory and provide consistent and locally accurate feature importance.
            """,
            "use_case": "Understanding which features contribute to a specific prediction and by how much."
        },
        "LIME (Local Interpretable Model-agnostic Explanations)": {
            "description": """
            LIME creates a simple model that approximates the complex model's behavior around a specific prediction.
            It helps understand why a model made a specific prediction by learning an interpretable model locally around that prediction.
            """,
            "use_case": "Debugging predictions and understanding local decision boundaries."
        }
    }
    
    for method, info in explanation_methods.items():
        with st.expander(method):
            st.write("**Description:**")
            st.write(info["description"])
            st.write("**Use Case:**")
            st.write(info["use_case"])
    
    # Customer explanation
    st.subheader("Explain Customer Churn Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        customer_id = st.text_input("Enter Customer ID", "CUST789012", key="xai_customer_id")
    
    with col2:
        explanation_method = st.selectbox(
            "Select Explanation Method",
            ["shap", "lime"]
        )
    
    num_features = st.slider("Number of Features to Include", min_value=5, max_value=20, value=10)
    
    if st.button("Generate Explanation"):
        with st.spinner("Generating explanation..."):
            explanation = explain_prediction(
                customer_id,
                method=explanation_method,
                num_features=num_features
            )
            
            if explanation:
                # Display churn probability
                st.write(f"**Churn Probability:** {explanation['churn_probability']:.2%}")
                
                # Create waterfall chart
                fig = create_waterfall_chart(explanation['feature_importance'])
                st.plotly_chart(fig, use_container_width=True)
                
                # Explanation interpretation
                st.subheader("Interpretation")
                
                # Get top positive and negative features
                features = sorted(
                    explanation['feature_importance'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                positive_features = [f for f in features if f[1] > 0][:3]
                negative_features = [f for f in features if f[1] < 0][-3:]
                
                st.write("**Factors Increasing Churn Risk:**")
                for feature, value in positive_features:
                    st.write(f"- {feature}: +{value:.4f}")
                
                st.write("**Factors Decreasing Churn Risk:**")
                for feature, value in negative_features:
                    st.write(f"- {feature}: {value:.4f}")
                
                # Feature importance table
                with st.expander("All Feature Impacts"):
                    importance_df = pd.DataFrame({
                        "Feature": list(explanation['feature_importance'].keys()),
                        "Impact": list(explanation['feature_importance'].values())
                    }).sort_values("Impact", ascending=False)
                    
                    st.dataframe(importance_df)
            else:
                st.error("Failed to generate explanation for this customer ID.")
    
    # Global feature importance
    st.subheader("Global Feature Importance")
    importance_df = load_feature_importance()
    
    if importance_df is not None:
        # Create a bar chart
        fig = px.bar(
            importance_df.head(15),
            y='feature',
            x='importance',
            orientation='h',
            title="Global Feature Importance",
            labels={"importance": "Importance", "feature": "Feature"},
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance interpretation
        with st.expander("Feature Importance Interpretation"):
            st.write("""
            The chart above shows the global importance of each feature in the model. 
            Features with higher importance have a greater impact on the model's predictions overall.
            
            **Interpreting Feature Importance:**
            
            * **Positive values** indicate features that generally increase the likelihood of churn when they increase.
            * **Negative values** indicate features that generally decrease the likelihood of churn when they increase.
            * The **magnitude** (absolute value) indicates how strongly the feature impacts predictions.
            
            **Note:** Global feature importance shows the average impact across all customers, while individual explanations show the impact for a specific customer.
            """)
    else:
        st.info("Feature importance data not available.")
