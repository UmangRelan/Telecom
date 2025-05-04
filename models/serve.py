"""
Telco Churn Model Serving API

FastAPI application for serving the trained churn prediction model,
with support for SHAP and LIME explanations.
"""
import os
import pickle
import json
import logging
from typing import Dict, List, Any, Optional, Union

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from feast import FeatureStore
from datetime import datetime
import time
import shap
import lime
import lime.lime_tabular
from prometheus_fastapi_instrumentator import Instrumentator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
MODEL_DIR = os.environ.get("MODEL_DIR", "./artifact")
FEAST_REPO_PATH = os.environ.get("FEAST_REPO_PATH", "../feast_repo")

# Models for request/response
class CustomerData(BaseModel):
    customer_id: str = Field(..., description="Unique identifier for the customer")

class BatchPredictionRequest(BaseModel):
    customer_ids: List[str] = Field(..., description="List of customer IDs to predict")

class PredictionResponse(BaseModel):
    customer_id: str = Field(..., description="Customer ID")
    churn_probability: float = Field(..., description="Probability of customer churning")
    churn_prediction: bool = Field(..., description="Churn prediction (True if likely to churn)")
    prediction_time: str = Field(..., description="Timestamp of prediction")

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    model_version: str = Field(..., description="Version of the model used")
    elapsed_time_ms: float = Field(..., description="Processing time in milliseconds")

class ExplanationResponse(BaseModel):
    customer_id: str = Field(..., description="Customer ID")
    churn_probability: float = Field(..., description="Probability of customer churning")
    feature_importance: Dict[str, float] = Field(..., description="Feature importance values")
    explanation_method: str = Field(..., description="Method used for explanation (SHAP or LIME)")
    prediction_time: str = Field(..., description="Timestamp of prediction")

class ModelInfo(BaseModel):
    model_version: str = Field(..., description="Version of the model")
    model_type: str = Field(..., description="Type of the model")
    training_date: str = Field(..., description="Date the model was trained")
    metrics: Dict[str, float] = Field(..., description="Performance metrics of the model")
    feature_importance: Optional[Dict[str, float]] = Field(None, description="Feature importance if available")

# Initialize FastAPI app
app = FastAPI(
    title="Telco Churn Prediction API",
    description="API for predicting customer churn in a telco company",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Prometheus instrumentation
Instrumentator().instrument(app).expose(app)

# Global variables for model and explainers
model = None
preprocessor = None
feature_names = None
shap_explainer = None
lime_explainer = None
feature_store = None
model_info = None

@app.on_event("startup")
async def startup_event():
    """Load model and explainers on startup."""
    global model, preprocessor, feature_names, shap_explainer, lime_explainer, feature_store, model_info
    
    try:
        # Load latest model
        model_path = os.path.join(MODEL_DIR, "latest_model.pkl")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Loaded model from {model_path}")
        
        # Load preprocessor
        preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.pkl")
        with open(preprocessor_path, "rb") as f:
            preprocessor = pickle.load(f)
        logger.info(f"Loaded preprocessor from {preprocessor_path}")
        
        # Load feature names
        feature_names_path = os.path.join(MODEL_DIR, "feature_names.json")
        with open(feature_names_path, "r") as f:
            feature_names = json.load(f)
        logger.info(f"Loaded {len(feature_names)} feature names")
        
        # Initialize Feast feature store
        feature_store = FeatureStore(repo_path=FEAST_REPO_PATH)
        logger.info(f"Initialized Feast feature store from {FEAST_REPO_PATH}")
        
        # Load model metrics
        metrics_path = os.path.join(MODEL_DIR, "latest_metrics.json")
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        
        # Get model type and version
        model_type = type(model).__name__
        model_version = os.path.getmtime(model_path)
        model_version_str = datetime.fromtimestamp(model_version).strftime("%Y-%m-%d %H:%M:%S")
        
        # Load feature importance if available
        feature_importance = None
        importance_path = os.path.join(MODEL_DIR, "feature_importance.csv")
        if os.path.exists(importance_path):
            importance_df = pd.read_csv(importance_path)
            feature_importance = dict(zip(importance_df["feature"], importance_df["importance"]))
        
        # Build model info
        model_info = {
            "model_version": model_version_str,
            "model_type": model_type,
            "training_date": model_version_str,
            "metrics": metrics,
            "feature_importance": feature_importance
        }
        
        # Initialize SHAP explainer
        if hasattr(model, "predict_proba"):
            # For tree-based models, use TreeExplainer
            if hasattr(model, "feature_importances_") and hasattr(model, "apply"):
                shap_explainer = shap.TreeExplainer(model)
                logger.info("Initialized SHAP TreeExplainer")
            else:
                # For other models, use KernelExplainer with a background dataset
                # (This would need actual data, so we'll skip for now)
                logger.info("SHAP KernelExplainer requires background data, skipping initialization")
                shap_explainer = None
        
        # Initialize LIME explainer
        # (This also needs training data for distribution, we'll initialize it on first use)
        lime_explainer = None
        logger.info("LIME explainer will be initialized on first use")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model is None or preprocessor is None or feature_store is None:
        raise HTTPException(status_code=503, detail="Model not loaded correctly")
    return {"status": "healthy", "model_loaded": True}

@app.get("/model-info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model."""
    if model_info is None:
        raise HTTPException(status_code=503, detail="Model info not available")
    return model_info

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: CustomerData):
    """
    Predict churn for a single customer.
    
    Args:
        data: CustomerData with customer_id
        
    Returns:
        PredictionResponse with churn probability and prediction
    """
    start_time = time.time()
    
    try:
        # Get features from Feast
        customer_df = pd.DataFrame({"customer_id": [data.customer_id]})
        customer_df["event_timestamp"] = datetime.now()
        
        # Get online features
        features = feature_store.get_online_features(
            entity_rows=[{"customer_id": data.customer_id}],
            features=[
                "cdr_statistics:daily_total_minutes",
                "cdr_statistics:daily_total_calls",
                "cdr_statistics:daily_total_sms",
                "cdr_statistics:avg_call_duration",
                "cdr_statistics:customer_service_calls",
                "cdr_statistics:dropped_calls_pct",
                "billing_features:monthly_bill_amount",
                "billing_features:payment_delay_days",
                "billing_features:total_charges_3m",
                "promotion_features:promo_response_rate",
                "customer_profile:tenure_days",
                "customer_profile:contract_type",
                "customer_profile:payment_method",
                "customer_profile:monthly_charges",
                "customer_profile:gender",
                "customer_profile:senior_citizen"
            ]
        ).to_df()
        
        # Preprocess features
        processed_features = preprocessor.transform(features)
        
        # Make prediction
        churn_probability = model.predict_proba(processed_features)[0, 1]
        churn_prediction = churn_probability >= 0.5
        
        # Create response
        prediction_time = datetime.now().isoformat()
        response = {
            "customer_id": data.customer_id,
            "churn_probability": float(churn_probability),
            "churn_prediction": bool(churn_prediction),
            "prediction_time": prediction_time
        }
        
        return response
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(data: BatchPredictionRequest):
    """
    Predict churn for multiple customers.
    
    Args:
        data: BatchPredictionRequest with list of customer_ids
        
    Returns:
        BatchPredictionResponse with list of predictions
    """
    start_time = time.time()
    
    try:
        predictions = []
        
        # Get features from Feast for all customers
        entity_rows = [{"customer_id": cid} for cid in data.customer_ids]
        
        # Get online features
        features = feature_store.get_online_features(
            entity_rows=entity_rows,
            features=[
                "cdr_statistics:daily_total_minutes",
                "cdr_statistics:daily_total_calls",
                "cdr_statistics:daily_total_sms",
                "cdr_statistics:avg_call_duration",
                "cdr_statistics:customer_service_calls",
                "cdr_statistics:dropped_calls_pct",
                "billing_features:monthly_bill_amount",
                "billing_features:payment_delay_days",
                "billing_features:total_charges_3m",
                "promotion_features:promo_response_rate",
                "customer_profile:tenure_days",
                "customer_profile:contract_type",
                "customer_profile:payment_method",
                "customer_profile:monthly_charges",
                "customer_profile:gender",
                "customer_profile:senior_citizen"
            ]
        ).to_df()
        
        # Preprocess features
        processed_features = preprocessor.transform(features)
        
        # Make predictions
        probabilities = model.predict_proba(processed_features)[:, 1]
        churn_predictions = probabilities >= 0.5
        
        # Create predictions list
        prediction_time = datetime.now().isoformat()
        for i, customer_id in enumerate(data.customer_ids):
            predictions.append({
                "customer_id": customer_id,
                "churn_probability": float(probabilities[i]),
                "churn_prediction": bool(churn_predictions[i]),
                "prediction_time": prediction_time
            })
        
        # Calculate elapsed time
        elapsed_time_ms = (time.time() - start_time) * 1000
        
        return {
            "predictions": predictions,
            "model_version": model_info["model_version"],
            "elapsed_time_ms": elapsed_time_ms
        }
    
    except Exception as e:
        logger.error(f"Error making batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.post("/explain", response_model=ExplanationResponse)
async def explain(
    data: CustomerData,
    method: str = Query("shap", enum=["shap", "lime"]),
    num_features: int = Query(10, ge=1, le=50)
):
    """
    Explain a churn prediction for a customer.
    
    Args:
        data: CustomerData with customer_id
        method: Explanation method ("shap" or "lime")
        num_features: Number of top features to include in explanation
        
    Returns:
        ExplanationResponse with feature importance
    """
    global lime_explainer
    
    try:
        # Get features from Feast
        customer_df = pd.DataFrame({"customer_id": [data.customer_id]})
        customer_df["event_timestamp"] = datetime.now()
        
        # Get online features
        features = feature_store.get_online_features(
            entity_rows=[{"customer_id": data.customer_id}],
            features=[
                "cdr_statistics:daily_total_minutes",
                "cdr_statistics:daily_total_calls",
                "cdr_statistics:daily_total_sms",
                "cdr_statistics:avg_call_duration",
                "cdr_statistics:customer_service_calls",
                "cdr_statistics:dropped_calls_pct",
                "billing_features:monthly_bill_amount",
                "billing_features:payment_delay_days",
                "billing_features:total_charges_3m",
                "promotion_features:promo_response_rate",
                "customer_profile:tenure_days",
                "customer_profile:contract_type",
                "customer_profile:payment_method",
                "customer_profile:monthly_charges",
                "customer_profile:gender",
                "customer_profile:senior_citizen"
            ]
        ).to_df()
        
        # Preprocess features
        processed_features = preprocessor.transform(features)
        
        # Make prediction
        churn_probability = model.predict_proba(processed_features)[0, 1]
        
        # Get explanation
        feature_importance = {}
        
        if method.lower() == "shap":
            if shap_explainer is None:
                raise HTTPException(status_code=400, detail="SHAP explainer not available")
            
            # Get SHAP values
            shap_values = shap_explainer.shap_values(processed_features)
            
            # For binary classification, shap_values might be a list of arrays
            if isinstance(shap_values, list):
                # Get values for positive class (churn)
                shap_values = shap_values[1]
            
            # Map SHAP values to feature names
            for i, name in enumerate(feature_names[:len(shap_values[0])]):
                feature_importance[name] = float(shap_values[0][i])
            
        elif method.lower() == "lime":
            # Initialize LIME explainer if not done yet
            if lime_explainer is None:
                lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    processed_features,
                    feature_names=feature_names[:processed_features.shape[1]],
                    class_names=["No Churn", "Churn"],
                    mode="classification"
                )
            
            # Get LIME explanation
            explanation = lime_explainer.explain_instance(
                processed_features[0], 
                model.predict_proba,
                num_features=num_features
            )
            
            # Extract feature importance
            for feature, importance in explanation.as_list():
                feature_importance[feature] = float(importance)
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported explanation method: {method}")
        
        # Sort feature importance by absolute value
        sorted_importance = dict(sorted(
            feature_importance.items(),
            key=lambda item: abs(item[1]),
            reverse=True
        )[:num_features])
        
        # Create response
        prediction_time = datetime.now().isoformat()
        response = {
            "customer_id": data.customer_id,
            "churn_probability": float(churn_probability),
            "feature_importance": sorted_importance,
            "explanation_method": method.lower(),
            "prediction_time": prediction_time
        }
        
        return response
    
    except Exception as e:
        logger.error(f"Error generating explanation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")

if __name__ == "__main__":
    # Run the FastAPI app with uvicorn
    uvicorn.run(
        "serve:app",
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
