"""
Telco Churn Model Training Script

This script loads features from Feast, trains multiple churn prediction models,
performs hyperparameter tuning with Optuna, and serializes the best model.
"""
import os
import logging
import json
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional

# Machine learning libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import optuna
import xgboost as xgb

# Feature store
from feast import FeatureStore

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TelcoChurnModelTrainer:
    """Telco Churn model training class."""
    
    def __init__(
        self,
        feast_repo_path: str = "../feast_repo",
        model_output_dir: str = "./artifact",
        training_dataset_path: Optional[str] = None,
        churn_label_path: Optional[str] = None,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """
        Initialize the model trainer.
        
        Args:
            feast_repo_path: Path to the Feast feature repository
            model_output_dir: Directory to save model artifacts
            training_dataset_path: Optional path to training dataset (if not using Feast)
            churn_label_path: Path to churn labels dataset
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.feast_repo_path = feast_repo_path
        self.model_output_dir = model_output_dir
        self.training_dataset_path = training_dataset_path
        self.churn_label_path = churn_label_path
        self.test_size = test_size
        self.random_state = random_state
        
        # Create output directory if it doesn't exist
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Initialize feature store if we're using Feast
        if not training_dataset_path:
            try:
                self.feature_store = FeatureStore(repo_path=feast_repo_path)
                logger.info(f"Initialized Feast feature store from {feast_repo_path}")
            except Exception as e:
                logger.error(f"Failed to initialize Feast feature store: {str(e)}")
                raise
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load features from Feast or file and churn labels.
        
        Returns:
            Tuple of features DataFrame and churn labels Series
        """
        if self.training_dataset_path:
            # Load from file
            features_df = pd.read_parquet(self.training_dataset_path)
            logger.info(f"Loaded features from {self.training_dataset_path}")
        else:
            # Load from Feast
            logger.info("Loading features from Feast feature store")
            
            # Get historical features from 90 days ago to today
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            
            # Load churn labels first to get entity list
            churn_labels = pd.read_csv(self.churn_label_path)
            entity_df = pd.DataFrame({"customer_id": churn_labels["customer_id"]})
            entity_df["event_timestamp"] = end_date
            
            # Get features from feature store
            training_df = self.feature_store.get_historical_features(
                entity_df=entity_df,
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
            
            features_df = training_df
            logger.info(f"Loaded {len(features_df)} records from Feast")
        
        # Load churn labels
        churn_labels = pd.read_csv(self.churn_label_path)
        logger.info(f"Loaded churn labels from {self.churn_label_path}")
        
        # Merge features with labels on customer_id
        merged_df = pd.merge(
            features_df, 
            churn_labels[["customer_id", "churn"]], 
            on="customer_id", 
            how="inner"
        )
        
        logger.info(f"Final dataset shape after merging: {merged_df.shape}")
        
        # Check for missing data
        missing_data = merged_df.isnull().sum()
        missing_cols = missing_data[missing_data > 0]
        if not missing_cols.empty:
            logger.warning(f"Missing data in columns: {missing_cols}")
        
        # Separate features and target
        X = merged_df.drop(["churn", "event_timestamp"], axis=1, errors='ignore')
        y = merged_df["churn"]
        
        return X, y
    
    def preprocess_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer]:
        """
        Preprocess data for model training.
        
        Args:
            X: Features DataFrame
            y: Target Series
            
        Returns:
            Tuple of X_train, X_test, y_train, y_test, and preprocessor
        """
        logger.info("Preprocessing data")
        
        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        # Identify numeric and categorical columns
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove customer_id from features
        if "customer_id" in numeric_features:
            numeric_features.remove("customer_id")
        if "customer_id" in categorical_features:
            categorical_features.remove("customer_id")
        
        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='drop'  # This drops customer_id
        )
        
        # Fit preprocessor on training data
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Save preprocessor for later use
        with open(os.path.join(self.model_output_dir, "preprocessor.pkl"), "wb") as f:
            pickle.dump(preprocessor, f)
        
        logger.info(f"Processed data shapes - X_train: {X_train_processed.shape}, X_test: {X_test_processed.shape}")
        
        # Save feature names for later reference
        feature_names = []
        
        # Get numeric feature names
        feature_names.extend(numeric_features)
        
        # Get one-hot encoded feature names
        ohe = preprocessor.named_transformers_['cat']
        if categorical_features:
            cat_feature_names = ohe.get_feature_names_out(categorical_features).tolist()
            feature_names.extend(cat_feature_names)
        
        with open(os.path.join(self.model_output_dir, "feature_names.json"), "w") as f:
            json.dump(feature_names, f)
        
        return X_train_processed, X_test_processed, y_train, y_test, preprocessor
    
    def train_logistic_regression(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[LogisticRegression, Dict[str, float]]:
        """
        Train a logistic regression model.
        
        Returns:
            Tuple of trained model and performance metrics
        """
        logger.info("Training Logistic Regression model")
        
        # Create and train model
        model = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            max_iter=1000,
            random_state=self.random_state
        )
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }
        
        logger.info(f"Logistic Regression performance: {metrics}")
        
        return model, metrics
    
    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[RandomForestClassifier, Dict[str, float]]:
        """
        Train a random forest model.
        
        Returns:
            Tuple of trained model and performance metrics
        """
        logger.info("Training Random Forest model")
        
        # Create and train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }
        
        logger.info(f"Random Forest performance: {metrics}")
        
        return model, metrics
    
    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[xgb.XGBClassifier, Dict[str, float]]:
        """
        Train an XGBoost model.
        
        Returns:
            Tuple of trained model and performance metrics
        """
        logger.info("Training XGBoost model")
        
        # Handle class imbalance
        scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
        
        # Create and train model
        model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }
        
        logger.info(f"XGBoost performance: {metrics}")
        
        return model, metrics
    
    def tune_with_optuna(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_trials: int = 20
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Perform hyperparameter tuning with Optuna.
        
        Returns:
            Tuple of best model and performance metrics
        """
        logger.info(f"Starting Optuna hyperparameter tuning with {n_trials} trials")
        
        def objective(trial):
            # Define the model type parameter
            model_type = trial.suggest_categorical("model_type", ["xgboost", "random_forest"])
            
            if model_type == "xgboost":
                # XGBoost parameters
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "gamma": trial.suggest_float("gamma", 0, 0.5),
                    "scale_pos_weight": np.sum(y_train == 0) / np.sum(y_train == 1),
                    "random_state": self.random_state
                }
                
                model = xgb.XGBClassifier(**params)
            else:
                # Random Forest parameters
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 5, 30),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                    "class_weight": "balanced",
                    "random_state": self.random_state
                }
                
                model = RandomForestClassifier(**params)
            
            # Cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                scoring="roc_auc", 
                cv=cv, 
                n_jobs=-1
            )
            
            return cv_scores.mean()
        
        # Create and run the study
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        # Get the best parameters
        best_params = study.best_params
        best_model_type = best_params.pop("model_type")
        logger.info(f"Best model type: {best_model_type}")
        logger.info(f"Best parameters: {best_params}")
        
        # Train the best model
        if best_model_type == "xgboost":
            # Add back the scale_pos_weight parameter
            best_params["scale_pos_weight"] = np.sum(y_train == 0) / np.sum(y_train == 1)
            best_params["random_state"] = self.random_state
            
            model = xgb.XGBClassifier(**best_params)
        else:
            best_params["class_weight"] = "balanced"
            best_params["random_state"] = self.random_state
            
            model = RandomForestClassifier(**best_params)
        
        # Train on full training set
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }
        
        logger.info(f"Optuna-tuned {best_model_type} performance: {metrics}")
        
        # Save the study results
        with open(os.path.join(self.model_output_dir, "optuna_study.pkl"), "wb") as f:
            pickle.dump(study, f)
        
        return model, metrics
    
    def select_best_model(
        self,
        models_and_metrics: List[Tuple[Any, Dict[str, float]]],
        metric: str = "f1"
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Select the best model based on a specific metric.
        
        Args:
            models_and_metrics: List of tuples (model, metrics)
            metric: Metric to use for selection
            
        Returns:
            Tuple of best model and its metrics
        """
        # Sort models by the specified metric (descending)
        sorted_models = sorted(
            models_and_metrics,
            key=lambda x: x[1][metric],
            reverse=True
        )
        
        best_model, best_metrics = sorted_models[0]
        model_name = type(best_model).__name__
        
        logger.info(f"Best model selected: {model_name} with {metric} = {best_metrics[metric]:.4f}")
        
        return best_model, best_metrics
    
    def save_model(self, model: Any, metrics: Dict[str, float]) -> str:
        """
        Save the trained model to disk.
        
        Args:
            model: Trained model
            metrics: Performance metrics
            
        Returns:
            Path to the saved model
        """
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = os.path.join(self.model_output_dir, f"model_{timestamp}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        # Save metrics
        metrics_path = os.path.join(self.model_output_dir, f"metrics_{timestamp}.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)
        
        # Save the latest model (for serving)
        latest_model_path = os.path.join(self.model_output_dir, "latest_model.pkl")
        with open(latest_model_path, "wb") as f:
            pickle.dump(model, f)
        
        # Save the latest metrics
        latest_metrics_path = os.path.join(self.model_output_dir, "latest_metrics.json")
        with open(latest_metrics_path, "w") as f:
            json.dump(metrics, f)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Latest model saved to {latest_model_path}")
        
        return latest_model_path
    
    def train(self) -> str:
        """
        Run the full training pipeline.
        
        Returns:
            Path to the saved best model
        """
        # Load data
        X, y = self.load_data()
        
        # Preprocess data
        X_train, X_test, y_train, y_test, preprocessor = self.preprocess_data(X, y)
        
        # Train multiple models
        lr_model, lr_metrics = self.train_logistic_regression(X_train, y_train, X_test, y_test)
        rf_model, rf_metrics = self.train_random_forest(X_train, y_train, X_test, y_test)
        xgb_model, xgb_metrics = self.train_xgboost(X_train, y_train, X_test, y_test)
        
        # Tune the best model with Optuna
        opt_model, opt_metrics = self.tune_with_optuna(X_train, y_train, X_test, y_test)
        
        # Collect all models and metrics
        models_and_metrics = [
            (lr_model, lr_metrics),
            (rf_model, rf_metrics),
            (xgb_model, xgb_metrics),
            (opt_model, opt_metrics)
        ]
        
        # Select the best model
        best_model, best_metrics = self.select_best_model(models_and_metrics)
        
        # Save the best model
        model_path = self.save_model(best_model, best_metrics)
        
        # Generate feature importance if the model supports it
        if hasattr(best_model, "feature_importances_"):
            self._save_feature_importance(best_model, X)
        
        return model_path
    
    def _save_feature_importance(self, model, X):
        """Save feature importance for the model."""
        if hasattr(model, "feature_importances_"):
            # Load feature names
            feature_names_path = os.path.join(self.model_output_dir, "feature_names.json")
            with open(feature_names_path, "r") as f:
                feature_names = json.load(f)
            
            # Get feature importances
            importances = model.feature_importances_
            
            # Ensure lengths match by truncating if necessary
            if len(importances) < len(feature_names):
                feature_names = feature_names[:len(importances)]
            elif len(importances) > len(feature_names):
                importances = importances[:len(feature_names)]
            
            # Create dataframe
            importance_df = pd.DataFrame({
                "feature": feature_names,
                "importance": importances
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values("importance", ascending=False)
            
            # Save to file
            importance_path = os.path.join(self.model_output_dir, "feature_importance.csv")
            importance_df.to_csv(importance_path, index=False)
            
            logger.info(f"Feature importance saved to {importance_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train telco churn prediction model")
    parser.add_argument("--feast-repo-path", type=str, default="../feast_repo",
                        help="Path to Feast feature repository")
    parser.add_argument("--model-output-dir", type=str, default="./artifact",
                        help="Directory to save model artifacts")
    parser.add_argument("--training-dataset-path", type=str, default=None,
                        help="Path to training dataset (if not using Feast)")
    parser.add_argument("--churn-label-path", type=str, required=True,
                        help="Path to churn labels dataset")
    
    args = parser.parse_args()
    
    trainer = TelcoChurnModelTrainer(
        feast_repo_path=args.feast_repo_path,
        model_output_dir=args.model_output_dir,
        training_dataset_path=args.training_dataset_path,
        churn_label_path=args.churn_label_path
    )
    
    model_path = trainer.train()
    print(f"Training complete. Model saved to {model_path}")
