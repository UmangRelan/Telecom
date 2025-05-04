Architecture
===========

This document describes the architecture of the Telco Churn Pipeline system.

Overview
--------

The Telco Churn Pipeline is designed as a modular, scalable, and robust system for
predicting customer churn in a telecommunications company. The architecture follows
a modern data-driven approach with clear separation of concerns between data engineering,
feature engineering, model training, and serving components.

.. code-block:: text

    ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
    │    Sources    │     │     Storage   │     │   Processing  │
    │   (S3/MinIO)  │────▶│ Silver & Gold │────▶│   (PySpark)   │
    └───────────────┘     │   (Parquet)   │     └───────────────┘
                          └───────────────┘             │
                                                        ▼
                          ┌───────────────┐     ┌───────────────┐
                          │  Data Quality │     │Feature Storage│
                          │     (Great    │◀────│    (Feast)    │
                          │ Expectations) │     └───────────────┘
                          └───────────────┘             │
                                                        ▼
    ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
    │  Monitoring   │     │Model Training │     │Model Serving  │
    │(Prometheus &  │◀────│  & Tuning     │◀────│   (FastAPI)   │
    │   Grafana)    │     │   (Optuna)    │     └───────────────┘
    └───────────────┘     └───────────────┘             │
                                                        ▼
                                                ┌───────────────┐
                                                │  Dashboard    │
                                                │  (Streamlit)  │
                                                └───────────────┘

Component Architecture
---------------------

Data Ingestion
~~~~~~~~~~~~~

Data ingestion is handled by Apache Airflow, which orchestrates the following processes:

1. Daily extraction of Call Detail Records (CDRs) from S3/MinIO
2. Triggering PySpark jobs for ETL processing
3. Validating processed data with Great Expectations
4. Materializing features to the Feast feature store

ETL Processing
~~~~~~~~~~~~~

The ETL pipeline uses PySpark for scalable data processing:

1. Reading raw CSV data from the data lake
2. Cleaning and transforming the data
3. Partitioning the data by date and customer ID
4. Writing the processed data to Parquet format

Feature Store
~~~~~~~~~~~~

Feast is used as the feature store to manage and serve features:

1. Features are defined and versioned in the feature repository
2. Data is materialized from offline storage to the online store
3. Features are served in real-time for model inference
4. Point-in-time correct feature retrieval for training

Model Training
~~~~~~~~~~~~~

The model training pipeline includes:

1. Feature extraction from the feature store
2. Preprocessing with scikit-learn pipelines
3. Training multiple model types (LogisticRegression, RandomForest, XGBoost)
4. Hyperparameter tuning with Optuna
5. Model evaluation and selection
6. Model serialization and versioning

Model Serving
~~~~~~~~~~~~

The trained model is served via a FastAPI application:

1. Real-time features are retrieved from Feast
2. Predictions are generated using the latest model
3. SHAP and LIME explanations are provided for transparency
4. Metrics are exposed for Prometheus monitoring

Monitoring and Alerting
~~~~~~~~~~~~~~~~~~~~~~

The system is monitored using:

1. Prometheus for metrics collection
2. Grafana for visualization and dashboards
3. Alerting for model drift and performance degradation

Infrastructure
-------------

The infrastructure is provisioned using Terraform and can be deployed to AWS or run
locally using Docker Compose:

1. S3/MinIO for data storage
2. PostgreSQL for the offline feature store and metadata
3. Redis for the online feature store
4. Kafka for streaming data (if applicable)
5. Prometheus and Grafana for monitoring

Deployment Topology
------------------

The system can be deployed in different environments:

Development
~~~~~~~~~~

* Local Docker Compose setup
* MinIO instead of S3
* Local PostgreSQL and Redis
* All components running on a single machine

Staging/Production
~~~~~~~~~~~~~~~~

* AWS infrastructure with S3, RDS, ElastiCache
* Amazon MSK for Kafka
* Amazon Managed Prometheus and Grafana
* Containerized deployments on ECS or Kubernetes

Data Flow
--------

1. Raw CDR data lands in the raw S3/MinIO bucket
2. Airflow DAG triggers the ETL process
3. PySpark jobs process the data and write to silver/gold buckets
4. Great Expectations validates the data quality
5. Feast materializes features from processed data
6. Training pipeline fetches features and trains models
7. FastAPI serves the model for real-time predictions
8. Streamlit dashboard visualizes results and insights
9. Prometheus monitors system performance and model metrics
10. Grafana displays dashboards and triggers alerts when needed
