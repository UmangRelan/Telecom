Telco Churn Pipeline Documentation
=================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   architecture
   data_schema
   operations
   modules/index

Introduction
-----------

The Telco Churn Pipeline is a comprehensive end-to-end data engineering and machine learning pipeline 
for predicting customer churn in a telecommunications company. This documentation provides details
on the architecture, data schema, operations, and code modules.

Key Features
-----------

* Automated data ingestion and processing
* Data quality validation with Great Expectations
* Feature engineering and storage with Feast
* Machine learning model training and hyperparameter tuning
* Model serving with FastAPI
* Monitoring and observability with Prometheus and Grafana
* Interactive dashboards with Streamlit

Getting Started
--------------

To run the pipeline locally, follow these steps:

1. Clone the repository
2. Install dependencies: ``pip install -r requirements.txt``
3. Run the bootstrap script: ``bash run.sh``

The script will set up the necessary infrastructure, initialize data pipelines, and start the services.

After running the script, you can access:

* Airflow UI: http://localhost:8080
* Streamlit Dashboard: http://localhost:8501
* FastAPI Swagger: http://localhost:8000/docs

Components
---------

The pipeline consists of the following components:

* **Data Ingestion**: Apache Airflow DAGs for automated data ingestion
* **Data Processing**: PySpark jobs for scalable data transformation
* **Data Quality**: Great Expectations suites for data validation
* **Feature Store**: Feast for feature management and serving
* **Model Training**: Scikit-learn and XGBoost models with Optuna tuning
* **Model Serving**: FastAPI application with SHAP and LIME explanations
* **Monitoring**: Prometheus and Grafana for metrics and visualizations
* **Dashboard**: Streamlit app for interactive visualization and exploration

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
