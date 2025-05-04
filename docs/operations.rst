Operations Guide
===============

This document provides instructions for operating and maintaining the Telco Churn Pipeline.

Initial Setup
------------

Prerequisites
~~~~~~~~~~~~

Before starting, ensure you have the following installed:

* Python 3.13
* Docker and Docker Compose
* Git

Repository Setup
~~~~~~~~~~~~~~~

Clone the repository and navigate to the project directory:

.. code-block:: bash

   git clone https://github.com/your-org/telco-pipeline.git
   cd telco-pipeline

Install dependencies:

.. code-block:: bash

   pip install -r requirements.txt

Bootstrap the Pipeline
~~~~~~~~~~~~~~~~~~~~~

Run the bootstrap script to set up all components:

.. code-block:: bash

   bash run.sh

This script will:

1. Start all necessary Docker containers
2. Initialize Airflow database
3. Set up Great Expectations
4. Bootstrap the Feast feature store
5. Launch the Streamlit dashboard

Access the running services:

* Airflow UI: http://localhost:8080
* Streamlit Dashboard: http://localhost:8501
* FastAPI Swagger: http://localhost:8000/docs
* MinIO Console: http://localhost:9001
* Grafana: http://localhost:3000

Daily Operations
--------------

Monitoring the Pipeline
~~~~~~~~~~~~~~~~~~~~~~

The pipeline should be monitored regularly to ensure all components are functioning correctly:

1. Check Airflow UI for any failed DAGs
2. Review Grafana dashboards for performance metrics
3. Monitor data quality reports from Great Expectations
4. Check model performance metrics in the Streamlit dashboard

Data Ingestion
~~~~~~~~~~~~~

The data ingestion pipeline is automated via Airflow. To manually trigger a DAG:

.. code-block:: bash

   cd telco-pipeline
   airflow dags trigger cdr_daily_ingest

To check the status of running DAGs:

.. code-block:: bash

   airflow dags list
   airflow dags state cdr_daily_ingest

Feature Engineering
~~~~~~~~~~~~~~~~~~

To materialize new features to the feature store:

.. code-block:: bash

   cd telco-pipeline
   python models/materialize_features.py --cdr-path s3://telco-silver/cdr/ --feast-repo-path feast_repo --days 90

Model Training
~~~~~~~~~~~~~

To manually train the churn prediction model:

.. code-block:: bash

   cd telco-pipeline
   python models/train.py --feast-repo-path feast_repo --model-output-dir models/artifact --churn-label-path data/churn_labels.csv

Model Serving
~~~~~~~~~~~~

The model serving API runs automatically as part of the Docker Compose setup. To restart it:

.. code-block:: bash

   docker-compose restart model-api

To access the model API directly:

.. code-block:: bash

   curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"customer_id": "CUST123456"}'

Maintenance Tasks
---------------

Database Backups
~~~~~~~~~~~~~~~

To backup the PostgreSQL database:

.. code-block:: bash

   docker-compose exec postgres pg_dump -U telco_admin telco_db > backup_$(date +%Y%m%d).sql

To restore from a backup:

.. code-block:: bash

   cat backup_20230515.sql | docker-compose exec -T postgres psql -U telco_admin telco_db

Updating Dependencies
~~~~~~~~~~~~~~~~~~~

To update Python dependencies:

1. Update the requirements.txt file
2. Rebuild the Docker images:

.. code-block:: bash

   docker-compose build --no-cache
   docker-compose up -d

Cleaning Up
~~~~~~~~~~

To clean temporary files and free disk space:

.. code-block:: bash

   # Remove temporary files
   rm -rf /tmp/telco_*
   
   # Clean Docker volumes
   docker-compose down -v
   
   # Clean up Airflow logs
   find ./logs -name "*.log" -type f -mtime +30 -delete

Troubleshooting
--------------

Common Issues
~~~~~~~~~~~~

1. **Failed Airflow DAGs**:
   
   * Check the logs in the Airflow UI
   * Ensure S3/MinIO is accessible
   * Verify that the PySpark jobs can access the data

2. **Model Serving API Errors**:
   
   * Check if Redis is running (`docker-compose ps redis`)
   * Ensure the model was trained and saved correctly
   * Verify that the feature store is properly configured

3. **Data Quality Failures**:
   
   * Review the Great Expectations validation results
   * Check for schema changes in the source data
   * Verify that the ETL jobs completed successfully

4. **Feast Connection Issues**:
   
   * Ensure Redis and PostgreSQL are running
   * Check connection strings in the feature_store.yaml file
   * Verify that features were materialized correctly

Logging
~~~~~~~

Logs for each component can be found in:

* Airflow: `logs/airflow/`
* PySpark: `logs/spark/`
* FastAPI: `logs/api/`
* Streamlit: `logs/dashboard/`

To view Docker container logs:

.. code-block:: bash

   docker-compose logs -f airflow-webserver
   docker-compose logs -f airflow-scheduler
   docker-compose logs -f model-api
   docker-compose logs -f dashboard

Monitoring Alerts
~~~~~~~~~~~~~~~

Alerts are configured in Grafana and can be sent to:

* Email
* Slack
* PagerDuty

To configure alert destinations:

1. Access Grafana at http://localhost:3000
2. Navigate to Alerting → Notification channels
3. Add your preferred notification method

Scaling the Pipeline
------------------

For larger datasets, the pipeline can be scaled in several ways:

Scaling PySpark
~~~~~~~~~~~~~~

To allocate more resources to PySpark jobs:

1. Update the SparkSubmitOperator configuration in `dags/cdr_ingest.py`:

   .. code-block:: python

      process_cdr_task = SparkSubmitOperator(
          task_id='process_cdr_data',
          application='/opt/airflow/etl/process_cdr.py',
          conn_id='spark_conn',
          executor_cores=4,  # Increase cores
          executor_memory='4g',  # Increase memory
          ...
      )

2. If using a standalone Spark cluster, add more worker nodes

Scaling Airflow
~~~~~~~~~~~~~~

To scale Airflow for more concurrent tasks:

1. Update the Airflow configuration to increase parallelism:

   .. code-block:: ini

      # airflow.cfg
      parallelism = 32
      dag_concurrency = 16
      max_active_runs_per_dag = 16

2. Add more Airflow worker nodes in the Docker Compose file

Disaster Recovery
---------------

Backup Strategy
~~~~~~~~~~~~~~

Regularly back up the following components:

1. PostgreSQL database (contains Airflow metadata and feature store data)
2. Model artifacts
3. Feature registry
4. Great Expectations suites and validations

The backup frequency should be:

* Database: Daily
* Model artifacts: After each training run
* Feature registry: After any changes
* Great Expectations: After any changes

Recovery Procedure
~~~~~~~~~~~~~~~~

In case of a system failure:

1. Restore the PostgreSQL database from the latest backup
2. Restore model artifacts to the models directory
3. Restore the Feast registry
4. Restart all services:

   .. code-block:: bash

      docker-compose down
      docker-compose up -d

5. Verify that all services are functioning correctly

Security Considerations
---------------------

Data Protection
~~~~~~~~~~~~~~

* All sensitive customer data should be encrypted at rest and in transit
* Access to the data should be restricted to authorized personnel only
* Regular security audits should be performed

Authentication and Authorization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* The Airflow UI requires authentication
* The FastAPI endpoints should be secured with API keys or tokens
* Access to the Streamlit dashboard should be restricted to authorized users

Compliance
~~~~~~~~~

Ensure the pipeline complies with relevant regulations such as:

* GDPR
* CCPA
* Industry-specific telecommunications regulations

Regular security updates should be applied to all components.
