Code Documentation
=================

This section provides detailed documentation for the various code modules in the Telco Churn Pipeline.

.. toctree::
   :maxdepth: 2
   :caption: Modules:

   dags
   etl
   feast
   models
   ge
   dashboard

Data Ingestion (DAGs)
-------------------

The `dags` module contains Airflow DAG definitions for automating data ingestion and processing.

.. automodule:: dags.cdr_ingest
   :members:
   :undoc-members:
   :show-inheritance:

ETL Processing
------------

The `etl` module contains PySpark jobs for data processing and transformation.

.. automodule:: etl.process_cdr
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: etl.benchmark
   :members:
   :undoc-members:
   :show-inheritance:

Feature Store
-----------

The `feast_repo` module defines the feature store entities and feature views.

.. automodule:: feast_repo.feature_repo
   :members:
   :undoc-members:
   :show-inheritance:

Model Training and Serving
------------------------

The `models` module contains code for training and serving machine learning models.

.. automodule:: models.train
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: models.serve
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: models.materialize_features
   :members:
   :undoc-members:
   :show-inheritance:

Data Quality
-----------

The `ge` module contains Great Expectations suites for data validation.

.. automodule:: ge.expectations.telco_suite
   :members:
   :undoc-members:
   :show-inheritance:

Dashboard
--------

The `dashboard` module contains the Streamlit dashboard for visualizing results.

.. automodule:: dashboard.app
   :members:
   :undoc-members:
   :show-inheritance:
