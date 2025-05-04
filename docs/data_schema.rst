Data Schema
===========

This document describes the data schema used throughout the Telco Churn Pipeline.

Raw Data
--------

Call Detail Records (CDRs)
~~~~~~~~~~~~~~~~~~~~~~~~~

CDRs are the primary data source for the pipeline, containing information about customer calls, SMS, and data usage.

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Column Name
     - Type
     - Description
   * - call_id
     - STRING
     - Unique identifier for the call record
   * - customer_id
     - STRING
     - Unique identifier for the customer
   * - call_time
     - TIMESTAMP
     - Date and time when the call/SMS/data session was initiated
   * - call_duration_seconds
     - INTEGER
     - Duration of the call/SMS/data session in seconds
   * - call_type
     - STRING
     - Type of communication (VOICE, SMS, DATA)
   * - destination_number
     - STRING
     - Phone number or service called/messaged
   * - network_type
     - STRING
     - Network used (2G, 3G, 4G, 5G, WIFI, ROAMING)
   * - cell_tower_id
     - STRING
     - Identifier for the cell tower used
   * - call_result
     - STRING
     - Outcome of the call (COMPLETED, BUSY, NO_ANSWER, FAILED, DROPPED)
   * - call_charge
     - FLOAT
     - Amount charged for the call/SMS/data session
   * - promotion_code
     - STRING
     - Promotion code applied, if any

Customer Profiles
~~~~~~~~~~~~~~~~

Customer profile data contains demographic and account information.

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Column Name
     - Type
     - Description
   * - customer_id
     - STRING
     - Unique identifier for the customer
   * - gender
     - STRING
     - Gender of the customer
   * - senior_citizen
     - INTEGER
     - Whether the customer is a senior citizen (1) or not (0)
   * - partner
     - STRING
     - Whether the customer has a partner (Yes/No)
   * - dependents
     - STRING
     - Whether the customer has dependents (Yes/No)
   * - tenure_days
     - INTEGER
     - Number of days the customer has been with the company
   * - contract_type
     - STRING
     - Type of contract (Month-to-month, One year, Two year)
   * - paperless_billing
     - STRING
     - Whether the customer has paperless billing (Yes/No)
   * - payment_method
     - STRING
     - Payment method used by the customer
   * - monthly_charges
     - FLOAT
     - Monthly charges for the customer
   * - total_charges
     - FLOAT
     - Total charges accumulated by the customer

Churn Labels
~~~~~~~~~~~

Churn label data indicates whether customers have churned.

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Column Name
     - Type
     - Description
   * - customer_id
     - STRING
     - Unique identifier for the customer
   * - churn
     - BOOLEAN
     - Whether the customer has churned (True) or not (False)
   * - churn_date
     - DATE
     - Date when the customer churned, if applicable

Processed Data
-------------

Silver Layer
~~~~~~~~~~~

The silver layer contains cleaned and partitioned data derived from the raw data.

CDR Silver Schema
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Column Name
     - Type
     - Description
   * - call_id
     - STRING
     - Unique identifier for the call record
   * - customer_id
     - STRING
     - Unique identifier for the customer
   * - call_time
     - TIMESTAMP
     - Date and time when the call/SMS/data session was initiated
   * - call_duration_seconds
     - INTEGER
     - Duration of the call/SMS/data session in seconds
   * - call_type
     - STRING
     - Type of communication (VOICE, SMS, DATA)
   * - destination_number
     - STRING
     - Phone number or service called/messaged
   * - network_type
     - STRING
     - Network used (2G, 3G, 4G, 5G, WIFI, ROAMING)
   * - cell_tower_id
     - STRING
     - Identifier for the cell tower used
   * - call_result
     - STRING
     - Outcome of the call (COMPLETED, BUSY, NO_ANSWER, FAILED, DROPPED)
   * - call_charge
     - FLOAT
     - Amount charged for the call/SMS/data session
   * - promotion_code
     - STRING
     - Promotion code applied, if any
   * - call_date
     - DATE
     - Extracted date from call_time
   * - year
     - INTEGER
     - Year extracted from call_date (partition column)
   * - month
     - INTEGER
     - Month extracted from call_date (partition column)
   * - day
     - INTEGER
     - Day extracted from call_date (partition column)
   * - hour
     - INTEGER
     - Hour extracted from call_time

Gold Layer
~~~~~~~~~

The gold layer contains aggregated and feature-engineered data ready for analytics and ML.

CDR Statistics Features
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Column Name
     - Type
     - Description
   * - customer_id
     - STRING
     - Unique identifier for the customer
   * - event_date
     - DATE
     - Date of the events
   * - event_timestamp
     - TIMESTAMP
     - Latest timestamp for the aggregated data
   * - daily_total_minutes
     - FLOAT
     - Total call minutes for the day
   * - daily_total_calls
     - INTEGER
     - Total number of calls for the day
   * - daily_total_sms
     - INTEGER
     - Total number of SMS messages for the day
   * - daily_total_data_mb
     - FLOAT
     - Total data usage in MB for the day
   * - avg_call_duration
     - FLOAT
     - Average call duration in seconds
   * - peak_hour_usage_pct
     - FLOAT
     - Percentage of usage during peak hours (17:00-21:00)
   * - weekend_usage_pct
     - FLOAT
     - Percentage of usage during weekends
   * - dropped_calls_pct
     - FLOAT
     - Percentage of calls that were dropped
   * - customer_service_calls
     - INTEGER
     - Number of calls to customer service
   * - created_timestamp
     - TIMESTAMP
     - Timestamp when the record was created

Billing Features
^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Column Name
     - Type
     - Description
   * - customer_id
     - STRING
     - Unique identifier for the customer
   * - event_date
     - DATE
     - Date of the events
   * - event_timestamp
     - TIMESTAMP
     - Latest timestamp for the aggregated data
   * - monthly_bill_amount
     - FLOAT
     - Monthly bill amount
   * - payment_delay_days
     - INTEGER
     - Number of days payment was delayed
   * - bill_fluctuation
     - FLOAT
     - Fluctuation in bill amount compared to previous month
   * - total_charges_3m
     - FLOAT
     - Total charges in the last 3 months
   * - overdue_payments_3m
     - INTEGER
     - Number of overdue payments in the last 3 months
   * - created_timestamp
     - TIMESTAMP
     - Timestamp when the record was created

Promotion Features
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Column Name
     - Type
     - Description
   * - customer_id
     - STRING
     - Unique identifier for the customer
   * - event_date
     - DATE
     - Date of the events
   * - event_timestamp
     - TIMESTAMP
     - Latest timestamp for the aggregated data
   * - promo_response_rate
     - FLOAT
     - Rate of response to promotions
   * - active_promotions_count
     - INTEGER
     - Number of active promotions
   * - days_since_last_promo
     - INTEGER
     - Days since last promotion was applied
   * - promo_discount_amount
     - FLOAT
     - Amount of discount from promotions
   * - created_timestamp
     - TIMESTAMP
     - Timestamp when the record was created

Feature Store
------------

The feature store organizes features for model training and serving.

Entities
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Entity
     - Description
   * - customer_id
     - Unique identifier for the customer

Feature Views
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Feature View
     - Features
   * - cdr_statistics
     - daily_total_minutes, daily_total_calls, daily_total_sms, daily_total_data_mb, avg_call_duration, peak_hour_usage_pct, weekend_usage_pct, dropped_calls_pct, customer_service_calls
   * - billing_features
     - monthly_bill_amount, payment_delay_days, bill_fluctuation, total_charges_3m, overdue_payments_3m
   * - promotion_features
     - promo_response_rate, active_promotions_count, days_since_last_promo, promo_discount_amount
   * - customer_profile
     - tenure_days, contract_type, payment_method, monthly_charges, total_charges, gender, senior_citizen, partner, dependents, paperless_billing

Model Input/Output
-----------------

Model Input Features
~~~~~~~~~~~~~~~~~~~

The final feature set used for model training and prediction.

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Feature Name
     - Type
     - Description
   * - daily_total_minutes
     - FLOAT
     - Total call minutes per day
   * - daily_total_calls
     - INTEGER
     - Total number of calls per day
   * - daily_total_sms
     - INTEGER
     - Total number of SMS messages per day
   * - avg_call_duration
     - FLOAT
     - Average call duration in seconds
   * - customer_service_calls
     - INTEGER
     - Number of calls to customer service
   * - dropped_calls_pct
     - FLOAT
     - Percentage of calls that were dropped
   * - monthly_bill_amount
     - FLOAT
     - Monthly bill amount
   * - payment_delay_days
     - INTEGER
     - Number of days payment was delayed
   * - total_charges_3m
     - FLOAT
     - Total charges in the last 3 months
   * - promo_response_rate
     - FLOAT
     - Rate of response to promotions
   * - tenure_days
     - INTEGER
     - Number of days the customer has been with the company
   * - contract_type
     - CATEGORICAL
     - Type of contract (One-hot encoded)
   * - payment_method
     - CATEGORICAL
     - Payment method used by the customer (One-hot encoded)
   * - monthly_charges
     - FLOAT
     - Monthly charges for the customer
   * - gender
     - CATEGORICAL
     - Gender of the customer (One-hot encoded)
   * - senior_citizen
     - INTEGER
     - Whether the customer is a senior citizen (1) or not (0)

Model Output
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Output
     - Type
     - Description
   * - churn_probability
     - FLOAT
     - Probability of customer churning (0-1)
   * - churn_prediction
     - BOOLEAN
     - Binary prediction of churn (True/False)

API Schemas
-----------

Prediction API
~~~~~~~~~~~~~

Request:

.. code-block:: json

   {
     "customer_id": "CUST123456"
   }

Response:

.. code-block:: json

   {
     "customer_id": "CUST123456",
     "churn_probability": 0.75,
     "churn_prediction": true,
     "prediction_time": "2023-05-15T14:32:10.123456"
   }

Batch Prediction API
~~~~~~~~~~~~~~~~~~~

Request:

.. code-block:: json

   {
     "customer_ids": ["CUST123456", "CUST789012"]
   }

Response:

.. code-block:: json

   {
     "predictions": [
       {
         "customer_id": "CUST123456",
         "churn_probability": 0.75,
         "churn_prediction": true,
         "prediction_time": "2023-05-15T14:32:10.123456"
       },
       {
         "customer_id": "CUST789012",
         "churn_probability": 0.25,
         "churn_prediction": false,
         "prediction_time": "2023-05-15T14:32:10.123456"
       }
     ],
     "model_version": "2023-05-10 09:15:32",
     "elapsed_time_ms": 125.45
   }

Explanation API
~~~~~~~~~~~~~~

Request:

.. code-block:: json

   {
     "customer_id": "CUST123456",
     "method": "shap",
     "num_features": 10
   }

Response:

.. code-block:: json

   {
     "customer_id": "CUST123456",
     "churn_probability": 0.75,
     "feature_importance": {
       "customer_service_calls": 0.35,
       "tenure_days": -0.25,
       "contract_type_month-to-month": 0.20,
       "monthly_charges": 0.15,
       "payment_delay_days": 0.10,
       "dropped_calls_pct": 0.08,
       "promo_response_rate": -0.07,
       "daily_total_minutes": 0.05,
       "payment_method_electronic_check": 0.04,
       "total_charges_3m": 0.03
     },
     "explanation_method": "shap",
     "prediction_time": "2023-05-15T14:32:10.123456"
   }
