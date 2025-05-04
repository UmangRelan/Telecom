"""
Feast Feature Repository Definition

This file defines the entities and features for the Telco Churn feature store,
including customer behavioral features used for churn prediction.
"""
from datetime import timedelta
import os

from feast import Entity, Feature, FeatureView, FileSource, ValueType
from feast.data_source import PushSource
from feast.field import Field
from feast.types import Float32, Int64, Int32, String

# Define the customer entity
customer = Entity(
    name="customer_id",
    value_type=ValueType.STRING,
    description="Unique customer identifier",
    join_keys=["customer_id"],
)

# Define data sources
cdr_source = FileSource(
    name="cdr_source",
    path="file:///tmp/telco_features/cdr_features.parquet",  # Path will be overridden at runtime
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

profile_source = FileSource(
    name="customer_profile_source",
    path="file:///tmp/telco_features/customer_profiles.parquet",  # Path will be overridden at runtime
    timestamp_field="event_timestamp",
)

# Define online sources for real-time serving
online_store_push_source = PushSource(
    name="online_features_source",
    batch_source=cdr_source,
)

# Define feature views
cdr_statistics_view = FeatureView(
    name="cdr_statistics",
    entities=[customer],
    ttl=timedelta(days=90),  # Features valid for 90 days
    schema=[
        Field(name="daily_total_minutes", dtype=Float32),
        Field(name="daily_total_calls", dtype=Int32),
        Field(name="daily_total_sms", dtype=Int32),
        Field(name="daily_total_data_mb", dtype=Float32),
        Field(name="avg_call_duration", dtype=Float32),
        Field(name="peak_hour_usage_pct", dtype=Float32),
        Field(name="weekend_usage_pct", dtype=Float32),
        Field(name="international_calls_count", dtype=Int32),
        Field(name="dropped_calls_pct", dtype=Float32),
        Field(name="customer_service_calls", dtype=Int32),
    ],
    source=cdr_source,
    online=True,
    tags={"team": "ml", "owner": "data-science"},
)

billing_view = FeatureView(
    name="billing_features",
    entities=[customer],
    ttl=timedelta(days=90),
    schema=[
        Field(name="monthly_bill_amount", dtype=Float32),
        Field(name="payment_delay_days", dtype=Int32),
        Field(name="bill_fluctuation", dtype=Float32),
        Field(name="total_charges_3m", dtype=Float32),
        Field(name="overdue_payments_3m", dtype=Int32),
    ],
    source=cdr_source,
    online=True,
    tags={"team": "ml", "owner": "data-science"},
)

promotion_view = FeatureView(
    name="promotion_features",
    entities=[customer],
    ttl=timedelta(days=180),
    schema=[
        Field(name="promo_response_rate", dtype=Float32),
        Field(name="active_promotions_count", dtype=Int32),
        Field(name="days_since_last_promo", dtype=Int32),
        Field(name="promo_discount_amount", dtype=Float32),
    ],
    source=cdr_source,
    online=True,
    tags={"team": "ml", "owner": "marketing"},
)

customer_profile_view = FeatureView(
    name="customer_profile",
    entities=[customer],
    ttl=timedelta(days=365),  # Longer TTL for profile data
    schema=[
        Field(name="tenure_days", dtype=Int32),
        Field(name="contract_type", dtype=String),
        Field(name="payment_method", dtype=String),
        Field(name="monthly_charges", dtype=Float32),
        Field(name="total_charges", dtype=Float32),
        Field(name="gender", dtype=String),
        Field(name="senior_citizen", dtype=Int32),
        Field(name="partner", dtype=String),
        Field(name="dependents", dtype=String),
        Field(name="paperless_billing", dtype=String),
    ],
    source=profile_source,
    online=True,
    tags={"team": "ml", "owner": "data-science"},
)

# Combined feature service for churn prediction
from feast import FeatureService

churn_prediction_fs = FeatureService(
    name="churn_prediction_service",
    features=[
        cdr_statistics_view,
        billing_view,
        promotion_view,
        customer_profile_view,
    ],
    tags={"owner": "ml-team", "stage": "production"},
)
