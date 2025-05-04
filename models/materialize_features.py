"""
Feature Materialization Script

This script materializes features from data sources to the Feast feature store,
making them available for online serving.
"""
import os
import sys
import logging
import argparse
from datetime import datetime, timedelta

import pandas as pd
from pyspark.sql import SparkSession
from feast import FeatureStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_spark_session():
    """Create and configure a Spark session."""
    return (SparkSession.builder
            .appName("TelcoFeatureMaterialization")
            .config("spark.sql.adaptive.enabled", "true")
            .getOrCreate())

def extract_cdr_features(spark, cdr_path, output_path):
    """
    Extract features from CDR data and save to Parquet.
    
    Args:
        spark: SparkSession
        cdr_path: Path to CDR Parquet data
        output_path: Path to save feature Parquet files
    """
    logger.info(f"Extracting features from CDR data at {cdr_path}")
    
    # Read CDR data
    cdr_df = spark.read.parquet(cdr_path)
    
    # Register as temp view for SQL
    cdr_df.createOrReplaceTempView("cdr")
    
    # Extract call statistics features
    cdr_stats_df = spark.sql("""
        SELECT
            customer_id,
            date(call_time) as event_date,
            MAX(call_time) as event_timestamp,
            SUM(CASE WHEN call_type = 'VOICE' THEN call_duration_seconds / 60.0 ELSE 0 END) as daily_total_minutes,
            SUM(CASE WHEN call_type = 'VOICE' THEN 1 ELSE 0 END) as daily_total_calls,
            SUM(CASE WHEN call_type = 'SMS' THEN 1 ELSE 0 END) as daily_total_sms,
            SUM(CASE WHEN call_type = 'DATA' THEN call_duration_seconds * 0.001 ELSE 0 END) as daily_total_data_mb,
            AVG(CASE WHEN call_type = 'VOICE' THEN call_duration_seconds ELSE NULL END) as avg_call_duration,
            SUM(CASE WHEN hour(call_time) BETWEEN 17 AND 21 THEN 1 ELSE 0 END) / COUNT(*) * 100 as peak_hour_usage_pct,
            SUM(CASE WHEN dayofweek(call_time) IN (1, 7) THEN 1 ELSE 0 END) / COUNT(*) * 100 as weekend_usage_pct,
            SUM(CASE WHEN call_result = 'DROPPED' THEN 1 ELSE 0 END) / 
                NULLIF(SUM(CASE WHEN call_type = 'VOICE' THEN 1 ELSE 0 END), 0) * 100 as dropped_calls_pct,
            SUM(CASE WHEN destination_number = 'CUSTOMER_SERVICE' THEN 1 ELSE 0 END) as customer_service_calls
        FROM cdr
        GROUP BY customer_id, date(call_time)
    """)
    
    # Extract billing features
    billing_df = spark.sql("""
        SELECT
            customer_id,
            date(call_time) as event_date,
            MAX(call_time) as event_timestamp,
            SUM(call_charge) as daily_charges,
            SUM(call_charge) / COUNT(DISTINCT date(call_time)) as avg_daily_charge
        FROM cdr
        GROUP BY customer_id, date(call_time)
    """)
    
    # Extract promotion features
    promo_df = spark.sql("""
        SELECT
            customer_id,
            date(call_time) as event_date,
            MAX(call_time) as event_timestamp,
            COUNT(DISTINCT promotion_code) as active_promotions_count,
            SUM(CASE WHEN promotion_code != 'NONE' THEN 1 ELSE 0 END) / 
                COUNT(*) * 100 as promo_usage_pct
        FROM cdr
        GROUP BY customer_id, date(call_time)
    """)
    
    # Add created timestamp
    cdr_stats_df = cdr_stats_df.withColumn("created_timestamp", 
                                         cdr_stats_df["event_timestamp"])
    billing_df = billing_df.withColumn("created_timestamp", 
                                      billing_df["event_timestamp"])
    promo_df = promo_df.withColumn("created_timestamp", 
                                  promo_df["event_timestamp"])
    
    # Save feature files
    cdr_stats_output = os.path.join(output_path, "cdr_features.parquet")
    billing_output = os.path.join(output_path, "billing_features.parquet")
    promo_output = os.path.join(output_path, "promotion_features.parquet")
    
    logger.info(f"Saving CDR statistics features to {cdr_stats_output}")
    cdr_stats_df.write.mode("overwrite").parquet(cdr_stats_output)
    
    logger.info(f"Saving billing features to {billing_output}")
    billing_df.write.mode("overwrite").parquet(billing_output)
    
    logger.info(f"Saving promotion features to {promo_output}")
    promo_df.write.mode("overwrite").parquet(promo_output)
    
    return {
        "cdr_features": cdr_stats_output,
        "billing_features": billing_output,
        "promotion_features": promo_output
    }

def materialize_features(feast_repo_path, feature_paths, start_date, end_date):
    """
    Materialize features to Feast online store.
    
    Args:
        feast_repo_path: Path to Feast feature repository
        feature_paths: Dictionary of feature paths
        start_date: Start date for materialization
        end_date: End date for materialization
    """
    logger.info(f"Initializing Feast feature store from {feast_repo_path}")
    store = FeatureStore(repo_path=feast_repo_path)
    
    logger.info(f"Materializing features from {start_date} to {end_date}")
    store.materialize(
        start_date=start_date,
        end_date=end_date
    )
    
    logger.info("Feature materialization complete")

def main():
    parser = argparse.ArgumentParser(description="Materialize features to Feast feature store")
    parser.add_argument("--cdr-path", type=str, required=True,
                        help="Path to CDR Parquet data")
    parser.add_argument("--output-path", type=str, default="/tmp/telco_features",
                        help="Path to save feature files")
    parser.add_argument("--feast-repo-path", type=str, default="../feast_repo",
                        help="Path to Feast feature repository")
    parser.add_argument("--days", type=int, default=90,
                        help="Number of days to materialize")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    
    # Create Spark session
    spark = create_spark_session()
    
    try:
        # Extract features
        feature_paths = extract_cdr_features(spark, args.cdr_path, args.output_path)
        
        # Update Feast paths
        feast_registry_path = os.path.join(args.feast_repo_path, "feature_store.yaml")
        
        # Materialize features
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
        
        materialize_features(args.feast_repo_path, feature_paths, start_date, end_date)
        
    finally:
        # Stop Spark session
        spark.stop()

if __name__ == "__main__":
    main()
