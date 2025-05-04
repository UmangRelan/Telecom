"""
Great Expectations Suite for Telco CDR data

This file defines data quality expectations for the telco CDR data,
checking for schema conformance, null counts, value ranges, and data drift.
"""
import great_expectations as ge
from great_expectations.core.expectation_suite import ExpectationSuite
from great_expectations.dataset import SparkDFDataset
from great_expectations.core.batch import BatchRequest
from great_expectations.data_context import DataContext
import pandas as pd
import os
import logging
import requests
from typing import Dict, Any, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_telco_expectation_suite(context: DataContext) -> ExpectationSuite:
    """
    Create and configure the expectation suite for Telco CDR data.
    
    Args:
        context: Great Expectations DataContext
        
    Returns:
        ExpectationSuite: The configured expectation suite
    """
    # Create a new suite
    suite_name = "telco_suite"
    context.create_expectation_suite(suite_name, overwrite_existing=True)
    
    # Get the suite
    suite = context.get_expectation_suite(suite_name)
    
    # Add expectations for column presence
    required_columns = [
        "call_id", "customer_id", "call_time", "call_duration_seconds",
        "call_type", "destination_number", "network_type", "call_result",
        "call_charge"
    ]
    
    for column in required_columns:
        suite.add_expectation(
            ge.core.expectation_configuration.ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": column}
            )
        )
    
    # Add expectations for data types
    suite.add_expectation(
        ge.core.expectation_configuration.ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_of_type",
            kwargs={"column": "call_id", "type_": "StringType"}
        )
    )
    suite.add_expectation(
        ge.core.expectation_configuration.ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_of_type",
            kwargs={"column": "customer_id", "type_": "StringType"}
        )
    )
    suite.add_expectation(
        ge.core.expectation_configuration.ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_of_type",
            kwargs={"column": "call_time", "type_": "TimestampType"}
        )
    )
    suite.add_expectation(
        ge.core.expectation_configuration.ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_of_type",
            kwargs={"column": "call_duration_seconds", "type_": "IntegerType"}
        )
    )
    suite.add_expectation(
        ge.core.expectation_configuration.ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_of_type",
            kwargs={"column": "call_charge", "type_": "FloatType"}
        )
    )
    
    # Add expectations for null values
    for column in ["call_id", "customer_id", "call_time"]:
        suite.add_expectation(
            ge.core.expectation_configuration.ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": column}
            )
        )
    
    # Some columns can have nulls but shouldn't have too many
    for column in ["call_duration_seconds", "call_type", "call_result", "call_charge"]:
        suite.add_expectation(
            ge.core.expectation_configuration.ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": column, "mostly": 0.95}
            )
        )
    
    # Add expectations for value ranges
    suite.add_expectation(
        ge.core.expectation_configuration.ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column": "call_duration_seconds", 
                "min_value": 0, 
                "max_value": 7200,  # Max 2 hours (in seconds)
                "mostly": 0.99
            }
        )
    )
    
    suite.add_expectation(
        ge.core.expectation_configuration.ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column": "call_charge", 
                "min_value": 0, 
                "max_value": 100,  # Max $100 charge
                "mostly": 0.99
            }
        )
    )
    
    # Add expectations for categorical values
    suite.add_expectation(
        ge.core.expectation_configuration.ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={
                "column": "call_type", 
                "value_set": ["VOICE", "SMS", "DATA"],
                "mostly": 0.99
            }
        )
    )
    
    suite.add_expectation(
        ge.core.expectation_configuration.ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={
                "column": "network_type", 
                "value_set": ["2G", "3G", "4G", "5G", "WIFI", "ROAMING"],
                "mostly": 0.99
            }
        )
    )
    
    suite.add_expectation(
        ge.core.expectation_configuration.ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={
                "column": "call_result", 
                "value_set": ["COMPLETED", "BUSY", "NO_ANSWER", "FAILED", "DROPPED"],
                "mostly": 0.99
            }
        )
    )
    
    # Row count expectations
    suite.add_expectation(
        ge.core.expectation_configuration.ExpectationConfiguration(
            expectation_type="expect_table_row_count_to_be_between",
            kwargs={"min_value": 1, "max_value": 100000000}
        )
    )
    
    # Save the suite
    context.save_expectation_suite(suite)
    logger.info(f"Created and saved expectation suite: {suite_name}")
    
    return suite

def send_slack_alert(webhook_url: str, message: str, validation_results: Dict[str, Any]) -> None:
    """
    Send an alert to Slack when data quality issues are detected.
    
    Args:
        webhook_url: Slack webhook URL
        message: Alert message
        validation_results: GE validation results
    """
    if not webhook_url:
        logger.warning("No Slack webhook URL provided, skipping alert")
        return
    
    # Extract failed expectations
    failed_expectations = []
    for result in validation_results.get("results", []):
        if not result.get("success", False):
            failed_expectations.append({
                "expectation_type": result.get("expectation_config", {}).get("expectation_type", "Unknown"),
                "column": result.get("expectation_config", {}).get("kwargs", {}).get("column", "N/A"),
                "details": result.get("result", {})
            })
    
    # Create message
    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*⚠️ Data Quality Alert*\n{message}"
            }
        },
        {
            "type": "divider"
        }
    ]
    
    # Add failed expectations
    for i, failure in enumerate(failed_expectations[:5]):  # Show at most 5 failures
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Failed Check #{i+1}*\nType: {failure['expectation_type']}\nColumn: {failure['column']}"
            }
        })
    
    # Add note if there are more failures
    if len(failed_expectations) > 5:
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"... and {len(failed_expectations) - 5} more failed checks"
                }
            ]
        })
    
    # Send to Slack
    payload = {
        "blocks": blocks
    }
    
    try:
        response = requests.post(webhook_url, json=payload)
        response.raise_for_status()
        logger.info("Slack alert sent successfully")
    except Exception as e:
        logger.error(f"Failed to send Slack alert: {str(e)}")

def detect_distribution_drift(
    context: DataContext,
    current_batch: BatchRequest,
    reference_batch: BatchRequest,
    threshold: float = 0.1
) -> Dict[str, Any]:
    """
    Detect distribution drift between current and reference data.
    
    Args:
        context: Great Expectations DataContext
        current_batch: Current data batch
        reference_batch: Reference data batch
        threshold: Threshold for KL divergence to trigger alert
        
    Returns:
        Dict with drift metrics and status
    """
    from scipy import stats
    import numpy as np
    
    # Get column profiles from both batches
    current_df = context.get_batch_dataframe(current_batch)
    reference_df = context.get_batch_dataframe(reference_batch)
    
    drift_metrics = {}
    drift_detected = False
    
    # Check numeric columns for distribution drift
    for column in current_df.select_dtypes(include=np.number).columns:
        if column in reference_df.columns:
            # Get values, handle nulls
            current_values = current_df[column].dropna().values
            reference_values = reference_df[column].dropna().values
            
            if len(current_values) > 100 and len(reference_values) > 100:
                # Calculate KL divergence
                current_hist, bin_edges = np.histogram(current_values, bins=50, density=True)
                reference_hist, _ = np.histogram(reference_values, bins=bin_edges, density=True)
                
                # Replace zeros to avoid division by zero in KL divergence
                current_hist = np.where(current_hist == 0, 1e-10, current_hist)
                reference_hist = np.where(reference_hist == 0, 1e-10, reference_hist)
                
                # Calculate KL divergence (relative entropy)
                kl_divergence = stats.entropy(current_hist, reference_hist)
                
                drift_metrics[column] = {
                    "kl_divergence": kl_divergence,
                    "drift_detected": kl_divergence > threshold
                }
                
                if kl_divergence > threshold:
                    drift_detected = True
                    logger.warning(f"Distribution drift detected in column {column} (KL div: {kl_divergence:.4f})")
    
    # Prepare result
    result = {
        "drift_detected": drift_detected,
        "column_metrics": drift_metrics
    }
    
    # Send alert if drift detected
    if drift_detected:
        webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
        if webhook_url:
            message = "Distribution drift detected in telco CDR data!"
            send_slack_alert(webhook_url, message, {"drift_metrics": drift_metrics})
    
    return result

if __name__ == "__main__":
    # Initialize GE context
    context = DataContext()
    
    # Create the expectation suite
    create_telco_expectation_suite(context)
    
    logger.info("Telco expectation suite created successfully")
