"""
Telco CDR Ingest DAG

This DAG pulls Call Detail Records from S3/MinIO, runs PySpark ETL,
validates with Great Expectations, and writes to silver Parquet files.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.hooks.S3_hook import S3Hook
import os
import logging

# Define default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'cdr_daily_ingest',
    default_args=default_args,
    description='Ingest CDR data from S3/MinIO, process with PySpark, validate with GE',
    schedule_interval='@daily',
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['telco', 'cdr', 'etl'],
)

# Define functions for the operators
def list_new_files(**kwargs):
    """List new CDR files that haven't been processed yet."""
    execution_date = kwargs['execution_date']
    date_str = execution_date.strftime('%Y-%m-%d')
    
    s3_hook = S3Hook(aws_conn_id='minio_conn')
    bucket_name = 'telco-raw'
    prefix = f'cdr/{date_str}/'
    
    files = s3_hook.list_keys(bucket_name=bucket_name, prefix=prefix)
    if not files:
        logging.info(f"No files found for date: {date_str}")
        return []
    
    logging.info(f"Found {len(files)} files to process: {files}")
    return files

def validate_with_great_expectations(**kwargs):
    """Validate processed data with Great Expectations."""
    import great_expectations as ge
    from great_expectations.data_context import DataContext
    
    ti = kwargs['ti']
    processed_files = ti.xcom_pull(task_ids='process_cdr_data')
    
    # Initialize GE context
    context = DataContext(os.path.join(os.environ.get('AIRFLOW_HOME', ''), '../ge'))
    
    validation_results = []
    for file_path in processed_files:
        # Create batch
        batch = context.get_batch({
            "path": file_path,
            "datasource": "telco_silver",
            "data_asset_name": "cdr_processed"
        }, "telco_suite")
        
        # Validate
        results = context.run_validation_operator(
            "action_list_operator",
            assets_to_validate=[batch],
            run_id=f"airflow-{kwargs['ts']}"
        )
        validation_results.append(results)
    
    # Check if all validations passed
    all_passed = all([result["success"] for result in validation_results])
    if not all_passed:
        raise Exception("Data quality validation failed")
    
    return validation_results

# Define the tasks
list_files_task = PythonOperator(
    task_id='list_new_files',
    python_callable=list_new_files,
    provide_context=True,
    dag=dag,
)

process_cdr_task = SparkSubmitOperator(
    task_id='process_cdr_data',
    application='/opt/airflow/etl/process_cdr.py',
    conn_id='spark_conn',
    verbose=False,
    executor_cores=2,
    executor_memory='2g',
    conf={
        'spark.driver.memory': '1g',
        'spark.dynamicAllocation.enabled': 'true',
        'spark.shuffle.service.enabled': 'true'
    },
    dag=dag,
)

validate_task = PythonOperator(
    task_id='validate_with_great_expectations',
    python_callable=validate_with_great_expectations,
    provide_context=True,
    dag=dag,
)

# Define the task dependencies
list_files_task >> process_cdr_task >> validate_task
