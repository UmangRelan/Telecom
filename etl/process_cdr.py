"""
CDR Processing ETL Job

This PySpark job reads raw CSV Call Detail Records,
cleans and transforms the data, and writes it as 
partitioned Parquet files.
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, year, month, dayofmonth, hour
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, TimestampType
import os
import sys
from datetime import datetime

def create_spark_session():
    """Create and configure a Spark session."""
    return (SparkSession.builder
            .appName("TelcoCDRProcessing")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.parquet.compression.codec", "snappy")
            .getOrCreate())

def define_schema():
    """Define the CDR data schema."""
    return StructType([
        StructField("call_id", StringType(), False),
        StructField("customer_id", StringType(), False),
        StructField("call_time", TimestampType(), False),
        StructField("call_duration_seconds", IntegerType(), True),
        StructField("call_type", StringType(), True),
        StructField("destination_number", StringType(), True),
        StructField("network_type", StringType(), True),
        StructField("cell_tower_id", StringType(), True),
        StructField("call_result", StringType(), True),
        StructField("call_charge", FloatType(), True),
        StructField("promotion_code", StringType(), True),
    ])

def process_cdr_data(spark, input_path, output_path):
    """
    Process CDR data:
    1. Read raw CSV data
    2. Clean and transform
    3. Write partitioned Parquet
    """
    # Define schema
    schema = define_schema()
    
    # Read data
    print(f"Reading CDR data from: {input_path}")
    df = spark.read.csv(
        input_path,
        header=True,
        schema=schema,
        mode="DROPMALFORMED"
    )
    
    # Perform basic data cleaning
    cleaned_df = df.filter(
        (col("call_duration_seconds") > 0) &
        (col("call_duration_seconds").isNotNull()) &
        (col("customer_id").isNotNull()) &
        (col("call_time").isNotNull())
    )
    
    # Extract date parts for partitioning
    partitioned_df = cleaned_df.withColumn("call_date", to_date("call_time")) \
        .withColumn("year", year("call_date")) \
        .withColumn("month", month("call_date")) \
        .withColumn("day", dayofmonth("call_date")) \
        .withColumn("hour", hour("call_time"))
    
    # Write to Parquet with partitioning
    print(f"Writing processed data to: {output_path}")
    (partitioned_df.write
        .partitionBy("year", "month", "day", "customer_id")
        .mode("overwrite")
        .parquet(output_path))
    
    processed_count = partitioned_df.count()
    print(f"Processed {processed_count} CDR records")
    
    return output_path

if __name__ == "__main__":
    # Parse command line arguments or use defaults
    if len(sys.argv) > 2:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
    else:
        # Default paths for local testing
        execution_date = datetime.now().strftime('%Y-%m-%d')
        input_path = f"s3a://telco-raw/cdr/{execution_date}/"
        output_path = f"s3a://telco-silver/cdr/dt={execution_date}/"
    
    # Create Spark session
    spark = create_spark_session()
    
    try:
        # Execute the processing
        process_cdr_data(spark, input_path, output_path)
    finally:
        # Stop the Spark session
        spark.stop()
