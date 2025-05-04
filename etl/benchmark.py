"""
ETL Performance Benchmark

This script generates synthetic CDR data and benchmarks the PySpark ETL process 
for performance testing and optimization.
"""
import time
import argparse
import os
import psutil
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
import uuid
import random

def create_spark_session():
    """Create and configure a Spark session for benchmarking."""
    return (SparkSession.builder
            .appName("TelcoCDRBenchmark")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.executor.memory", "4g")
            .config("spark.driver.memory", "2g")
            .config("spark.executor.cores", "2")
            .getOrCreate())

def generate_synthetic_data(num_rows, output_path):
    """
    Generate synthetic CDR data for benchmarking.
    
    Args:
        num_rows: Number of rows to generate
        output_path: Path to write CSV data
    """
    print(f"Generating {num_rows} synthetic CDR records...")
    
    # Define possible values for categorical fields
    call_types = ["VOICE", "SMS", "DATA"]
    network_types = ["4G", "5G", "3G", "WIFI"]
    call_results = ["COMPLETED", "BUSY", "NO_ANSWER", "FAILED"]
    promo_codes = ["NONE", "SUMMER23", "LOYALTY", "WEEKEND", "FAMILY"]
    
    # Generate customer IDs (fewer than rows to simulate multiple calls per customer)
    num_customers = max(100, int(num_rows / 100))
    customer_ids = [f"CUST{i:06d}" for i in range(num_customers)]
    
    # Generate data
    data = {
        "call_id": [str(uuid.uuid4()) for _ in range(num_rows)],
        "customer_id": [random.choice(customer_ids) for _ in range(num_rows)],
        "call_time": [
            (datetime.now() - timedelta(days=random.randint(0, 30), 
                                       hours=random.randint(0, 23),
                                       minutes=random.randint(0, 59))).strftime("%Y-%m-%d %H:%M:%S")
            for _ in range(num_rows)
        ],
        "call_duration_seconds": [random.randint(5, 1800) for _ in range(num_rows)],
        "call_type": [random.choice(call_types) for _ in range(num_rows)],
        "destination_number": [f"+1{random.randint(2000000000, 9999999999)}" for _ in range(num_rows)],
        "network_type": [random.choice(network_types) for _ in range(num_rows)],
        "cell_tower_id": [f"TOWER{random.randint(1, 1000):04d}" for _ in range(num_rows)],
        "call_result": [random.choice(call_results) for _ in range(num_rows)],
        "call_charge": [round(random.uniform(0.01, 15.00), 2) for _ in range(num_rows)],
        "promotion_code": [random.choice(promo_codes) for _ in range(num_rows)]
    }
    
    # Create DataFrame and write to CSV
    df = pd.DataFrame(data)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Generated data written to: {output_path}")
    return output_path

def benchmark_etl_job(spark, input_path, output_path):
    """
    Benchmark the ETL job and measure performance metrics.
    
    Args:
        spark: SparkSession
        input_path: Path to input CSV data
        output_path: Path to write processed Parquet data
    
    Returns:
        dict: Performance metrics
    """
    # Get baseline memory usage
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Start timer
    start_time = time.time()
    
    # Import the ETL processing module
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from process_cdr import process_cdr_data
    
    # Run the ETL job
    process_cdr_data(spark, input_path, output_path)
    
    # Calculate metrics
    end_time = time.time()
    end_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Count records in the result
    result_df = spark.read.parquet(output_path)
    record_count = result_df.count()
    
    # Gather Spark metrics
    spark_metrics = spark.sparkContext.statusTracker().getExecutorInfos()
    
    # Compile and return performance metrics
    metrics = {
        "execution_time_seconds": end_time - start_time,
        "memory_usage_mb": end_memory - start_memory,
        "record_count": record_count,
        "records_per_second": record_count / (end_time - start_time),
        "spark_executors": len(spark_metrics)
    }
    
    return metrics

def run_benchmark(row_counts=[10_000, 100_000, 1_000_000, 10_000_000], repetitions=3):
    """
    Run benchmarks for multiple data sizes and repetitions.
    
    Args:
        row_counts: List of data sizes to benchmark
        repetitions: Number of times to repeat each benchmark
    """
    # Create temporary directories
    base_input_dir = "/tmp/telco_benchmark/input"
    base_output_dir = "/tmp/telco_benchmark/output"
    os.makedirs(base_input_dir, exist_ok=True)
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Create Spark session
    spark = create_spark_session()
    
    # Store results
    benchmark_results = []
    
    try:
        for num_rows in row_counts:
            print(f"\n{'='*50}\nBenchmarking with {num_rows} rows\n{'='*50}")
            
            repetition_metrics = []
            for rep in range(1, repetitions + 1):
                print(f"\nRepetition {rep}/{repetitions}")
                
                # Generate unique paths for this run
                run_id = f"{num_rows}_{rep}"
                input_path = f"{base_input_dir}/cdr_{run_id}.csv"
                output_path = f"{base_output_dir}/cdr_{run_id}"
                
                # Generate data
                generate_synthetic_data(num_rows, input_path)
                
                # Run benchmark
                metrics = benchmark_etl_job(spark, input_path, output_path)
                metrics["row_count"] = num_rows
                metrics["repetition"] = rep
                
                # Print metrics
                print(f"Execution time: {metrics['execution_time_seconds']:.2f} seconds")
                print(f"Memory usage: {metrics['memory_usage_mb']:.2f} MB")
                print(f"Processing rate: {metrics['records_per_second']:.2f} records/second")
                
                repetition_metrics.append(metrics)
                benchmark_results.append(metrics)
            
            # Calculate and print average metrics for this row count
            avg_exec_time = sum(m["execution_time_seconds"] for m in repetition_metrics) / repetitions
            avg_memory = sum(m["memory_usage_mb"] for m in repetition_metrics) / repetitions
            avg_rate = sum(m["records_per_second"] for m in repetition_metrics) / repetitions
            
            print(f"\nAverage metrics for {num_rows} rows:")
            print(f"Execution time: {avg_exec_time:.2f} seconds")
            print(f"Memory usage: {avg_memory:.2f} MB")
            print(f"Processing rate: {avg_rate:.2f} records/second")
    
    finally:
        # Save results to CSV
        results_df = pd.DataFrame(benchmark_results)
        results_path = "/tmp/telco_benchmark/benchmark_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\nDetailed benchmark results saved to: {results_path}")
        
        # Stop Spark session
        spark.stop()

if __name__ == "__main__":
    import sys
    
    parser = argparse.ArgumentParser(description="Benchmark PySpark ETL performance")
    parser.add_argument("--row-counts", type=int, nargs="+", 
                        default=[10_000, 100_000, 1_000_000],
                        help="List of row counts to benchmark")
    parser.add_argument("--repetitions", type=int, default=3,
                        help="Number of repetitions for each benchmark")
    
    args = parser.parse_args()
    
    run_benchmark(row_counts=args.row_counts, repetitions=args.repetitions)
