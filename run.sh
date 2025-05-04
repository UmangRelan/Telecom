#!/bin/bash

# Telco Churn Pipeline Bootstrap Script
# This script sets up the entire telco churn pipeline environment

set -e  # Exit on error

# Create necessary directories
mkdir -p infra/prometheus infra/grafana/provisioning/datasources infra/grafana/provisioning/dashboards
mkdir -p models/artifact logs/{airflow,spark,api,dashboard}
mkdir -p /tmp/telco_features /tmp/telco_benchmark

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print section header
section() {
    echo -e "\n${GREEN}==== $1 ====${NC}\n"
}

# Print info message
info() {
    echo -e "${YELLOW}INFO: $1${NC}"
}

# Print error message
error() {
    echo -e "${RED}ERROR: $1${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
section "Checking prerequisites"

# Check if Docker is installed
if ! command_exists docker; then
    error "Docker is required but not installed. Please install Docker and try again."
    exit 1
fi

# Check if Docker Compose is installed
if ! command_exists docker-compose; then
    error "Docker Compose is required but not installed. Please install Docker Compose and try again."
    exit 1
fi

# Check if Python is installed
if ! command_exists python3; then
    error "Python 3 is required but not installed. Please install Python 3 and try again."
    exit 1
fi

info "All prerequisites are satisfied"

# Create prometheus.yml config file
section "Setting up Prometheus configuration"
cat > infra/prometheus/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'model-api'
    static_configs:
      - targets: ['model-api:8000']
    metrics_path: /metrics

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
EOF
info "Created Prometheus configuration"

# Create Grafana datasource config
section "Setting up Grafana configuration"
cat > infra/grafana/provisioning/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF
info "Created Grafana datasource configuration"

# Create Grafana dashboard config
mkdir -p infra/grafana/provisioning/dashboards
cat > infra/grafana/provisioning/dashboards/dashboard.yml << EOF
apiVersion: 1

providers:
  - name: 'Default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    options:
      path: /etc/grafana/provisioning/dashboards
EOF
info "Created Grafana dashboard configuration"

# Starting Docker Compose
section "Starting Docker Compose services"
docker-compose up -d
info "Docker Compose services started"

# Wait for services to be ready
section "Waiting for services to be ready"
info "Waiting for PostgreSQL to be ready..."
until docker-compose exec -T postgres pg_isready -U ${PGUSER:-postgres} > /dev/null 2>&1; do
    echo -n "."
    sleep 2
done
echo ""
info "PostgreSQL is ready"

info "Waiting for Redis to be ready..."
until docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; do
    echo -n "."
    sleep 2
done
echo ""
info "Redis is ready"

info "Waiting for Airflow webserver to be ready..."
until curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health | grep -q "200"; do
    echo -n "."
    sleep 5
done
echo ""
info "Airflow webserver is ready"

# Initialize MinIO buckets and data
section "Setting up MinIO buckets"
info "Creating MinIO buckets..."
# Install MinIO client
if ! command_exists mc; then
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl -O https://dl.min.io/client/mc/release/linux-amd64/mc
        chmod +x mc
        ./mc --version
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        curl -O https://dl.min.io/client/mc/release/darwin-amd64/mc
        chmod +x mc
        ./mc --version
    else
        info "MinIO client not installed. Please manually create the buckets."
    fi
fi

if command_exists mc || [ -f "./mc" ]; then
    MC_CMD="mc"
    if [ -f "./mc" ]; then
        MC_CMD="./mc"
    fi
    # Configure MinIO client
    $MC_CMD config host add myminio http://localhost:9000 minio minio123
    # Create buckets
    $MC_CMD mb myminio/telco-raw
    $MC_CMD mb myminio/telco-silver
    $MC_CMD mb myminio/telco-gold
    $MC_CMD mb myminio/telco-models
    info "MinIO buckets created"
fi

# Initialize Great Expectations
section "Initializing Great Expectations"
info "Setting up Great Expectations..."
if ! command_exists great_expectations; then
    pip install great_expectations
fi

if [ ! -d "ge/great_expectations" ]; then
    mkdir -p ge
    cd ge
    great_expectations init
    cd ..
    info "Great Expectations initialized"
else
    info "Great Expectations already initialized"
fi

# Initialize Feast feature store
section "Initializing Feast feature store"
info "Setting up Feast feature store..."
if ! command_exists feast; then
    pip install feast
fi

cd feast_repo
feast apply
cd ..
info "Feast feature store initialized"

# Generate sample data for testing
section "Generating sample data for testing"
info "Generating sample data..."
mkdir -p data

# Create sample churn labels
cat > data/churn_labels.csv << EOF
customer_id,churn,churn_date
CUST000001,0,
CUST000002,1,2023-05-10
CUST000003,0,
CUST000004,1,2023-04-22
CUST000005,0,
CUST000006,0,
CUST000007,1,2023-05-15
CUST000008,0,
CUST000009,0,
CUST000010,1,2023-04-01
CUST123456,1,2023-05-01
CUST789012,0,
EOF
info "Created sample churn labels"

# Run ETL benchmark to generate synthetic data
info "Running ETL benchmark to generate synthetic data..."
cd etl
mkdir -p /tmp/telco_benchmark/input
mkdir -p /tmp/telco_benchmark/output
python benchmark.py --row-counts 1000 --repetitions 1
cd ..
info "Generated synthetic data"

# Trigger Airflow DAG
section "Triggering initial Airflow DAG"
info "Triggering cdr_daily_ingest DAG..."
docker-compose exec -T airflow-webserver airflow dags unpause cdr_daily_ingest
docker-compose exec -T airflow-webserver airflow dags trigger cdr_daily_ingest
info "Airflow DAG triggered"

# Start the Streamlit dashboard
section "Starting Streamlit dashboard"
info "Streamlit dashboard is running at http://localhost:5000"

# Print final instructions
section "🚀 Pipeline is now live!"
echo "Airflow UI: http://localhost:8080 (username: airflow, password: airflow)"
echo "Streamlit Dashboard: http://localhost:5000"
echo "FastAPI Swagger: http://localhost:8000/docs"
echo "MinIO Console: http://localhost:9001 (username: minio, password: minio123)"
echo "Grafana: http://localhost:3000 (username: admin, password: admin)"
echo "Prometheus: http://localhost:9090"
echo ""
echo "To stop the pipeline, run: docker-compose down"
echo "To restart the pipeline, run: bash run.sh"
