# S3 / MinIO outputs
output "s3_raw_bucket" {
  description = "Name of the raw data S3 bucket"
  value       = aws_s3_bucket.telco_raw.bucket
}

output "s3_silver_bucket" {
  description = "Name of the silver data S3 bucket"
  value       = aws_s3_bucket.telco_silver.bucket
}

output "s3_gold_bucket" {
  description = "Name of the gold data S3 bucket"
  value       = aws_s3_bucket.telco_gold.bucket
}

output "s3_models_bucket" {
  description = "Name of the models S3 bucket"
  value       = aws_s3_bucket.telco_models.bucket
}

# PostgreSQL outputs
output "postgres_endpoint" {
  description = "Endpoint of the PostgreSQL database"
  value       = aws_db_instance.postgres.endpoint
}

output "postgres_connection_string" {
  description = "Connection string for PostgreSQL"
  value       = "postgresql://${var.postgres_username}:${var.postgres_password}@${aws_db_instance.postgres.endpoint}/${var.postgres_db_name}"
  sensitive   = true
}

# Kafka outputs
output "kafka_bootstrap_servers" {
  description = "Bootstrap servers for Kafka"
  value       = aws_msk_cluster.kafka.bootstrap_brokers
}

output "kafka_zookeeper_connect_string" {
  description = "Zookeeper connection string for Kafka"
  value       = aws_msk_cluster.kafka.zookeeper_connect_string
}

# Redis outputs
output "redis_endpoint" {
  description = "Endpoint of the Redis cluster"
  value       = aws_elasticache_cluster.redis.cache_nodes.0.address
}

output "redis_port" {
  description = "Port of the Redis cluster"
  value       = aws_elasticache_cluster.redis.cache_nodes.0.port
}

output "redis_connection_string" {
  description = "Connection string for Redis"
  value       = "redis://${aws_elasticache_cluster.redis.cache_nodes.0.address}:${aws_elasticache_cluster.redis.cache_nodes.0.port}"
}

# Prometheus outputs
output "prometheus_endpoint" {
  description = "Endpoint of the Prometheus workspace"
  value       = aws_prometheus_workspace.prometheus.prometheus_endpoint
}

# Grafana outputs
output "grafana_endpoint" {
  description = "Endpoint of the Grafana workspace"
  value       = aws_grafana_workspace.grafana.endpoint
}
