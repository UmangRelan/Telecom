variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-east-1"
}

variable "project" {
  description = "Project name"
  type        = string
  default     = "telco-churn"
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "dev"
}

variable "vpc_id" {
  description = "VPC ID for resources"
  type        = string
  default     = "vpc-12345678" # Replace with your VPC ID or use data sources
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16" # Replace with your VPC CIDR
}

variable "subnet_ids" {
  description = "Subnet IDs for resources"
  type        = list(string)
  default     = ["subnet-12345678", "subnet-87654321"] # Replace with your subnet IDs or use data sources
}

# PostgreSQL variables
variable "postgres_instance_class" {
  description = "RDS instance class for PostgreSQL"
  type        = string
  default     = "db.t3.micro"
}

variable "postgres_allocated_storage" {
  description = "Allocated storage for PostgreSQL in GB"
  type        = number
  default     = 20
}

variable "postgres_db_name" {
  description = "PostgreSQL database name"
  type        = string
  default     = "telco_db"
}

variable "postgres_username" {
  description = "PostgreSQL username"
  type        = string
  default     = "telco_admin"
}

variable "postgres_password" {
  description = "PostgreSQL password"
  type        = string
  sensitive   = true
  default     = "please-change-me" # Should be provided externally through AWS Secrets Manager or similar
}

variable "postgres_publicly_accessible" {
  description = "Whether PostgreSQL should be publicly accessible"
  type        = bool
  default     = false
}

# Kafka variables
variable "kafka_instance_type" {
  description = "Instance type for Kafka brokers"
  type        = string
  default     = "kafka.t3.small"
}

variable "kafka_volume_size" {
  description = "Volume size for Kafka brokers in GB"
  type        = number
  default     = 20
}

# Redis variables
variable "redis_node_type" {
  description = "Node type for Redis"
  type        = string
  default     = "cache.t3.micro"
}
