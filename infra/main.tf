#####################################################
# Telco Churn Infrastructure
#####################################################

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.16"
    }
  }
  required_version = ">= 1.0.0"
}

provider "aws" {
  region = var.aws_region
  
  # For local development with MinIO instead of AWS
  # Uncomment these when using MinIO
  # skip_credentials_validation = true
  # skip_metadata_api_check     = true
  # skip_requesting_account_id  = true
  # endpoints {
  #   s3 = "http://localhost:9000"
  # }
}

###########################
# S3 / MinIO
###########################

resource "aws_s3_bucket" "telco_raw" {
  bucket = "${var.project}-raw-${var.environment}"
  
  tags = {
    Name        = "${var.project}-raw"
    Environment = var.environment
    Project     = var.project
  }
}

resource "aws_s3_bucket" "telco_silver" {
  bucket = "${var.project}-silver-${var.environment}"
  
  tags = {
    Name        = "${var.project}-silver"
    Environment = var.environment
    Project     = var.project
  }
}

resource "aws_s3_bucket" "telco_gold" {
  bucket = "${var.project}-gold-${var.environment}"
  
  tags = {
    Name        = "${var.project}-gold"
    Environment = var.environment
    Project     = var.project
  }
}

resource "aws_s3_bucket" "telco_models" {
  bucket = "${var.project}-models-${var.environment}"
  
  tags = {
    Name        = "${var.project}-models"
    Environment = var.environment
    Project     = var.project
  }
}

###########################
# RDS PostgreSQL
###########################

resource "aws_db_subnet_group" "default" {
  name       = "${var.project}-db-subnet-group"
  subnet_ids = var.subnet_ids
  
  tags = {
    Name        = "${var.project}-db-subnet-group"
    Environment = var.environment
    Project     = var.project
  }
}

resource "aws_security_group" "postgres" {
  name        = "${var.project}-postgres-sg"
  description = "Security group for PostgreSQL"
  vpc_id      = var.vpc_id
  
  ingress {
    description = "PostgreSQL from VPC"
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name        = "${var.project}-postgres-sg"
    Environment = var.environment
    Project     = var.project
  }
}

resource "aws_db_instance" "postgres" {
  identifier             = "${var.project}-postgres"
  engine                 = "postgres"
  engine_version         = "14.3"
  instance_class         = var.postgres_instance_class
  allocated_storage      = var.postgres_allocated_storage
  storage_type           = "gp2"
  db_name                = var.postgres_db_name
  username               = var.postgres_username
  password               = var.postgres_password
  db_subnet_group_name   = aws_db_subnet_group.default.name
  vpc_security_group_ids = [aws_security_group.postgres.id]
  publicly_accessible    = var.postgres_publicly_accessible
  skip_final_snapshot    = true
  
  tags = {
    Name        = "${var.project}-postgres"
    Environment = var.environment
    Project     = var.project
  }
}

###########################
# Amazon MSK (Kafka)
###########################

resource "aws_security_group" "kafka" {
  name        = "${var.project}-kafka-sg"
  description = "Security group for Kafka"
  vpc_id      = var.vpc_id
  
  ingress {
    description = "Kafka from VPC"
    from_port   = 9092
    to_port     = 9092
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name        = "${var.project}-kafka-sg"
    Environment = var.environment
    Project     = var.project
  }
}

resource "aws_msk_cluster" "kafka" {
  cluster_name           = "${var.project}-kafka"
  kafka_version          = "2.8.0"
  number_of_broker_nodes = 3
  
  broker_node_group_info {
    instance_type   = var.kafka_instance_type
    client_subnets  = var.subnet_ids
    security_groups = [aws_security_group.kafka.id]
    storage_info {
      ebs_storage_info {
        volume_size = var.kafka_volume_size
      }
    }
  }
  
  tags = {
    Name        = "${var.project}-kafka"
    Environment = var.environment
    Project     = var.project
  }
}

###########################
# ElastiCache (Redis)
###########################

resource "aws_security_group" "redis" {
  name        = "${var.project}-redis-sg"
  description = "Security group for Redis"
  vpc_id      = var.vpc_id
  
  ingress {
    description = "Redis from VPC"
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name        = "${var.project}-redis-sg"
    Environment = var.environment
    Project     = var.project
  }
}

resource "aws_elasticache_subnet_group" "default" {
  name       = "${var.project}-redis-subnet-group"
  subnet_ids = var.subnet_ids
  
  tags = {
    Name        = "${var.project}-redis-subnet-group"
    Environment = var.environment
    Project     = var.project
  }
}

resource "aws_elasticache_cluster" "redis" {
  cluster_id           = "${var.project}-redis"
  engine               = "redis"
  node_type            = var.redis_node_type
  num_cache_nodes      = 1
  parameter_group_name = "default.redis6.x"
  engine_version       = "6.2"
  port                 = 6379
  subnet_group_name    = aws_elasticache_subnet_group.default.name
  security_group_ids   = [aws_security_group.redis.id]
  
  tags = {
    Name        = "${var.project}-redis"
    Environment = var.environment
    Project     = var.project
  }
}

###########################
# Amazon Managed Prometheus
###########################

resource "aws_prometheus_workspace" "prometheus" {
  alias = "${var.project}-prometheus"
  
  tags = {
    Name        = "${var.project}-prometheus"
    Environment = var.environment
    Project     = var.project
  }
}

###########################
# Amazon Managed Grafana
###########################

resource "aws_grafana_workspace" "grafana" {
  name                     = "${var.project}-grafana"
  account_access_type      = "CURRENT_ACCOUNT"
  authentication_providers = ["AWS_SSO"]
  permission_type          = "SERVICE_MANAGED"
  role_arn                 = aws_iam_role.grafana.arn
  
  data_sources = ["PROMETHEUS"]
  
  tags = {
    Name        = "${var.project}-grafana"
    Environment = var.environment
    Project     = var.project
  }
}

resource "aws_iam_role" "grafana" {
  name = "${var.project}-grafana-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "grafana.amazonaws.com"
        }
      }
    ]
  })
  
  tags = {
    Name        = "${var.project}-grafana-role"
    Environment = var.environment
    Project     = var.project
  }
}

resource "aws_iam_role_policy_attachment" "grafana_prometheus" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonPrometheusQueryAccess"
  role       = aws_iam_role.grafana.name
}
