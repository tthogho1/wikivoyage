data "aws_region" "current" {}
data "aws_caller_identity" "current" {}

resource "aws_ecs_cluster" "this" {
  name = var.name

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

resource "aws_service_discovery_http_namespace" "this" {
  name = "${var.name}.local"
}

resource "aws_cloudwatch_log_group" "webapp" {
  name              = "/ecs/${var.name}-webapp"
  retention_in_days = var.log_retention_days
}

resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${var.name}-article-db-api"
  retention_in_days = var.log_retention_days
}

# ---- IAM ----

data "aws_iam_policy_document" "ecs_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "execution" {
  name               = "${var.name}-ecs-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

resource "aws_iam_role_policy_attachment" "execution_managed" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

data "aws_iam_policy_document" "execution_secrets" {
  statement {
    actions = ["secretsmanager:GetSecretValue"]
    resources = [
      aws_secretsmanager_secret.openai_api_key.arn,
      aws_secretsmanager_secret.zilliz_uri.arn,
      aws_secretsmanager_secret.zilliz_token.arn,
    ]
  }
}

resource "aws_iam_role_policy" "execution_secrets" {
  name   = "secrets-read"
  role   = aws_iam_role.execution.id
  policy = data.aws_iam_policy_document.execution_secrets.json
}

resource "aws_iam_role" "task" {
  name               = "${var.name}-ecs-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

# ---- webapp task + service ----

resource "aws_ecs_task_definition" "webapp" {
  family                   = "${var.name}-webapp"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.webapp_cpu
  memory                   = var.webapp_memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([
    {
      name      = "webapp"
      image     = "${aws_ecr_repository.webapp.repository_url}:${var.webapp_image_tag}"
      essential = true
      portMappings = [{
        containerPort = var.webapp_port
        protocol      = "tcp"
        name          = "http"
      }]
      environment = [
        { name = "PORT", value = tostring(var.webapp_port) },
        { name = "NODE_ENV", value = "production" },
        { name = "OPENAI_MODEL", value = var.openai_model },
        # Read by webapp/lib/env.ts. article_db.api is reached through Service Connect, not the ALB:
        # the host must be the api service's client_alias dns_name (the HTTP namespace has no DNS,
        # so "<name>.<namespace>" would not resolve).
        { name = "RETRIEVAL_URL", value = "http://article-db-api:${var.api_port}" },
      ]
      secrets = [
        { name = "OPENAI_API_KEY", valueFrom = aws_secretsmanager_secret.openai_api_key.arn },
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.webapp.name
          "awslogs-region"        = data.aws_region.current.name
          "awslogs-stream-prefix" = "webapp"
        }
      }
    }
  ])
}

resource "aws_ecs_service" "webapp" {
  name            = "${var.name}-webapp"
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.webapp.arn
  desired_count   = var.webapp_desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = local.subnet_ids
    security_groups  = [aws_security_group.webapp.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.webapp.arn
    container_name    = "webapp"
    container_port    = var.webapp_port
  }

  service_connect_configuration {
    enabled   = true
    namespace = aws_service_discovery_http_namespace.this.arn
  }

  depends_on = [aws_lb_listener.http]
}

# ---- article-db-api task + service (internal only) ----

resource "aws_ecs_task_definition" "api" {
  family                   = "${var.name}-article-db-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.api_cpu
  memory                   = var.api_memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn

  ephemeral_storage {
    size_in_gib = var.api_ephemeral_storage_gb
  }

  container_definitions = jsonencode([
    {
      name      = "article-db-api"
      image     = "${aws_ecr_repository.api.repository_url}:${var.api_image_tag}"
      essential = true
      portMappings = [{
        containerPort = var.api_port
        protocol      = "tcp"
        name          = "http"
      }]
      environment = [
        { name = "PORT", value = tostring(var.api_port) },
      ]
      secrets = [
        { name = "ZILLIZ_URI", valueFrom = aws_secretsmanager_secret.zilliz_uri.arn },
        { name = "ZILLIZ_TOKEN", valueFrom = aws_secretsmanager_secret.zilliz_token.arn },
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.api.name
          "awslogs-region"        = data.aws_region.current.name
          "awslogs-stream-prefix" = "api"
        }
      }
    }
  ])
}

resource "aws_ecs_service" "api" {
  name            = "${var.name}-article-db-api"
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = var.api_desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = local.subnet_ids
    security_groups  = [aws_security_group.api.id]
    assign_public_ip = true # no NAT gateway in this minimal setup; SG still blocks all public ingress
  }

  service_connect_configuration {
    enabled   = true
    namespace = aws_service_discovery_http_namespace.this.arn

    service {
      port_name      = "http"
      discovery_name = "article-db-api"
      client_alias {
        port     = var.api_port
        dns_name = "article-db-api"
      }
    }
  }
}
