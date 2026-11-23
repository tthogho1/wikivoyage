resource "aws_ecr_repository" "webapp" {
  name                 = "${var.name}-webapp"
  image_tag_mutability = "MUTABLE"
  force_delete         = var.ecr_force_delete

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "api" {
  name                 = "${var.name}-article-db-api"
  image_tag_mutability = "MUTABLE"
  force_delete         = var.ecr_force_delete

  image_scanning_configuration {
    scan_on_push = true
  }
}

locals {
  ecr_keep_last_10_policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 10 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}

resource "aws_ecr_lifecycle_policy" "webapp" {
  repository = aws_ecr_repository.webapp.name
  policy     = local.ecr_keep_last_10_policy
}

resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name
  policy     = local.ecr_keep_last_10_policy
}
