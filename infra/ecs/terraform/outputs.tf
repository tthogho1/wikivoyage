output "alb_dns_name" {
  description = "Point your domain's DNS (CNAME) here, or use as a CloudFront origin"
  value       = aws_lb.this.dns_name
}

output "ecr_webapp_repository_url" {
  value = aws_ecr_repository.webapp.repository_url
}

output "ecr_api_repository_url" {
  value = aws_ecr_repository.api.repository_url
}

output "ecs_cluster_name" {
  value = aws_ecs_cluster.this.name
}

output "ecs_webapp_service_name" {
  value = aws_ecs_service.webapp.name
}

output "ecs_api_service_name" {
  value = aws_ecs_service.api.name
}
