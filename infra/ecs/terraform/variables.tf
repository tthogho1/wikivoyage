variable "region" {
  description = "AWS region"
  type        = string
  default     = "ap-northeast-1"
}

variable "name" {
  description = "Name prefix for every resource"
  type        = string
  default     = "wikivoyage"
}

variable "vpc_id" {
  description = "VPC to use; empty = the region's default VPC"
  type        = string
  default     = ""
}

# ---- Networking ----

variable "public_subnet_ids" {
  description = "Subnets for the ALB and the webapp tasks (public); empty = all default-for-AZ subnets of the VPC"
  type        = list(string)
  default     = []
}

variable "alb_ingress_cidrs" {
  description = "CIDRs allowed to reach the ALB on 80/443"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

# ---- webapp service (Next.js) ----

variable "webapp_port" {
  type    = number
  default = 3000
}

variable "webapp_cpu" {
  description = "Fargate vCPU units (256 = .25 vCPU)"
  type        = number
  default     = 512
}

variable "webapp_memory" {
  description = "Fargate memory (MB)"
  type        = number
  default     = 1024
}

variable "webapp_desired_count" {
  type    = number
  default = 1
}

# ---- article-db-api service (FastAPI, Qwen3 + SigLIP2 on CPU) ----

variable "api_port" {
  type    = number
  default = 8001
}

variable "api_cpu" {
  description = "4 vCPU: both models stay resident in memory on CPU"
  type        = number
  default     = 4096
}

variable "api_memory" {
  description = "16 GB RAM: minimum for Qwen3-Embedding-0.6B + SigLIP2-so400m resident"
  type        = number
  default     = 16384
}

variable "api_desired_count" {
  type    = number
  default = 1
}

variable "api_ephemeral_storage_gb" {
  description = "Fargate task ephemeral storage; default 20GB is enough for the model cache (~6GB) + image"
  type        = number
  default     = 21
}

# ---- Images ----

variable "webapp_image_tag" {
  type    = string
  default = "latest"
}

variable "api_image_tag" {
  type    = string
  default = "latest"
}

variable "ecr_force_delete" {
  description = "Allow `terraform destroy` to delete ECR repos even if they still contain images"
  type        = bool
  default     = false
}

# ---- Secrets (article_db + openai), read from your local .env by Ansible; set here only if you prefer Terraform-managed secrets ----

variable "openai_api_key" {
  type      = string
  default   = ""
  sensitive = true
}

variable "zilliz_uri" {
  type      = string
  default   = ""
  sensitive = true
}

variable "zilliz_token" {
  type      = string
  default   = ""
  sensitive = true
}

variable "openai_model" {
  type    = string
  default = "gpt-4o-mini"
}

variable "log_retention_days" {
  type    = number
  default = 14
}
