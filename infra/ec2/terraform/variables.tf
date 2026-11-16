variable "region" {
  description = "AWS region"
  type        = string
  default     = "ap-northeast-1"
}

variable "name" {
  description = "Name prefix for every resource"
  type        = string
  default     = "wikivoyage-app"
}

variable "instance_type" {
  description = "Both models stay in memory (~7 GB on CPU); 16 GB RAM is the minimum"
  type        = string
  default     = "m7i.xlarge"
}

variable "root_volume_gb" {
  description = "Root disk: OS, Python venv with torch, models (~6 GB), node_modules, build"
  type        = number
  default     = 60
}

variable "ssh_public_key_path" {
  description = "Public key installed for the ubuntu user"
  type        = string
  default     = "~/.ssh/wikivoyage-ec2.pub"
}

variable "ssh_private_key_path" {
  description = "Matching private key, written into the Ansible inventory"
  type        = string
  default     = "~/.ssh/wikivoyage-ec2"
}

variable "ssh_cidr" {
  description = "CIDR allowed to SSH, e.g. \"203.0.113.10/32\" (your IP)"
  type        = string

  validation {
    condition     = can(cidrhost(var.ssh_cidr, 0)) && var.ssh_cidr != "0.0.0.0/0"
    error_message = "ssh_cidr must be a valid CIDR and must not be 0.0.0.0/0."
  }
}

variable "vpc_id" {
  description = "VPC to use; empty = the region's default VPC"
  type        = string
  default     = ""
}

variable "subnet_id" {
  description = "Public subnet to use; empty = the first default-for-AZ subnet of the VPC"
  type        = string
  default     = ""
}
