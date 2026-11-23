data "aws_vpc" "this" {
  default = var.vpc_id == ""
  id      = var.vpc_id == "" ? null : var.vpc_id
}

data "aws_subnets" "public" {
  count = length(var.public_subnet_ids) == 0 ? 1 : 0

  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.this.id]
  }
  filter {
    name   = "default-for-az"
    values = ["true"]
  }
}

locals {
  subnet_ids = length(var.public_subnet_ids) > 0 ? var.public_subnet_ids : data.aws_subnets.public[0].ids
}

resource "aws_security_group" "alb" {
  name        = "${var.name}-alb"
  description = "ALB: public HTTP/HTTPS in, anything out"
  vpc_id      = data.aws_vpc.this.id
}

resource "aws_vpc_security_group_ingress_rule" "alb_http" {
  for_each          = toset(var.alb_ingress_cidrs)
  security_group_id = aws_security_group.alb.id
  cidr_ipv4         = each.value
  ip_protocol       = "tcp"
  from_port         = 80
  to_port           = 80
}

resource "aws_vpc_security_group_ingress_rule" "alb_https" {
  for_each          = toset(var.alb_ingress_cidrs)
  security_group_id = aws_security_group.alb.id
  cidr_ipv4         = each.value
  ip_protocol       = "tcp"
  from_port         = 443
  to_port           = 443
}

resource "aws_vpc_security_group_egress_rule" "alb_all" {
  security_group_id = aws_security_group.alb.id
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1"
}

resource "aws_security_group" "webapp" {
  name        = "${var.name}-webapp"
  description = "webapp Fargate tasks: traffic from the ALB only"
  vpc_id      = data.aws_vpc.this.id
}

resource "aws_vpc_security_group_ingress_rule" "webapp_from_alb" {
  security_group_id            = aws_security_group.webapp.id
  referenced_security_group_id = aws_security_group.alb.id
  ip_protocol                   = "tcp"
  from_port                     = var.webapp_port
  to_port                       = var.webapp_port
}

resource "aws_vpc_security_group_egress_rule" "webapp_all" {
  security_group_id = aws_security_group.webapp.id
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1"
}

resource "aws_security_group" "api" {
  name        = "${var.name}-article-db-api"
  description = "article_db.api Fargate tasks: traffic from webapp only, no public ingress"
  vpc_id      = data.aws_vpc.this.id
}

resource "aws_vpc_security_group_ingress_rule" "api_from_webapp" {
  security_group_id            = aws_security_group.api.id
  referenced_security_group_id = aws_security_group.webapp.id
  ip_protocol                   = "tcp"
  from_port                     = var.api_port
  to_port                       = var.api_port
}

resource "aws_vpc_security_group_egress_rule" "api_all" {
  security_group_id = aws_security_group.api.id
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1"
}
