resource "aws_lb" "this" {
  name               = var.name
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = local.subnet_ids
}

resource "aws_lb_target_group" "webapp" {
  name        = "${var.name}-webapp"
  port        = var.webapp_port
  protocol    = "HTTP"
  vpc_id      = data.aws_vpc.this.id
  target_type = "ip"

  health_check {
    path                = "/"
    healthy_threshold   = 2
    unhealthy_threshold = 5
    interval            = 30
    timeout             = 10
    matcher             = "200-399"
  }
}

# Plain HTTP by default. For HTTPS, add an ACM certificate ARN and an
# aws_lb_listener on 443 with ssl_policy + certificate_arn, and redirect 80 -> 443.
resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.this.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.webapp.arn
  }
}
