# Secrets Manager entries for the 3 required app secrets. Values can be set via
# terraform.tfvars (sensitive) or left blank and populated out-of-band (AWS console /
# `aws secretsmanager put-secret-value`) — Terraform won't overwrite a blank value
# back over one set out-of-band because `ignore_changes` is set below.

resource "aws_secretsmanager_secret" "openai_api_key" {
  name                    = "${var.name}/OPENAI_API_KEY"
  recovery_window_in_days = 0
}

resource "aws_secretsmanager_secret" "zilliz_uri" {
  name                    = "${var.name}/ZILLIZ_URI"
  recovery_window_in_days = 0
}

resource "aws_secretsmanager_secret" "zilliz_token" {
  name                    = "${var.name}/ZILLIZ_TOKEN"
  recovery_window_in_days = 0
}

resource "aws_secretsmanager_secret_version" "openai_api_key" {
  secret_id     = aws_secretsmanager_secret.openai_api_key.id
  secret_string = var.openai_api_key != "" ? var.openai_api_key : "CHANGEME"
  lifecycle {
    ignore_changes = [secret_string]
  }
}

resource "aws_secretsmanager_secret_version" "zilliz_uri" {
  secret_id     = aws_secretsmanager_secret.zilliz_uri.id
  secret_string = var.zilliz_uri != "" ? var.zilliz_uri : "CHANGEME"
  lifecycle {
    ignore_changes = [secret_string]
  }
}

resource "aws_secretsmanager_secret_version" "zilliz_token" {
  secret_id     = aws_secretsmanager_secret.zilliz_token.id
  secret_string = var.zilliz_token != "" ? var.zilliz_token : "CHANGEME"
  lifecycle {
    ignore_changes = [secret_string]
  }
}
