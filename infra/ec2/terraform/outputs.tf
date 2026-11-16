output "public_ip" {
  description = "Elastic IP: point your domain's DNS A record here"
  value       = aws_eip.app.public_ip
}

output "instance_id" {
  value = aws_instance.app.id
}

output "ssh" {
  value = "ssh -i ${var.ssh_private_key_path} ubuntu@${aws_eip.app.public_ip}"
}
