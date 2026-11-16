# EC2 app host (Terraform + Ansible)

Runs the chat app on one EC2 instance in `ap-northeast-1`:

```
Internet ──443──▶ nginx (Let's Encrypt, rate limit on /api/chat, optional basic auth)
                    └─▶ webapp (Next.js, 127.0.0.1:3000) ──▶ OpenAI
                           └─▶ article_db.api (FastAPI, 127.0.0.1:8001, Qwen3 + SigLIP2 on CPU) ──▶ Zilliz
```

| | |
|---|---|
| `terraform/` | m7i.xlarge (4 vCPU / 16 GB, ~$0.25/h), Ubuntu 24.04, 60 GB encrypted gp3, Elastic IP, security group (22 from your IP, 80/443 public), IAM role with SSM only. Writes `ansible/inventory.ini` |
| `ansible/` | Roles `base` (packages, swap, auto security updates, service user), `app` (git checkout, CPU torch venv, model download, Next.js build, systemd units), `nginx` (TLS, proxy, rate limit) |

Only nginx listens publicly; the API and the web app are bound to 127.0.0.1. Secrets come from
your local repo-root `.env` and are written to `/etc/wikivoyage/app.env` (mode 0640); they are not in
Terraform state or in git.

## Before you start

- **Key pair:** a dedicated SSH key: `ssh-keygen -t ed25519 -f ~/.ssh/wikivoyage-ec2 -N ''`.
- **AWS:** credentials with EC2/IAM permissions (`aws sts get-caller-identity`).
- **Domain:** you can create a DNS A record for it.
- **`.env`:** in the repo root, with `ZILLIZ_URI`, `ZILLIZ_TOKEN` and a working `OPENAI_API_KEY`.
- **Code:** the branch in `ansible/group_vars/all.yml` (`app_branch`, default `qa`) must be pushed to
  GitHub. The server clones it from there.

## Deploy

```bash
# 1. Infrastructure
cd infra/ec2/terraform
cp terraform.tfvars.example terraform.tfvars   # set ssh_cidr = "<your IP>/32"
terraform init
terraform apply                                # prints public_ip

# 2. DNS: create an A record  chat.example.com → <public_ip>, wait until it resolves
dig +short chat.example.com

# 3. Software (first run ~15 min: torch, ~6 GB of models, npm build)
cd ../ansible
ansible-playbook site.yml -e domain=chat.example.com -e letsencrypt_email=you@example.com
```

Then open `https://chat.example.com`.

## Every day

| Task | Command |
|---|---|
| Deploy new code (after `git push`) | `ansible-playbook site.yml -e domain=… -e letsencrypt_email=…` (pulls `app_branch`, rebuilds, restarts what changed) |
| Change secrets | Edit `.env`, rerun the playbook |
| Logs | `ssh … journalctl -u wikivoyage-api -f` / `-u wikivoyage-web` / `/var/log/nginx/` |
| Status | `ssh … systemctl status wikivoyage-api wikivoyage-web nginx` |
| Stop paying for compute | `aws ec2 stop-instances --instance-ids $(terraform output -raw instance_id)` (disk and Elastic IP still cost a little) |
| Remove everything | `terraform destroy` |

## Settings

- **Ansible** (`ansible/group_vars/all.yml`): `app_branch`, `chat_rate_limit` (default 10 requests/min
  per IP on `/api/chat`, burst 5), and `swap_size_mb`.
- **Basic auth:** set `APP_BASIC_AUTH_USER` and `APP_BASIC_AUTH_PASSWORD` in `.env`. The site is public
  and every question costs OpenAI tokens, so turn this on unless you want open access.
- **Terraform** (`terraform.tfvars`): `instance_type`, `root_volume_gb`, `vpc_id`/`subnet_id` (default
  VPC otherwise), and key paths.
- **Login if SSH is locked out:** the instance has the SSM agent and role, so
  `aws ssm start-session --target <instance_id>` works.

Search on CPU takes about 1–2 s per question; the OpenAI answer streams after that.
