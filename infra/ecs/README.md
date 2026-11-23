# ECS Fargate deployment (ALB + two services)

Alternative to [../ec2](../ec2) (single EC2 host). This stack runs the app as two
independently scaled ECS Fargate services behind an Application Load Balancer:

```
Internet ──443──▶ ALB (public) ──▶ webapp service (Fargate, Next.js, :3000)
                                        │  (Service Connect, private DNS)
                                        ▼
                                   article-db-api service (Fargate, FastAPI, :8001)
                                        │                            │
                                        ▼                            ▼
                                     OpenAI                       Zilliz Cloud
```

- `webapp` is the only service reachable from the internet (via the ALB).
- `article-db-api` has no public ingress; it's reached only from `webapp` via
  [ECS Service Connect](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/service-connect.html)
  at `http://article-db-api:8001` (passed to the webapp as `RETRIEVAL_URL`).
- Both services run on Fargate (no EC2 instances to patch). The API task is sized
  4 vCPU / 16 GB RAM to hold the Qwen3 + SigLIP2 models in memory (CPU-only).
- Secrets (`OPENAI_API_KEY`, `ZILLIZ_URI`, `ZILLIZ_TOKEN`) live in AWS Secrets
  Manager and are injected as task secrets — never baked into images or stored
  in Terraform state.
- Docker images are built and pushed to two private ECR repos. Terraform creates
  the repos/cluster/services; Ansible builds+pushes images and triggers rollouts
  (Fargate has no servers to configure, so Ansible's job here is CI-style
  orchestration, not host configuration).

## Layout

```
terraform/   VPC lookup, ECR, ALB, Secrets Manager, ECS cluster/services
ansible/     Build & push images, force new ECS deployments
```

## Deploy workflow

1. `cd terraform && terraform init && terraform apply`
   - Creates the ECR repos first. Outputs `ecr_webapp_repository_url`,
     `ecr_api_repository_url`, `alb_dns_name`.
   - On the **first apply**, the ECS services will fail to stabilize because the
     ECR repos are empty — push images first (step 2), then `terraform apply`
     again (or just let Ansible build+push, then re-run apply / let the service
     stabilize).
2. `cd ../ansible && ansible-playbook site.yml -e aws_region=ap-northeast-1`
   - Builds `webapp/Dockerfile` and `article_db/Dockerfile`, pushes to ECR,
     forces a new deployment of both ECS services.
3. Point your domain's DNS (CNAME) at the `alb_dns_name` output. Put CloudFront
   in front of the ALB if you want a CDN / WAF / custom domain with ACM at the
   edge — not included here, add an `aws_cloudfront_distribution` pointed at
   the ALB as origin if needed.

## Day-2 operations

| Task | Command |
|---|---|
| Redeploy after a code change | re-run `ansible-playbook site.yml` |
| Change a secret | update it in Secrets Manager (or `terraform apply` after editing `terraform.tfvars`), then force a new deployment |
| Logs | CloudWatch Logs groups `/ecs/wikivoyage-webapp`, `/ecs/wikivoyage-article-db-api` |
| Scale | change `webapp_desired_count` / `api_desired_count` in `terraform.tfvars`, `terraform apply` |
| Teardown | `terraform destroy` (then delete the ECR repos' images if `force_delete` is false) |

See [terraform/variables.tf](terraform/variables.tf) for all tunables (CPU/memory,
desired counts, container ports, secret ARNs/values).
