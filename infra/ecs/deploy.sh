#!/usr/bin/env bash
# Build or tear down the whole ECS stack in one command.
#
#   ./deploy.sh up      [-y]   terraform apply → set secrets from .env → ansible build/push/deploy
#   ./deploy.sh down    [-y]   empty the ECR repos → terraform destroy
#   ./deploy.sh status         terraform outputs + ECS service state
#
# -y skips the confirmation prompts (terraform -auto-approve).
#
# Needs terraform, aws, docker (+buildx), ansible-playbook and git; the toolbox image has them all:
#   docker compose -f infra/ecs/toolbox/compose.yml run --rm deploy ./deploy.sh up
#
# Overrides (env): AWS_REGION (default ap-northeast-1), STACK_NAME (default wikivoyage),
# ENV_FILE (default <repo>/.env). Other Terraform variables: terraform/terraform.tfvars or TF_VAR_*.
# Secrets: OPENAI_API_KEY, ZILLIZ_URI, ZILLIZ_TOKEN are taken from the environment, else from ENV_FILE.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
TF_DIR="$HERE/terraform"
ANSIBLE_DIR="$HERE/ansible"

REGION="${AWS_REGION:-ap-northeast-1}"
NAME="${STACK_NAME:-wikivoyage}"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"
SECRET_KEYS=(OPENAI_API_KEY ZILLIZ_URI ZILLIZ_TOKEN)

YES=false
CMD="${1:-}"
shift || true
for arg in "$@"; do
  case "$arg" in
    -y|--yes) YES=true ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

TF_VARS=(-var "region=$REGION" -var "name=$NAME")
APPROVE=()
$YES && APPROVE=(-auto-approve)

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

require_tools() {
  local missing=()
  for t in "$@"; do command -v "$t" >/dev/null || missing+=("$t"); done
  ((${#missing[@]} == 0)) || die "missing tools: ${missing[*]} (use the toolbox: infra/ecs/toolbox)"
}

confirm() {
  $YES && return 0
  read -r -p "$1 [y/N] " ans
  [[ "$ans" == [yY] || "$ans" == [yY][eE][sS] ]] || die "aborted"
}

# Value of KEY from the environment, else from ENV_FILE (KEY=VALUE lines; quotes stripped).
# The file is parsed, not sourced, so values with $ or spaces are taken literally.
secret_value() {
  local key="$1" val="${!1:-}"
  if [[ -z "$val" && -f "$ENV_FILE" ]]; then
    val="$(grep -E "^[[:space:]]*(export[[:space:]]+)?${key}=" "$ENV_FILE" | tail -n1 | cut -d= -f2- || true)"
    val="${val%$'\r'}"
    val="${val#\"}"; val="${val%\"}"
    val="${val#\'}"; val="${val%\'}"
  fi
  printf '%s' "$val"
}

tf_init() { terraform -chdir="$TF_DIR" init -input=false >/dev/null; }

push_secrets() {
  log "Secrets Manager ($NAME/*)"
  local key id new cur missing=()
  for key in "${SECRET_KEYS[@]}"; do
    id="$NAME/$key"
    new="$(secret_value "$key")"
    cur="$(aws secretsmanager get-secret-value --region "$REGION" --secret-id "$id" \
             --query SecretString --output text 2>/dev/null || true)"
    if [[ -z "$new" ]]; then
      if [[ -z "$cur" || "$cur" == "CHANGEME" ]]; then missing+=("$key"); fi
      echo "  $key: not in env or $ENV_FILE — left as is"
    elif [[ "$new" == "$cur" ]]; then
      echo "  $key: unchanged"
    else
      aws secretsmanager put-secret-value --region "$REGION" --secret-id "$id" \
        --secret-string "$new" >/dev/null
      echo "  $key: updated"
    fi
  done
  ((${#missing[@]} == 0)) || die "secrets still unset: ${missing[*]}. Put them in $ENV_FILE or export them, then re-run."
}

cmd_up() {
  require_tools terraform aws docker ansible-playbook git
  aws sts get-caller-identity --query Arn --output text >/dev/null || die "AWS credentials not available"

  log "terraform apply (region=$REGION, name=$NAME)"
  tf_init
  terraform -chdir="$TF_DIR" apply -input=false "${TF_VARS[@]}" "${APPROVE[@]}"

  push_secrets

  # Builds both images (linux/amd64), pushes :<git sha> and :latest, forces a new deployment of
  # both services and waits for them to stabilise. The task definitions use :latest.
  log "ansible: build, push, deploy"
  (cd "$ANSIBLE_DIR" && ansible-playbook site.yml -e "aws_region=$REGION" -e "name=$NAME")

  log "done"
  echo "  URL: http://$(terraform -chdir="$TF_DIR" output -raw alb_dns_name)"
}

cmd_down() {
  require_tools terraform aws
  confirm "Destroy the '$NAME' ECS stack in $REGION, including all images in its ECR repos?"

  # force_delete is false by default and destroy reads it from state, so empty the repos first.
  log "emptying ECR repos"
  local repo ids
  for repo in "$NAME-webapp" "$NAME-article-db-api"; do
    ids="$(aws ecr list-images --region "$REGION" --repository-name "$repo" \
             --query 'imageIds[*]' --output json 2>/dev/null || echo '[]')"
    if [[ "$ids" == "[]" ]]; then
      echo "  $repo: empty or absent"
      continue
    fi
    # batch-delete-image takes at most 100 ids per call
    local n; n="$(jq length <<<"$ids")"
    for ((i = 0; i < n; i += 100)); do
      aws ecr batch-delete-image --region "$REGION" --repository-name "$repo" \
        --image-ids "$(jq -c ".[$i:$((i + 100))]" <<<"$ids")" >/dev/null
    done
    echo "  $repo: deleted $n image(s)"
  done

  log "terraform destroy"
  tf_init
  terraform -chdir="$TF_DIR" destroy -input=false "${TF_VARS[@]}" -auto-approve
  log "done"
}

cmd_status() {
  require_tools terraform aws
  log "terraform outputs"
  terraform -chdir="$TF_DIR" output 2>/dev/null || echo "  (no state — stack not created)"
  log "ECS services"
  aws ecs describe-services --region "$REGION" --cluster "$NAME" \
    --services "$NAME-webapp" "$NAME-article-db-api" \
    --query 'services[].{service:serviceName,status:status,desired:desiredCount,running:runningCount,rollout:deployments[0].rolloutState}' \
    --output table 2>/dev/null || echo "  (cluster $NAME not found)"
}

case "$CMD" in
  up) cmd_up ;;
  down) require_tools jq; cmd_down ;;
  status) cmd_status ;;
  *) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
