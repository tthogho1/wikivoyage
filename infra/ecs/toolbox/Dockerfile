# Deploy toolbox for infra/ecs: Terraform, AWS CLI v2, Docker CLI + buildx, Ansible
# (+ community.docker), git. Run Terraform and the Ansible playbook from inside it so
# nothing has to be installed on the host except Docker.
#
# Build (from the repo root):
#   docker build -t wikivoyage-deploy infra/ecs/toolbox
#
# Run (from the repo root) — or use infra/ecs/toolbox/compose.yml:
#   docker run --rm -it \
#     -v "$PWD":/workspace \
#     -v "$HOME/.aws":/root/.aws \
#     -v /var/run/docker.sock:/var/run/docker.sock \
#     -e AWS_PROFILE \
#     wikivoyage-deploy
#
# Images are built by the HOST's Docker daemon through the mounted socket; this image
# only carries the docker client and the buildx plugin.

FROM python:3.12-slim-bookworm

ARG TERRAFORM_VERSION=1.9.8
ARG ANSIBLE_CORE_VERSION=2.17.5
# Set by BuildKit: amd64 or arm64. Lets the same Dockerfile build natively on Apple Silicon.
ARG TARGETARCH

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl unzip git less jq \
    && rm -rf /var/lib/apt/lists/*

# Terraform (>= 1.6 required by infra/ecs/terraform/versions.tf)
RUN curl -fsSLo /tmp/terraform.zip \
        "https://releases.hashicorp.com/terraform/${TERRAFORM_VERSION}/terraform_${TERRAFORM_VERSION}_linux_${TARGETARCH}.zip" \
    && unzip -q /tmp/terraform.zip -d /usr/local/bin \
    && rm /tmp/terraform.zip \
    && terraform version

# AWS CLI v2 (used by Terraform's credential chain and by site.yml: sts, ecr, ecs)
RUN case "${TARGETARCH}" in amd64) a=x86_64 ;; arm64) a=aarch64 ;; *) echo "unsupported arch ${TARGETARCH}"; exit 1 ;; esac \
    && curl -fsSLo /tmp/awscli.zip "https://awscli.amazonaws.com/awscli-exe-linux-${a}.zip" \
    && unzip -q /tmp/awscli.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/aws /tmp/awscli.zip \
    && aws --version

# Docker CLI + buildx plugin (community.docker.docker_image_build runs `docker buildx build`)
COPY --from=docker:27.3.1-cli /usr/local/bin/docker /usr/local/bin/docker
COPY --from=docker/buildx-bin:0.17.1 /buildx /usr/local/libexec/docker/cli-plugins/docker-buildx

# Ansible + the collection listed in infra/ecs/ansible/requirements.yml
RUN pip install --no-cache-dir "ansible-core==${ANSIBLE_CORE_VERSION}" \
    && ansible-galaxy collection install --collections-path /usr/share/ansible/collections \
        "community.docker:>=3.6.0,<4.0.0"

# Fargate tasks default to X86_64, so app images must be linux/amd64 even when built on an
# arm64 Mac. buildx honours this variable; override with -e DOCKER_DEFAULT_PLATFORM=... if needed.
ENV DOCKER_DEFAULT_PLATFORM=linux/amd64 \
    TF_PLUGIN_CACHE_DIR=/root/.terraform.d/plugin-cache \
    ANSIBLE_COLLECTIONS_PATH=/usr/share/ansible/collections

# The repo is bind-mounted with the host user's uid; site.yml runs `git rev-parse` on it.
RUN git config --system --add safe.directory '*' \
    && mkdir -p "${TF_PLUGIN_CACHE_DIR}"

WORKDIR /workspace/infra/ecs
CMD ["bash"]
