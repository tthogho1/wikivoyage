"""
.env の内容を Hugging Face Spaces の Secret に一括登録するスクリプト。

Usage:
    HF_TOKEN=hf_xxx python scripts/set_hf_secrets.py
    または huggingface-cli login 済みの場合はそのまま実行。
"""
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi
except ImportError:
    sys.exit("huggingface_hub が未インストールです: pip install huggingface_hub")

# -----------------------------------------------------------------------
# 設定
# -----------------------------------------------------------------------
REPO_ID = "tthogho1/wikivoyage"   # HF Spaces の repo_id
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

# Chatbot に不要なローカル専用キーはスキップ
SKIP_KEYS = {"WIKIVOYAGE_PAGES_DIR", "S3_BUCKET", "AWS_REGION",
             "CHUNK_METHOD", "CHUNK_SIZE", "CHUNK_OVERLAP",
             "EMBEDDING_BATCH_SIZE", "ZILLIZ_BATCH_SIZE"}

# -----------------------------------------------------------------------
# .env を読み込む（dotenv 不要のシンプル実装）
# -----------------------------------------------------------------------
def load_env_file(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            # 引用符を除去
            value = value.strip().strip('"').strip("'")
            result[key.strip()] = value
    return result

# -----------------------------------------------------------------------
# メイン
# -----------------------------------------------------------------------
def main():
    token = os.getenv("HF_TOKEN")
    api = HfApi(token=token)  # token=None の場合はキャッシュ済みトークンを使用

    env_vars = load_env_file(ENV_FILE)
    print(f"Loaded {len(env_vars)} keys from {ENV_FILE.name}\n")

    ok, skipped = [], []
    for key, value in env_vars.items():
        if key in SKIP_KEYS:
            skipped.append(key)
            continue
        try:
            api.add_space_secret(repo_id=REPO_ID, key=key, value=value)
            print(f"  ✅  {key}")
            ok.append(key)
        except Exception as e:
            print(f"  ❌  {key}: {e}")

    print(f"\nDone — {len(ok)} secrets set, {len(skipped)} skipped.")
    if skipped:
        print("Skipped:", ", ".join(skipped))

if __name__ == "__main__":
    main()
