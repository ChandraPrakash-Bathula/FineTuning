"""Shared HF setup. Fixes this machine's broken SSL_CERT_FILE and resolves a token."""
import os


def fix_ssl():
    """Homebrew leaves SSL_CERT_FILE pointing at a file that no longer exists."""
    cur = os.environ.get("SSL_CERT_FILE")
    if cur and not os.path.exists(cur):
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())


def _from_env_file(path):
    """Minimal .env reader. Accepts HF_TOKEN, HUGGINGFACE_TOKEN or a bare `hf`,
    with or without surrounding quotes."""
    if not os.path.exists(path):
        return None
    wanted = ("hf", "hf_token", "huggingface_token", "hugging_face_hub_token")
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip().lower().lstrip("$") not in wanted:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if value:
            return value
    return None


def token():
    """Token lookup, in priority order:
    HF_TOKEN env -> repo .hf_token -> backend/.env -> repo .env -> stored login."""
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok.strip()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local = os.path.join(here, ".hf_token")
    if os.path.exists(local):
        return open(local).read().strip()
    for candidate in (os.path.join(here, "backend", ".env"), os.path.join(here, ".env")):
        found = _from_env_file(candidate)
        if found:
            return found
    return None


fix_ssl()

REPO = "ChandraPrakashBathula/wikisql-qwen7b-finetuning-lab"
VARIANTS = (["full_ft"]
            + [f"lora_r{r}" for r in (4, 8, 16, 32, 64)]
            + [f"qlora_r{r}" for r in (4, 8, 16, 32, 64)])
