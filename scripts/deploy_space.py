"""Push this project to a Hugging Face Space.

Two flavours:

  static   free CPU Space, the whole lesson. Uploads frontend/dist only.
           Everything works except asking the models a new question.

  docker   GPU Space. Uploads the sources plus deploy/Dockerfile; the Space
           builds the frontend and runs FastAPI, serving the lesson and the
           inference API on one origin.

  HF_TOKEN=hf_xxx python3 scripts/deploy_space.py static <user>/<space-name>
  HF_TOKEN=hf_xxx python3 scripts/deploy_space.py docker <user>/<space-name>

A write-scoped token is required to create or push to a Space.
"""
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hf import token  # noqa: E402  (also repairs this machine's SSL_CERT_FILE)

from huggingface_hub import HfApi  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# NEVER upload these. A Space repo is a git repo; anything copied in is published.
# Hugging Face's own secret scanner is a backstop, not a substitute for this list.
SECRET_PATTERNS = (
    ".env", ".env.*", "*.env", ".hf_token", "*.pem", "*.key",
    "id_rsa*", "credentials*", "secrets*", ".netrc", ".git", ".gitignore",
    "__pycache__", "*.pyc", "node_modules", "dist",
)


def _safe_ignore(_dir, names):
    """shutil.copytree filter: drop secrets and build junk."""
    import fnmatch
    drop = set()
    for n in names:
        for pat in SECRET_PATTERNS:
            if fnmatch.fnmatch(n, pat):
                drop.add(n)
    return drop


def _assert_no_secrets(root):
    """Last line of defence: refuse to upload anything that smells like a key."""
    import re
    token_re = re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")
    bad = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            full = os.path.join(dirpath, f)
            rel = os.path.relpath(full, root)
            import fnmatch
            if any(fnmatch.fnmatch(f, p) for p in SECRET_PATTERNS):
                bad.append(f"{rel} (matches a secret filename pattern)")
                continue
            if os.path.getsize(full) > 2_000_000:
                continue
            try:
                text = open(full, "r", errors="ignore").read()
            except OSError:
                continue
            if token_re.search(text):
                bad.append(f"{rel} (contains something shaped like an HF token)")
    if bad:
        raise SystemExit("REFUSING TO UPLOAD — secrets found in the staged tree:\n  "
                         + "\n  ".join(bad))


def build_frontend():
    print("building frontend…")
    subprocess.run(["npm", "run", "build"], cwd=os.path.join(ROOT, "frontend"), check=True)
    dist = os.path.join(ROOT, "frontend", "dist")
    if not os.path.isdir(dist):
        sys.exit(f"MISSING ARTIFACT\nRequired: a production build\n"
                 f"Expected location: {dist}\nRun `npm run build` in frontend/.")
    return dist


def stage_static(tmp):
    dist = build_frontend()
    shutil.copytree(dist, tmp, dirs_exist_ok=True)
    shutil.copyfile(os.path.join(ROOT, "deploy", "README.static.md"),
                    os.path.join(tmp, "README.md"))
    return "static"


def stage_docker(tmp):
    for rel in ("backend", "frontend/src", "frontend/public"):
        shutil.copytree(os.path.join(ROOT, rel), os.path.join(tmp, rel),
                        dirs_exist_ok=True, ignore=_safe_ignore)
    for rel in ("frontend/package.json", "frontend/package-lock.json",
                "frontend/index.html", "frontend/vite.config.ts",
                "frontend/tsconfig.json", "frontend/tsconfig.app.json",
                "frontend/tsconfig.node.json"):
        shutil.copyfile(os.path.join(ROOT, rel), os.path.join(tmp, rel))
    os.makedirs(os.path.join(tmp, "data"), exist_ok=True)
    shutil.copyfile(os.path.join(ROOT, "data", "registry.json"),
                    os.path.join(tmp, "data", "registry.json"))
    shutil.copyfile(os.path.join(ROOT, "deploy", "Dockerfile"),
                    os.path.join(tmp, "Dockerfile"))
    shutil.copyfile(os.path.join(ROOT, "deploy", "README.docker.md"),
                    os.path.join(tmp, "README.md"))
    return "docker"


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("static", "docker"):
        sys.exit(__doc__)
    flavour, repo_id = sys.argv[1], sys.argv[2]
    # A Space is outward-facing. Default to private; --public is explicit.
    rest = sys.argv[3:]
    private = "--public" not in rest
    # Docker Spaces are refused on free cpu-basic without PRO, but paid hardware
    # is allowed. Pass --hardware to create one directly on a paid tier.
    hardware = None
    for i, a in enumerate(rest):
        if a == "--hardware" and i + 1 < len(rest):
            hardware = rest[i + 1]

    tok = token()
    if not tok:
        sys.exit("No HF token. Set HF_TOKEN (write scope) or write one to .hf_token.")

    api = HfApi(token=tok)
    with tempfile.TemporaryDirectory() as tmp:
        sdk = stage_static(tmp) if flavour == "static" else stage_docker(tmp)

        _assert_no_secrets(tmp)

        total = sum(os.path.getsize(os.path.join(r, f))
                    for r, _, fs in os.walk(tmp) for f in fs)
        n = sum(len(fs) for _, _, fs in os.walk(tmp))
        print(f"staged {n} files, {total / 1e6:.1f} MB")

        # Only create when missing: create_repo re-validates the free-tier rule
        # even with exist_ok=True, which 402s for an existing Docker Space.
        try:
            api.repo_info(repo_id, repo_type="space")
            print(f"space {repo_id} exists, uploading into it")
        except Exception:
            api.create_repo(repo_id, repo_type="space", space_sdk=sdk,
                            private=private, space_hardware=hardware)
            print(f"created {repo_id}" + (f" on {hardware}" if hardware else ""))
        api.upload_folder(folder_path=tmp, repo_id=repo_id, repo_type="space",
                          commit_message=f"Deploy fine-tuning laboratory ({flavour})")

    url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"\ndeployed ({'private' if private else 'public'}): {url}")
    if flavour == "docker":
        print("Set hardware under Settings -> a 24 GB GPU for LOAD_BASES=both,\n"
              "or a 16 GB T4 with LOAD_BASES=qlora. On CPU it still serves the lesson.")


if __name__ == "__main__":
    main()
