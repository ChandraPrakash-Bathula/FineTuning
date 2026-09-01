"""Pull every Phase 1 artifact the app needs out of the private HF repo into data/.

Discovers paths by listing the repo rather than assuming them, so a slightly
different layout than the build template describes still works.

    HF_TOKEN=hf_xxx python3 scripts/fetch_artifacts.py
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hf import REPO, VARIANTS, token  # noqa: E402

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
IMG_EXT = (".png", ".jpg", ".jpeg", ".svg", ".webp", ".pdf")


def grab(api_token, remote, local):
    path = hf_hub_download(REPO, remote, token=api_token)
    os.makedirs(os.path.dirname(local), exist_ok=True)
    shutil.copyfile(path, local)
    return os.path.getsize(local)


def main():
    tok = token()  # None is fine: the repo is public

    files = HfApi().list_repo_files(REPO, token=tok)
    print(f"{len(files)} files in {REPO}\n")

    # registry.json, wherever it lives
    registries = [f for f in files if os.path.basename(f) == "registry.json"]
    if not registries:
        print("!! no registry.json found — the app has no numbers without it")
    for f in registries:
        n = grab(tok, f, os.path.join(DATA, "registry.json"))
        print(f"registry   {f}  ({n:,} bytes)")

    # The repo's embedded generation_probes cover only 25 questions per variant
    # and are superseded by the 1,000-question exports that build_probes.py
    # consumes. Deliberately not downloaded -- two probe sources disagreeing is
    # worse than one.
    probes = [f for f in files if os.path.basename(f) == "generation_probes.json"]
    print(f"\nskipping {len(probes)} embedded 25-probe files "
          f"(superseded by the exports build_probes.py reads)")

    # figures
    figs = [f for f in files if f.lower().endswith(IMG_EXT)]
    print(f"\nfigures ({len(figs)})")
    for f in sorted(figs):
        grab(tok, f, os.path.join(DATA, "figures", os.path.basename(f)))
        print(f"  {f}")

    # anything else JSON that isn't a weight/config file, for inspection
    known = set(registries) | set(probes)
    other = [f for f in files
             if f.endswith(".json") and f not in known
             and os.path.basename(f) not in (
                 "config.json", "adapter_config.json", "generation_config.json",
                 "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                 "model.safetensors.index.json", "added_tokens.json")]
    if other:
        print(f"\nother json ({len(other)})")
        for f in sorted(other):
            grab(tok, f, os.path.join(DATA, "extra", f.replace("/", "__")))
            print(f"  {f}")

    # a manifest of the full remote listing, so the layout is inspectable offline
    with open(os.path.join(DATA, "repo_files.json"), "w") as fh:
        json.dump(sorted(files), fh, indent=2)
    print(f"\nwrote data/repo_files.json")


if __name__ == "__main__":
    main()
