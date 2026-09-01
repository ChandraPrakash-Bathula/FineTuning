"""Inference backend -- the GPU half of the lab.

The frontend is fully usable without this: it reads registry.json and the
pre-generated probes. This process only adds what needs a 7B model in VRAM,
namely answering a question the student typed themselves.

    pip install -r backend/requirements.txt
    uvicorn backend.app:app --host 0.0.0.0 --port 8000

Set HF_TOKEN only if the weights repo is private again.
"""
import functools
import json
import os
import sys
import threading
import time
import traceback
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sql_exec import execute, render_sql  # noqa: E402

REPO = "ChandraPrakashBathula/wikisql-qwen7b-finetuning-lab"
BASE = "Qwen/Qwen2.5-7B-Instruct"
RANKS = (4, 8, 16, 32, 64)
LORA_ADAPTERS = [f"lora_r{r}" for r in RANKS]
QLORA_ADAPTERS = [f"qlora_r{r}" for r in RANKS]
ADAPTERS = LORA_ADAPTERS + QLORA_ADAPTERS
TOKEN = os.environ.get("HF_TOKEN")
LOAD_FULL_FT = os.environ.get("LOAD_FULL_FT", "0") == "1"

# Which bases to hold in VRAM. The bf16 base is ~15.2 GB and the 4-bit one ~4 GB,
# so "both" needs a 24 GB card. On a 16 GB T4, "qlora" alone fits comfortably.
#   both  (default)  LoRA + QLoRA          >= 24 GB
#   qlora            QLoRA adapters only   >= 16 GB  (or less)
#   lora             LoRA adapters only    >= 24 GB
LOAD_BASES = os.environ.get("LOAD_BASES", "both").lower()
WANT_LORA = LOAD_BASES in ("both", "lora")
WANT_QLORA = LOAD_BASES in ("both", "qlora")

# Serve the built frontend from the same process when it is present, so a single
# Space can host the lesson and the API on one origin.
FRONTEND_DIST = os.environ.get(
    "FRONTEND_DIST",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "frontend", "dist"))

SYSTEM = ('You translate questions into SQL. Answer with one SQL query and '
          'nothing else. Refer to columns as col0, col1, ... and the table '
          'as "table".')

app = FastAPI(title="Fine-Tuning Lab")

# Loading the bases means downloading ~16 GB and materialising three models. Done
# lazily that cost lands on whoever clicks first, as a multi-minute hang with no
# feedback. So warm up at startup in a background thread and report progress.
_warm = {"state": "cold", "detail": "", "started": 0.0, "seconds": 0.0}
_warm_lock = threading.Lock()


def _warmup():
    with _warm_lock:
        if _warm["state"] in ("loading", "ready"):
            return
        _warm.update(state="loading", detail="loading bases", started=time.time())
    try:
        models()
        with _warm_lock:
            _warm.update(state="ready", detail="",
                         seconds=round(time.time() - _warm["started"], 1))
    except Exception as exc:
        traceback.print_exc()
        with _warm_lock:
            _warm.update(state="error", detail=f"{type(exc).__name__}: {exc}",
                         seconds=round(time.time() - _warm["started"], 1))


@app.on_event("startup")
def _start_warmup():
    """Only when a GPU is actually present -- on CPU this would never finish."""
    if os.environ.get("WARMUP", "auto") == "off":
        return
    try:
        import torch
        if not torch.cuda.is_available():
            with _warm_lock:
                _warm.update(state="no_gpu", detail="no CUDA device")
            return
    except Exception:
        with _warm_lock:
            _warm.update(state="no_gpu", detail="torch unavailable")
        return
    threading.Thread(target=_warmup, daemon=True).start()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ---------------------------------------------------------------- registry ---

@functools.lru_cache(maxsize=1)
def registry() -> dict:
    local = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "registry.json")
    if os.path.exists(local):
        return json.load(open(local))
    from huggingface_hub import hf_hub_download
    return json.load(open(hf_hub_download(REPO, "registry/registry.json", token=TOKEN)))


@app.get("/api/registry")
def get_registry():
    return registry()


# ---------------------------------------------------------------- examples ---

@functools.lru_cache(maxsize=1)
def dev_examples() -> list[dict]:
    """Held-out dev rows, WITH table rows so generated SQL can actually run."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikisql", split="validation")
    out = []
    for i, row in enumerate(ds.select(range(500))):
        t = row["table"]
        out.append({
            "id": i,
            "question": row["question"],
            "table": {"header": t["header"], "types": t["types"], "rows": t["rows"]},
            "gold_sql": render_sql(row["sql"]),
        })
    return out


@app.get("/api/examples")
def get_examples(n: int = 20):
    return {"examples": dev_examples()[: max(1, min(n, 200))]}


# --------------------------------------------------------------- inference ---

class Table(BaseModel):
    header: list[str]
    types: list[str] | None = None
    rows: list[list[Any]] = []


class RunRequest(BaseModel):
    variants: list[str]
    example_id: int | None = None
    question: str | None = None
    table: Table | None = None


@functools.lru_cache(maxsize=1)
def models():
    """Bases loaded ONCE, adapters attached by name and switched with set_adapter().

    TWO bases, not one. A QLoRA adapter was trained to correct a 4-bit NF4 model;
    serving it on the bf16 base measurably loses accuracy (about 1 point on the
    held-out set -- see the QLoRA page's serving comparison, generated from
    regen.log vs regen_q.log). So LoRA adapters go on the bf16 base and QLoRA
    adapters go on a 4-bit base, which is how the registry evaluated them.

    Cost: ~15 GB for the bf16 base plus ~5 GB for the 4-bit one, instead of the
    ~150 GB that loading eleven separate models would take.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(BASE)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    lora = None
    if WANT_LORA:
        bf16_base = AutoModelForCausalLM.from_pretrained(
            BASE, dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
        lora = PeftModel.from_pretrained(
            bf16_base, REPO, subfolder=LORA_ADAPTERS[0],
            adapter_name=LORA_ADAPTERS[0], token=TOKEN)
        for name in LORA_ADAPTERS[1:]:
            lora.load_adapter(REPO, subfolder=name, adapter_name=name, token=TOKEN)

    # matches the training-time quantisation recorded in registry.json:
    # {"bits": 4, "quant_type": "nf4", "double_quant": true, "compute_dtype": "bfloat16"}
    qlora = None
    if WANT_QLORA:
        nf4 = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        q4_base = AutoModelForCausalLM.from_pretrained(
            BASE, quantization_config=nf4, device_map="auto",
            attn_implementation="sdpa")
        qlora = PeftModel.from_pretrained(
            q4_base, REPO, subfolder=QLORA_ADAPTERS[0],
            adapter_name=QLORA_ADAPTERS[0], token=TOKEN)
        for name in QLORA_ADAPTERS[1:]:
            qlora.load_adapter(REPO, subfolder=name, adapter_name=name, token=TOKEN)

    full = None
    if LOAD_FULL_FT:
        full = AutoModelForCausalLM.from_pretrained(
            REPO, subfolder="full_ft", dtype=torch.bfloat16,
            device_map="auto", token=TOKEN)
    return tok, lora, qlora, full


def build_prompt(question: str, headers: list[str], types: list[str]) -> str:
    schema = "\n".join(f"  col{i}: {h} ({t})"
                       for i, (h, t) in enumerate(zip(headers, types)))
    return f'Table "table":\n{schema}\n\nQuestion: {question.strip()}\n\nSQL:'


def _require_gpu():
    """Fail fast. Without CUDA, models() would try to pull ~16 GB and then die
    loading it into host RAM -- the request would hang for minutes first."""
    try:
        import torch
    except ImportError:
        raise HTTPException(503, "torch is not installed on this backend")
    if not torch.cuda.is_available():
        raise HTTPException(
            503, "This backend has no CUDA device, so it cannot generate. "
                 "Recorded probe answers still work; live generation needs a GPU.")


def generate_sql(variant_id: str, question: str, headers, types, max_new_tokens=128) -> str:
    import torch
    _require_gpu()
    tok, lora, qlora, full = models()
    if variant_id == "full_ft":
        if full is None:
            raise HTTPException(503, "full_ft not loaded; start with LOAD_FULL_FT=1")
        m = full
    elif variant_id in QLORA_ADAPTERS:
        if qlora is None:
            raise HTTPException(
                503, f"QLoRA base not loaded (LOAD_BASES={LOAD_BASES}); "
                     f"restart with LOAD_BASES=both or qlora")
        qlora.set_adapter(variant_id)   # milliseconds, no reload
        m = qlora
    elif variant_id in LORA_ADAPTERS:
        if lora is None:
            raise HTTPException(
                503, f"LoRA base not loaded (LOAD_BASES={LOAD_BASES}); "
                     f"restart with LOAD_BASES=both or lora")
        lora.set_adapter(variant_id)
        m = lora
    else:
        raise HTTPException(404, f"unknown variant {variant_id!r}")

    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_prompt(question, headers, types)}]
    ids = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt").to(m.device)
    with torch.no_grad():
        out = m.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    text = tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()
    return text.split("\n")[0].strip().strip("`")


@app.post("/api/run")
def run(req: RunRequest):
    if req.example_id is not None:
        ex = next((e for e in dev_examples() if e["id"] == req.example_id), None)
        if ex is None:
            raise HTTPException(404, f"no example {req.example_id}")
        question, table, gold = ex["question"], ex["table"], ex["gold_sql"]
    elif req.question and req.table:
        question = req.question
        table = req.table.model_dump()
        table["types"] = table.get("types") or ["text"] * len(table["header"])
        gold = None          # student's own table: nothing to score against
    else:
        raise HTTPException(422, "give example_id, or question + table")

    gold_run = execute(table, gold) if gold else {"rows": None, "error": None}

    results = []
    for vid in req.variants:
        try:
            sql = generate_sql(vid, question, table["header"], table["types"])
        except HTTPException:
            raise
        except Exception as exc:
            results.append({"variant_id": vid, "predicted_sql": None,
                            "error": f"{type(exc).__name__}: {exc}"})
            continue
        pred = execute(table, sql)
        results.append({
            "variant_id": vid,
            "predicted_sql": sql,
            "gold_sql": gold,
            "result_rows": pred["rows"],
            "gold_rows": gold_run["rows"],
            "exact_match": gold is not None and sql.strip() == gold.strip(),
            "execution_correct": (gold is not None and pred["error"] is None
                                  and pred["rows"] == gold_run["rows"]),
            "invalid_sql": pred["error"] is not None,
            "error": pred["error"],
        })

    # No gold means no verdict. The UI must show the query and rows only.
    return {"question": question, "scoreable": gold is not None, "results": results}


@app.get("/api/health")
def health():
    """Cheap: never touches the GPU, so the frontend can poll it on load."""
    available = []
    if WANT_LORA:
        available += LORA_ADAPTERS
    if WANT_QLORA:
        available += QLORA_ADAPTERS
    if LOAD_FULL_FT:
        available.append("full_ft")
    with _warm_lock:
        w = dict(_warm)
    elapsed = round(time.time() - w["started"], 1) if w["state"] == "loading" else w["seconds"]
    return {
        "ok": True,
        "adapters": available,
        "full_ft_loaded": LOAD_FULL_FT,
        "load_bases": LOAD_BASES,
        "serving": {"lora": "bf16", "qlora": "4-bit nf4"},
        # cold | loading | ready | error | no_gpu
        "models": w["state"],
        "models_detail": w["detail"],
        "models_seconds": elapsed,
    }


# Mounted last so /api/* always wins over the static routes.
if os.path.isdir(FRONTEND_DIST):
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
