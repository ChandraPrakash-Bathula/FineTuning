---
title: Fine-Tuning Laboratory
emoji: 🔬
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Interactive Fine-Tuning Laboratory

The lesson plus live inference. FastAPI serves the built frontend and the API on
one origin.

**Hardware.** The bf16 base is ~15.2 GB, the 4-bit NF4 base ~4 GB.

| `LOAD_BASES` | Serves | Minimum |
|---|---|---|
| `qlora` | QLoRA adapters | Nvidia T4 — 16 GB |
| `both` (default) | LoRA + QLoRA | Nvidia L4 / A10G — 24 GB |
| `both` + `LOAD_FULL_FT=1` | all eleven | L40S — 48 GB |

Set these under Settings → Variables:

```
LOAD_BASES=both
LOAD_FULL_FT=1
```

On an L40S that is ~35 GB of weights in 48 GB. The bases warm up in a background
thread at startup, so the Space downloads ~16 GB and loads three models before the
first request rather than during it; `/api/health` reports `models: loading | ready`
and the UI shows the progress.

On CPU hardware the Space still serves the whole lesson and reports live inference
as unavailable — it will not try to load a 7B model into RAM.
