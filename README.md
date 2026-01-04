# the-rag-auditor

# RAG Auditor — Day10/Day11 

This repo contains scripts to:
1) build calibration datasets (SciFact / FEVER),
2) generate PR + reliability plots,
3) export routing thresholds to `configs/thresholds.yaml`,
4) (Day11) use thresholds for routing decisions.

---

## Quick start 

### 0) Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
