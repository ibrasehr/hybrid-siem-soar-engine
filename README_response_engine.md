# Governed SOAR response engine v2 (Harshith)

Unzip into the repo root (next to correlation_engine.py). This REPLACES the earlier
response_engine/, dashboard/ and tests/test_response_engine.py.

    pip install streamlit jsonschema pytest
    python -m pytest -q tests          # 95 response-engine tests (+ Ibra's 3 if his tests/ is present)
    streamlit run dashboard/app.py

## What it does
| Tier | When | Action |
|---|---|---|
| 0 LOG_ONLY | confidence < review_floor (0.50) | recorded, nothing executed |
| 1 REVIEW   | below auto threshold, OR the BDI gate holds it | analyst queue (capacity-limited, overflow kept) |
| 2 AUTO_SOFT| auto_threshold <= conf < hard_threshold | rate_limit_flow / step_up_mfa |
| 3 AUTO_HARD| conf >= hard_threshold (0.97) | block_flow / suspend_session |

* BDI = asset criticality (1-5) x action severity (soft 2, hard 5). Over `bdi_limit` (12) -> held for an analyst.
* Fixed action catalog: anything outside it raises UnknownAction (no free-form actions).
* Ledger: snapshot before acting, exact inverse stored, revert verifies the restore and refuses on StateDrift.
* Feedback log: approve / reject / revert outcomes; suggest_threshold() only SUGGESTS, never applies.
* Missing / NaN / out-of-range confidence raises, so it can never trigger an action.
* Decisions never read ground truth (tested).

## Modules
policy.py  catalog.py  assets.py  engine.py  ledger.py  review_queue.py  feedback.py  pipeline.py  evaluate.py  adapters.py

## Quick use
    from response_engine import Policy, load_alerts, run_pipeline
    r = run_pipeline(load_alerts("alerts.json"), Policy(auto_threshold=0.9),
                     approve_ids=[...], revert_ids=[...], feedback_path="feedback_log.csv")
    r.decisions, r.ledger.to_frame(), r.queue.queue, r.feedback.to_frame()

## Real evaluation set (Ibra's CLI, held-out 30% split)
    python correlation_engine.py --train-csv data/UNSW_NB15_training-set.csv \
        --eval-split --threshold 0.5 \
        --alerts-json scratch/sample.json --alerts-full data/alerts_full.jsonl
    python -m response_engine.evaluate data/alerts_full.jsonl            # BDI gate on
    python -m response_engine.evaluate data/alerts_full.jsonl --no-bdi-gate --out results/sweep_no_gate.csv

## Limits to state in the report
* UNSW-NB15 flows have no host/IP: actions target the flow; asset criticality is SIMULATED
  (stable hash). Pass AssetRegistry.from_csv(...) to use a real inventory.
* The 100-alert alerts.json is biased (mostly >=0.9). Report numbers only from alerts_full.jsonl.
* One active containment per target. Actions are simulated state changes, not real firewall calls.
* Feedback loop = the data pipe plus a threshold suggestion; automatic retraining is future work.
