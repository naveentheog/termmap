# Experiments (how the final model was found)

Run from the repo root with `PYTHONPATH=.:experiments`, e.g.
`PYTHONPATH=.:experiments python experiments/run_fast.py ft4_tagpred_bi_bi3_truesub_llmft --base v7 --seeds 3`

| Script | Purpose |
|---|---|
| run_retrieval_cv.py, run_dense_candidates.py, run_hybrid_cv.py, build_v7.py | candidate generation + recall reports |
| run_rerank_cv.py, run_rerank_v4.py, run_rerank_v5.py, run_nosub.py | LightGBM reranker experiments |
| run_crossencoder.py, run_ce_*.py, run_rerank_ce.py | zero-shot cross-encoder experiments |
| export_*_input.py | inputs for the Kaggle / GPU fine-tuning scripts (kaggle/, llm/) |
| run_fast.py | the out-of-fold experiment runner used for every row of the results table |
| train_models*.py | earlier final-model trainers (superseded by ../train_final.py) |
