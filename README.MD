# NLP/LLM-Based Intimate Partner Violence (IPV) Detection

This repository tracks experiments that benchmark large language models on IPV detection tasks. The goal is to keep the evaluation pipeline stable while we iterate on prompts, datasets, and model families (open-source and proprietary). Everything needed to rerun an experiment—from prompts to metrics to sample code—lives inside this repo so collaborators can pick up progress quickly.

## Key Resources
- [Dataset/reddit_data.csv](Dataset/reddit_data.csv): Labeled Reddit sentences from the DETECTIPV 2022 work; used as the public ground truth during Week 1 experiments.
- [1_LLM_Eval/prompts.md](1_LLM_Eval/prompts.md): Canonical prompt templates for binary vs. multitype detection.
- [1_LLM_Eval/eval_llm_pipeline.py](1_LLM_Eval/eval_llm_pipeline.py): Reusable evaluation module that prints metrics, exports ROC/AUC/precision-residual plots, and builds multitype confidence waterfalls.
- [1_LLM_Eval/eval_pipeline_sample_use.ipynb](1_LLM_Eval/eval_pipeline_sample_use.ipynb): Notebook walkthrough showing how to call the pipeline for both binary and multitype runs.
- [1_LLM_Eval/test_results/](1_LLM_Eval/test_results/): JSON outputs from each prompt/model combo plus generated figures (`figs/`) and cached comparisons.
- [1_LLM_Eval/qwen2.ipynb](1_LLM_Eval/qwen2.ipynb), [1_LLM_Eval/qwen3.ipynb](1_LLM_Eval/qwen3.ipynb), [1_LLM_Eval/qwen3_(1) (1).ipynb](1_LLM_Eval/qwen3_(1)%20(1).ipynb): Development notebooks for Qwen-based experiments (prompt iterations, inference scripts, and exploratory plots).

## Progress Log
### Week 1 – Baseline With Open Instruction-Tuned Models
Focus: run Llama 3.1 (8B) and Qwen 2.5 (7B) against the labeled Reddit dataset for a like-for-like comparison against the DETECTIPV 2022 paper.
- **Binary IPV detection**: prompts and evaluation captured in [1_LLM_Eval/qwen2.ipynb](1_LLM_Eval/qwen2.ipynb) and serialized outputs inside [1_LLM_Eval/test_results/binary_*.json](1_LLM_Eval/test_results).
- **Subtype (multilabel) detection**: prompts + parsing logic iterated in [1_LLM_Eval/qwen3.ipynb](1_LLM_Eval/qwen3.ipynb) and [1_LLM_Eval/qwen3_(1) (1).ipynb](1_LLM_Eval/qwen3_(1)%20(1).ipynb), producing `multilabel_*.json` result files.
- **Evaluation parity**: [1_LLM_Eval/eval_llm_pipeline.py](1_LLM_Eval/eval_llm_pipeline.py) now matches the DETECTIPV metric stack (accuracy, precision, F1, ROC-AUC) and introduces confidence-based waterfall plots to visualize subtype predictions.
- **Sample usage doc**: [1_LLM_Eval/eval_pipeline_sample_use.ipynb](1_LLM_Eval/eval_pipeline_sample_use.ipynb) documents how to plug new JSON outputs into the shared pipeline, keeping future experiments consistent.

### Week 2 (In Progress) – Proprietary LLM Sampling & New Data
Focus: broaden coverage beyond open models and stress-test on a newer dataset.
1. **Sampling + evaluation on proprietary LLMs (e.g., GPT-5)**  
   - Reuse the same JSON schema so results drop directly into the evaluation pipeline.  
   - Capture both binary and multitype prompts to compare against Week 1 baselines.
2. **Explore prediction quality on the newer dataset**  
   - Clean/ingest the additional corpus (encrypted or kept private as needed).  
   - Run spot checks in notebooks, then formalize via the pipeline once labeling is aligned.

## Next Steps
1. **Finalize GPT-5 (or other proprietary) sampling plan**: document prompts + temperature settings alongside the existing prompt collection in [1_LLM_Eval/prompts.md](1_LLM_Eval/prompts.md).
2. **Extend the evaluation cache**: log proprietary outputs under `1_LLM_Eval/test_results/` with timestamped filenames so they can be replayed without rerunning expensive calls.
3. **Integrate the new dataset**: add a sanitized CSV/Parquet placeholder (or loader script) plus clear instructions for accessing the private source.
4. **Benchmark summary**: once Week 2 runs complete, update this README with comparative charts (binary vs multitype, open vs proprietary) to give stakeholders a quick snapshot.

Feel free to open [1_LLM_Eval/eval_pipeline_sample_use.ipynb](1_LLM_Eval/eval_pipeline_sample_use.ipynb) and rerun the cells whenever new JSON predictions are available—the figures and CSVs will regenerate automatically under [1_LLM_Eval/test_results/figs/](1_LLM_Eval/test_results/figs/).
