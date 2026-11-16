# Independent code review

Reviewer: separate implementation agent (`build_reasoning_legal_repos`); not the author of this repository. Review date: 2026-11-16.

## Round 1 — correctness and research methods

Scope: source, CLI, configurations, dataset provenance, scoring, NVIDIA SDK boundaries, model/data split handling, notebooks and documentation. This is code review; GPU training is not validated merely by reviewing its recipe.

### OF-1 [P1] AutoModel training and HF evaluation prompts differed

The pinned dataset concatenates context+space+question+space; inference added Context:/Question:/Answer: prefixes and newlines. Align exact serialization and add a parity test before interpreting adapter effects.

Status at first review: changes requested; owner notified.

### OF-2 [P1] evaluate could run a model and alter evidence mode

Require --results, preserve its original mode/config/model/source digest, and reject failed source runs. Never execute inference when a rescore input is absent.

Status at first review: changes requested; owner notified.

### OF-3 [P2] Tutorials needed actual financial input/result selection

Add task/report selection and a saved-result selector. Separate authored examples, NIM baseline, small model and original Omni-FinAI results visibly.

Status at first review: changes requested; owner notified.

## Strengths verified in source

Original 70B and small teaching model are explicitly distinct. Public model revisions and licenses are identified. Authored company IDs are disjoint across splits. The trainer checks nonzero LoRA-B tensors and reloads the exported adapter; these checks must pass on real hardware before training is called validated.

## Round 2 — readability, reproducibility and regression verification

OF-1 is resolved: inference uses exactly `context + " " + question + " "`, matching the pinned AutoModel 0.7 non-chat ColumnMapped dataset; a regression test covers this boundary. OF-2 is resolved: evaluation requires saved successful evidence and preserves source mode/configuration/hash without generating outputs. OF-3 is resolved: notebooks expose task/model selection and a saved-result path.

The current small teaching profile is explicitly **Qwen2.5 0.5B Instruct**, separate from original Omni-FinAI 70B. The earlier gated Llama failure is retained rather than relabeled as a success. The original model card confirms its Llama 3.1 Community License and the linked Omni-FinAI paper. Company identifiers remain disjoint across train/validation/test, and numeric-summary coverage is not presented as overall summary quality. The AutoModel trainer requires nonzero adapter updates, reload and a fixed test comparison before declaring execution successful.

Independently reran the regression suite after fixes: **12 tests passed** using `python -m unittest discover -s tests -v`. Reviewed the formatted implementation, notebook mode controls, sources, licenses and failure paths.

**Review disposition:** reported code, scoring and teaching defects are resolved. Notebook bootstrap/publication and real GPU/model executions remain separately reported in the repository validation evidence; this review does not turn an unexecuted recipe or a model download into a successful inference/training result.

## Verified training-environment update

Reviewed the explicitly named Qwen2.5 0.5B Instruct teaching profile separately from the NIM generator and original Omni-FinAI model. The prior gated-model failure is retained. Independently fetched the pinned [official AutoModel source](https://github.com/NVIDIA-NeMo/Automodel/blob/c57935fc64faea913ce9f5e38ea7e57e6416cc93/nemo_automodel/components/_peft/lora_kernel.py): `lora_db_kernel` uses autotune key `["M", "K"]`, and has no `S` argument. The supplied patch performs that single compatibility backport, checks version 0.7.0, keeps a backup and reports hashes. Whole upstream file SHA-256 verified: `ed3833f157cc11328c165a362b2ab02e3ae8b48f522e88b93826a059765e49c8`. The resulting training environment is accurately described as a patched 0.7 release, not an untouched wheel. GPU output remains subject to the separately recorded adapter/export/reload checks.

## Focused final review — checkpoint chat formatting

The standalone HF demo now defaults to the selected checkpoint's shipped chat template with an assistant-generation prefix. It refuses this mode when the tokenizer has no template. The template is rendered once and tokenized with `add_special_tokens=False`, preventing an extra BOS token. Decoding slices off all input tokens, and generation keeps the checkpoint's EOS configuration. The original model revision remains pinned.

The AutoModel before/after path intentionally retains the exact plain ColumnMapped training prompt and the same held-out inputs; its numerical results are not compared as if they used the new standalone chat format. CLI records explicitly distinguish `checkpoint-chat` and `plain`. The prior three-question, all-zero plain-prompt baseline is retained under its original evidence path and labelled accurately in the README.

Independently reran all **13 tests**, including the new no-duplicate-BOS regression and existing plain-prompt parity test: all passed. Source review found no unresolved defect in this delta. The separate bounded checkpoint-chat GPU run has now completed; its actual outputs were independently checked as described below.

### Actual checkpoint-chat run and inference-process isolation

Independently checked [the separate checkpoint-chat evidence](evidence/original-70b-chat-inference.json) against the raw result from GPU job 14146. It records live execution, successful completion and `checkpoint-chat` in both configuration and result. The original 70B model revision, seed, generation limit, three question IDs and expected checks match the retained plain baseline. Recomputing the existing scoring returns **0/3 passing checks**. The generated text includes repetition or irrelevant text; these failures remain visible. Decoding excludes input tokens, so such repetition comes from model generation rather than counting the prompt as an answer.

Reviewed `hf_worker.py` and the training launcher update: baseline and adapter inference now run in separate, bounded child processes with checked exit codes, followed by process exit before the next stage. This prevents the 70B baseline's CUDA model allocation from remaining in the training process. Both child evaluations explicitly use the unchanged plain training prompt and the same test file; adapter inference additionally receives the exported adapter path. The parent still checks split separation and adapter updates. All **13 tests passed again**, and source compilation passed. The original 70B AutoModel retry is separately recorded as running; this review does not mark that training successful before its gradient, export, reload and fixed-test evidence exists.

## Final original-70B execution evidence review

The previously pending original-70B run is now verified. A separate read-only scheduler query confirmed job **14147 COMPLETED, exit 0:0**, elapsed 8m38s; the recorded pipeline duration is 515.599s. The saved training log contains **12 optimizer updates (0–11), all with nonzero gradient norms**, validation and checkpoint export. An independent CPU read of the actual exported adapter confirmed **1,120 tensors, including 560/560 nonzero LoRA-B tensors**, size 207,244,360 bytes and SHA-256 `753a81db0ccf18c7db604ac660715029c56b4a8c5cc73d8af479c8639c53894d`. The after-inference worker reloads that exported adapter and has completed successfully.

Regenerated the authored test examples from the recorded seed and verified their hash equals the before, after and final result hashes. Both records contain the same 12 IDs and checks. Independently recomputed the results: extraction **0/4 → 0/4**, numerical QA **0/4 → 0/4**, summary numeric coverage **0/4 → 4/4**. Repetitive and nonconforming answers remain visible; numeric coverage is explicitly not a broad summary-quality or financial-capability claim. The prior failed attempts remain separate. See [independent verification](evidence/automodel-original/independent-verification.json).

Reviewed the final notebook/documentation changes: source checkout is pinned to `v0.1.0`; training bootstrap selects a separate pinned small or distributed environment before SDK imports; original70B rejects the wrong environment selection. Optional Colab secret handling catches only the small import/read block and does not swallow model failures. Bounded one- and four-GPU Slurm scripts require an explicitly prepared `AUTOMODEL_ENV`. The three notebooks retain **17 executed local tutorial cells and no error outputs**; public clone/install remains a separate publication check. Application/test hashes are unchanged from the independently passing 13-test review. **Final disposition: approved for the documented scope; no unresolved reported finding.**
