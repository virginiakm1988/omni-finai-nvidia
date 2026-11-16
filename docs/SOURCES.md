# Sources and licensing

- [Omni-FinAI 70B model](https://huggingface.co/ichanchiu/Llama-3.1-Omni-FinAI-70B)
- [Project paper / SSRN abstract 5004298](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5004298)
- [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel)
- [Official AutoModel SFT/PEFT guide](https://docs.nvidia.com/nemo/automodel/v0.7/recipes-e2e-examples/sft-peft)
- [Small teaching base: Qwen2.5 0.5B Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

## Version pins checked for this implementation

- NeMo Fabric: Python distributions `nemo-fabric==0.6.1`, `nemo-fabric-adapters-common==0.6.1`. Adapter contract cross-checked against public commit `910218f0891749c9364e814596e452f7258e23bb` and tested against the installed release.
- NeMo Relay: `nemo-relay==0.11.1`; actual scoped tool/LLM instrumentation, local ATOF sink.
- NeMo Data Designer: `data-designer==0.10.0`; structured LLM column, seeded dataframe, explicit run configuration.
- NeMo Helix: source commit `6386f9f0a84e8c4b71c4852fec9d7fae1aaf6b11`, package `packages/nhx_evals_sdk`. No full service platform is required for captured-trial scoring.
- NeMo AutoModel: `nemo-automodel==0.7.0`; recipe/API compared against official public source `c57935fc64faea913ce9f5e38ea7e57e6416cc93` and the v0.7 documentation.
- Omni-FinAI 70B revision: `8f9e1ccba75181c7ca95df77e8313a76bfdd4a50`.
- Small Qwen2.5 0.5B Instruct revision: `7ae557604adf67be50417f59c2c2f167def9a775`.

Only the SDKs applicable to this repository are installed. Hosted NIM model IDs are recorded with each run; their server implementation is provider-managed and cannot be frozen by a client package lock. These tutorials are independent companion implementations. NVIDIA SDKs and related public code retain their own Apache-2.0 notices. Llama-derived checkpoints retain their model licenses and use policies. No external weights or paper text are redistributed here.

## Method map

The original profile selects the pinned 70B Omni-FinAI checkpoint. The small profile selects a different, explicitly labelled Qwen2.5 0.5B Instruct base for training mechanics. The authored financial reports use disjoint company IDs across training, validation, and test sets. AutoModel trains LoRA on training rows, validates separately, exports HF-compatible adapter weights, verifies nonzero LoRA-B updates, reloads them with PEFT, and scores the identical fixed test set.

This companion does not reproduce the paper training corpus, large-scale optimization, or benchmark results. It uses small synthetic financial reports with checkable arithmetic. The small profile is not Omni-FinAI. An optional NIM baseline is also labelled as a different model. A short LoRA run may reduce accuracy; the recorded result is retained.

The overview diagram is newly authored in NVIDIA-inspired green, black, and neutral colors; it is not an official NVIDIA architecture figure or endorsement.

## Checkpoint prompting audit

The pinned Omni-FinAI model card describes a finance-pretrained foundation for subsequent task fine-tuning, derived from Llama-3.1-70B-Instruct. Its shipped tokenizer includes the Llama chat template, and its generation configuration stops on token IDs128001,128008,128009. The standalone HF demo defaults to that shipped chat template and decodes only newly generated tokens. An explicit `--prompt-format plain` baseline uses the exact context-space-question-space format used by the AutoModel ColumnMapped training recipe; before/after adapter evaluation always uses that matching plain format. Plain-prompt baseline scores are not presented as the checkpoint-chat interface. No examples are modified to match expected answers.
