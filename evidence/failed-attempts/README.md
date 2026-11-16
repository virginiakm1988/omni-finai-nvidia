# Preserved failed attempts

The small teaching profile first encountered a gated model access denial, then an SDK import incompatibility. Both original error records are retained. The final small profile is explicitly a different, ungated Qwen base.

Original70B job14138 completed its HF demo but failed AutoModel distributed startup because PyTorch2.15 removed an internal FSDP interface. Its separate failure artifacts remain in the parent evidence directory. A fresh PyTorch2.10 environment resolved that issue.

Job14144 then reached FSDP training but failed with CUDA out-of-memory: the parent baseline evaluation process retained its model allocation. Its workers were explicitly cancelled after the failure, so the original run JSON remains `running` as written before process termination; it is not a successful or still-active run. Scheduler records show CANCELLED, elapsed00:08:41 (batch signal15). The unedited sanitized record and actual failure log are preserved here.

The implementation now performs before/after inference in isolated subprocesses, releasing all worker model allocations before training. Separate job14147 completed with scheduler exit0 in00:08:38. The completed original70B result is `../automodel-original/result.json`; it does not overwrite any earlier attempt.
