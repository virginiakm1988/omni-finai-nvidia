# Omni-FinAI with NVIDIA NeMo AutoModel：快速入門

原始模型路線實際使用 Omni-FinAI 70B。小模型路線使用不同的 Qwen2.5 0.5B Instruct，專門教學 LoRA 訓練流程，不能當成 Omni-FinAI 的實測。訓練前後使用相同測試集，保存與重新載入 adapter，並照實呈現數值是否改善。

## 操作

1. 建立這個 repo 專用的 Python 3.12 環境，避免不同專案的 `project_demo` 套件互相覆蓋。
2. `python -m pip install -e .`
3. `python -m project_demo prepare --output-dir runs/preparation`
4. `python -m project_demo demo --mode fixture --output-dir runs/fixture`
5. 開啟 [互動 Colab](https://colab.research.google.com/github/virginiakm1988/omni-finai-nvidia/blob/v0.1.0/notebooks/01_interactive_demo.ipynb)，先看範例，再依需求切換真實執行。

`fixture` 是教學範例，不是實際模型輸出。`live` 失敗時會保存錯誤，不會自動換成範例成功結果。每次執行都有獨立輸出目錄、設定與時間紀錄。API key 請使用環境變數或 Colab Secrets；不要寫進程式或公開 repo。

GPU 訓練請使用 Taipei-1 Slurm。先跑 1 GPU 的小模型，再以獨立 4 GPU 作業驗證 70B；不得把小模型結果標示為原模型。

完整步驟、驗收、限制與來源請見 [英文 README](README.md) 與 [Playbook](docs/PLAYBOOK.md)。

## 實際執行結果

原始 Omni-FinAI 70B 已在 Taipei-1 的 4 張 H100 完成 12 次 AutoModel LoRA 更新、adapter 儲存／重載，以及同一組 12 筆測試的前後比較。嚴格資訊抽取與數值問答皆維持 0/4；摘要數值涵蓋率由 0/4 變成 4/4，但輸出仍有重複文字，不能視為整體摘要品質提升。不同基底的 Qwen 0.5B 教學也已完成真實訓練，兩者結果分開保存。請參閱 [原模型實測](evidence/automodel-original/result.json) 與 [失敗紀錄](evidence/failed-attempts/README.md)。
