#!/usr/bin/env bash
if [ -f /venv/main/bin/activate ]; then
    source /venv/main/bin/activate
fi
export HF_ENDPOINT=https://huggingface.co
export HF_HOME=/workspace/.hf_home
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=30
export PYTHONUNBUFFERED=1
cd /workspace
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
    --output phobert_base_finished_20261023 \
    --output-dir /workspace/phobert_base_runner_20261023 \
    /workspace/baselineHierarchicalSupCon_PhoBERTbase_Vast_RTX3090.ipynb \
    > /workspace/phobert_base_runner_20261023/runner.log 2>&1
result=$?
printf '%s\n' "$result" > /workspace/phobert_base_runner_20261023/exit_code.txt
echo "Notebook exited with code $result. See runner.log."
exec bash
