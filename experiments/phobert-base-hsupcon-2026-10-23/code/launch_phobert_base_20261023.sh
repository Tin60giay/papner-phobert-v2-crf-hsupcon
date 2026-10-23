#!/usr/bin/env bash
set -euo pipefail

NOTEBOOK=/workspace/baselineHierarchicalSupCon_PhoBERTbase_Vast_RTX3090.ipynb
RUN_DIR=/workspace/phobert_base_runner_20261023
TRAIN_LOG=/workspace/phobert_base_hsupcon_20261023/outputs/phobert_base_crf_hierarchical_supcon_20261023/train.log
SESSION=phobert_base_20261023

test -f "$NOTEBOOK"
command -v tmux >/dev/null
if [ -f /venv/main/bin/activate ]; then
    source /venv/main/bin/activate
fi
command -v jupyter >/dev/null
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session already exists: $SESSION; refusing to start a duplicate."
    exit 0
fi
mkdir -p "$RUN_DIR" "$(dirname "$TRAIN_LOG")"
touch "$TRAIN_LOG" "$RUN_DIR/runner.log"
cat > "$RUN_DIR/execute.sh" <<'EXECUTE'
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
EXECUTE

tmux new-session -d -s "$SESSION" -n train "bash '$RUN_DIR/execute.sh'"
tmux new-window -t "$SESSION" -n progress "tail -n 25 -F '$TRAIN_LOG' '$RUN_DIR/runner.log'"
echo "Started in tmux: $SESSION"
echo "Progress: tmux attach -t $SESSION"
