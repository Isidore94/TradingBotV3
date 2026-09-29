#!/usr/bin/env bash
# Runs ON the GPU host (piped over ssh by run_ai_jobs.ps1): make sure the desk's
# Ollama is answering on 127.0.0.1:11434, starting it in tmux if it is not.
set -u
probe() { curl -fsS -m 3 http://127.0.0.1:11434/api/version >/dev/null 2>&1; }
if probe; then echo "ollama: already up"; exit 0; fi
tmux kill-session -t ollama 2>/dev/null
tmux new -d -s ollama "OLLAMA_MODELS=/home/aaron/models/ollama OLLAMA_HOST=127.0.0.1:11434 OLLAMA_CONTEXT_LENGTH=65536 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_KEEP_ALIVE=30m LD_LIBRARY_PATH=/usr/lib/wsl/lib ollama serve >> \$HOME/ollama-tradingbot.log 2>&1"
for _ in $(seq 1 30); do
  if probe; then echo "ollama: started"; exit 0; fi
  sleep 1
done
echo "ollama: FAILED to answer within 30s"
exit 1
