#!/usr/bin/env bash
# 远端：安装 SheetSage2 转谱环境（独立 venv，与 YuE2 的 torch 2.10 隔离）
set -x

cd /content/YuE || exit 1

uv venv --python 3.11 --seed .venv-sheetsage2
.venv-sheetsage2/bin/python -m pip install -q huggingface-hub==0.36.0
.venv-sheetsage2/bin/huggingface-cli download m-a-p/SheetSage2 --local-dir models/SheetSage2
.venv-sheetsage2/bin/python -m pip install -q torch==2.8.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu126
.venv-sheetsage2/bin/python -m pip install -q -r models/SheetSage2/requirements.txt

echo SHEETSAGE_SETUP_DONE
