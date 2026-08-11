#!/bin/bash
set -e

# I don't have any SM8x-based GPU hardware, so I haven't developed support for it yet.
export FLASH_ATTENTION_DISABLE_SM80="TRUE"
# Compiling it takes too much time; it might work, but I haven't tested it myself.
export FLASH_ATTENTION_DISABLE_FP8="TRUE"

# export FLASH_ATTENTION_DISABLE_SOFTCAP="TRUE"
# export FLASH_ATTENTION_DISABLE_LOCAL="TRUE"
# export FLASH_ATTENTION_DISABLE_BACKWARD="TRUE"
# export FLASH_ATTENTION_DISABLE_APPENDKV="TRUE"

export FLASH_ATTENTION_FORCE_BUILD="TRUE"

# export FLASH_ATTENTION_DISABLE_HDIM64="TRUE"
# export FLASH_ATTENTION_DISABLE_HDIM96="TRUE"
# export FLASH_ATTENTION_DISABLE_HDIM128="TRUE"
# export FLASH_ATTENTION_DISABLE_HDIM192="TRUE"

export MAX_JOBS=32

cd hopper
# rm -rf __pycache__/ flash_attn_3.egg-info/ build/ dist/
python setup.py bdist_wheel
pip install --no-deps --force-reinstall dist/*.whl
