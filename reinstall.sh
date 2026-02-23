#!/bin/bash
set -e

rm dist/*.whl || true
rm hopper/dist/*.whl || true

export FLASH_ATTENTION_DISABLE_SM80="TRUE"
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
python setup.py bdist_wheel
pip install --no-deps --force-reinstall dist/*.whl

export MAX_JOBS=10
export TORCH_CUDA_ARCH_LIST="8.0;9.0;9.0a"

python setup.py bdist_wheel
pip install --no-deps --force-reinstall dist/*.whl
