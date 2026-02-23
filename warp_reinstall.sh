set -xe

time ./reinstall.sh > my_install.log 2>&1
grep -i "error" my_install.log
tail -n 10 my_install.log

cd tests
# python -m pytest -q -s test_flash_attn.py::test_flash_attn_splitkv > test_flash_attn_splitkv.log
python -m pytest -q -s test_flash_attn.py > test_flash_attn_all.log
# python -m pytest -q -s test_flash_attn.py::test_flash_attn_qkvpacked > test_flash_attn_qkvpacked.log

export FLASH_ATTENTION_DISABLE_SM80="TRUE"
export FLASH_ATTENTION_DISABLE_FP8="TRUE"
cd ../hopper
python -m pytest -q -s test_flash_attn.py::test_flash_attn_output > test_all_output3
python -m pytest -q -s test_flash_attn.py::test_flash_attn_varlen_output > test_all_varlen3
