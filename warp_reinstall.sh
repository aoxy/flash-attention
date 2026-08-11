time ./reinstall.sh > my_install.log 2>&1
grep -i "error" my_install.log
tail -n 10 my_install.log

cd hopper
python -m pytest -q -s test_flash_attn.py::test_flash_attn_output > test_all_output3
python -m pytest -q -s test_flash_attn.py::test_flash_attn_varlen_output > test_all_varlen3