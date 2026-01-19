time ./reinstall.sh > my_install.log 2>&1
grep -i "error" my_install.log
tail -n 10 my_install.log