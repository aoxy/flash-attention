python setup.py bdist_wheel

cd hopper
python setup.py bdist_wheel

cd ../csrc/fused_dense_lib
python setup.py bdist_wheel

cd ../..
cp dist/*.whl ./whls/
cp hopper/dist/*.whl ./whls/
cp csrc/fused_dense_lib/dist/*.whl ./whls/

pip install ./whls/*.whl
