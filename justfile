capture target:
    @echo 'Capturing {{target}}'
    nsys profile -o profile-{{without_extension(target)}} --cuda-memory-usage=true --force-overwrite=true --trace=cuda,nvtx --python-sampling=true uv run {{target}}

pylibcudf:
    just capture pylibcudf-simple.py

numba:
    just capture numba-simple.py

nvcomp-simple:
    just capture nvcomp-simple.py

nvcomp-compute:
    just capture nvcomp-compute.py

nvcomp-multi-device:
    just capture nvcomp-multi-device.py

# Run multi-device nvcomp example without profiling
run-multi-device:
    uv run nvcomp-multi-device.py

zarr-shards:
    just capture zarr_shards.py
