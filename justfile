capture-nvcomp:
    nsys profile -o nvcomp-simple --cuda-memory-usage=true --force-overwrite=true --trace=cuda,nvtx --python-sampling=true uv run nvcomp-simple.py
