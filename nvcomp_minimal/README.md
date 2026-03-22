# nvcomp-minimal

Minimal Cython wrapper for nvcomp zstd decompression.

## Build Requirements

This package requires the following C/C++ libraries to be available at build time:

- **nvcomp** (>=5.0): NVIDIA compression library
- **CUDA toolkit**: For CUDA runtime headers

### Building with Pixi

The easiest way to build this package is using pixi, which will automatically provide all build dependencies:

```bash
cd nvcomp_minimal
pixi install
pixi run build
```

### Building with pip/uv in a conda/pixi environment

If you're installing this as a dependency in another project (e.g., via a local path reference in pyproject.toml), make sure you're in a conda or pixi environment that has the required build dependencies:

```bash
# Create/activate a pixi environment with build dependencies
pixi install nvcomp libnvcomp-dev

# Then install your project
uv pip install -e .
```

The setup.py will automatically detect nvcomp from:
1. `CONDA_PREFIX` environment variable (conda/pixi environments)
2. `PIXI_PROJECT_ROOT` environment variable (pixi environments)
3. `nvidia-nvcomp-cu12` PyPI wheel
4. System paths (`/usr/local/nvcomp`, `/opt/nvcomp`, etc.)
5. `NVCOMP_HOME` environment variable (manual override)

### Building without a package manager

If you have nvcomp installed in a non-standard location, set the `NVCOMP_HOME` environment variable:

```bash
export NVCOMP_HOME=/path/to/nvcomp
pip install -e .
```

## Runtime Dependencies

At runtime, the package requires:
- `cupy-cuda12x`: For GPU array support
- `numpy`: For CPU array support
- `nvcomp` shared library (libnvcomp.so.5): Must be available in LD_LIBRARY_PATH or system library paths
