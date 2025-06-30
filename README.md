# cuda-streams-sample

This is a scratch workspace exploring CUDA streams

## numba-simple

See [numba-simple.py](./numba-simple.py). This workload

1. Allocates a *pinned* host buffer (per stream)
2. Schedules a host to device copy (per stream)
3. Launches a compute kernel (per stream)

Things look good!

![](https://github.com/user-attachments/assets/92c9fbcf-5975-457d-af5d-1064439bdb0a)


In particular, note that

- We pipeline host-to-device memory copies and compute (vertical overlap of the teal and blue bars)
- We have good utilization of the memory system (no gaps in the teal)
- We have good utilization of the compute system (no gaps in the blue)


## pylibcudf-sample

pylibcudf is still working to provide an API that expose streams
(https://github.com/rapidsai/cudf/issues/18962).
[pylibcudf-simple.py](./pylibcudf-simple.py) is a rough attempt. Things look
less good here, but this could easily be user-error. This example

1. Launches a `plc.io.parquet.read_parquet` (one per stream).
2. Launches a `numba.cuda` kernel on the data (one per stream)

See https://github.com/TomAugspurger/cuda-streams-sample/issues/3 for an analysis.