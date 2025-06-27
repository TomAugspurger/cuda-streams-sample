import pylibcudf as plc
import pyarrow as pa
import pyarrow.parquet as pq
import numba.cuda
import rmm.pylibrmm.stream
import nvtx


@numba.cuda.jit
def my_kernel(a, out) -> None:
    for i in range(len(a)):
        out[0] += a[i]


def main():
    N_STREAMS = 4
    N_ROWS = 1000

    # https://github.com/rapidsai/cudf/issues/17082
    # AFAICT, the pylibcudf.interop APIs don't support streams currently.

    # However, `pylibcudf.io.parquet.read_parquet` *does* support streams.
    # https://docs.rapids.ai/api/cudf/nightly/pylibcudf/api_docs/io/parquet/
    # So we can perhaps use that?

    table = pa.Table.from_pydict({"a": list(range(N_ROWS))})
    pq.write_table(table, "test.parquet")

    opts = plc.io.parquet.ParquetReaderOptions.builder(
        plc.io.types.SourceInfo(["test.parquet"])
    ).build()

    streams = [numba.cuda.stream() for _ in range(N_STREAMS)]

    # warmup
    with nvtx.annotate("warmup"):
        plc.io.parquet.read_parquet(opts)
        my_kernel[256, 1](
            numba.cuda.device_array(1, dtype="int64"),
            numba.cuda.device_array(1, dtype="int64"),
        )

    with nvtx.annotate("benchmark"):
        for stream in streams:
            # Does read_parquet block? Kinda seems like it...
            df = plc.io.parquet.read_parquet(
                opts, stream=rmm.pylibrmm.stream.Stream(stream)
            )
            out = numba.cuda.device_array(1, dtype="int64", stream=stream)  # type: ignore[arg-type]
            # Column.data() has a type of '|u1' so we're operating on the raw bytes of the array
            # This is clearly wrong (the sum is incorrect), but good enough for our testing.
            my_kernel[256, 1, stream](df.columns[0].data(), out)  # type: ignore[call-overload]

        # and synchronize
        for stream in streams:
            stream.synchronize()

    print("done", out.copy_to_host()[0])


if __name__ == "__main__":
    main()
