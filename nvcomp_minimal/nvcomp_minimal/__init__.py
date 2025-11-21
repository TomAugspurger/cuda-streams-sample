"""Minimal nvcomp wrapper for zstd decompression."""

from .zstd import ZstdCodec, DecompressedArray

__all__ = ["ZstdCodec", "DecompressedArray"]
