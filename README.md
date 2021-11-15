Pruning elements in SIMD vectors

This crate is a port of Daniel Lemire's C library [simdprune](https://github.com/lemire/simdprune/).

The mask "marks" values in the input for deletion. So if
the mask is odd, then the first value is marked for deletion.
This function produces a new vector that start with all
values that have not been deleted.

Passing a mask of 0 would simply copy the provided vector.

Note that this is the opposite of the mask behavior of AVX512 VCOMPRESS/VPCOMRESS instructions.

# Examples

See [`prune_epi32`].

# Features

All features below are enabled by default.

* **std** - Enables the standard library. Disabling this enables the `no_std` crate attribute.
* **large_tables** - Enables functions like [`prune_epi8`] which require large tables (>1MB).
Disabling this may speed up compilation.
