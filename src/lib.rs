//! Pruning elements in SIMD vectors
//!
//! This crate is a port of Daniel Lemire's C library [simdprune](https://github.com/lemire/simdprune/).
//!
//! The mask "marks" values in the input for deletion. So if
//! the mask is odd, then the first value is marked for deletion.
//! This function produces a new vector that start with all
//! values that have not been deleted.
//!
//! Passing a mask of 0 would simply copy the provided vector.
//!
//! Note that this is the opposite of the mask behavior of AVX512 VCOMPRESS/VPCOMRESS instructions.
//!
//! # Examples
//!
//! See [`prune_epi32`].
//!
//! # Features
//!
//! All features below are enabled by default.
//!
//! * **std** - Enables the standard library. Disabling this enables the `no_std` crate attribute.
//! * **large_tables** - Enables functions like [`prune_epi8`] which require large tables (>1MB).
//! Disabling this may speed up compilation.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "large_tables")]
mod large_tables;
mod tables;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(feature = "large_tables")]
use large_tables::mask128_epi8;
use tables::*;

/// Prune 8-bit values.
///
/// Values corresponding to a 1-bit in the mask are removed from output
///
/// The table used for this operation occupies 1 MB.
///
/// The last value not deleted is used to pad the result.
///
/// Requires the `large_tables` feature (enabled by default).
///
/// Trick: by leaving the highest bit (`1 << 15`) to zero whether
/// you want to delete the last value or not, you can end up using
/// only the first half of the table (which limits cache usage).
///
/// # Panics
/// Panics if `mask` is not in `[0, 1 << 16)`.
/// # Examples
/// See [`prune_epi32`].
#[target_feature(enable = "ssse3")]
#[cfg(feature = "large_tables")]
#[cfg_attr(docsrs, doc(cfg(feature = "large_tables")))]
#[inline]
pub unsafe fn prune_epi8(x: __m128i, mask: i32) -> __m128i {
    let ptr = mask128_epi8[16 * mask as usize..].as_ptr().cast();
    _mm_shuffle_epi8(x, _mm_loadu_si128(ptr))
}

#[inline]
unsafe fn left_shift_bytes(x: __m128i, count: i32) -> __m128i {
    // we'd like to shift by count bytes, but it can't be done directly without immediates
    let p1 = _mm_sll_epi64(x, _mm_cvtsi64_si128(count as i64 * 8));
    let p2 = _mm_srl_epi64(
        _mm_unpacklo_epi64(_mm_setzero_si128(), x),
        _mm_cvtsi64_si128(64 - count as i64 * 8),
    );
    _mm_or_si128(p1, p2)
}

/// Prune 8-bit values. Like [`prune_epi8`] but uses a 2kB table.
///
/// Values corresponding to a 1-bit in the mask are removed from output
///
/// Note that this will be faster if you enable the `popcnt` instruction set feature flag,
/// available on SSE4.2 and later.
///
/// # Panics
/// Panics if `mask` is not in `[0, 1 << 16)`.
/// # Examples
/// See [`prune_epi32`].
#[target_feature(enable = "ssse3")]
#[inline]
pub unsafe fn thinprune_epi8(x: __m128i, mask: i32) -> __m128i {
    let mask1 = mask & 0xFF;
    let pop = 8 - mask1.count_ones();
    let mask2 = mask as u32 >> 8; // we want a logical shift here
    let m1 = _mm_loadl_epi64(thintable_epi8[mask1 as usize..].as_ptr().cast());
    let m2 = _mm_loadl_epi64(thintable_epi8[mask2 as usize..].as_ptr().cast());
    let m2add = _mm_add_epi8(m2, _mm_set1_epi8(8));
    let m2shifted = left_shift_bytes(m2add, pop as i32);
    let shufmask = _mm_or_si128(m2shifted, m1);
    _mm_shuffle_epi8(x, shufmask)
}

/// Prune 8-bit values. Like [`prune_epi8`] but uses a <1kB table.
///
/// Values corresponding to a 1-bit in the mask are removed from output
///
/// credit: @animetosho
///
/// # Panics
/// Panics if `mask` is not in `[0, 1 << 16)`.
/// # Examples
/// See [`prune_epi32`].
#[target_feature(enable = "ssse3")]
#[inline]
pub unsafe fn skinnyprune_epi8(x: __m128i, mask: i32) -> __m128i {
    let mask1 = mask & 0xFF;
    // we want a logical shift here
    let mask2 = mask as u32 >> 8;
    // reference impl uses _mm_loadh_pi but since Rust removed __m64 support,
    // we use _mm_loadh_pd here.
    let ptr1 = thintable_epi8[mask1 as usize..].as_ptr().cast();
    let ptr2 = thintable_epi8[mask2 as usize..].as_ptr().cast();
    let mut shufmask =
        _mm_castpd_si128(_mm_loadh_pd(_mm_castsi128_pd(_mm_loadl_epi64(ptr1)), ptr2));
    shufmask = _mm_add_epi8(shufmask, _mm_set_epi32(0x0808_0808, 0x0808_0808, 0, 0));
    let pruned = _mm_shuffle_epi8(x, shufmask);
    let popx2 = BitsSetTable256mul2[mask1 as usize];
    let compactmask = _mm_loadu_si128(pshufb_combine_table[popx2 as usize * 8..].as_ptr().cast());
    _mm_shuffle_epi8(pruned, compactmask)
}

/// Prune 8-bit values.
///
/// Values corresponding to a 1-bit in the mask are removed from output
///
/// The table used for this operation occupies 4 kB.
///
/// The last value not deleted is used to pad the result.
///
/// Trick: by leaving the highest bit (`1 << 7`) to zero whether
/// you want to delete the last value or not, you can end up using
/// only the first half of the table (which limits cache usage).
///
/// # Panics
/// Panics if `mask` is not in `[0, 1 << 8)`.
/// # Examples
/// See [`prune_epi32`].
#[target_feature(enable = "ssse3")]
#[inline]
pub unsafe fn prune_epi16(x: __m128i, mask: i32) -> __m128i {
    let ptr = mask128_epi16[16 * mask as usize..].as_ptr().cast();
    _mm_shuffle_epi8(x, _mm_loadu_si128(ptr))
}

/// Prune 32-bit integer values.
///
/// Values corresponding to a 1-bit in the mask are removed from output
///
/// # Panics
/// Panics if `mask` is not in `[0, 1 << 4)`.
///
/// # Examples
///
/// ```
/// # #[cfg(target_arch = "x86")] use core::arch::x86::*;
/// # #[cfg(target_arch = "x86_64")] use core::arch::x86_64::*;
/// use simdprune::prune_epi32;
///
/// unsafe {
///     let input = _mm_set_epi32(3, 2, 1, 0);
///     let mask = 0b1010;
///     let pruned = prune_epi32(input, mask);
///     let mut buf = [0_u32; 4];
///     _mm_storeu_si128(buf.as_mut_ptr().cast(), pruned);
///     assert_eq!(&buf[..4 - mask.count_ones() as usize], [0, 2]);
/// }
#[target_feature(enable = "ssse3")]
#[inline]
pub unsafe fn prune_epi32(x: __m128i, mask: i32) -> __m128i {
    let ptr = mask128_epi32[16 * mask as usize..].as_ptr().cast();
    _mm_shuffle_epi8(x, _mm_loadu_si128(ptr))
}

/// Prune 32-bit floating-point values.
///
/// Values corresponding to a 1-bit in the mask are removed from output
///
/// # Panics
/// Panics if `mask` is not in `[0, 1 << 4)`.
/// # Examples
/// See [`prune_epi32`].
#[inline]
#[target_feature(enable = "ssse3")]
pub unsafe fn prune_ps(x: __m128, mask: i32) -> __m128 {
    _mm_castsi128_ps(prune_epi32(_mm_castps_si128(x), mask))
}

/// Prune 32-bit integer values.
///
/// Values corresponding to a 1-bit in the mask are removed from output
///
/// # Panics
/// Panics if `mask` is not in `[0, 1 << 8)`.
/// # Examples
/// See [`prune_epi32`].
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn prune256_epi32(x: __m256i, mask: i32) -> __m256i {
    let ptr = mask256_epi32[8 * mask as usize..].as_ptr().cast();
    _mm256_permutevar8x32_epi32(x, _mm256_loadu_si256(ptr))
}

/// Prune 32-bit floating-point values.
///
/// Values corresponding to a 1-bit in the mask are removed from output
///
/// # Panics
/// Panics if `mask` is not in `[0, 1 << 8)`.
/// # Examples
/// See [`prune_epi32`].
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn prune256_ps(x: __m256, mask: i32) -> __m256 {
    let ptr = mask256_epi32[8 * mask as usize..].as_ptr().cast();
    _mm256_permutevar8x32_ps(x, _mm256_loadu_si256(ptr))
}

/// Prune 32-bit floating-point values. Uses 64bit `pdep/pext` to save a step in unpacking.
///
/// source:
/// <http://stackoverflow.com/questions/36932240/avx2-what-is-the-most-efficient-way-to-pack-left-based-on-a-mask>
///
/// ***Note that `_pdep_u64` is very slow on AMD Ryzen.***
///
/// # Panics
/// Panics if `mask` is not in `[0, 1 << 8)`.
/// # Examples
/// See [`prune_epi32`].
#[target_feature(enable = "avx2,bmi2")]
#[inline]
pub unsafe fn pext_prune256_epi32(src: __m256i, mask: u64) -> __m256i {
    assert!(mask < 1 << 8);
    let mut expanded_mask = _pdep_u64(mask, 0x0101_0101_0101_0101); // unpack each bit to a byte
    expanded_mask *= 0xFF;
    let identity_indices = 0x0706_0504_0302_0100;
    let wanted_indices = _pext_u64(identity_indices, expanded_mask);
    let bytevec = _mm_cvtsi64_si128(wanted_indices as i64);
    let shufmask = _mm256_cvtepu8_epi32(bytevec);
    _mm256_permutevar8x32_epi32(src, shufmask)
}
