#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use std::convert::TryFrom;

use simdprune::*;

fn slow_prune<T: Default + Copy>(input: &[T], mask: i32) -> Vec<T> {
    let mut idx = 0;
    let mut expected = input.to_vec();
    expected.retain(|_| {
        let flag = (mask as u32 >> idx) & 1 == 0;
        idx += 1;
        flag
    });
    expected
}

fn test_128<T>(func: unsafe fn(__m128i, i32) -> __m128i, length: usize)
where
    T: Default + Copy + std::cmp::PartialEq + std::fmt::Debug + TryFrom<usize>,
    <T as TryFrom<usize>>::Error: std::fmt::Debug,
{
    let mut buf = vec![T::default(); length];
    let input: Vec<_> = (0..length).map(|x| T::try_from(x).unwrap()).collect();

    for mask in 0..1 << length {
        unsafe {
            let input_vec = _mm_loadu_si128(input.as_ptr().cast());
            _mm_storeu_si128(buf.as_mut_ptr().cast(), func(input_vec, mask));

            let expected = slow_prune(&input, mask);
            let result = &buf[..mask.count_zeros() as usize - (32 - length)];
            assert_eq!(expected, result, "\n mask: {:#0w$b}", mask, w = length + 2);
        }
    }
}

fn test_256<T>(func: unsafe fn(__m256i, i32) -> __m256i, length: usize)
where
    T: Default + Copy + std::cmp::PartialEq + std::fmt::Debug + TryFrom<usize>,
    <T as TryFrom<usize>>::Error: std::fmt::Debug,
{
    let mut buf = vec![T::default(); length];
    let input: Vec<_> = (0..length).map(|x| T::try_from(x).unwrap()).collect();

    for mask in 0..1 << length {
        unsafe {
            let input_vec = _mm256_loadu_si256(input.as_ptr().cast());
            _mm256_storeu_si256(buf.as_mut_ptr().cast(), func(input_vec, mask));

            let expected = slow_prune(&input, mask);
            let result = &buf[..mask.count_zeros() as usize - (32 - length)];
            assert_eq!(expected, result, "\n mask: {:#0w$b}", mask, w = length + 2);
        }
    }
}

#[test]
#[cfg(feature = "large_tables")]
#[ignore] // expensive
fn test_8() {
    test_128::<u8>(prune_epi8, 16);
}

#[test]
#[ignore] // expensive
fn test_thin_8() {
    test_128::<u8>(thinprune_epi8, 16);
}

#[test]
#[ignore] // expensive
fn test_skinny_8() {
    test_128::<u8>(skinnyprune_epi8, 16);
}

#[test]
fn test_16() {
    test_128::<u16>(prune_epi16, 8);
}

#[test]
fn test_32() {
    test_128::<u32>(prune_epi32, 4);
}

#[test]
fn test256_32() {
    test_256::<u32>(prune256_epi32, 8);
}
