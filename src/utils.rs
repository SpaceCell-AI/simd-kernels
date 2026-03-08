// Copyright (c) 2025 SpaceCell Enterprises Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing available. See LICENSE and LICENSING.md.

//! # **Utility Functions** - *SIMD Processing and Memory Management Utilities*
//!
//! Core utilities supporting SIMD kernel implementations with efficient memory handling,
//! bitmask operations, and performance-critical helper functions.

use std::simd::{Mask, MaskElement, SimdElement};

use minarrow::{Bitmask, Vec64};

/// Extracts a core::SIMD `Mask<M, N>` for a batch of N lanes from a Minarrow `Bitmask`.
///
/// - `mask_bytes`: packed Arrow validity bits (LSB=index 0, bit=1 means valid)
/// - `offset`: starting index (bit offset into the mask)
/// - `logical_len`: number of logical bits in the mask
/// - `M`: SIMD mask type (e.g., i64 for f64, i32 for f32, i8 for i8)
///
/// Returns: SIMD Mask<M, N> representing validity for these N lanes.
/// Bits outside the logical length (i.e., mask is shorter than offset+N)
/// are treated as valid.
#[inline(always)]
pub fn bitmask_to_simd_mask<const N: usize, M>(
    mask_bytes: &[u8],
    offset: usize,
    logical_len: usize,
) -> Mask<M, N>
where
    M: MaskElement + SimdElement,
{
    let lane_limit = (offset + N).min(logical_len);
    let n_lanes = lane_limit - offset;
    let mut bits: u64 = 0;
    for j in 0..n_lanes {
        let idx = offset + j;
        let byte = mask_bytes[idx >> 3];
        if ((byte >> (idx & 7)) & 1) != 0 {
            bits |= 1u64 << j;
        }
    }
    if n_lanes < N {
        bits |= !0u64 << n_lanes;
    }
    Mask::<M, N>::from_bitmask(bits)
}

/// Converts a SIMD `Mask<M, N>` to a Minarrow `Bitmask` for the given logical length.
/// Used at the end of a block operation within SIMD-accelerated kernel functions.
#[inline(always)]
pub fn simd_mask_to_bitmask<const N: usize, M>(mask: Mask<M, N>, len: usize) -> Bitmask
where
    M: MaskElement + SimdElement,
{
    let mut bits = Vec64::with_capacity((len + 7) / 8);
    bits.resize((len + 7) / 8, 0);

    let word = mask.to_bitmask();
    let bytes = word.to_le_bytes();

    let n_bytes = (len + 7) / 8;
    bits[..n_bytes].copy_from_slice(&bytes[..n_bytes]);

    if len % 8 != 0 {
        let last = n_bytes - 1;
        let mask_byte = (1u8 << (len % 8)) - 1;
        bits[last] &= mask_byte;
    }

    Bitmask {
        bits: bits.into(),
        len,
    }
}

/// Bulk-ORs a local bitmask block (from a SIMD mask or similar) into the global Minarrow bitmask at the correct byte offset.
/// The block (`block_mask`) is expected to contain at least ceil(n_lanes/8) bytes,
/// with the bit-packed validity bits starting from position 0.
///
/// Used to streamline repetitive boilerplate and ensure consistency across kernel null-mask handling.
///
/// ### Parameters
/// - `out_mask`: mutable reference to the output/global Bitmask
/// - `block_mask`: reference to the local Bitmask containing the block's bits
/// - `offset`: starting bit offset in the global mask
/// - `n_lanes`: number of bits in this block (usually SIMD lane count)
#[inline(always)]
pub fn write_global_bitmask_block(
    out_mask: &mut Bitmask,
    block_mask: &Bitmask,
    offset: usize,
    n_lanes: usize,
) {
    let n_bytes = (n_lanes + 7) / 8;
    let base = offset / 8;
    let block_bytes = &block_mask.bits[..n_bytes];
    for b in 0..n_bytes {
        if base + b < out_mask.bits.len() {
            out_mask.bits[base + b] |= block_bytes[b];
        }
    }
}

/// Determines whether nulls are present given an optional null count and mask reference.
/// Avoids computing mask cardinality to preserve performance guarantees.
#[inline(always)]
pub fn has_nulls(null_count: Option<usize>, mask: Option<&Bitmask>) -> bool {
    match null_count {
        Some(n) => n > 0,
        None => mask.is_some(),
    }
}
