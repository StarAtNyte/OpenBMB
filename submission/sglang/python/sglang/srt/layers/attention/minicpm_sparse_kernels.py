import torch
import triton
import triton.language as tl
from functools import lru_cache


# TODO. Now only page size == 1 is supported. Consider extend to page size > 1
@triton.jit
def compress_k_complete_kernel_new(
    key_cache_ptr,
    token_table_ptr,
    cu_new_k_token_nums_ptr,
    history_compress_k_token_nums_ptr,
    k_stride,
    compressed_k_table_ptr,
    cu_new_compress_k_token_nums_ptr,
    cu_total_compress_k_token_nums_ptr,
    total_compress_k_token_nums_ptr,
    full_compressed_k_ptr,
    batch_size,
    max_chunks_per_seq,
    token_table_cols,
    compressed_k_table_cols,
    head_num_k: tl.constexpr,
    head_dim: tl.constexpr,
    kernel_size: tl.constexpr,
    kernel_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    max_grid_chunks: tl.constexpr,
 ):
    """
    Optimized single-kernel: each (batch, chunk, head) thread independently
    computes mean pooling and stores its own result — no head-0-loop redundancy.

    Grid: (batch_size, min(max_total_chunks, max_grid_chunks), head_num_k)
    - PHASE 1 (history chunks): each head reads from key_cache[compressed_idx, head_idx, :]
      and writes directly to full_compressed_k.
    - PHASE 3/4 (new chunks): each head computes mean pooling over kernel_size tokens
      for its own head_idx and stores to key_cache[new_compressed_k_indices, head_idx, :]
      and full_compressed_k. No inter-head loops.
    """
    batch_idx = tl.program_id(0)
    grid_chunk_idx = tl.program_id(1)
    head_idx = tl.program_id(2)

    chunk_stride = max_grid_chunks

    if batch_idx >= batch_size or head_idx >= head_num_k:
        return

    history_compress = tl.load(history_compress_k_token_nums_ptr + batch_idx)

    cu_new_k_start = tl.load(cu_new_k_token_nums_ptr + batch_idx)
    cu_new_k_end = tl.load(cu_new_k_token_nums_ptr + batch_idx + 1)
    new_k_count = cu_new_k_end - cu_new_k_start
    new_chunks_in_seq = tl.where(
        new_k_count >= kernel_size,
        (new_k_count - kernel_size) // kernel_stride + 1,
        0
    )

    history_chunks_in_seq = history_compress
    total_chunks_in_seq = history_chunks_in_seq + new_chunks_in_seq

    cu_total_start = tl.load(cu_total_compress_k_token_nums_ptr + batch_idx)

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < head_dim

    chunk_in_seq = grid_chunk_idx

    while chunk_in_seq < total_chunks_in_seq:
        is_history_chunk = chunk_in_seq < history_chunks_in_seq

        if is_history_chunk:
            # ====================================================================
            # PHASE 1: History chunk — each head copies its own slice
            # ====================================================================
            history_chunk_idx = chunk_in_seq
            global_full_idx = cu_total_start + history_chunk_idx

            full_compressed_idx = tl.load(
                compressed_k_table_ptr + batch_idx * compressed_k_table_cols + history_chunk_idx
            ).to(tl.int32)

            src_offset = full_compressed_idx * head_num_k * head_dim + head_idx * head_dim
            x = tl.load(key_cache_ptr + src_offset + offs, mask=mask, other=0.0).to(tl.float32)

            out_offset = global_full_idx * head_num_k * head_dim + head_idx * head_dim
            tl.store(full_compressed_k_ptr + out_offset + offs, x, mask=mask)

        else:
            # ====================================================================
            # PHASE 2: New chunk
            # ====================================================================
            new_chunk_idx = chunk_in_seq - history_chunks_in_seq
            y = new_chunk_idx * kernel_stride + history_compress * k_stride

            if y < token_table_cols:
                compressed_table_y = new_chunk_idx + history_compress

                if compressed_table_y < compressed_k_table_cols:
                    new_compressed_k_indices = tl.load(
                        compressed_k_table_ptr + batch_idx * compressed_k_table_cols + compressed_table_y
                    ).to(tl.int32)

                    # ====================================================================
                    # PHASE 3: Mean pooling — this head only, no loop over heads
                    # ====================================================================
                    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

                    for token_offset in range(kernel_size):
                        token_y = (new_chunk_idx * kernel_stride + token_offset) + history_compress * k_stride

                        if token_y < token_table_cols:
                            token_k_indices = tl.load(
                                token_table_ptr + batch_idx * token_table_cols + token_y
                            ).to(tl.int32)
                        else:
                            token_k_indices = 0

                        key_base_offset = token_k_indices * head_num_k * head_dim + head_idx * head_dim
                        x = tl.load(key_cache_ptr + key_base_offset + offs, mask=mask, other=0.0).to(tl.float32)
                        acc += x

                    acc = acc / kernel_size

                    # ====================================================================
                    # PHASE 4: Store compressed mean to key_cache[new_compressed_k_indices, head_idx, :]
                    # Each head stores its own result — no head-0 loop
                    # ====================================================================
                    store_offset = new_compressed_k_indices * head_num_k * head_dim + head_idx * head_dim
                    tl.store(key_cache_ptr + store_offset + offs, acc, mask=mask)

                    # ====================================================================
                    # PHASE 5: Copy from key_cache to full_compressed_k — each head its own slice
                    # (re-use acc since we just wrote it — no need to re-load from key_cache)
                    # ====================================================================
                    global_full_idx = cu_total_start + history_compress + new_chunk_idx
                    out_offset = global_full_idx * head_num_k * head_dim + head_idx * head_dim
                    tl.store(full_compressed_k_ptr + out_offset + offs, acc, mask=mask)

        chunk_in_seq += chunk_stride


@triton.jit
def compress_k_complete_kernel_new_padded(
    key_cache_ptr,
    token_table_ptr,
    cu_new_k_token_nums_ptr,
    history_compress_k_token_nums_ptr,
    k_stride,
    compressed_k_table_ptr,
    cu_new_compress_k_token_nums_ptr,
    cu_total_compress_k_token_nums_ptr,
    total_compress_k_token_nums_ptr,
    full_compressed_k_ptr,
    batch_size,
    max_chunks_per_seq,
    token_table_cols,
    compressed_k_table_cols,
    head_num_k: tl.constexpr,
    head_dim: tl.constexpr,
    kernel_size: tl.constexpr,
    kernel_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    max_grid_chunks: tl.constexpr,
):
    """
    Padded layout version: stores compressed keys in batch-major order.
    Optimized: each head thread stores its own result — no head-0 loop redundancy.

    Output layout: full_compressed_k[batch_idx * max_chunks_per_seq + chunk_idx]
    """
    batch_idx = tl.program_id(0)
    grid_chunk_idx = tl.program_id(1)
    head_idx = tl.program_id(2)

    chunk_stride = max_grid_chunks

    if batch_idx >= batch_size or head_idx >= head_num_k:
        return

    history_compress = tl.load(history_compress_k_token_nums_ptr + batch_idx)

    cu_new_k_start = tl.load(cu_new_k_token_nums_ptr + batch_idx)
    cu_new_k_end = tl.load(cu_new_k_token_nums_ptr + batch_idx + 1)
    new_k_count = cu_new_k_end - cu_new_k_start
    new_chunks_in_seq = tl.where(
        new_k_count >= kernel_size,
        (new_k_count - kernel_size) // kernel_stride + 1,
        0
    )

    history_chunks_in_seq = history_compress
    total_chunks_in_seq = history_chunks_in_seq + new_chunks_in_seq

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < head_dim

    chunk_in_seq = grid_chunk_idx

    while chunk_in_seq < total_chunks_in_seq:
        is_history_chunk = chunk_in_seq < history_chunks_in_seq

        if is_history_chunk:
            # ====================================================================
            # PHASE 1: History chunk — each head copies its own slice (PADDED)
            # ====================================================================
            history_chunk_idx = chunk_in_seq
            global_full_idx = batch_idx * max_chunks_per_seq + history_chunk_idx

            full_compressed_idx = tl.load(
                compressed_k_table_ptr + batch_idx * compressed_k_table_cols + history_chunk_idx
            ).to(tl.int32)

            src_offset = full_compressed_idx * head_num_k * head_dim + head_idx * head_dim
            x = tl.load(key_cache_ptr + src_offset + offs, mask=mask, other=0.0).to(tl.float32)

            out_offset = global_full_idx * head_num_k * head_dim + head_idx * head_dim
            tl.store(full_compressed_k_ptr + out_offset + offs, x, mask=mask)

        else:
            # ====================================================================
            # PHASE 2: New chunk
            # ====================================================================
            new_chunk_idx = chunk_in_seq - history_chunks_in_seq
            y = new_chunk_idx * kernel_stride + history_compress * k_stride

            if y < token_table_cols:
                compressed_table_y = new_chunk_idx + history_compress

                if compressed_table_y < compressed_k_table_cols:
                    new_compressed_k_indices = tl.load(
                        compressed_k_table_ptr + batch_idx * compressed_k_table_cols + compressed_table_y
                    ).to(tl.int32)

                    # ====================================================================
                    # PHASE 3: Mean pooling — this head only
                    # ====================================================================
                    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

                    for token_offset in range(kernel_size):
                        token_y = (new_chunk_idx * kernel_stride + token_offset) + history_compress * k_stride

                        if token_y < token_table_cols:
                            token_k_indices = tl.load(
                                token_table_ptr + batch_idx * token_table_cols + token_y
                            ).to(tl.int32)
                        else:
                            token_k_indices = 0

                        key_base_offset = token_k_indices * head_num_k * head_dim + head_idx * head_dim
                        x = tl.load(key_cache_ptr + key_base_offset + offs, mask=mask, other=0.0).to(tl.float32)
                        acc += x

                    acc = acc / kernel_size

                    # ====================================================================
                    # PHASE 4: Store to key_cache — each head its own slot
                    # ====================================================================
                    store_offset = new_compressed_k_indices * head_num_k * head_dim + head_idx * head_dim
                    tl.store(key_cache_ptr + store_offset + offs, acc, mask=mask)

                    # ====================================================================
                    # PHASE 5: Copy to full_compressed_k (PADDED) — re-use acc
                    # ====================================================================
                    global_full_idx = batch_idx * max_chunks_per_seq + history_compress + new_chunk_idx
                    out_offset = global_full_idx * head_num_k * head_dim + head_idx * head_dim
                    tl.store(full_compressed_k_ptr + out_offset + offs, acc, mask=mask)

        chunk_in_seq += chunk_stride


"""Fused CUDA kernel for sparse_page_table to flashinfer format conversion.

This module provides a CUDA graph compatible conversion from MiniCPM's
sparse_page_table format to FlashInfer's kv_indices + kv_indptr format.
"""

import os
from typing import Tuple

import torch
import triton
import triton.language as tl


# Environment variable to select implementation
# Set USE_TRITON_KERNEL=1 to use Triton (CUDA graph compatible)
# Set USE_TRITON_KERNEL=0 to use PyTorch reference (slower, not CUDA graph compatible)
# Default is "1" - always use Triton kernel for CUDA graph compatibility
USE_TRITON_KERNEL = os.environ.get("USE_TRITON_KERNEL", "1") == "1"

# Environment variable to enable comparison between PyTorch and Triton implementations
# Set COMPARE_PYTORCH_TRITON=1 to validate Triton outputs against PyTorch reference
_COMPARISON_ENABLED = os.environ.get("COMPARE_PYTORCH_TRITON", "0") == "1"


#
# Alternative: Two-kernel approach for better performance with large batches
# Kernel 1: Compute cumulative sum (parallel scan)
# Kernel 2: Flatten and fill


@triton.jit
def cumsum_kernel(
    cache_seqlens_ptr,
    kv_indptr_ptr,
    sparse_bs: tl.constexpr,
):
    """Compute cumulative sum using parallel scan algorithm."""
    # Simple sequential implementation for now
    # TODO: Implement parallel scan for better performance
    cumsum = 0
    tl.store(kv_indptr_ptr, 0)

    for i in range(sparse_bs):
        val = tl.load(cache_seqlens_ptr + i)
        cumsum += val
        tl.store(kv_indptr_ptr + i + 1, cumsum)


@triton.jit
def flatten_and_fill_kernel(
    sparse_page_table_ptr,
    cache_seqlens_ptr,
    kv_indptr_ptr,
    kv_indices_ptr,
    kv_last_page_len_ptr,
    max_sparse_tokens: tl.constexpr,
    sparse_bs: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 256,
):
    """Flatten sparse_page_table and fill kv_last_page_len."""
    pid = tl.program_id(axis=0)

    if pid >= sparse_bs:
        return

    # Get offset and num_valid
    offset = tl.load(kv_indptr_ptr + pid)
    num_valid = tl.load(cache_seqlens_ptr + pid)

    # Copy valid entries
    num_loops = tl.cdiv(num_valid, BLOCK_SIZE)
    for i in range(num_loops):
        idx = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = idx < num_valid

        src_idx = pid * max_sparse_tokens + idx
        data = tl.load(sparse_page_table_ptr + src_idx, mask=mask, other=0)

        dst_idx = offset + idx
        tl.store(kv_indices_ptr + dst_idx, data, mask=mask)

    # Fill kv_last_page_len
    tl.store(kv_last_page_len_ptr + pid, 1)


def convert_sparse_to_flashinfer_two_kernel(
    sparse_page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
):
    """Two-kernel version for potentially better performance."""
    sparse_bs = cache_seqlens.shape[0]
    max_sparse_tokens = sparse_page_table.shape[1]

    # Kernel 1: Compute cumulative sum
    cumsum_kernel[(1,)](
        cache_seqlens,
        kv_indptr,
        sparse_bs=sparse_bs,
    )

    # Kernel 2: Flatten and fill
    BLOCK_SIZE = 256
    flatten_and_fill_kernel[(sparse_bs,)](
        sparse_page_table,
        cache_seqlens,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        max_sparse_tokens=max_sparse_tokens,
        sparse_bs=sparse_bs,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return kv_indptr, kv_indices, kv_last_page_len


# ============================================================================
# PyTorch Reference Implementation (for testing and fallback)
# ============================================================================


def convert_sparse_to_flashinfer_pytorch(
    sparse_page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch reference implementation for sparse_page_table conversion.

    This is the reference implementation used for testing and verification.
    It is NOT CUDA graph compatible due to intermediate allocations.

    Args:
        sparse_page_table: [sparse_bs, max_sparse_tokens] - Valid entries at start
        cache_seqlens: [sparse_bs] - Number of valid entries per row
        kv_indptr: Pre-allocated [sparse_bs + 1] buffer for output
        kv_indices: Pre-allocated [sparse_bs * max_sparse_tokens] buffer for output
        kv_last_page_len: Pre-allocated [sparse_bs] buffer for output

    Returns:
        Tuple of (kv_indptr, kv_indices, kv_last_page_len) - modified in-place
    """
    sparse_bs = cache_seqlens.shape[0]

    # Compute cumulative sum for kv_indptr
    kv_indptr[0] = 0
    kv_indptr[1:] = torch.cumsum(cache_seqlens, dim=0)

    # Flatten sparse_page_table based on cache_seqlens
    idx = 0
    for i in range(sparse_bs):
        num_valid = cache_seqlens[i].item()
        if num_valid > 0:
            kv_indices[idx : idx + num_valid] = sparse_page_table[i, :num_valid]
            idx += num_valid

    # Fill kv_last_page_len with ones
    kv_last_page_len.fill_(1)

    return kv_indptr, kv_indices, kv_last_page_len


# ============================================================================
# Unified Interface
# ============================================================================


def convert_sparse_page_table_to_flashinfer(
    sparse_page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert sparse_page_table to FlashInfer format.

    This is the main entry point that selects between PyTorch reference
    implementation and Triton kernel based on USE_TRITON_KERNEL env var.

    Args:
        sparse_page_table: [sparse_bs, max_sparse_tokens] - Valid entries at start
        cache_seqlens: [sparse_bs] - Number of valid entries per row
        kv_indptr: Pre-allocated [sparse_bs + 1] buffer for output
        kv_indices: Pre-allocated [sparse_bs * max_sparse_tokens] buffer for output
        kv_last_page_len: Pre-allocated [sparse_bs] buffer for output

    Returns:
        Tuple of (kv_indptr, kv_indices, kv_last_page_len) - modified in-place

    """
    if True:
        return convert_sparse_to_flashinfer_two_kernel(
            sparse_page_table,
            cache_seqlens,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
        )
    else:
        return convert_sparse_to_flashinfer_pytorch(
            sparse_page_table,
            cache_seqlens,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
        )
