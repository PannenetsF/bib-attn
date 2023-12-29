import torch
import triton
import triton.language as tl

@triton.jit
def bib_gqa_ref_qk_mul_kernel_length_mask_fused(
    q_tensor, q_bs, q_H, q_h,
    k_tensor, k_D, k_H, k_h,
    score_tensor, s_bs, s_H, s_L,
    block_to_rslc, b2r_bs, b2r_blk,
    HEAD_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    '''
    the b2l is the block to length in the block, which is the used length of the block
    '''
    block_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    cur_batch = tl.load(block_to_rslc + block_idx * b2r_bs + b2r_blk * 0)
    cur_start = tl.load(block_to_rslc + block_idx * b2r_bs + b2r_blk * 1)
    cur_length = tl.load(block_to_rslc + block_idx * b2r_bs + b2r_blk * 2)
    cur_chunk = tl.load(block_to_rslc + block_idx * b2r_bs + b2r_blk * 3)

    head_off = tl.arange(0, HEAD_DIM)
    chunk_off = tl.arange(0, CHUNK_SIZE)

    q_off = cur_batch * q_bs + head_idx * q_H + head_off * q_h
    q_vec = tl.load(q_tensor + q_off) # [head_dim]


    k_len = cur_chunk * CHUNK_SIZE + chunk_off
    k_off = (k_len[:, None] + cur_start) * k_D + head_idx * k_H + head_off[None, :] * k_h
    k_mask = k_len < cur_length
    k_mat = tl.load(k_tensor + k_off, mask=k_mask[:, None], other=0.0)
    
    score = tl.sum(q_vec[None, :] * k_mat, axis=1)
    score_off = cur_batch * s_bs + head_idx * s_H + k_len * s_L
    tl.store(score_tensor + score_off, score, mask=k_mask)

@torch.no_grad()
def bib_gqa_ref_qk_mul_length_mask_fused(
    q_tensor,
    k_tensor,
    score_tensor,
    block_to_rslc,
    CHUNK_SIZE=64,
):
    block_num = block_to_rslc.shape[0]
    head_num = q_tensor.shape[1]

    grid = lambda META: (block_num, head_num)
    meta = {
        'HEAD_NUM': head_num,
        'HEAD_DIM': q_tensor.shape[2],
        'CHUNK_SIZE': CHUNK_SIZE,
    }
    bib_gqa_ref_qk_mul_kernel_length_mask_fused[grid](
        q_tensor, q_tensor.stride(0), q_tensor.stride(1), q_tensor.stride(2),
        k_tensor, k_tensor.stride(0), k_tensor.stride(1), k_tensor.stride(2),
        score_tensor, score_tensor.stride(0), score_tensor.stride(1), score_tensor.stride(2),
        # block_to_request, block_to_request.stride(0),
        # block_to_start, block_to_start.stride(0),
        # block_to_length, block_to_length.stride(0),
        block_to_rslc, block_to_rslc.stride(0), block_to_rslc.stride(1),
        HEAD_NUM=head_num,
        HEAD_DIM=q_tensor.shape[2],
        CHUNK_SIZE=CHUNK_SIZE,
    )



@triton.jit
def bib_gqa_ref_qk_mul_kernel_length_mask(
    q_tensor, q_bs, q_H, q_h,
    k_tensor, k_D, k_H, k_h,
    score_tensor, s_bs, s_H, s_L,
    block_to_request, b2r_bs,
    block_to_start, b2s_bs,
    block_to_length, b2l_bs,
    block_to_chunk, b2c_bs,
    HEAD_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    '''
    the b2l is the block to length in the block, which is the used length of the block
    '''
    block_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    cur_batch = tl.load(block_to_request + block_idx * b2r_bs)
    cur_start = tl.load(block_to_start + block_idx * b2s_bs)
    cur_length = tl.load(block_to_length + block_idx * b2l_bs)
    cur_chunk = tl.load(block_to_chunk + block_idx * b2c_bs)

    head_off = tl.arange(0, HEAD_DIM)
    chunk_off = tl.arange(0, CHUNK_SIZE)

    q_off = cur_batch * q_bs + head_idx * q_H + head_off * q_h
    q_vec = tl.load(q_tensor + q_off) # [head_dim]

    k_len = cur_chunk * CHUNK_SIZE + chunk_off
    k_off = (k_len[:, None] + cur_start) * k_D + head_idx * k_H + head_off[None, :] * k_h
    k_mask = k_len < cur_length
    k_mat = tl.load(k_tensor + k_off, mask=k_mask[:, None], other=0.0)
    score = tl.sum(q_vec[None, :] * k_mat, axis=1)
    score_off = cur_batch * s_bs + head_idx * s_H + k_len * s_L
    tl.store(score_tensor + score_off, score, mask=k_mask)

@torch.no_grad()
def bib_gqa_ref_qk_mul_length_mask(
    q_tensor,
    k_tensor,
    score_tensor,
    block_to_request,
    block_to_start,
    block_to_length,
    block_to_chunk,
    CHUNK_SIZE=64,
):
    block_num = block_to_request.shape[0]
    head_num = q_tensor.shape[1]

    grid = lambda META: (block_num, head_num)
    meta = {
        'HEAD_NUM': head_num,
        'HEAD_DIM': q_tensor.shape[2],
        'CHUNK_SIZE': CHUNK_SIZE,
    }
    bib_gqa_ref_qk_mul_kernel_length_mask[grid](
        q_tensor, q_tensor.stride(0), q_tensor.stride(1), q_tensor.stride(2),
        k_tensor, k_tensor.stride(0), k_tensor.stride(1), k_tensor.stride(2),
        score_tensor, score_tensor.stride(0), score_tensor.stride(1), score_tensor.stride(2),
        block_to_request, block_to_request.stride(0),
        block_to_start, block_to_start.stride(0),
        block_to_length, block_to_length.stride(0),
        block_to_chunk, block_to_chunk.stride(0),
        HEAD_NUM=head_num,
        HEAD_DIM=q_tensor.shape[2],
        CHUNK_SIZE=CHUNK_SIZE,
    )
