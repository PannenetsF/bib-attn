import torch

import triton
import triton.language as tl
import math

from balance import balance as balance_block, validate as validate_block

@triton.jit
def bib_attention_score_kernel(
    q_tensor, 
    q_bs, q_H, q_h,
    k_cache,
    kc_D, kc_H, kc_h,
    k_start,
    ks_bs,
    k_length,
    kl_bs,
    score_tensor,
    st_bs, st_H, st_nc, st_sc,
    chunk_to_req,
    cr_bs, cr_nc,
    chunk_to_block,
    cb_bs, cb_nc,
    CHUNK_NUM: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    HEAD_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    chunk_idx = tl.program_id(2)

    is_active_block = (chunk_idx < CHUNK_NUM) and (batch_idx < BATCH_SIZE) # the block is indexed
    is_computing = is_active_block and (head_idx < HEAD_NUM) # the block is indexed and the head is indexed

    chunk_to_req_index = batch_idx * cr_bs + chunk_idx * cr_nc
    chunk_to_block_index = batch_idx * cb_bs + chunk_idx * cb_nc

    req_index = tl.load(chunk_to_req + chunk_to_req_index, mask=is_computing, other=0)
    block_index = tl.load(chunk_to_block + chunk_to_block_index, mask=is_computing, other=0)

    is_occupied = (req_index >= 0) and (block_index >= 0)
    is_computing = is_computing and is_occupied

    q_addr = q_tensor + req_index * q_bs + head_idx * q_H + q_h * tl.arange(0, HEAD_DIM)
    q_vec = tl.load(q_addr, mask=is_computing, other=0.)

    k_start_index = tl.load(k_start + req_index, mask=is_computing, other=0)
    k_length_scalar = tl.load(k_length + req_index, mask=is_computing, other=0)


    for ck_idx in tl.static_range(CHUNK_SIZE):
        this_length = ck_idx + block_index * CHUNK_SIZE
        is_valid = this_length < k_length_scalar
        is_valid_k = is_computing and is_valid
        k_index = k_start_index + this_length
        k_addr = k_cache + k_index * kc_D + head_idx * kc_H + kc_h * tl.arange(0, HEAD_DIM)
        k_vec = tl.load(k_addr, mask=is_valid_k, other=0.)
        
        this_score = tl.sum(k_vec * q_vec, axis=0)
        this_score = tl.where(is_computing, this_score, 0.0)
        score_offset = req_index * st_bs + head_idx * st_H + block_index * st_nc + ck_idx * st_sc
        tl.store(score_tensor + score_offset, this_score, mask=is_valid_k)

@torch.no_grad()
def attention_score(
    q_tensor,
    k_cache,
    k_start,
    k_length,
    score_tensor,
    chunk_size=4,
    debug_dict={}
):
    bs, H, h = q_tensor.shape
    max_length = k_length.max().item()
    assert score_tensor.shape[-1] == chunk_size
    chunk_num = math.floor((max_length + chunk_size - 1) / chunk_size)
    

    assert max_length == debug_dict['max_length']
    assert chunk_num == debug_dict['chunk_num']
    assert chunk_size == debug_dict['chunk_size']
    grid = (bs, H, chunk_num)
    chunk_to_req = torch.zeros(bs, chunk_num, dtype=torch.int32).cuda() - 1
    chunk_to_block = torch.zeros(bs, chunk_num, dtype=torch.int32).cuda() - 1
    for i in range(bs):
        num_chunk = math.floor((k_length[i] + chunk_size - 1) / chunk_size)
        chunk_to_req[i][:num_chunk] = i
        chunk_to_block[i][:num_chunk] = torch.arange(0, num_chunk).cuda() 

    bib_attention_score_kernel[grid](
        q_tensor,
        q_tensor.stride(0), q_tensor.stride(1), q_tensor.stride(2), # bs, H, h
        k_cache,
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), # D, H, h
        k_start,
        k_start.stride(0), # bs
        k_length,
        k_length.stride(0), # bs
        score_tensor,
        score_tensor.stride(0), score_tensor.stride(1), score_tensor.stride(2), score_tensor.stride(3),
        chunk_to_req,
        chunk_to_req.stride(0), chunk_to_req.stride(1),
        chunk_to_block,
        chunk_to_block.stride(0), chunk_to_block.stride(1),
        chunk_num,
        chunk_size,
        H,
        h,
        bs
    )



@torch.no_grad()
def attention_score_balance(
    q_tensor,
    k_cache,
    k_start,
    k_length,
    score_tensor,
    chunk_size=4,
    debug_dict={}
):
    bs, H, h = q_tensor.shape
    max_length = k_length.max().item()
    assert score_tensor.shape[-1] == chunk_size
    chunk_num = math.floor((max_length + chunk_size - 1) / chunk_size)
    

    assert max_length == debug_dict['max_length']
    assert chunk_num == debug_dict['chunk_num']
    assert chunk_size == debug_dict['chunk_size']
    chunk_to_req = torch.zeros(bs, chunk_num, dtype=torch.int32).cuda() - 1
    chunk_to_block = torch.zeros(bs, chunk_num, dtype=torch.int32).cuda() - 1
    rounded_k_length = torch.ceil(k_length.float() / chunk_size).to(torch.int32)
    assert rounded_k_length.max().item() == chunk_num
    chunk_to_req, chunk_to_block = balance_block(rounded_k_length.cpu())
    chunk_to_req, chunk_to_block = chunk_to_req.cuda(), chunk_to_block.cuda()
    grid = (bs, H, chunk_to_req.shape[-1])

    bib_attention_score_kernel[grid](
        q_tensor,
        q_tensor.stride(0), q_tensor.stride(1), q_tensor.stride(2), # bs, H, h
        k_cache,
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), # D, H, h
        k_start,
        k_start.stride(0), # bs
        k_length,
        k_length.stride(0), # bs
        score_tensor,
        score_tensor.stride(0), score_tensor.stride(1), score_tensor.stride(2), score_tensor.stride(3),
        chunk_to_req,
        chunk_to_req.stride(0), chunk_to_req.stride(1),
        chunk_to_block,
        chunk_to_block.stride(0), chunk_to_block.stride(1),
        chunk_num,
        chunk_size,
        H,
        h,
        bs
    )

@torch.no_grad()
def attention_score_lazy_balance(
    q_tensor,
    k_cache,
    k_start,
    k_length,
    score_tensor,
    chunk_size=4,
    debug_dict={},
    chunk_to_block=None,
    chunk_to_req=None,
):
    bs, H, h = q_tensor.shape
    max_length = k_length.max().item()
    assert score_tensor.shape[-1] == chunk_size
    chunk_num = math.floor((max_length + chunk_size - 1) / chunk_size)
    

    assert max_length == debug_dict['max_length']
    assert chunk_num == debug_dict['chunk_num']
    assert chunk_size == debug_dict['chunk_size']
    
    if chunk_to_block is None and chunk_to_req is None:
        chunk_to_req = torch.zeros(bs, chunk_num, dtype=torch.int32).cuda() - 1
        chunk_to_block = torch.zeros(bs, chunk_num, dtype=torch.int32).cuda() - 1
        rounded_k_length = torch.ceil(k_length.float() / chunk_size).to(torch.int32)
        assert rounded_k_length.max().item() == chunk_num
        chunk_to_req, chunk_to_block = balance_block(rounded_k_length.cpu())
        chunk_to_req, chunk_to_block = chunk_to_req.cuda(), chunk_to_block.cuda()
    elif chunk_to_block is None or chunk_to_req is None:
        raise ValueError("chunk_to_block and chunk_to_req should be both None or both not None")
    grid = (bs, H, chunk_to_req.shape[-1])

    bib_attention_score_kernel[grid](
        q_tensor,
        q_tensor.stride(0), q_tensor.stride(1), q_tensor.stride(2), # bs, H, h
        k_cache,
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), # D, H, h
        k_start,
        k_start.stride(0), # bs
        k_length,
        k_length.stride(0), # bs
        score_tensor,
        score_tensor.stride(0), score_tensor.stride(1), score_tensor.stride(2), score_tensor.stride(3),
        chunk_to_req,
        chunk_to_req.stride(0), chunk_to_req.stride(1),
        chunk_to_block,
        chunk_to_block.stride(0), chunk_to_block.stride(1),
        chunk_num,
        chunk_size,
        H,
        h,
        bs
    )
    return chunk_to_block, chunk_to_req

def _cos_of_tensors(a, b):
    assert a.shape == b.shape
    B, H, C, S = a.shape
    total_cos = torch.nn.functional.cosine_similarity(a.reshape(B, -1), b.reshape(B, -1), dim=-1).mean()
    return total_cos

@torch.no_grad()
def attention_score_torch_test(
    q_tensor,
    k_cache,
    k_start,
    k_length,
    triton_score_tensor,
):
    score_tensor = torch.zeros_like(triton_score_tensor)
    bs, H, h = q_tensor.shape
    score_tensor = score_tensor.reshape(bs, H, -1) # bs, H, l
    max_length = k_length.max().item()
    chunk_size = score_tensor.shape[-1]
    chunk_num = math.floor((max_length + chunk_size - 1) / chunk_size)
    for i in range(bs):
        this_q = q_tensor[i] # H, h
        this_k_start = k_start[i]
        this_k_length = k_length[i]
        this_k_cache = k_cache[this_k_start:this_k_start + this_k_length] # l, H, h
        this_k_cache = this_k_cache.permute(1, 0, 2).contiguous() # H, l, h
        this_q = this_q.reshape(H, h, 1) # H, h, 1
        this_score = torch.matmul(this_k_cache, this_q) # H, l, 1
        this_score = this_score.contiguous()
        score_tensor[i, :, :this_k_length] = this_score[:, :, 0]
    score_tensor = score_tensor.reshape_as(triton_score_tensor)
    cos = _cos_of_tensors(score_tensor, triton_score_tensor)
    assert cos.item() > 0.9999, f'cosine similarity is {cos.item()}'
