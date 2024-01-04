import math

import torch

from bib_decoding import bib_gqa_ref_request_max_reduce, bib_gqa_ref_qkv_score_kernel


@torch.no_grad()
def bib_decoding_debug(
        q_tensor,
        k_tensor,
        v_tensor,
        score_tensor,
        request_to_block,
        block_to_request,
        block_to_start,
        block_to_length,
        block_to_chunk,
        sm_scale,
        deno_tensor=None,
        nume_tensor=None,
        CHUNK_SIZE=64,
        step=None
):
    r'''
    step 1: 
        deno: get max scale and exp sum of each block 
        nume: get the inner product of scale and value, (reduce on the Length dim)
    '''
    block_num = block_to_request.shape[0]
    chunk_num = request_to_block.shape[1]
    batch_size = q_tensor.shape[0]
    head_num = q_tensor.shape[1]
    head_dim = q_tensor.shape[2]

    if not deno_tensor:
        deno_tensor = torch.zeros((block_num, head_num, 2), dtype=torch.float32, device=q_tensor.device)
    if not nume_tensor:
        nume_tensor = torch.zeros((block_num, head_num, q_tensor.shape[2]), dtype=torch.float32, device=q_tensor.device)

    grid = lambda META: (block_num, head_num)
    bib_gqa_ref_qkv_score_kernel[grid](
        q_tensor, q_tensor.stride(0), q_tensor.stride(1), q_tensor.stride(2),
        k_tensor, k_tensor.stride(0), k_tensor.stride(1), k_tensor.stride(2),
        v_tensor, v_tensor.stride(0), v_tensor.stride(1), v_tensor.stride(2),
        deno_tensor, deno_tensor.stride(0), deno_tensor.stride(1), deno_tensor.stride(2),
        nume_tensor, nume_tensor.stride(0), nume_tensor.stride(1), nume_tensor.stride(2),
        block_to_request, block_to_request.stride(0),
        block_to_start, block_to_start.stride(0),
        block_to_length, block_to_length.stride(0),
        block_to_chunk, block_to_chunk.stride(0),
        sm_scale=sm_scale,
        HEAD_NUM=head_num,
        HEAD_DIM=head_dim,
        CHUNK_SIZE=CHUNK_SIZE,
    )

    if step == 1: return deno_tensor, nume_tensor

    grid = lambda META: (2 ** (math.ceil(math.log2(batch_size))), head_num)
    bib_gqa_ref_request_max_reduce[grid](
        request_to_block, request_to_block.stride(0), request_to_block.stride(1),
        deno_tensor, deno_tensor.stride(0), deno_tensor.stride(1), deno_tensor.stride(2),
        nume_tensor, nume_tensor.stride(0), nume_tensor.stride(1), nume_tensor.stride(2),
        score_tensor, score_tensor.stride(0), score_tensor.stride(1), score_tensor.stride(2),
        BATCH_SIZE=batch_size,
        CHUNK_NUM=chunk_num,
        HEAD_DIM=head_dim,
        ROUND_CHUNK_NUM=2 ** (math.ceil(math.log2(chunk_num)))
    )

    if step == 2: return score_tensor


def _cos_of_tensors(a, b):
    assert a.shape == b.shape, f'{a.shape} vs {b.shape}'
    B = a.shape[0]
    total_cos = torch.nn.functional.cosine_similarity(a.reshape(B, -1), b.reshape(B, -1), dim=-1).mean()
    return total_cos


def bib_step1(
        q_tensor,
        k_tensor,
        v_tensor,
        block_to_request,
        block_to_start,
        block_to_length,
        block_to_chunk,
        sm_scale
):
    block_num = block_to_request.shape[0]
    batch_size = q_tensor.shape[0]
    head_num = q_tensor.shape[1]
    head_dim = q_tensor.shape[2]

    deno_tensor = torch.zeros((block_num, head_num, 2), dtype=torch.float32, device=q_tensor.device)
    nume_tensor = torch.zeros((block_num, head_num, q_tensor.shape[2]), dtype=torch.float32, device=q_tensor.device)

    block_idx = 0
    for req, start, length, chunk in zip(block_to_request, block_to_start, block_to_length, block_to_chunk):
        q_vec = q_tensor[req]  # H, h
        q_mat = q_vec.reshape(head_num, 1, h)  # H, 1, h
        l_start = chunk * chunk_size
        l_end = min(l_start + chunk_size, length)
        k_mat = k_tensor[start + l_start: start + l_end]  # L, H, h
        k_mat = k_mat.permute(1, 2, 0).contiguous()  # H, h, L
        score_mat = sm_scale * torch.matmul(q_mat, k_mat)  # H, 1, L
        score_mat = score_mat.reshape(head_num, l_end - l_start)

        score_max = score_mat.max(dim=1, keepdim=True)[0]  # H,
        score_exp = (score_mat - score_max).exp()  # H, L
        score_exp_sum = score_exp.sum(dim=1)

        v_mat = v_tensor[start + l_start: start + l_end]  # L, H, h
        v_mat = v_mat.permute(1, 2, 0).contiguous()  # H, h, L
        score_exp = score_exp.reshape(head_num, 1, l_end - l_start)  # H, 1, L
        score_exp_v = (v_mat * score_exp).sum(dim=2)  # H, h

        deno_tensor[block_idx, :, 0] = score_max[:, 0]
        deno_tensor[block_idx, :, 1] = score_exp_sum

        nume_tensor[block_idx] = score_exp_v

        block_idx += 1
    return deno_tensor, nume_tensor


def bib_step2(
        deno_tensor,
        nume_tensor,
        request_to_block,
        batch_size,
        head_num,
        head_dim
):
    score_tensor = torch.zeros((batch_size, head_num, head_dim), dtype=torch.float32, device=torch.device(0))
    for batch_idx in range(batch_size):
        blocks = request_to_block[batch_idx]
        blocks = blocks[blocks != -1]
        s_max = deno_tensor[blocks, :, 0]  # L, H
        s_exp_sum = deno_tensor[blocks, :, 1]  # L, H
        s_exp_v = nume_tensor[blocks, :, :]  # L, H, h

        g_max = s_max.max(dim=0, keepdim=True)[0]  # H
        rescale = (s_max - g_max).exp()  # L, H
        s_exp_sum *= rescale
        s_exp_sum = s_exp_sum.sum(dim=0).reshape(head_dim, 1)  # H, 1
        s_exp_v = s_exp_v * rescale.unsqueeze(-1)  # L, H, h
        s_exp_v = s_exp_v.sum(dim=0)  # H, h
        score = s_exp_v / s_exp_sum
        score_tensor[batch_idx] = score

    return score_tensor


def decoding(
        q_tensor,
        k_tensor,
        v_tensor,
        k_length,
        k_start,
        sm_scale,
):
    bs, H, h = q_tensor.shape
    max_L = k_length.max()
    sm_mat = torch.zeros((bs, H, max_L), dtype=torch.float32, device=torch.device('cuda:0'))

    for bidx in range(bs):
        q_vec = q_tensor[bidx]  # H, h
        start = k_start[bidx]
        dur = k_length[bidx]
        k_vec = k_tensor[start: start + dur]  # L, H, h

        q_vec = q_vec.reshape(H, h, 1)
        k_vec = k_vec.permute(1, 0, 2).contiguous()  # H, L, h

        sm_mat[bidx, :, :dur] = torch.matmul(k_vec, q_vec).reshape(H, dur)  # H, L
        sm_mat[bidx, :, dur:] = -1e9

    sm_mat *= sm_scale
    sm_mat = sm_mat.softmax(dim=-1)

    attn = torch.zeros((bs, H, h), dtype=torch.float32, device=torch.device('cuda:0'))
    for bidx in range(bs):
        sm = sm_mat[bidx]  # H, L
        sm = sm.permute(1, 0).contiguous().reshape(-1, H, 1)

        start = k_start[bidx]
        dur = k_length[bidx]
        v_vec = v_tensor[start: start + dur]  # L, H, h

        score = sm[:dur] * v_vec
        score = score.sum(dim=0)  # H, h
        attn[bidx] = score

    return attn


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    bs = 3
    H = 8
    h = 8
    chunk_size = 16
    max_L = 4 * chunk_size
    D = bs * max_L

    q_shape = (bs, H, h)
    k_shape = (D, H, h)
    score_shape = (bs, H, h)

    q_tensor = torch.randn(q_shape, dtype=torch.float32).cuda()
    k_tensor = torch.randn(k_shape, dtype=torch.float32).cuda()
    v_tensor = torch.randn(k_shape, dtype=torch.float32).cuda()
    score_tensor = torch.zeros(score_shape, dtype=torch.float32).cuda()
    triton_score_tensor = torch.zeros(score_shape, dtype=torch.float32).cuda()

    k_length = torch.randint(2, max_L, (bs,), dtype=torch.int32).cuda()
    k_start = k_length.cumsum(dim=0)
    k_start = torch.cat([torch.tensor([0]).cuda(), k_start[:-1]]).contiguous()

    block_to_length = []
    block_to_request = []
    block_to_start = []
    block_to_chunk = []
    for req_idx, (leng, start) in enumerate(zip(k_length, k_start)):
        for chunk_idx in range(math.ceil(leng / chunk_size)):
            block_to_length.append(leng)
            block_to_start.append(start)
            block_to_chunk.append(chunk_idx)
            block_to_request.append(req_idx)
    request_to_block = []
    r2b_dict = {}
    for block_idx, req_idx in enumerate(block_to_request):
        if req_idx not in r2b_dict:
            r2b_dict[req_idx] = []
        r2b_dict[req_idx].append(block_idx)
    max_leng_r2b = max([len(r2b_dict[k]) for k in r2b_dict])
    max_leng_r2b = 2 ** (math.ceil(math.log2(max_leng_r2b)))
    for req_idx in range(bs):
        request_to_block.append(r2b_dict[req_idx] + [-1] * (max_leng_r2b - len(r2b_dict[req_idx])))
    request_to_block = torch.tensor(request_to_block, dtype=torch.int32).cuda()
    block_to_length = torch.tensor(block_to_length, dtype=torch.int32).cuda()
    block_to_start = torch.tensor(block_to_start, dtype=torch.int32).cuda()
    block_to_chunk = torch.tensor(block_to_chunk, dtype=torch.int32).cuda()
    block_to_request = torch.tensor(block_to_request, dtype=torch.int32).cuda()

    deno, nume = bib_decoding_debug(
        q_tensor,
        k_tensor,
        v_tensor,
        score_tensor,
        request_to_block,
        block_to_request,
        block_to_start,
        block_to_length,
        block_to_chunk,
        sm_scale=1 / math.sqrt(H),
        CHUNK_SIZE=chunk_size,
        step=1
    )

    deno_torch, nume_torch = bib_step1(
        q_tensor,
        k_tensor,
        v_tensor,
        block_to_request,
        block_to_start,
        block_to_length,
        block_to_chunk,
        sm_scale=1 / math.sqrt(H),
    )

    assert _cos_of_tensors(deno_torch, deno) > 0.99, 'failed at step 1'
    assert _cos_of_tensors(nume_torch, nume) > 0.99, 'failed at step 1'

    score = bib_decoding_debug(
        q_tensor,
        k_tensor,
        v_tensor,
        score_tensor,
        request_to_block,
        block_to_request,
        block_to_start,
        block_to_length,
        block_to_chunk,
        sm_scale=1 / math.sqrt(H),
        CHUNK_SIZE=chunk_size,
        step=2
    )

    score_torch = bib_step2(
        deno_torch,
        nume_torch,
        request_to_block,
        bs, H, h
    )

    score_ref = decoding(
        q_tensor,
        k_tensor,
        v_tensor,
        k_length,
        k_start,
        sm_scale=1 / math.sqrt(H)
    )
    assert _cos_of_tensors(score_torch,
                           score_ref) > 0.99, f'failed at  validation {_cos_of_tensors(score_torch, score_ref)}'
