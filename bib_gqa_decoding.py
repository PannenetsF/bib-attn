import torch
import math
import triton
import triton.language as tl


@triton.jit
def bib_gqa_qkv_mul_kernel(
        q_tensor, q_bs, q_H, q_h,
        k_tensor, k_D, k_H, k_h,
        v_tensor, v_D, v_H, v_h,
        deno_tensor, deno_bs, deno_H, deno_dim,  # 2dim, max and sum
        nume_tensor, nume_bs, nume_H, nume_h,
        block_to_request, b2r_bs,
        block_to_start, b2s_bs,
        block_to_length, b2l_bs,
        block_to_chunk, b2c_bs,
        sm_scale: tl.constexpr,
        Q_HEAD_NUM: tl.constexpr,
        HEAD_NUM: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        CHUNK_SIZE: tl.constexpr,
):
    block_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    cur_batch = tl.load(block_to_request + block_idx * b2r_bs)
    cur_start = tl.load(block_to_start + block_idx * b2s_bs)
    cur_length = tl.load(block_to_length + block_idx * b2l_bs)
    cur_chunk = tl.load(block_to_chunk + block_idx * b2c_bs)

    head_off = tl.arange(0, HEAD_DIM)
    chunk_off = tl.arange(0, CHUNK_SIZE)
    q_head_num_off = tl.arange(0, Q_HEAD_NUM)
    q_head_mask = q_head_num_off < GROUP_SIZE

    q_head_idx = kv_head_idx * GROUP_SIZE + q_head_num_off
    q_off = cur_batch * q_bs + q_head_idx[:, None] * q_H + head_off[None, :] * q_h  # [H, h]
    q_vec = tl.load(q_tensor + q_off)  # [head_num, head_dim]

    k_len = cur_chunk * CHUNK_SIZE + chunk_off
    k_off = (k_len[None, :] + cur_start) * k_D + kv_head_idx * k_H + head_off[:, None] * k_h  # [chunk_size, head_dim]
    v_off = (k_len[:, None] + cur_start) * v_D + kv_head_idx * v_H + head_off[None, :] * v_h
    k_mask = k_len < cur_length
    k_mat = tl.load(k_tensor + k_off, mask=k_mask[None, :], other=0.0)
    score = tl.dot(q_vec, k_mat)  # [H, chunk_size]
    score = tl.where(q_head_mask[:, None] & k_mask[None, :], score * sm_scale, -1e9)  # [H, chunk_size]

    v_mat = tl.load(v_tensor + v_off, mask=k_mask[:, None], other=0.0)  # [chunk_size, head_dim]
    s_max = tl.max(score, axis=1)
    s_exp = tl.exp(score - s_max[:, None])
    s_exp_sum = tl.sum(s_exp, axis=1)

    s_exp_v = tl.dot(s_exp, v_mat)

    # save s_max, s_exp_sum, s_exp_v
    # shape (1, ), (1, ), (head_dim, )
    # tl.store(deno_tensor + block_idx * deno_bs + q_head_idx * deno_H + 0 * deno_dim, s_max, mask=q_head_mask)
    # tl.store(deno_tensor + block_idx * deno_bs + q_head_idx * deno_H + 1 * deno_dim, s_exp_sum, mask=q_head_mask)
    tl.store(nume_tensor + block_idx * nume_bs + q_head_idx[:, None] * nume_H + head_off[None, :] * nume_h, s_exp_v,
             mask=q_head_mask[:, None])


@triton.jit
def bib_gqa_ref_request_max_reduce(
        request_to_block, r2b_bs, r2b_blk,
        deno_tensor, deno_bs, deno_H, deno_dim,  # 2dim, max and sum
        nume_tensor, nume_bs, nume_H, nume_h,
        out_tensor, out_bs, out_H, out_h,
        BATCH_SIZE: tl.constexpr,
        CHUNK_NUM: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        ROUND_CHUNK_NUM: tl.constexpr
):
    request_id = tl.program_id(0)
    head_idx = tl.program_id(1)

    request_mask = tl.where(request_id < BATCH_SIZE, 1, 0)

    for i in range(0, request_mask, 1):
        chunk_off = tl.arange(0, ROUND_CHUNK_NUM)
        head_off = tl.arange(0, HEAD_DIM)
        block_idx = tl.load(request_to_block + request_id * r2b_bs + r2b_blk * chunk_off, mask=chunk_off < CHUNK_NUM,
                            other=-1)
        block_mask = block_idx != -1
        s_max = tl.load(deno_tensor + block_idx * deno_bs + head_idx * deno_H + 0 * deno_dim, mask=block_mask,
                        other=-1e9)  # (CN, )
        s_exp_sum = tl.load(deno_tensor + block_idx * deno_bs + head_idx * deno_H + 1 * deno_dim, mask=block_mask,
                            other=0)  # (CN, )
        s_exp_v = tl.load(nume_tensor + block_idx[:, None] * nume_bs + head_idx * nume_H + head_off[None, :] * nume_h,
                          mask=block_mask[:, None], other=0.)  # (CN, h)
        s_g_max = tl.max(s_max, axis=0)  # 1
        rescale = tl.exp(s_max - s_g_max)  # CN
        s_exp_sum = s_exp_sum * rescale  # CN
        s_exp_sum = tl.sum(s_exp_sum)  # 1

        s_exp_v = s_exp_v * rescale[:, None]  # CN, h
        s_exp_v = tl.sum(s_exp_v, 0)
        s_exp_v = s_exp_v / s_exp_sum
        tl.store(out_tensor + request_id * out_bs + head_idx * out_H + head_off * out_h, s_exp_v)


@torch.no_grad()
def bib_gqa_decoding(
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
        GROUP_SIZE=4,
        CHUNK_SIZE=64,
):
    r'''
    step 1: 
        deno: get max scale and exp sum of each block 
        nume: get the inner product of scale and value, (reduce on the Length dim)
    step 2:
        sync all deno
            deno: get global max, and rescale the exp sum
    '''
    block_num = block_to_request.shape[0]
    chunk_num = request_to_block.shape[1]
    batch_size = q_tensor.shape[0]
    head_num = q_tensor.shape[1]
    head_dim = q_tensor.shape[2]
    k_head_num = k_tensor.shape[1]

    assert GROUP_SIZE < head_num, f'you should use MQA kernel rather than the GQA kernel k={k_head_num} q={head_num}'

    if deno_tensor is None:
        deno_tensor = torch.zeros((block_num, head_num, 2), dtype=torch.float32, device=q_tensor.device)
    if nume_tensor is None:
        nume_tensor = torch.zeros((block_num, head_num, head_dim), dtype=torch.float32, device=q_tensor.device)

    grid = lambda META: (block_num, head_num)
    bib_gqa_qkv_mul_kernel[grid](
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
        Q_HEAD_NUM=min(16, triton.next_power_of_2(head_dim)),
        CHUNK_SIZE=CHUNK_SIZE,
        GROUP_SIZE=GROUP_SIZE
    )
    #
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


@torch.no_grad()
def bib_gqa_decoding_stage1(
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
        GROUP_SIZE=4,
        CHUNK_SIZE=64,
):
    r'''
    step 1:
        deno: get max scale and exp sum of each block
        nume: get the inner product of scale and value, (reduce on the Length dim)
    step 2:
        sync all deno
            deno: get global max, and rescale the exp sum
    '''
    block_num = block_to_request.shape[0]
    chunk_num = request_to_block.shape[1]
    batch_size = q_tensor.shape[0]
    head_num = q_tensor.shape[1]
    head_dim = q_tensor.shape[2]
    k_head_num = k_tensor.shape[1]

    assert GROUP_SIZE < head_num, f'you should use MQA kernel rather than the GQA kernel k={k_head_num} q={head_num}'

    if deno_tensor is None:
        deno_tensor = torch.zeros((block_num, head_num, 2), dtype=torch.float32, device=q_tensor.device)
    if nume_tensor is None:
        nume_tensor = torch.zeros((block_num, head_num, head_dim), dtype=torch.float32, device=q_tensor.device)

    grid = lambda META: (block_num, head_num)
    bib_gqa_qkv_mul_kernel[grid](
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
        Q_HEAD_NUM=min(16, triton.next_power_of_2(head_dim)),
        CHUNK_SIZE=CHUNK_SIZE,
        GROUP_SIZE=GROUP_SIZE
    )


if __name__ == '__main__':
    import math

    bs = 3
    H = 64
    G = 4
    h = 16
    chunk_size = 16
    max_L = 1024
    D = bs * max_L

    q_shape = (bs, H, h)
    k_shape = (D, H // G, h)
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

    for c in BiBConfig().all_config():
        print(f'trying {c}')
        bib_gqa_decoding(
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
            GROUP_SIZE=G,
            config=c
        )
