import math
from unittest import TestCase, main

import torch

from bib_gqa_decoding import bib_gqa_decoding


def bib_gqa_decoding_run(bs, H, h, G, chunk_size, max_L):
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
    )
    return q_tensor, k_tensor, v_tensor, k_length, score_tensor


def gqa_torch(q, k, v, sm_scale, length):
    bs, H_q, h = q.shape
    H_kv = k.shape[1]
    group = H_q // H_kv

    attn_tensor = torch.zeros_like(q)

    for idx in range(bs):
        k_start = length[:idx].sum()
        k_end = k_start + length[idx]
        L = length[idx]
        q_cur = q[idx]  # H_q, h
        k_cur = k[k_start:k_end]  # L, H_kv, h
        v_cur = v[k_start:k_end]  # L, H_kv, h
        k_cur = k_cur[:, :, None, :].expand(L, H_kv, group, h).reshape(L, H_kv * group, h)
        v_cur = v_cur[:, :, None, :].expand(L, H_kv, group, h).reshape(L, H_kv * group, h)

        k_cur = k_cur.permute(1, 0, 2)  # H_kv * group, L, h
        v_cur = v_cur.permute(1, 0, 2)  # H_kv * group, L, h
        q_cur = q_cur.reshape(H_q, h, 1)  # H_q, h, 1
        attn = torch.matmul(k_cur, q_cur)  # H_q, L, 1
        attn = attn.reshape(H_q, L)
        attn = attn * sm_scale
        attn = torch.softmax(attn, dim=1)
        attn = attn.reshape(H_q, L, 1)
        attn = attn * v_cur
        attn = attn.sum(dim=1)
        attn_tensor[idx] = attn

    return attn_tensor


def _cos_of_tensors(a, b):
    assert a.shape == b.shape, f'{a.shape} vs {b.shape}'
    B = a.shape[0]
    total_cos = torch.nn.functional.cosine_similarity(a.reshape(B, -1), b.reshape(B, -1), dim=-1).mean()
    return total_cos


def bib_gqa_decoding_verify(bs, H, h, G, chunk_size, max_L):
    q_tensor, k_tensor, v_tensor, k_length, score_tensor = bib_gqa_decoding_run(bs, H, h, G, chunk_size, max_L)
    attn_tensor = gqa_torch(q_tensor, k_tensor, v_tensor, 1 / math.sqrt(H), k_length)
    return _cos_of_tensors(attn_tensor, score_tensor)


model_config = [
    dict(H=64, G=8, h=128),
    # dict(H=32, G=32, h=128),
]
shape_config = [
    dict(bs=4, chunk_size=64, max_L=512),
    dict(bs=64, chunk_size=64, max_L=512),
    dict(bs=4, chunk_size=64, max_L=1024),
    dict(bs=64, chunk_size=64, max_L=1024),
    dict(bs=4, chunk_size=64, max_L=2048),
    dict(bs=64, chunk_size=64, max_L=2048),
]


class TestBiBDecoding(TestCase):
    def test_run(self):
        for mc in model_config:
            for sc in shape_config:
                bib_gqa_decoding_run(**mc, **sc)

    def test_acc(self):
        for mc in model_config:
            for sc in shape_config:
                self.assertGreaterEqual(bib_gqa_decoding_verify(**mc, **sc), 0.99)


if __name__ == '__main__':
    main()
