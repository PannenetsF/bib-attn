import math
from unittest import TestCase, main

import torch

from bib_decoding import bib_decoding


def _cos_of_tensors(a, b):
    assert a.shape == b.shape, f'{a.shape} vs {b.shape}'
    B = a.shape[0]
    total_cos = torch.nn.functional.cosine_similarity(a.reshape(B, -1), b.reshape(B, -1), dim=-1).mean()
    return total_cos


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


def bib_decoding_test(bs=3, H=8, h=8, chunk_size=16, max_L=1024):
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

    bib_decoding(
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
    )

    score_ref = decoding(
        q_tensor,
        k_tensor,
        v_tensor,
        k_length,
        k_start,
        sm_scale=1 / math.sqrt(H)
    )
    return _cos_of_tensors(score_tensor, score_ref)


class TestBiBDecoding(TestCase):
    def test_run(self):
        bib_decoding_test(bs=3, H=8, h=8, chunk_size=16, max_L=1024)
        bib_decoding_test(bs=37, H=8, h=8, chunk_size=16, max_L=1024)

    def test_acc(self):
        cos = bib_decoding_test(bs=3, H=8, h=8, chunk_size=16, max_L=1024)
        self.assertGreater(cos, 0.99)
        cos = bib_decoding_test(bs=43, H=32, h=32, chunk_size=32, max_L=2048)
        self.assertGreater(cos, 0.99)


if __name__ == '__main__':
    main()
