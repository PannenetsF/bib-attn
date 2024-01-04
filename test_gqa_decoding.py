import math

import torch

from gqa_decoding import gqa_token_decode_attention_flash_decoding

if __name__ == "__main__":
    bs = 4
    H = 16
    h = 16
    D = 24 * 1000
    chunk_size = 16
    max_L = 4 * chunk_size
    torch.cuda.manual_seed_all(0)
    torch.manual_seed(0)
    k_length = torch.randint(2, max_L, (bs,), dtype=torch.int32).cuda()
    k_start = k_length.cumsum(dim=0)
    k_start = torch.cat([torch.tensor([0]).cuda(), k_start[:-1]]).contiguous()
    q_tensor = torch.randn(bs, H, h, dtype=torch.float32).cuda()  # d
    k_cache = torch.randn(D, H, h, dtype=torch.float32).cuda()  # d
    v_cache = torch.randn(D, H, h, dtype=torch.float32).cuda()  # d
    max_length = k_length.max().item()
    chunk_num = math.floor((max_length + chunk_size - 1) / chunk_size)
    score_tensor = torch.zeros_like(q_tensor)
    req_to_tokens = torch.zeros(bs, max_length, dtype=torch.int32).cuda()
    for i in range(bs):
        req_to_tokens[i][:k_length[i]] = torch.arange(0, k_length[i]).cuda() + k_start[i]
    b_req_idx = torch.zeros(bs, dtype=torch.int32).cuda()
    for i in range(bs):
        b_req_idx[i] = i
    b_start_loc = torch.zeros(bs, dtype=torch.int32).cuda()
    for i in range(bs):
        b_start_loc[i] = k_start[i]
    b_seqlen = torch.zeros(bs, dtype=torch.int32).cuda()
    for i in range(bs):
        b_seqlen[i] = k_length[i]
    BLOCK_SEQ = 128
    mid_o = torch.empty([bs, H, max_length // BLOCK_SEQ + 1, h], dtype=torch.float32,
                        device="cuda")
    mid_o_logexpsum = torch.empty([bs, H, max_length // BLOCK_SEQ + 1], dtype=torch.float32,
                                  device="cuda")
    gqa_token_decode_attention_flash_decoding(q_tensor, k_cache, v_cache, score_tensor, mid_o, mid_o_logexpsum,
                                              b_seqlen, req_to_tokens, b_req_idx,
                                              bs, max_length, H, H)
