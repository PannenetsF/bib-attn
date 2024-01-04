import math

import torch

from lightllm_score import token_att_fwd, lightllm_attention

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
    score_tensor = torch.zeros(H, D).cuda()
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

    token_att_fwd(q_tensor, k_cache, score_tensor, req_to_tokens, b_req_idx, b_start_loc, b_seqlen, max_length)


    def convert_score_tensor(score_tensor):
        target_score = torch.zeros(bs, H, max_length).cuda()
        for i in range(bs):
            this_score = score_tensor[:, k_start[i]:k_start[i] + k_length[i]]  # H, L
            target_score[i][:, :k_length[i]] = this_score
        return target_score


    target_score = convert_score_tensor(score_tensor)


    def _cos_of_tensors(a, b):
        assert a.shape == b.shape
        B = a.shape[0]
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
        score_tensor = score_tensor.reshape(bs, H, -1)  # bs, H, l
        max_length = k_length.max().item()
        chunk_size = score_tensor.shape[-1]
        chunk_num = math.floor((max_length + chunk_size - 1) / chunk_size)
        for i in range(bs):
            this_q = q_tensor[i]  # H, h
            this_k_start = k_start[i]
            this_k_length = k_length[i]
            this_k_cache = k_cache[this_k_start:this_k_start + this_k_length]  # l, H, h
            this_k_cache = this_k_cache.permute(1, 0, 2).contiguous()  # H, l, h
            this_q = this_q.reshape(H, h, 1)  # H, h, 1
            this_score = torch.matmul(this_k_cache, this_q)  # H, l, 1
            this_score = this_score.contiguous()
            score_tensor[i, :, :this_k_length] = this_score[:, :, 0]
        score_tensor = score_tensor.reshape_as(triton_score_tensor)
        cos = _cos_of_tensors(score_tensor, triton_score_tensor)
        assert cos.item() > 0.9999, f'cosine similarity is {cos.item()}'


    attention_score_torch_test(
        q_tensor,
        k_cache,
        k_start,
        k_length,
        target_score.detach().clone().contiguous()
    )

    output_tensor = torch.zeros(bs, H, h).cuda()
    lightllm_attention(
        q_tensor,
        k_cache,
        v_cache,
        output_tensor,
        req_to_tokens,
        b_req_idx,
        b_start_loc,
        b_seqlen,
        max_length,
        H, h, D
    )
