import torch
import math
from bib_gqa_ref import bib_gqa_ref_qk_mul_length_mask, bib_gqa_ref_qk_mul_length_mask_fused

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

@torch.no_grad()
def bib_gqa_ref_qk_mul_base(
    q_tensor,
    k_tensor,
    score_tensor,
    block_to_request,
    block_to_start,
    block_to_length,
    block_to_chunk,
    chunk_size,
):
    block_to_request, block_to_start, block_to_length, block_to_chunk = \
        block_to_request.tolist(), block_to_start.tolist(), block_to_length.tolist(), block_to_chunk.tolist()
    for req, start, length, chunk in zip(block_to_request, block_to_start, block_to_length, block_to_chunk):
        q_vec = q_tensor[req]
        l_start = chunk * chunk_size
        l_end = min(l_start + chunk_size, length)
        for k_idx in range(l_start, l_end):
            k_vec = k_tensor[start + k_idx]
            score = torch.sum(q_vec * k_vec, dim=-1)
            score_tensor[req, :, k_idx] = score

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    bs = 4
    H = 8
    h = 8
    chunk_size = 16
    max_L = 4 * chunk_size
    D = bs * max_L

    q_shape = (bs, H, h)
    k_shape = (D, H, h)
    score_shape = (bs, H, max_L)
    
    q_tensor = torch.randn(q_shape, dtype=torch.float32).cuda()
    k_tensor = torch.randn(k_shape, dtype=torch.float32).cuda()
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
    
    block_to_length = torch.tensor(block_to_length, dtype=torch.int32).cuda()
    block_to_start = torch.tensor(block_to_start, dtype=torch.int32).cuda()
    block_to_chunk = torch.tensor(block_to_chunk, dtype=torch.int32).cuda()
    block_to_request = torch.tensor(block_to_request, dtype=torch.int32).cuda()

    bib_gqa_ref_qk_mul_base(
        q_tensor,
        k_tensor,
        score_tensor,
        block_to_request,
        block_to_start,
        block_to_length,
        block_to_chunk,
        chunk_size,
    )

    attention_score_torch_test(
        q_tensor,
        k_tensor,
        k_start,
        k_length,
        score_tensor.detach().clone()
    )

    bib_gqa_ref_qk_mul_length_mask(
        q_tensor,
        k_tensor,
        triton_score_tensor,
        block_to_request,
        block_to_start,
        block_to_length,
        block_to_chunk,
        chunk_size,
    )

    torch.cuda.synchronize()

    cos = _cos_of_tensors(score_tensor, triton_score_tensor)
    assert cos.item() > 0.9999, f'cosine similarity is {cos.item()}'

    triton_score_tensor = torch.zeros(score_shape, dtype=torch.float32).cuda()
    block_to_rslc = torch.stack([block_to_request, block_to_start, block_to_length, block_to_chunk], dim=1).cuda()

    bib_gqa_ref_qk_mul_length_mask_fused(
        q_tensor,
        k_tensor,
        triton_score_tensor,
        block_to_rslc,
        chunk_size,
    )

    torch.cuda.synchronize()

    cos = _cos_of_tensors(score_tensor, triton_score_tensor)
    assert cos.item() > 0.9999, f'cosine similarity is {cos.item()}'

    