import torch
import math

from bib_cont_score import attention_score, attention_score_torch_test

if __name__ == "__main__":
    bs = 4
    H = 8
    h = 8
    D = 24 * 1000
    chunk_size = 16
    max_L = 4 * chunk_size
    torch.cuda.manual_seed_all(0)
    torch.manual_seed(0)
    k_length = torch.randint(2, max_L, (bs,), dtype=torch.int32).cuda()
    k_start = k_length.cumsum(dim=0)
    k_start = torch.cat([torch.tensor([0]).cuda(), k_start[:-1]]).contiguous()
    q_tensor = torch.randn(bs, H, h, dtype=torch.float32).cuda()
    k_cache = torch.randn(D, H, h, dtype=torch.float32).cuda()
    max_length = k_length.max().item()
    chunk_num = math.floor((max_length + chunk_size - 1) / chunk_size)
    score_tensor = torch.zeros(bs, H, chunk_num, chunk_size).cuda()
    
    def _check(*args):
        for idx, arg in enumerate(args):
            assert arg.is_contiguous(), f'args {idx} is not contiguous'
    _check(
        q_tensor,
        k_cache,
        k_start,
        k_length,
        score_tensor
    )

    attention_score(
        q_tensor,
        k_cache,
        k_start,
        k_length,
        score_tensor,
        chunk_size=score_tensor.shape[-1],
        debug_dict=dict(
            max_length=max_length,
            chunk_num=chunk_num,
            chunk_size=chunk_size,
            bs=bs,
        )
    )
    torch.cuda.synchronize()
    attention_score_torch_test(
        q_tensor,
        k_cache,
        k_start,
        k_length,
        score_tensor.detach().clone()
    )