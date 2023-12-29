import os
import torch
import triton

import math
from lightllm_score import token_att_fwd
from bib_cont_score import attention_score_lazy_balance
from balance import validate as validate_param
from bib_gqa_ref import bib_gqa_ref_qk_mul, bib_gqa_ref_qk_mul_remove_block, bib_gqa_ref_qk_mul_remove_k_load, bib_gqa_ref_qk_mul_length_mask, bib_gqa_ref_qk_mul_length_mask_fused


def prepare_light(min_length, max_length, batch, head_num, head_dim, chunk_size, mean_ratio, std_ratio, outlier_ratio):
    k_norm_length = torch.randn(size=(batch,)) * std_ratio + mean_ratio
    k_length = (k_norm_length.clamp(0, 1) * max_length).round().to(torch.int32).clamp(min_length, max_length)
    num_outlier = math.ceil(outlier_ratio * batch)
    idx_outlier = torch.multinomial(torch.ones(batch) / batch, num_outlier, replacement=False)
    for idx in idx_outlier:
        k_length[idx] = max_length
    D = 2 ** k_length.sum().log2().ceil().to(torch.int32)
    k_start = k_length.cumsum(dim=0)
    k_start = torch.cat([torch.tensor([0]), k_start[:-1]]).contiguous()
    q_tensor = torch.randn(batch, head_num, head_dim, dtype=torch.float32)
    k_cache = torch.randn(D, head_num, head_dim, dtype=torch.float32)
    max_length = k_length.max().item()
    chunk_num = math.floor((max_length + chunk_size - 1) / chunk_size)
    score_tensor = torch.zeros(head_num, D)
    req_to_tokens = torch.zeros(batch, max_length, dtype=torch.int32)
    for i in range(batch):
        req_to_tokens[i][:k_length[i]] = torch.arange(0, k_length[i]) + k_start[i]
    b_req_idx = torch.zeros(batch, dtype=torch.int32)
    for i in range(batch):
        b_req_idx[i] = i
    b_start_loc = torch.zeros(batch, dtype=torch.int32)
    for i in range(batch):
        b_start_loc[i] = k_start[i]
    b_seqlen = torch.zeros(batch, dtype=torch.int32)
    for i in range(batch):
        b_seqlen[i] = k_length[i]
    return q_tensor, k_cache, score_tensor, req_to_tokens, b_req_idx, b_start_loc, b_seqlen, max_length



def prepare_bib(min_length, max_length, batch, head_num, head_dim, chunk_size, mean_ratio, std_ratio, outlier_ratio):
    k_norm_length = torch.randn(size=(batch,)) * std_ratio + mean_ratio
    k_length = (k_norm_length.clamp(0, 1) * max_length).round().to(torch.int32).clamp(min_length, max_length)
    num_outlier = math.ceil(outlier_ratio * batch)
    idx_outlier = torch.multinomial(torch.ones(batch) / batch, num_outlier, replacement=False)
    for idx in idx_outlier:
        k_length[idx] = max_length
    D = 2 ** k_length.sum().log2().ceil().to(torch.int32)
    k_start = k_length.cumsum(dim=0)
    k_start = torch.cat([torch.tensor([0]), k_start[:-1]]).contiguous()
    q_tensor = torch.randn(batch, head_num, head_dim, dtype=torch.float32)
    k_cache = torch.randn(D, head_num, head_dim, dtype=torch.float32)
    max_length = k_length.max().item()
    chunk_num = math.floor((max_length + chunk_size - 1) / chunk_size)
    score_tensor = torch.zeros(batch, head_num, chunk_num, chunk_size)
    chunk_to_block, chunk_to_req = None, None
    args = q_tensor, k_cache, k_start, k_length, score_tensor, chunk_size, {}, chunk_to_block, chunk_to_req
    args = [arg.cuda().contiguous() if isinstance(arg, torch.Tensor) else arg for arg in args]
    chunk_to_block, chunk_to_req = attention_score_lazy_balance(*args)
    assert validate_param(torch.ceil(k_length.float() / chunk_size).to(torch.int32), chunk_to_req, chunk_to_block)
    args = q_tensor, k_cache, k_start, k_length, score_tensor, chunk_size, {}, chunk_to_block, chunk_to_req
    return args

def prepare_bib_gqa(min_length, max_length, batch, head_num, head_dim, chunk_size, mean_ratio, std_ratio, outlier_ratio):
    k_norm_length = torch.randn(size=(batch,)) * std_ratio + mean_ratio
    k_length = (k_norm_length.clamp(0, 1) * max_length).round().to(torch.int32).clamp(min_length, max_length)
    num_outlier = math.ceil(outlier_ratio * batch)
    idx_outlier = torch.multinomial(torch.ones(batch) / batch, num_outlier, replacement=False)
    for idx in idx_outlier:
        k_length[idx] = max_length
    D = 2 ** k_length.sum().log2().ceil().to(torch.int32)
    k_start = k_length.cumsum(dim=0)
    k_start = torch.cat([torch.tensor([0]), k_start[:-1]]).contiguous()
    max_length = k_length.max().item()

    q_tensor = torch.randn(batch, head_num, head_dim, dtype=torch.float32)
    k_cache = torch.randn(D, head_num, head_dim, dtype=torch.float32)
    score_tensor = torch.zeros(batch, head_num, max_length, dtype=torch.float32)

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

    args = q_tensor, k_cache, score_tensor, block_to_request, block_to_start, block_to_length, block_to_chunk, chunk_size
    return args


def prepare_bib_gqa_fused(min_length, max_length, batch, head_num, head_dim, chunk_size, mean_ratio, std_ratio, outlier_ratio):
    k_norm_length = torch.randn(size=(batch,)) * std_ratio + mean_ratio
    k_length = (k_norm_length.clamp(0, 1) * max_length).round().to(torch.int32).clamp(min_length, max_length)
    num_outlier = math.ceil(outlier_ratio * batch)
    idx_outlier = torch.multinomial(torch.ones(batch) / batch, num_outlier, replacement=False)
    for idx in idx_outlier:
        k_length[idx] = max_length
    D = 2 ** k_length.sum().log2().ceil().to(torch.int32)
    k_start = k_length.cumsum(dim=0)
    k_start = torch.cat([torch.tensor([0]), k_start[:-1]]).contiguous()
    max_length = k_length.max().item()

    q_tensor = torch.randn(batch, head_num, head_dim, dtype=torch.float32)
    k_cache = torch.randn(D, head_num, head_dim, dtype=torch.float32)
    score_tensor = torch.zeros(batch, head_num, max_length, dtype=torch.float32)

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
    block_to_rslc = torch.cat([block_to_request.unsqueeze(1), block_to_start.unsqueeze(1), block_to_length.unsqueeze(1), block_to_chunk.unsqueeze(1)], dim=1).cuda()

    args = q_tensor, k_cache, score_tensor, block_to_rslc, chunk_size
    return args

BENCH_TERM = ['lightllm', 'gqa-len-mask', 'gqa-fused']

def bench(max_length, head_num, head_dim, chunk_size, mean_ratio=0.3, std_ratio=0.3, outlier_ratio=0.01, min_length=10, save_prefix='.'):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['batch'],  
            x_vals=[2**i for i in range(1, 9, 1)],  
            x_log=True,  
            line_arg='provider',  
            line_vals=BENCH_TERM,  
            line_names=BENCH_TERM,
            # styles=[('blue', '-'), ('red', '-')],  
            # line_vals=['lightllm', 'bib', 'gqa'],  
            # line_names=['lightllm', 'bib', 'gqa'],  
            # styles=[('blue', '-'), ('green', '-'), ('red', '-')],  
            ylabel='latency',  
            plot_name=f'attn-C{chunk_size}',  
            args={},  
        ))
    def benchmark(batch, provider):
        print(f'benching batch = {batch}, provide = {provider}')
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'lightllm':
            args = prepare_light(min_length, max_length, batch, head_num, head_dim, chunk_size, mean_ratio, std_ratio, outlier_ratio)
            args = [arg.cuda().contiguous() if isinstance(arg, torch.Tensor) else arg for arg in args]
            p50, p20, p80 = triton.testing.do_bench(lambda: token_att_fwd(*args), quantiles=quantiles)
        elif provider == 'bib':
            args = prepare_bib(min_length, max_length, batch, head_num, head_dim, chunk_size, mean_ratio, std_ratio, outlier_ratio)
            args = [arg.cuda().contiguous() if isinstance(arg, torch.Tensor) else arg for arg in args]
            p50, p20, p80 = triton.testing.do_bench(lambda: attention_score_lazy_balance(*args), quantiles=quantiles)
        elif 'gqa' in provider:
            preparefn = prepare_bib_gqa
            if provider == 'gqa':
                callfn = bib_gqa_ref_qk_mul
            elif provider == 'gqa-remove-k':
                callfn = bib_gqa_ref_qk_mul_remove_k_load
            elif provider == 'gqa-remove-block':
                callfn = bib_gqa_ref_qk_mul_remove_block
            elif provider == 'gqa-len-mask':
                callfn = bib_gqa_ref_qk_mul_length_mask
            elif provider == 'gqa-fused':
                preparefn = prepare_bib_gqa_fused
                callfn = bib_gqa_ref_qk_mul_length_mask_fused
            else:
                raise NotImplementedError
            args = preparefn(min_length, max_length, batch, head_num, head_dim, chunk_size, mean_ratio, std_ratio, outlier_ratio)
            args = [arg.cuda().contiguous() if isinstance(arg, torch.Tensor) else arg for arg in args]
            p50, p20, p80 = triton.testing.do_bench(lambda: callfn(*args), quantiles=quantiles)
        return p50, p20, p80
    save = f'{save_prefix}/max-{max_length}-H-{head_num}-h-{head_dim}-mean-{mean_ratio}-std-{std_ratio}-out-{outlier_ratio}/'
    os.makedirs(save, exist_ok=True)
    benchmark.run(show_plots=False, print_data=False, save_path=save)
    print(f'save to {save}')

if __name__ == '__main__':
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    bench(3072, 32, 32, 256, mean_ratio=0.1)
    bench(3072, 32, 32, 256, mean_ratio=0.3)
    bench(3072, 32, 32, 256, mean_ratio=0.5)