import math
import os
from functools import partial

import numpy as np
import torch
import triton

from bib_gqa_decoding import bib_gqa_decoding
from gqa_decoding import gqa_token_decode_attention_flash_decoding

BENCH_TERM = ['gqa', 'bib', ]
bib_gqa_fns = dict(gqa=gqa_token_decode_attention_flash_decoding, bib=bib_gqa_decoding,)
                   # bib_stage1=bib_gqa_decoding_stage1)

ARGS_TERMS = {
    'batch': int,
    'length': int,
    'mean': float,
    'std': float,
    'outlier': float,
    'chunk': int,
    'G': int,
    'H': int,
    'h': int
}


def prepare_gqa(min_length, length, batch, G, H, h, chunk, mean, std, outlier):
    assert H % G == 0
    k_norm_length = torch.randn(size=(batch,)) * std + mean
    k_length = (k_norm_length.clamp(0, 1) * length).round().to(torch.int32).clamp(min_length, length)
    num_outlier = math.ceil(outlier * batch)
    idx_outlier = torch.multinomial(torch.ones(batch) / batch, num_outlier, replacement=False)
    for idx in idx_outlier:
        k_length[idx] = length
    D = 2 ** k_length.sum().log2().ceil().to(torch.int32)
    k_start = k_length.cumsum(dim=0)
    k_start = torch.cat([torch.tensor([0]), k_start[:-1]]).contiguous()
    q_tensor = torch.randn(batch, H, h, dtype=torch.float32)
    k_cache = torch.randn(D, H // G, h, dtype=torch.float32)
    v_cache = torch.randn(D, H // G, h, dtype=torch.float32)
    length = k_length.max().item()
    output_tensor = torch.ones_like(q_tensor)
    req_to_tokens = torch.zeros(batch, length, dtype=torch.int32)
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

    BLOCK_SEQ = chunk
    mid_o = torch.empty([batch, H, length // BLOCK_SEQ + 1, h], dtype=torch.float32,
                        device="cuda")
    mid_o_logexpsum = torch.empty([batch, H, length // BLOCK_SEQ + 1], dtype=torch.float32,
                                  device="cuda")
    return q_tensor, k_cache, v_cache, output_tensor, mid_o, mid_o_logexpsum, b_seqlen, req_to_tokens, b_req_idx, batch, length, H, h


def prepare_bib_gqa(min_length, length, batch, G, H, h, chunk, mean, std, outlier):
    D = batch * length
    q_shape = (batch, H, h)
    k_shape = (D, H // G, h)
    score_shape = (batch, H, h)

    q_tensor = torch.randn(q_shape, dtype=torch.float32).cuda()
    k_tensor = torch.randn(k_shape, dtype=torch.float32).cuda()
    v_tensor = torch.randn(k_shape, dtype=torch.float32).cuda()
    score_tensor = torch.zeros(score_shape, dtype=torch.float32).cuda()
    triton_score_tensor = torch.zeros(score_shape, dtype=torch.float32).cuda()

    k_length = torch.randint(2, length, (batch,), dtype=torch.int32).cuda()
    k_start = k_length.cumsum(dim=0)
    k_start = torch.cat([torch.tensor([0]).cuda(), k_start[:-1]]).contiguous()

    block_to_length = []
    block_to_request = []
    block_to_start = []
    block_to_chunk = []
    for req_idx, (leng, start) in enumerate(zip(k_length, k_start)):
        for chunk_idx in range(math.ceil(leng / chunk)):
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
    for req_idx in range(batch):
        request_to_block.append(r2b_dict[req_idx] + [-1] * (max_leng_r2b - len(r2b_dict[req_idx])))
    request_to_block = torch.tensor(request_to_block, dtype=torch.int32).cuda()
    block_to_length = torch.tensor(block_to_length, dtype=torch.int32).cuda()
    block_to_start = torch.tensor(block_to_start, dtype=torch.int32).cuda()
    block_to_chunk = torch.tensor(block_to_chunk, dtype=torch.int32).cuda()
    block_to_request = torch.tensor(block_to_request, dtype=torch.int32).cuda()
    block_num = block_to_request.shape[0]
    deno_tensor = torch.zeros((block_num, H, 2), dtype=torch.float32, device=q_tensor.device)
    nume_tensor = torch.zeros((block_num, H, h), dtype=torch.float32, device=q_tensor.device)

    args = q_tensor, k_tensor, v_tensor, score_tensor, request_to_block, block_to_request, block_to_start, block_to_length, block_to_chunk, 1 / math.sqrt(
        H), deno_tensor, nume_tensor, G, chunk
    return args


def parse_args(args=None):
    import argparse
    parser = argparse.ArgumentParser(description='Argument groups example')

    # bs, len, mean, std, chunk, H, h

    parser.add_argument('mode', choices=list(ARGS_TERMS.keys()))
    for _t in ARGS_TERMS:
        parser.add_argument(f'--{_t}', type=float, default=None)
        parser.add_argument(f'--{_t}_start', type=float)
        parser.add_argument(f'--{_t}_end', type=float)
        parser.add_argument(f'--{_t}_num', type=int, default=10)
        parser.add_argument(f'--{_t}_pot', action='store_true')
        parser.add_argument(f'--{_t}_log10', action='store_true')
    args = parser.parse_args(args)
    for _t, _type in ARGS_TERMS.items():
        _d = getattr(args, _t, None)
        if not _d: continue
        setattr(args, _t, _type(_d))

    return args


def _get_split(args):
    sweep = []
    term = args.mode
    start = getattr(args, f'{term}_start')
    end = getattr(args, f'{term}_end')
    num = getattr(args, f'{term}_num')
    pot = getattr(args, f'{term}_pot')
    log10 = getattr(args, f'{term}_log10')
    if log10:
        start = 10 ** start
        end = 10 ** end
    split = np.linspace(start, end, num)
    if pot:
        split = 2 ** (np.ceil(np.log2(split - 1)))
        split = split.astype(np.int32)
    dtype = ARGS_TERMS[term]
    split = split.tolist()
    split = [dtype(s) for s in split]
    uniq = []
    for s in split:
        if s not in uniq:
            uniq.append(s)
    return uniq


def _get_name_and_fn(args, split, base):
    term = args.mode
    start = getattr(args, f'{term}_start')
    end = getattr(args, f'{term}_end')
    num = getattr(args, f'{term}_num')
    log10 = getattr(args, f'{term}_log10')
    dtype = ARGS_TERMS[term]
    start, end = dtype(start), dtype(end)
    sweep = f'sw_{term}_{start}:{end}:{num}'

    other = ''
    for t in ARGS_TERMS:
        if t != args.mode:
            dtype = ARGS_TERMS[t]
            other += f'_{t}_{dtype(getattr(args, t))}'
    name = sweep + other

    def _bench(args):
        @triton.testing.perf_report(
            triton.testing.Benchmark(
                x_names=['val'],
                x_vals=split,
                x_log=log10,
                line_arg='provider',
                line_vals=BENCH_TERM,
                line_names=BENCH_TERM,
                ylabel='latency',
                plot_name=f'{term}',
                args={"args": args},
            ))
        def benchmark(val, provider, args):
            print(f'benching {term} = {val}, provide = {provider}')
            quantiles = [0.5, 0.2, 0.8]
            if provider.startswith('bib'):
                prepare_fn = prepare_bib_gqa
                call_fn = bib_gqa_fns[provider]
            elif provider.startswith('gqa'):
                prepare_fn = prepare_gqa
                call_fn = gqa_token_decode_attention_flash_decoding
            else:
                raise NotImplementedError
            kwargs = {}
            for t in ARGS_TERMS:
                d = getattr(args, t)
                if d is None:
                    assert t == term
                kwargs[t] = d
            kwargs[term] = val
            args = prepare_fn(10, **kwargs)
            args = [arg.cuda().contiguous() if isinstance(arg, torch.Tensor) else arg for arg in args]
            p50, p20, p80 = triton.testing.do_bench(lambda: call_fn(*args), quantiles=quantiles)
            return p50, p20, p80

        return benchmark

    bm = _bench(args)
    if base is not None:
        name = os.path.join(base, name)
    os.makedirs(name, exist_ok=True)
    bm.run(show_plots=False, print_data=False, save_path=name)


def main(args=None, base=None):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    args = parse_args(args)
    split = _get_split(args)
    _get_name_and_fn(args, split, base)


if __name__ == '__main__':
    main()
