import torch
import math

def balance(lengths):
    bs = lengths.shape[0]
    avg_length = math.ceil(lengths.sum() / bs)
    sorted_lengths, sorted_idx = torch.sort(lengths, descending=True)
    batch_to_req = torch.zeros(bs, avg_length, dtype=torch.int32) - 1
    batch_to_block = torch.zeros(bs, avg_length, dtype=torch.int32) - 1
    left_ptr = bs - 1
    left_start = 0

    for right_ptr in range(bs):
        if right_ptr > left_ptr:
            break
        if (right_ptr == left_ptr) and left_start > 0:
            break
        if sorted_lengths[right_ptr] >= avg_length:
            batch_to_req[sorted_idx[right_ptr]][:avg_length] = sorted_idx[right_ptr]
            batch_to_block[sorted_idx[right_ptr]][:avg_length] = torch.arange(0, avg_length)
            right_residual = sorted_lengths[right_ptr] - avg_length
            while right_residual > 0:
                if left_start == 0:
                    batch_to_req[sorted_idx[left_ptr]][:sorted_lengths[left_ptr]] = sorted_idx[left_ptr]
                    batch_to_block[sorted_idx[left_ptr]][:sorted_lengths[left_ptr]] = torch.arange(0, sorted_lengths[left_ptr])
                    left_start = sorted_lengths[left_ptr]
                if left_start < avg_length:
                    left_cap = avg_length - left_start
                    if left_cap > right_residual:
                        batch_to_req[sorted_idx[left_ptr]][left_start:left_start+right_residual] = sorted_idx[right_ptr]
                        batch_to_block[sorted_idx[left_ptr]][left_start:left_start+right_residual] = torch.arange(0, right_residual) + (sorted_lengths[right_ptr] - right_residual)
                        left_start += right_residual
                        right_residual = 0
                    else:
                        batch_to_req[sorted_idx[left_ptr]][left_start:] = sorted_idx[right_ptr]
                        batch_to_block[sorted_idx[left_ptr]][left_start:] = torch.arange(0, left_cap) + (sorted_lengths[right_ptr] - right_residual)
                        right_residual -= left_cap
                        left_ptr -= 1
                        left_start = 0
                else:
                    raise ValueError("")
        else:
            batch_to_req[sorted_idx[right_ptr]][:sorted_lengths[right_ptr]] = sorted_idx[right_ptr]
            batch_to_block[sorted_idx[right_ptr]][:sorted_lengths[right_ptr]] = torch.arange(0, sorted_lengths[right_ptr])
    return batch_to_req, batch_to_block

def validate(lengths, batch_to_req, batch_to_block):
    lengths, batch_to_req, batch_to_block = lengths.cpu(), batch_to_req.cpu(), batch_to_block.cpu()
    flag = True
    for idx, l in enumerate(lengths):
        num_cond = ((batch_to_req == idx).sum() == l).item()
        mask = batch_to_req == idx
        length_cond = ((batch_to_block[mask].sort()[0]) == torch.arange(l)).all()
        flag = flag and num_cond
        flag = flag and length_cond
    return flag
