import torch
from balance import balance, validate

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    max_length = 20
    bs = 10 
    lengths = torch.randint(1, max_length, (bs, ))
    batch_to_req, batch_to_block = balance(lengths)
    flag = validate(lengths, batch_to_req, batch_to_block)
    print(batch_to_block)
    print(lengths.sum(), bs, flag)
