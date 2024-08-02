#!/usr/bin/env python

import sys
import torch
import numpy as np
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

def parse():
    if len(sys.argv) < 5:
        print(f"Usage: {sys.argv[0]} batchsize seqlen headdim causal")
        sys.exit(1)

    batchsize = int(sys.argv[1])
    seqlen = int(sys.argv[2])
    headdim = int(sys.argv[3])
    causal = (sys.argv[4] == "True")
    return batchsize, seqlen, headdim, causal


if __name__ == "__main__":
    cuda = torch.device('cuda:0')
    dtype = torch.float16

    hiddendim = 2048

    # batchsize = 16
    # seqlen = 1024
    # headdim = 128
    batchsize, seqlen, headdim, causal = parse()

    nheads = hiddendim // headdim
    # if nheads is different for kv than q, it does multi-query or
    # grouped-query attention
    nheads_kv = nheads
    torch.random.manual_seed(0)
    q = torch.randn(batchsize, seqlen, nheads,    headdim, device=cuda, dtype=dtype, requires_grad=True)
    k = torch.randn(batchsize, seqlen, nheads_kv, headdim, device=cuda, dtype=dtype, requires_grad=True)
    v = torch.randn(batchsize, seqlen, nheads_kv, headdim, device=cuda, dtype=dtype, requires_grad=True)

    # forward-pass only
    print(f"executing flash_attn_func() with batchsize={batchsize}, seqlen={seqlen}, nheads={nheads}, headdim={headdim}, causal={causal}")
    repeats = 3
    for i in range(repeats):
        out = flash_attn_func(q, k, v, causal=causal)
    # print(out)
