#https://github.com/SmerkyG/gptcore/blob/main/model/experimental/rwkv_inner.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

# 32 is optimal chunk length (longer will use too much memory, shorter is inefficient)
def rwkv_inner(r,k,v,w,u,kv_state,chunk_len=32):
    """
    expects
    r : (B,H,L,K)
    k : (B,H,L,K)
    v : (B,H,L,V)
    w : (B,H,L,K) or (1,H,L,K)
    u : (1,H,1,K)
    kv_state : (B,H,K,V)
    """
    B,H,L,K = k.size()
    V = v.size(-1)
    T = chunk_len

    if L == 1:
        kv = k @ v
        out = r @ (kv_state + u * kv)
        kv_state = w * kv_state + kv
        return out, kv_state
    else:
        # FIXME - support fast path for non-exact multiples
        # ensure it's an exact multiple
        if L % T != 0:
            T = 1

        N = L // T

        # this has to be done to avoid numerical instability (inf/NaN) when w is used as a divisor up to chunk_length//2 places away (so precision_min_val^(T//2) has to be in fp range)
        precision_dtype, precision_min_val = torch.float32, 0.02 # good for fp32 
        #precision_dtype, precision_min_val = torch.float64, 1e-10 # good for fp64
        
        w = w.clamp(precision_min_val)

        # calculate cumulative decay in log space where it won't overflow
        w_log = w.float().log() # (1,H,L,K) or (B,H,L,K)

        # chunked view of w_log
        wc_log = w_log.view(w.size(0),H,N,T,K)
        wc_log_cum = wc_log.cumsum(dim=-2)

        # chunked view of shifted_w_log
        shifted_wc_log_cum = torch.cat([torch.zeros_like(wc_log_cum[:,:,:,:1,:]), wc_log_cum[:,:,:,:-1,:]], dim=-2)


        # NOTE - we have to apply the decay weight from TWO ahead.. ONE ahead gets no decay (log==0)
        # pre-applied weights
        # left side is prior chunk (w_inter), right side is current chunk (w_intra)
        # without u...
        # w0   w1   w2   w3   | w4   w5   w6   w7          
        # w1:4 w2:4 w3:4 w4:4 | w4:5 w4:6 w4:7 w4:8
        # with u...
        # w0   w1   w2   w3   | w4   w5   w6   w7          
        # w1:4 w2:4 w3:4 w4:4 | w4:4 w4:5 w4:6 w4:7

        # ws decays the entire current state (representing t-1) to the prior block (t-2)
        ws = wc_log.sum(dim=-2, keepdim=True) # 1HN1K or BHN1K
        # w_inter is the decay to the end of the current block, since it will be applied at the next iteration when current (t) becomes prior (t-1)
        # this formula because e.g. w1:4 = w0:4 - w0:1
        w_inter = ws - wc_log_cum # 1HNTK or BHNTK (w^(T-1) ... w^0)
        # w_intra is the decay from the beginning of the current block (t), since it will be applied to current queries (t) against prior state (representing keys+values up to but not including block t)
        # this formula because e.g. w1:3 = w0:3 - w0
        w_intra = wc_log_cum - wc_log # 1HNTK or BHNTK (w^0 ... w^(T-2))

        ws = list(ws.mT.exp().to(r.dtype).unbind(dim=-3)) # N x 1HK1 or BHK1 !!NOTE THE .mT HERE!!
        w_inter = w_inter.exp().to(r.dtype) # 1HNTK or BHNTK
        w_intra = w_intra.exp().to(r.dtype) # 1HNTK or BHNTK

        # chunked view of r, k, v
        r = r.view(B,H,N,T,K) 
        k = k.view(B,H,N,T,K) 
        v = v.view(B,H,N,T,V)
        u = u.unsqueeze(2).to(r.dtype) # (1,H,1,1,K)

        # parallel calculation of all intra-chunk attention contributions
        wc_log_offset = shifted_wc_log_cum[:,:,:,T//2:T//2+1,:] # B,H,N,1,K
        r_decay = (shifted_wc_log_cum - wc_log_offset).to(precision_dtype).exp() # B,H,N,T,K
        k_inv_decay = (wc_log_offset - wc_log_cum).to(precision_dtype).exp() # B,H,N,T,K
        a = ((r*r_decay) @ (k*k_inv_decay).mT).to(r.dtype).tril(-1) # B,H,N,T,T
        # add u term to attention (NOTE - the tril(-1) above zeroed the diagonal)
        a = a + torch.einsum('bhntk,bhntk->bhnt', r, u * k).diag_embed()
        out = a @ v # BHNTV
        # alternate way of adding in u
        #outc = outc + torch.einsum('bhntk,bhntk,bhntv->bhntv', rc, u.unsqueeze(-2) * kc, vc) 

        # parallel precalculation of chunked (k*wk).mT@v for use in recurrent state calc below
        wkv = (k * w_inter).mT @ v # BHNKV
        wkv = list(wkv.unbind(dim=-3)) # N x BHKV

        # recurrent calculation of all states
        states = []
        for i in range(N):
            states.append(kv_state)
            kv_state = kv_state * ws[i] + wkv[i] # BHKV
            # equivalent non-precalced version
            #wkv = (k[...,i,:,:] * wk[...,i,:,:]).mT @ v[...,i,:,:]
            #kv_state = kv_state * ws[i] + wkv
        states = torch.stack(states, dim=2) # BHNKV       

        # parallel application of all r to states
        out = out + (r * w_intra) @ states # BHNTV
        out = out.view(B,H,L,V)
        return out, kv_state
            
