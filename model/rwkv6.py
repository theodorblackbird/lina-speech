from dataclasses import dataclass
from typing import Optional
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.attentive_rnn import AttentiveRNN
from model.crossatt import BlindCrossAttention, CrossAttention
from model.base_blocks import MixingBlock, SwiGLU
from model.rwkv_inner import rwkv_inner
from einops import rearrange
from fla.ops.gla import fused_chunk_gla

from torch.utils.cpp_extension import load


#adapted from https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v5/src/model.py

def __noop(ob):
    return ob


MyFunction = __noop


class WKV_6(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, H, r, k, v, w, u):
        with torch.no_grad():
            assert r.dtype == torch.bfloat16
            assert k.dtype == torch.bfloat16
            assert v.dtype == torch.bfloat16
            assert w.dtype == torch.bfloat16
            assert u.dtype == torch.bfloat16
            assert HEAD_SIZE == C // H
            ctx.B = B
            ctx.T = T
            ctx.C = C
            ctx.H = H
            assert r.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert w.is_contiguous()
            assert u.is_contiguous()
            ew = (-torch.exp(w.float())).contiguous()
            ctx.save_for_backward(r, k, v, ew, u)
            y = torch.empty(
                (B, T, C),
                device=r.device,
                dtype=torch.bfloat16,
                memory_format=torch.contiguous_format,
            )  # .uniform_(-100, 100)
            wkv6_cuda.forward(B, T, C, H, r, k, v, ew, u, y)
            return y

    @staticmethod
    def backward(ctx, gy):
        with torch.no_grad():
            assert gy.dtype == torch.bfloat16
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            assert gy.is_contiguous()
            r, k, v, ew, u = ctx.saved_tensors
            gr = torch.empty(
                (B, T, C),
                device=gy.device,
                requires_grad=False,
                dtype=torch.bfloat16,
                memory_format=torch.contiguous_format,
            )  # .uniform_(-100, 100)
            gk = torch.empty(
                (B, T, C),
                device=gy.device,
                requires_grad=False,
                dtype=torch.bfloat16,
                memory_format=torch.contiguous_format,
            )  # .uniform_(-100, 100)
            gv = torch.empty(
                (B, T, C),
                device=gy.device,
                requires_grad=False,
                dtype=torch.bfloat16,
                memory_format=torch.contiguous_format,
            )  # .uniform_(-100, 100)
            gw = torch.empty(
                (B, T, C),
                device=gy.device,
                requires_grad=False,
                dtype=torch.bfloat16,
                memory_format=torch.contiguous_format,
            )  # .uniform_(-100, 100)
            gu = torch.empty(
                (B, C),
                device=gy.device,
                requires_grad=False,
                dtype=torch.bfloat16,
                memory_format=torch.contiguous_format,
            )  # .uniform_(-100, 100)
            wkv6_cuda.backward(B, T, C, H, r, k, v, ew, u, gy, gr, gk, gv, gw, gu)
            gu = torch.sum(gu, 0).view(H, C // H)
            return (None, None, None, None, gr, gk, gv, gw, gu)


def RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u):
    return WKV_6.apply(B, T, C, H, r, k, v, w, u)


class RWKV_TMix_x060(nn.Module):
    def __init__(self, d_model, n_head, layer_id, n_layer, head_size_divisor: int=8):
        super().__init__()
        self.layer_id = layer_id

        self.head_size = d_model // n_head
        self.head_size_divisor = head_size_divisor
        self.n_head = n_head
        assert d_model % self.n_head == 0
        global wkv6_cuda
        wkv6_cuda = load(
            name="wkv6",
            sources=["cuda/wkv6_op.cpp", f"cuda/wkv6_cuda.cu"],
            verbose=True,
            extra_cuda_cflags=[
                "-res-usage",
                "--use_fast_math",
                "-O3",
                "-Xptxas -O3",
                "--extra-device-vectorization",
                f"-D_N_={self.head_size}",
                f"-D_T_={int(os.environ['RWKV_CTXLEN'])}",
            ],
        )
        with torch.no_grad():
            ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, d_model)
            for i in range(d_model):
                ddd[0, 0, i] = i / d_model

            # fancy time_mix
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(
                1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            )
            self.time_maa_r = nn.Parameter(
                1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0)
            )
            self.time_maa_g = nn.Parameter(
                1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0)
            )

            TIME_MIX_EXTRA_DIM = 32  # generate TIME_MIX for w,k,v,r,g
            self.time_maa_w1 = nn.Parameter(
                torch.zeros(d_model, TIME_MIX_EXTRA_DIM * 5)
            )
            self.time_maa_w2 = nn.Parameter(
                torch.zeros(5, TIME_MIX_EXTRA_DIM, d_model).uniform_(-0.01, 0.01)
            )

            # fancy time_decay
            decay_speed = torch.ones(d_model)
            for n in range(d_model):
                decay_speed[n] = -6 + 5 * (n / (d_model - 1)) ** (
                    0.7 + 1.3 * ratio_0_to_1
                )
            self.time_decay = nn.Parameter(decay_speed.reshape(1, 1, d_model))

            TIME_DECAY_EXTRA_DIM = 64
            self.time_decay_w1 = nn.Parameter(
                torch.zeros(d_model, TIME_DECAY_EXTRA_DIM)
            )
            self.time_decay_w2 = nn.Parameter(
                torch.zeros(TIME_DECAY_EXTRA_DIM, d_model).uniform_(-0.01, 0.01)
            )

            tmp = torch.zeros(d_model)
            for n in range(d_model):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (d_model - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)

        self.value = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model, bias=False)
        self.ln_x = nn.GroupNorm(
            self.n_head, d_model, eps=(1e-5) * (self.head_size_divisor**2)
        )

    def wkv(self, r, k, v, w):
            w = torch.exp(-torch.exp(w.double()))

            B,T,C = r.size()
            if self.kv_state is None:
                    self.kv_state = torch.zeros(B, self.n_head, self.head_size, self.head_size, device=r.device).double()
            
            #int64_t B, int64_t T, int64_t K, int64_t H, torch::Tensor &state, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &y
            
            # xr = (( v ).view(B,T,self.n_head,self.head_size)*(k*r*self.time_faaaa.view(1,1,C)).view(B,T,self.n_head,self.head_size).sum(-1,True)).view(B,T,C)

            # torch.ops.wkv6f.forward(B,T,self.head_size,self.n_head,self.kv_state,r,k,v,w,xr)
            # return xr
            B,T,C = r.size()
            K = self.head_size
            H = self.n_head
            
            k = k.view(B,T,H,K).transpose(0,1).reshape(T,B,H,1,K).double()
            r = r.view(B,T,H,K).transpose(0,1).reshape(T,B,H,1,K).double()
            v = v.view(B,T,H,K).transpose(0,1).reshape(T,B,H,1,K).double()
            w = w.view(B,T,H,K).transpose(0,1).reshape(T,B,H,1,K).to(r.dtype)
            
         
            
            # xr = (( v )*(k*r.view(T,B,H,1,K)*self.time_faaaa.view(1,1,H,1,K)).sum(-1,True)).reshape(-1,T,K)

            kk = k.reshape(T,-1,K).transpose(0,1).transpose(1,2)
            vv = v.reshape(T,-1,K).transpose(0,1)
            
            r = r.reshape(H*B,T,-1)
            xr = torch.zeros(B*H,T,K,device=r.device).double()
            for i in range(T):
                
                xr[:,i] = torch.bmm(r[:,i:i+1],self.kv_state[:B].view(B,-1,K,K).add((v[i].view(B,-1,1,K).mul(k[i].view(B,-1,K,1)).mul(self.time_faaaa.view(1,-1,K,1)))).view(-1,K,K)).view(B*H,K)
                
                # calculate the effects time has on the state
                torch.mul(self.kv_state[:B].view(-1,H,K,K),w[i].view(B,H,K,1),out=self.kv_state[:B].view(-1,H,K,K))
                
                #calculate the effects k,v have on the state
                torch.baddbmm(self.kv_state[:B].view(-1,K,K),kk[:,:,i:i+1],vv[:,i:i+1],out=self.kv_state[:B].view(-1,K,K))
            
            
            return xr.view(T,B,H,K).transpose(0,1).reshape(B,T,H*K).float()


    @MyFunction
    def jit_func(self, x):
        B, T, C = x.size()

        if self.inference:
            state = self.state
            if state is None:
                state = torch.zeros_like(x)
        else:
            state = self.time_shift(x)

        xx = state - x

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B * T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)
        xg = x + xx * (self.time_maa_g + mg)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = self.time_decay + ww
        
        if self.inference:
            self.state = x

        return r, k, v, g, w

    @MyFunction
    def jit_func_2(self, x, g):
        B, T, C = x.size()
        x = x.view(B * T, C)

        x = self.ln_x(x).view(B, T, C)
        x = self.output(x * g)
        return x

    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head

        r, k, v, g, w = self.jit_func(x)
        if self.inference:
            x = self.wkv(r, k, v, w)
        else:
            x = RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w.bfloat16(), u=self.time_faaaa.bfloat16())

        return self.jit_func_2(x, g)


class RWKV_CMix_x060(nn.Module):
    def __init__(self, d_model, layer_id, n_layer):
        super().__init__()
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        dim_ffn = int((d_model * 3.5) // 32 * 32)

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, d_model)
            for i in range(d_model):
                ddd[0, 0, i] = i / d_model
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(d_model, dim_ffn, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(dim_ffn, d_model, bias=False)
        self.inference = False
        self.state = None

    @MyFunction
    def forward(self, x):
        if self.inference:
            state = self.state
            if state is None:
                state = torch.zeros_like(x)
        else:
            state = self.time_shift(x)

        xx =  state - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        if self.inference:
            self.state = x
        return torch.sigmoid(self.receptance(xr)) * kv



def exists(x):
    return x is not None

class AttentiveRWKV6(AttentiveRNN):
    def __init__(
        self,
        d_model,
        n_layer,
        d_context,
        heads,
        dropout_att=0.0,
        d_blind=128,
        blind=False,
        light=False,
    ):
        super().__init__()
        block = lambda dim, heads, lay_i, n_lay: MixingBlock(lambda: RWKV_TMix_x060(dim, heads, lay_i, n_lay),
                                                   lambda: SwiGLU(dim) if light else RWKV_CMix_x060(dim, lay_i, n_lay),
                                                   lambda: nn.LayerNorm(dim),)
        self.encoder = nn.Sequential(*[block(d_model, heads, i, n_layer) for i in range(n_layer)])
        self.decoder = nn.Sequential(*[block(d_model, heads, i, n_layer) for i in range(n_layer)])
        self.cross_att = (
            BlindCrossAttention(
                d_model, d_context, d_model, heads, block(d_blind, 1, 0, 2), dropout_att, pos_dim=d_blind,
            )
            if blind
            else CrossAttention(d_model, d_context, d_model, heads, dropout_att)
        )

    def forward(self, x, ctx, x_mask=None, ctx_mask=None):
        if exists(x_mask) and exists(ctx_mask):
            mask = rearrange(x_mask, "b i -> b i 1") * rearrange(
                ctx_mask, "b j -> b 1 j"
            )
            mask = rearrange(mask, "b i j -> b 1 i j")
        else:
            mask = None
        y = torch.utils.checkpoint.checkpoint_sequential(self.encoder, len(self.encoder), x, use_reentrant=False)
        v, att = self.cross_att(y, ctx, mask=mask)
        y = torch.utils.checkpoint.checkpoint_sequential(self.decoder, len(self.decoder), y + v, use_reentrant=False)
 
        return y, att

    def init_state(self, max_seqlen=1000):
        for be, bd in zip(self.encoder, self.decoder):
            be.tmix.inference, bd.tmix.inference = True, True
            be.cmix.inference, bd.cmix.inference = True, True
            be.tmix.state, bd.tmix.state = None, None
            be.cmix.state, bd.cmix.state = None, None
            be.tmix.kv_state, bd.tmix.kv_state = None, None
        self.cross_att.pos_net.tmix.inference = True
        self.cross_att.pos_net.cmix.inference = True
        self.cross_att.pos_net.tmix.state = None
        self.cross_att.pos_net.cmix.state = None
        self.cross_att.pos_net.tmix.kv_state = None


    def step(self, y_embd, x_enc, time_step):
        y_embd = self.encoder(y_embd)
        v, att = self.cross_att(y_embd, x_enc, time_step=time_step)
        y_embd = y_embd + v
        y_embd = self.decoder(y_embd)
        return y_embd, att
