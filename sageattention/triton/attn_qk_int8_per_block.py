"""
Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch, math
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, q_scale, kv_len,
                    K_ptrs, K_scale_ptr, V_ptrs, stride_kn, stride_vn, 
                    start_m,  
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  
                    t_idx=None):
    log = 0
    nof_kv_tiles = tl.cdiv(kv_len, BLOCK_N)
    qk_ratio = BLOCK_M // BLOCK_N
    j_bias = start_m * qk_ratio
    # pv_thr = -8 if t_idx < 10 else -6 if t_idx < 20 else -4
    pv_thr = -1000
    for j_ in range(nof_kv_tiles):
        # linear indexing
        j = j_

        # # linear indexing starting at diag
        # j = (j_ + j_bias) % nof_kv_tiles

        # # radial indexing - starts at diag and alternates around it
        # sign = 2 * (j_ % 2) - 1
        # mag = (j_ + 1) // 2
        # j_wo_bias = sign * mag
        # j = (nof_kv_tiles + j_wo_bias + j_bias) % nof_kv_tiles

        # # radial indexing with sink
        # if j_bias == 0:
        #     sign = 2 * (j_ % 2) - 1
        #     mag = (j_ + 1) // 2
        #     j_wo_bias = sign * mag
        #     j = (nof_kv_tiles + j_wo_bias + j_bias) % nof_kv_tiles
        # else:
        #     if j_ < qk_ratio:
        #         j = j_
        #     else:
        #         sign = 2 * (j_ % 2) - 1
        #         mag = (j_ - qk_ratio + 1) // 2
        #         j_wo_bias = sign * mag
        #         j = qk_ratio + (nof_kv_tiles + j_wo_bias + j_bias  - 2 * qk_ratio) % (nof_kv_tiles - qk_ratio)

        kv_start = j * BLOCK_N
        kv_stop = kv_len - kv_start
        k_scale = tl.load(K_scale_ptr + j)
        k = tl.load(K_ptrs + kv_start * stride_kn, mask = offs_n[None, :] < kv_stop)
        qk = tl.dot(q, k).to(tl.float32) * q_scale * k_scale
        
        m_local = tl.max(qk, 1)
        m_ij = tl.maximum(m_i, m_local)
        do_pv = tl.max(m_local - m_ij) > pv_thr  # current p is far from zero (sparge condition)
        log += 1 - do_pv

        if do_pv:
            qk = qk - m_ij[:, None]
            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
        
            alpha = tl.math.exp2(m_i - m_ij)
            m_i = m_ij
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]

            p = p.to(tl.float16)
            v = tl.load(V_ptrs + kv_start * stride_vn, mask = offs_n[:, None] < kv_stop)
            acc += tl.dot(p, v, out_dtype=tl.float16)  
    return acc, l_i, m_i, log

@triton.jit
def _attn_fwd(Q, K, V, Q_scale, K_scale, Out, Lse, 
              stride_qz, stride_qh, stride_qn,
              stride_kz, stride_kh, stride_kn,  
              stride_vz, stride_vh, stride_vn,  
              stride_oz, stride_oh, stride_on,  
              qo_len, kv_len, H: tl.constexpr, num_kv_groups: tl.constexpr,
              HEAD_DIM: tl.constexpr,  
              BLOCK_M: tl.constexpr,  
              BLOCK_N: tl.constexpr,  
              STAGE: tl.constexpr,
              RETURN_LSE: tl.constexpr,
              log, t_idx=None):
    start_m = tl.program_id(0)

    off_z = tl.program_id(2).to(tl.int64)
    off_h = tl.program_id(1).to(tl.int64)

    q_scale_offset = (off_z * H + off_h) * tl.cdiv(qo_len, BLOCK_M)
    k_scale_offset = (off_z * (H // num_kv_groups) + off_h // num_kv_groups) * tl.cdiv(kv_len, BLOCK_N)  
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    Q_ptrs = Q + (off_z * stride_qz + off_h * stride_qh) + offs_m[:, None] * stride_qn + offs_k[None, :]
    Q_scale_ptr = Q_scale + q_scale_offset + start_m
    K_ptrs = K + (off_z * stride_kz + (off_h // num_kv_groups) * stride_kh) + offs_n[None, :] * stride_kn + offs_k[:, None] 
    K_scale_ptr = K_scale + k_scale_offset
    V_ptrs = V + (off_z * stride_vz + (off_h // num_kv_groups) * stride_vh) + offs_n[:, None] * stride_vn + offs_k[None, :]
    O_block_ptr = Out + (off_z * stride_oz + off_h * stride_oh) + offs_m[:, None] * stride_on + offs_k[None, :]
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    q = tl.load(Q_ptrs, mask = offs_m[:, None] < qo_len)
    q_scale = tl.load(Q_scale_ptr)
    acc, l_i, m_i, log_inner = _attn_fwd_inner(acc, l_i, m_i, q, q_scale, kv_len, K_ptrs, K_scale_ptr, V_ptrs, stride_kn, stride_vn,
                                    start_m,  
                                    BLOCK_M, HEAD_DIM, BLOCK_N,  
                                    4 - STAGE, offs_m, offs_n,
                                    t_idx=t_idx 
                                    )
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask = (offs_m[:, None] < qo_len))

    tl.store(log + start_m, log_inner)

    if RETURN_LSE:
        lse_ptrs = Lse + (off_z * qo_len * H + off_h * qo_len) + offs_m
        l_i = tl.log2(l_i) + m_i
        tl.store(lse_ptrs, l_i, mask = (offs_m < qo_len))

def forward(q, k, v, q_scale, k_scale, tensor_layout="HND", output_dtype=torch.float16, return_lse=False, t_idx=None):
    log = torch.zeros(256, dtype=torch.int32, device='cuda')

    BLOCK_M = 128
    BLOCK_N = 64
    stage = 1

    o = torch.empty(q.shape, dtype=output_dtype, device=q.device)

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(1), v.stride(2)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(1), o.stride(2)
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(2), v.stride(1)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(2), o.stride(1)
    else:
        raise ValueError(f"tensor_layout {tensor_layout} not supported")
    
    HEAD_DIM_K = head_dim
    num_kv_groups = h_qo // h_kv

    if return_lse:
        lse = torch.empty([b, h_qo, qo_len], dtype=torch.float32, device=q.device)
    else:
        lse = torch.empty([0], dtype=torch.float32, device='cpu')

    grid = (triton.cdiv(qo_len, BLOCK_M), h_qo, b)
    _attn_fwd[grid](
        q, k, v, q_scale, k_scale, o, lse,
        stride_bz_q, stride_h_q, stride_seq_q, 
        stride_bz_k, stride_h_k, stride_seq_k,  
        stride_bz_v, stride_h_v, stride_seq_v,  
        stride_bz_o, stride_h_o, stride_seq_o,
        qo_len, kv_len,
        h_qo, num_kv_groups,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM_K,  
        STAGE=stage, RETURN_LSE=return_lse,
        num_warps=4 if head_dim == 64 else 8,
        num_stages=3 if head_dim == 64 else 4,
        log=log, t_idx=t_idx)

    print(f'{int(100*log.sum() / (kv_len / BLOCK_N) / (qo_len / BLOCK_M))}%', end=',')
    return o, lse

if __name__ == "__main__":
    print("hello world")
    kv_len = 32760
    BLOCK_M = 128
    BLOCK_N = 64
    nof_kv_tiles = kv_len // BLOCK_N + 1
    qk_ratio = BLOCK_M // BLOCK_N
    j_bias = 2 * qk_ratio
    for j_ in range(nof_kv_tiles):
        # radial indexing with sink
        if j_bias == 0:
            sign = 2 * (j_ % 2) - 1
            mag = (j_ + 1) // 2
            j_wo_bias = sign * mag
            j = (nof_kv_tiles + j_wo_bias + j_bias) % nof_kv_tiles
        else:
            if j_ < qk_ratio:
                j = j_
                j_wo_bias = 0
            else:
                sign = 2 * (j_ % 2) - 1
                mag = (j_ - qk_ratio + 1) // 2
                j_wo_bias = sign * mag
                j = qk_ratio + (nof_kv_tiles + j_wo_bias + j_bias  - 2 * qk_ratio) % (nof_kv_tiles - qk_ratio)
        print(f'{j_} -> {j} ({j_wo_bias})')