# 344 TFLOPS
import os
import torch, math
import triton
import triton.language as tl
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

@triton.jit
def _attn_fwd_inner(acc, l_i, q, q_scale, #
                    K_ptrs, K_scale_ptr, V_ptrs, DtS_ptrs, #
                    start_m, s_max, #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr):
    # s_max = 0.0
    lo, hi = 0, N_CTX
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_mask = offs_n[None, :] < (N_CTX - start_n)   # 这个很关键
        k = tl.load(K_ptrs, mask = k_mask)
        dts = tl.load(DtS_ptrs, mask = k_mask)
        k_scale = tl.load(K_scale_ptr)
        qk = tl.dot(q, k).to(tl.float32) * q_scale * k_scale + dts.to(tl.float32) # + tl.where(k_mask, 0, -1.0e6)
        qk = qk - s_max
        p = tl.math.exp2(qk)
        # accumulate the partial sums for the softmax denominator across all BLOCK_N chunks
        l_ij = tl.sum(p, 1)
        l_i += l_ij
        v = tl.load(V_ptrs, mask = offs_n[:, None] < (N_CTX - start_n))
        p = p.to(tl.float16)
        
        acc += tl.dot(p, v, out_dtype=tl.float16)  
        K_ptrs += BLOCK_N * HEAD_DIM
        K_scale_ptr += 1
        V_ptrs += BLOCK_N * HEAD_DIM
        DtS_ptrs += BLOCK_N
    return acc, l_i

@triton.jit
def _attn_fwd(Q, K, V, DtS, Q_scale, K_scale, Q_max, K_max, Out,  #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              Z, H, N_Dts, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr  #
              ):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    vk_offset = qvk_offset // stride_qm
    q_scale_offset = off_hz * tl.cdiv(N_CTX, BLOCK_M)
    # print("q_scale_offset", q_scale_offset.to(tl.float32))
    k_scale_offset = off_hz * tl.cdiv(N_CTX, BLOCK_N)  # tl.cdiv(vk_offset, BLOCK_N)
    
    
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    Q_ptrs = Q + qvk_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    Q_scale_ptr = Q_scale + q_scale_offset + start_m
    K_ptrs = K + qvk_offset + offs_k[:, None] + offs_n[None, :] * stride_kn
    K_scale_ptr = K_scale + k_scale_offset
    V_ptrs = V + qvk_offset + offs_n[:, None] * stride_qm + offs_k[None, :] * stride_qk
    DtS_ptrs = DtS + (off_z * H * N_Dts + off_h * N_Dts + start_m) * N_CTX + offs_n[None, :]
    O_block_ptr = Out + qvk_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    # initialize pointer to m and l
    # m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_ptrs, mask = offs_m[:, None] < N_CTX)
    q_scale = tl.load(Q_scale_ptr)
    Q_max_ptr = Q_max + off_hz
    K_max_ptr = K_max + off_hz
    q_max = tl.load(Q_max_ptr)
    k_max = tl.load(K_max_ptr)
    s_max = q_max * k_max
    acc, l_i = _attn_fwd_inner(acc, l_i, q, q_scale, K_ptrs, K_scale_ptr, V_ptrs, DtS_ptrs, #
                                    start_m, s_max,#
                                    BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                    4 - STAGE, offs_m, offs_n, N_CTX #
                                    )
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask = (offs_m[:, None] < N_CTX))


def forward(q, k, v, delta_s, q_scale, k_scale, q_max, k_max):  # q_max: [B, H, 1]  
    BLOCK_M = 128
    BLOCK_N = 64
    HEAD_DIM_K = k.shape[-1]
    o = torch.empty_like(q, dtype=torch.float16)
    stage = 1
    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
    _attn_fwd[grid](
        q, k, v, delta_s, q_scale, k_scale, q_max, k_max, o,  #
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
        q.shape[0], q.shape[1],  #
        delta_s.shape[-2],
        N_CTX=q.shape[2],  #
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM_K,  #
        STAGE=stage,  #
        num_warps=4,  #
        num_stages=3)
    return o
