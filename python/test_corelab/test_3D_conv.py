import os

#os.environ['TRITON_INTERPRET'] = '1'
from functools import partial

import torch
import triton
import triton.language as tl

DEVICE = "cuda"
@triton.jit
def bmm_dot_kernel(
    k_size,
    a_ptr, b_ptr, c_ptr,
    stride_a0, stride_a1, stride_a2,
    stride_b0, stride_b1, stride_b2,
    stride_c0, stride_c1, stride_c2,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr,
):
    pid_m, pid_n, pid_batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    rm_vec        = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    rn_vec        = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    rk_vec        = tl.arange(0, BLOCK_SIZE)

    rbatch_vec_   = pid_batch * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    rbatch_vec3d  = tl.expand_dims(tl.expand_dims(rbatch_vec_, 1), 1)
    
    offs_a_mat = a_ptr + rbatch_vec3d*stride_a0 + tl.expand_dims(rm_vec, 1)*stride_a1 + tl.expand_dims(rk_vec, 0)*stride_a2
    offs_b_mat = b_ptr + rbatch_vec3d*stride_b0 + tl.expand_dims(rk_vec, 1)*stride_b1 + tl.expand_dims(rn_vec, 0)*stride_b2

    acc = tl.zeros((BLOCK_SIZE_BATCH, BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    for _ in range(0, k_size, BLOCK_SIZE):
        a = tl.load(offs_a_mat)
        b = tl.load(offs_b_mat)
        acc = tl.dot(a, b, acc)

        offs_a_mat += BLOCK_SIZE * stride_a2
        offs_b_mat += BLOCK_SIZE * stride_b1

    c = c_ptr + rbatch_vec3d*stride_c0 + tl.expand_dims(rm_vec, 1)*stride_c1 + tl.expand_dims(rn_vec, 0)*stride_c2
    
    tl.store(c, acc)

def bmm_dot_op(a: torch.Tensor, b: torch.Tensor, matmul_k_fn):
    batch_size=a.shape[0]
    k_size = a.shape[2]
    m = a.shape[1]
    n = b.shape[2]
    assert m == n # m and n are equal in this test

    BLOCK_SIZE=16 # block sizes for m, n, k are all equal
    BLOCK_SIZE_BATCH=2 # can be 1, 2, 4, 8, 16

    c = torch.zeros(batch_size, m, n, device=a.device, dtype=a.dtype)

    grid = (m // BLOCK_SIZE, n // BLOCK_SIZE, batch_size // BLOCK_SIZE_BATCH)
    matmul_k_fn[grid](
        k_size,
        a, b, c,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1), c.stride(2),
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_SIZE_BATCH=BLOCK_SIZE_BATCH
    )
    return c


bmm_dot = partial(bmm_dot_op, matmul_k_fn=bmm_dot_kernel)

if __name__ == '__main__':
    dtype=torch.float32
    k = 128 # can be 16, 32, 128, 256... etc.
    
    mn = 32
    batch_size = 16
    a = torch.rand(batch_size, mn, k, dtype=dtype, device='cuda')
    b = torch.rand(batch_size, k, mn, dtype=dtype, device='cuda')

    # print(a.shape, b.shape)
    c_ref = torch.bmm(a, b)
    c_test = bmm_dot(a, b)

    print(c_ref)
    print(c_test)
    allclose = torch.allclose(c_test, c_ref, rtol=0.1, atol=0.1)
    assert allclose
    print(f'Test COMPLETE! {allclose=}')
