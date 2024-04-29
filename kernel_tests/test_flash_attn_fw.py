import numpy as np
import time

import torch
import sys
sys.path.append('./')  # Adjust path as necessary

from test_utils import TestDecorator
kt = TestDecorator()

torch.manual_seed(41)

import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps
backend = minitorch.TensorBackend(CudaKernelOps)

def create_causal_mask(bs, nh, seq_len):
    """
    return a 1x1xTxt triangular causal mask for Q @ K^T (which will get broadcasted to BxHxTxT)
    """
    mask = -np.finfo(np.float32).max * np.triu(np.ones((1, 1, seq_len, seq_len), dtype=np.float32), 1) # This should be ok, but may be problematic
    # mask = -np.finfo(datatype).max * np.triu(np.ones((bs, nh, seq_len, seq_len), dtype=datatype), 1) 
    return minitorch.tensor_from_numpy(mask, backend=backend)  

@kt.case(atol=1e-3, rtol=1e-3, ntest=3)
def test_launch_flash_attn_mask():
  batch_size = 128
  nhead = 8
  seq_len = 256
  head_dim = 32
  print(
      "(batch_size, nhead, seq_len, head_dim"
      f"): ({batch_size}, {nhead}, {seq_len},{head_dim})"
  )
  q = kt.rand((batch_size, nhead, seq_len, head_dim))
  k = kt.rand((batch_size, nhead, seq_len, head_dim))
  v = kt.rand((batch_size, nhead, seq_len, head_dim))
  
  # print("============q is===========\n",q[:,:,:,:10])

  def custom():
    q_mt = minitorch.tensor(q.clone().tolist(), backend=backend, requires_grad=True)
    k_mt = minitorch.tensor(k.clone().tolist(), backend=backend, requires_grad=True)
    v_mt = minitorch.tensor(v.clone().tolist(), backend=backend, requires_grad=True)
    mask_mt =create_causal_mask(batch_size, nhead, seq_len)
    
    start_time = time.time()
    cust_out = q_mt.flash_attn(q_mt,k_mt,v_mt,mask_mt,minitorch.tensor_from_numpy(np.array(1)))
    end_time = time.time()

    cust_out = torch.tensor(cust_out._tensor._storage).float().cuda()
    return [
        cust_out,
    ], end_time - start_time

  def baseline_fuse():
    q_mt = minitorch.tensor(q.clone().tolist(), backend=backend, requires_grad=True)
    k_mt = minitorch.tensor(k.clone().tolist(), backend=backend, requires_grad=True)
    v_mt = minitorch.tensor(v.clone().tolist(), backend=backend, requires_grad=True)
    mask_mt = create_causal_mask(batch_size, nhead, seq_len)

    start_time = time.time()
    attn_score = q_mt@ k_mt.permute(0, 1, 3, 2) / (head_dim ** 0.5)
    attn_score += mask_mt
    mask_zero = minitorch.zeros_tensor_from_numpy(shape=(batch_size,seq_len),backend=backend)
    attn_score = attn_score.attn_softmax(mask_zero)
    bsl_out = attn_score@v_mt
    end_time = time.time()

    bsl_out = torch.tensor(bsl_out._tensor._storage).float().cuda()
    return kt.norm_res_list(bsl_out), end_time - start_time
  
  def baseline_no_fuse():
    q_mt = minitorch.tensor(q.clone().tolist(), backend=backend, requires_grad=True)
    k_mt = minitorch.tensor(k.clone().tolist(), backend=backend, requires_grad=True)
    v_mt = minitorch.tensor(v.clone().tolist(), backend=backend, requires_grad=True)
    mask_mt = create_causal_mask(batch_size, nhead, seq_len)

    start_time = time.time()
    attn_score = q_mt@ k_mt.permute(0, 1, 3, 2) / (head_dim ** 0.5)
    attn_score += mask_mt
    attn_score = minitorch.nn.softmax(attn_score, dim=3)
    bsl_out = attn_score@v_mt
    end_time = time.time()

    bsl_out = torch.tensor(bsl_out._tensor._storage).float().cuda()
    return kt.norm_res_list(bsl_out), end_time - start_time
     

  return baseline_fuse, baseline_no_fuse



kt.init(device="cuda:0", nhead=8)
kt.run(
  'test_launch_flash_attn_mask'
)