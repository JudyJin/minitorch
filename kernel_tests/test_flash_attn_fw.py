import numpy as np
import time

import torch
import sys
sys.path.append('/home/zhaojin/minitorch')  # Adjust path as necessary

from test_utils import TestDecorator
kt = TestDecorator()

import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps
backend = minitorch.TensorBackend(CudaKernelOps)


@kt.case(atol=1e-3, rtol=1e-3, ntest=5)
def test_launch_flash_attn():
  batch_size, seq_len = kt.bs_sl()
  nhead = kt.nhead
  print(
      "(batch_size, nhead, seq_len, head_dim"
      f"): ({batch_size}, {nhead}, {seq_len},{64}"
  )
  q = kt.rand((batch_size, nhead, seq_len, 64))
  k = kt.rand((batch_size, nhead, seq_len, 64))
  v = kt.rand((batch_size, nhead, seq_len, 64))

  def custom():
    q_mt = minitorch.tensor(q.clone().tolist(), backend=backend, requires_grad=True)
    k_mt = minitorch.tensor(k.clone().tolist(), backend=backend, requires_grad=True)
    v_mt = minitorch.tensor(v.clone().tolist(), backend=backend, requires_grad=True)

    start_time = time.time()
    cust_out = q_mt.flash_attn(q_mt,k_mt,v_mt,minitorch.tensor_from_numpy(np.array(0)))
    end_time = time.time()

    cust_out = torch.tensor(cust_out._tensor._storage).float().cuda()
    return [
        cust_out,
    ], end_time - start_time

  def baseline():
    q_mt = minitorch.tensor(q.clone().tolist(), backend=backend, requires_grad=True)
    k_mt = minitorch.tensor(k.clone().tolist(), backend=backend, requires_grad=True)
    v_mt = minitorch.tensor(v.clone().tolist(), backend=backend, requires_grad=True)

    start_time = time.time()
    attn_score = q_mt@ k_mt.permute(0, 1, 3, 2)
    attn_score = minitorch.nn.softmax(attn_score,dim=3)
    bsl_out = attn_score@v_mt
    # if not is_dec_self_attn_infer:
    #   res = minitorch.nn.softmax(inp_mt + mask_mt, dim=3)
    # else:
    #   res = minitorch.nn.softmax(inp_mt, dim=3)
    end_time = time.time()

    bsl_out = torch.tensor(bsl_out._tensor._storage).float().cuda()
    return kt.norm_res_list(bsl_out), end_time - start_time

  return custom, baseline


kt.init(device="cuda:0", nhead=8)
kt.run(
  'test_launch_flash_attn'
)