import numpy as np
import time

import torch
import sys
sys.path.append('/home/zhaojin/minitorch')  # Adjust path as necessary

from test_utils import TestDecorator
kt = TestDecorator()

torch.manual_seed(41)

import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps
backend = minitorch.TensorBackend(CudaKernelOps)


@kt.case(atol=1e-3, rtol=1e-3, ntest=1)
def test_launch_flash_attn_bw():
  batch_size = 8
  nhead = 24
  seq_len = 16
  head_dim = 64
  print(
      "(batch_size, nhead, seq_len, head_dim"
      f"): ({batch_size}, {nhead}, {seq_len},{head_dim})"
  )
  # Create input tensors
  q = kt.rand((batch_size, nhead, seq_len, head_dim))
  k = kt.rand((batch_size, nhead, seq_len, head_dim))
  v = kt.rand((batch_size, nhead, seq_len, head_dim))
  out_grad = kt.rand((batch_size, nhead, seq_len, head_dim))
  
  # Create minitorch tensors
  def custom():
    q_mt = minitorch.tensor(q.clone().tolist(), backend=backend, requires_grad=True)
    k_mt = minitorch.tensor(k.clone().tolist(), backend=backend, requires_grad=True)
    v_mt = minitorch.tensor(v.clone().tolist(), backend=backend, requires_grad=True)
    out_grad_mt = minitorch.tensor(out_grad.clone().tolist(), backend=backend, requires_grad=True)
    cust_out = q_mt.flash_attn(q_mt,k_mt,v_mt,minitorch.tensor_from_numpy(np.array(0)))
    # Compute gradients
    start_time = time.time()
    print(1)
    cust_out.backward(out_grad_mt)
    print(1)
    end_time = time.time()
    # Test backward
    q_grad = torch.tensor(q_mt.grad.to_numpy(), dtype=torch.float32).cuda()
    k_grad = torch.tensor(k_mt.grad.to_numpy(), dtype=torch.float32).cuda()
    v_grad = torch.tensor(v_mt.grad.to_numpy(), dtype=torch.float32).cuda()
    print(q_grad)
    return [q_grad, k_grad, v_grad], end_time - start_time
  
  def baseline():
    # q_mt = minitorch.tensor(q.clone().tolist(), backend=backend, requires_grad=True)
    # k_mt = minitorch.tensor(k.clone().tolist(), backend=backend, requires_grad=True)
    # v_mt = minitorch.tensor(v.clone().tolist(), backend=backend, requires_grad=True)
    # out_grad_mt = minitorch.tensor(out_grad.clone().tolist(), backend=backend, requires_grad=True)
    # # Compute gradients backwards
    # start_time = time.time()
    # cust_out = q_mt.flash_attn(q_mt,k_mt,v_mt,minitorch.tensor_from_numpy(np.array(0)))
    # cust_out.backward(out_grad_mt)

    # q_grad = torch.tensor(q_mt.grad.to_numpy(), dtype=torch.float32).cuda()
    # k_grad = torch.tensor(k_mt.grad.to_numpy(), dtype=torch.float32).cuda()
    # v_grad = torch.tensor(v_mt.grad.to_numpy(), dtype=torch.float32).cuda()
    # end_time = time.time()
    q_grad = torch.tensor(q.clone().tolist(), dtype=torch.float32).cuda()
    k_grad = torch.tensor(k.clone().tolist(), dtype=torch.float32).cuda()
    v_grad = torch.tensor(v.clone().tolist(), dtype=torch.float32).cuda()
    end_time = start_time = time.time()  
    return [q_grad, k_grad, v_grad], end_time - start_time
  
  return custom, baseline
  # return custom
    
kt.init(device="cuda:0", nhead=8)
kt.run(
  'test_launch_flash_attn_bw'
)