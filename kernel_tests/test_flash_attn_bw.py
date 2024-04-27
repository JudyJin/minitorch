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


@kt.case(atol=1e-3, rtol=1e-3, ntest=3)
def test_launch_flash_attn_bw():
  batch_size = 128
  nhead = 8
  seq_len = 256
  head_dim = 16
  print(
      "(batch_size, nhead, seq_len, head_dim"
      f"): ({batch_size}, {nhead}, {seq_len},{head_dim})"
  )
  # Create input tensors
  q = kt.rand((batch_size, nhead, seq_len, head_dim))
  k = kt.rand((batch_size, nhead, seq_len, head_dim))
  v = kt.rand((batch_size, nhead, seq_len, head_dim))
  out_grad = kt.rand((batch_size, nhead, seq_len, head_dim))
  mask_zero = kt.zeros((batch_size, nhead, seq_len,seq_len))
  mask = kt.dec_self_attn_mask(seq_len) * -1e8
  
  # Create minitorch tensors
  def custom():
    q_mt = minitorch.tensor(q.clone().tolist(), backend=backend, requires_grad=True)
    k_mt = minitorch.tensor(k.clone().tolist(), backend=backend, requires_grad=True)
    v_mt = minitorch.tensor(v.clone().tolist(), backend=backend, requires_grad=True)
    mask_zero_mt = minitorch.tensor(mask_zero.clone().tolist(), backend=backend, requires_grad=True)

    out_grad_mt = minitorch.tensor(out_grad.clone().tolist(), backend=backend, requires_grad=True)
    cust_out = q_mt.flash_attn(q_mt,k_mt,v_mt, mask_zero_mt, minitorch.tensor_from_numpy(np.array(0)))
    # Compute gradients
    start_time = time.time()
    cust_out.backward(out_grad_mt)
    end_time = time.time()
    # Test backward
    q_grad = torch.tensor(q_mt.grad.to_numpy(), dtype=torch.float32).cuda()
    k_grad = torch.tensor(k_mt.grad.to_numpy(), dtype=torch.float32).cuda()
    v_grad = torch.tensor(v_mt.grad.to_numpy(), dtype=torch.float32).cuda()
    # print("custom", q_grad)
    return [q_grad, k_grad, v_grad], end_time - start_time
  
  def baseline():
    q_mt = minitorch.tensor(q.clone().tolist(), backend=backend, requires_grad=True)
    k_mt = minitorch.tensor(k.clone().tolist(), backend=backend, requires_grad=True)
    v_mt = minitorch.tensor(v.clone().tolist(), backend=backend, requires_grad=True)
    out_grad_mt = minitorch.tensor(out_grad.clone().tolist(), backend=backend, requires_grad=True)

    attn_score = q_mt@ k_mt.permute(0, 1, 3, 2) / (head_dim ** 0.5)
    mask_ = minitorch.zeros_tensor_from_numpy(shape=(batch_size,seq_len),backend=backend)
    attn_score = attn_score.attn_softmax(mask_)
    # attn_score = minitorch.nn.softmax(attn_score,dim=3)
    bsl_out = attn_score@v_mt

    start_time = time.time()
    bsl_out.backward(out_grad_mt)
    end_time = time.time() 

    # Test backward
    q_grad = torch.tensor(q_mt.grad.to_numpy(), dtype=torch.float32).cuda()
    k_grad = torch.tensor(k_mt.grad.to_numpy(), dtype=torch.float32).cuda()
    v_grad = torch.tensor(v_mt.grad.to_numpy(), dtype=torch.float32).cuda()
    # print("baseline", q_grad)
    return [q_grad, k_grad, v_grad], end_time - start_time

  def baseline_non_fuse():
    q_mt = minitorch.tensor(q.clone().tolist(), backend=backend, requires_grad=True)
    k_mt = minitorch.tensor(k.clone().tolist(), backend=backend, requires_grad=True)
    v_mt = minitorch.tensor(v.clone().tolist(), backend=backend, requires_grad=True)
    out_grad_mt = minitorch.tensor(out_grad.clone().tolist(), backend=backend, requires_grad=True)

    attn_score = q_mt@ k_mt.permute(0, 1, 3, 2) / (head_dim ** 0.5)
    attn_score = minitorch.nn.softmax(attn_score,dim=3)
    bsl_out = attn_score@v_mt

    start_time = time.time()
    bsl_out.backward(out_grad_mt)
    end_time = time.time() 

    # Test backward
    q_grad = torch.tensor(q_mt.grad.to_numpy(), dtype=torch.float32).cuda()
    k_grad = torch.tensor(k_mt.grad.to_numpy(), dtype=torch.float32).cuda()
    v_grad = torch.tensor(v_mt.grad.to_numpy(), dtype=torch.float32).cuda()
    # print("baseline", q_grad)
    return [q_grad, k_grad, v_grad], end_time - start_time
  
  return custom, baseline
  # return custom

@kt.case(atol=1e-3, rtol=1e-3, ntest=3)
def test_launch_flash_attn_bw_mask():
  batch_size = 128
  nhead = 8
  seq_len = 256
  head_dim = 32
  print(
      "(batch_size, nhead, seq_len, head_dim"
      f"): ({batch_size}, {nhead}, {seq_len},{head_dim})"
  )
  # Create input tensors
  q = kt.rand((batch_size, nhead, seq_len, head_dim))
  k = kt.rand((batch_size, nhead, seq_len, head_dim))
  v = kt.rand((batch_size, nhead, seq_len, head_dim))
  out_grad = kt.rand((batch_size, nhead, seq_len, head_dim))
  mask = kt.dec_self_attn_mask(seq_len) * -1e8
  
  # Create minitorch tensors
  def custom():
    q_mt = minitorch.tensor(q.clone().tolist(), backend=backend, requires_grad=True)
    k_mt = minitorch.tensor(k.clone().tolist(), backend=backend, requires_grad=True)
    v_mt = minitorch.tensor(v.clone().tolist(), backend=backend, requires_grad=True)
    mask_mt = minitorch.tensor(mask.clone().tolist(), backend=backend, requires_grad=True)

    out_grad_mt = minitorch.tensor(out_grad.clone().tolist(), backend=backend, requires_grad=True)
    cust_out = q_mt.flash_attn(q_mt,k_mt,v_mt,mask_mt,minitorch.tensor_from_numpy(np.array(1)))
    # Compute gradients
    start_time = time.time()
    cust_out.backward(out_grad_mt)
    end_time = time.time()
    # Test backward
    q_grad = torch.tensor(q_mt.grad.to_numpy(), dtype=torch.float32).cuda()
    k_grad = torch.tensor(k_mt.grad.to_numpy(), dtype=torch.float32).cuda()
    v_grad = torch.tensor(v_mt.grad.to_numpy(), dtype=torch.float32).cuda()
    # print("custom", q_grad)
    return [q_grad, k_grad, v_grad], end_time - start_time
  
  def baseline():
    q_mt = minitorch.tensor(q.clone().tolist(), backend=backend, requires_grad=True)
    k_mt = minitorch.tensor(k.clone().tolist(), backend=backend, requires_grad=True)
    v_mt = minitorch.tensor(v.clone().tolist(), backend=backend, requires_grad=True)
    out_grad_mt = minitorch.tensor(out_grad.clone().tolist(), backend=backend, requires_grad=True)

    mask_ = mask.unsqueeze(0).unsqueeze(0)
    mask_mt = minitorch.tensor(mask_.clone().tolist(), backend=backend, requires_grad=True)

    attn_score = q_mt@ k_mt.permute(0, 1, 3, 2) / (head_dim ** 0.5)
    attn_score = attn_score.attn_softmax(mask_mt)
    # attn_score = minitorch.nn.softmax(attn_score,dim=3)
    bsl_out = attn_score@v_mt

    start_time = time.time()
    bsl_out.backward(out_grad_mt)
    end_time = time.time() 

    # Test backward
    q_grad = torch.tensor(q_mt.grad.to_numpy(), dtype=torch.float32).cuda()
    k_grad = torch.tensor(k_mt.grad.to_numpy(), dtype=torch.float32).cuda()
    v_grad = torch.tensor(v_mt.grad.to_numpy(), dtype=torch.float32).cuda()
    # print("baseline", q_grad)
    return [q_grad, k_grad, v_grad], end_time - start_time

  def baseline_non_fuse():
    q_mt = minitorch.tensor(q.clone().tolist(), backend=backend, requires_grad=True)
    k_mt = minitorch.tensor(k.clone().tolist(), backend=backend, requires_grad=True)
    v_mt = minitorch.tensor(v.clone().tolist(), backend=backend, requires_grad=True)
    out_grad_mt = minitorch.tensor(out_grad.clone().tolist(), backend=backend, requires_grad=True)
    mask_mt = minitorch.tensor(mask.clone().tolist(), backend=backend, requires_grad=True)

    attn_score = q_mt@ k_mt.permute(0, 1, 3, 2) / (head_dim ** 0.5)
    attn_score = minitorch.nn.softmax(attn_score + mask_mt, dim=3)
    # attn_score = minitorch.nn.softmax(attn_score,dim=3)
    bsl_out = attn_score@v_mt

    start_time = time.time()
    bsl_out.backward(out_grad_mt)
    end_time = time.time() 

    # Test backward
    q_grad = torch.tensor(q_mt.grad.to_numpy(), dtype=torch.float32).cuda()
    k_grad = torch.tensor(k_mt.grad.to_numpy(), dtype=torch.float32).cuda()
    v_grad = torch.tensor(v_mt.grad.to_numpy(), dtype=torch.float32).cuda()
    # print("baseline", q_grad)
    return [q_grad, k_grad, v_grad], end_time - start_time
  
  return custom, baseline_non_fuse
  # return custom
    
kt.init(device="cuda:0", nhead=8)
kt.run(
  # 'test_launch_flash_attn_bw',
  'test_launch_flash_attn_bw_mask'
)