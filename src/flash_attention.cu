#include <math.h>
#include <iostream>

#include <cub/block/block_load.cuh>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include "includes/block_reduce.h"
#include "includes/kernels.h"

#include <cooperative_groups.h>

namespace flash_attn{
namespace cuda{


template <typename T>
__global__ void flash_attn_fw(const T *Q, const T* K, const T* V, T* O, T* L, T* M, int seq_len, int head_dim,const T * masks, bool is_causal) {
    int batch_id = blockIdx.y;
    int head_id = blockIdx.z;
    int batch_size = gridDim.y;
    int nhead = gridDim.z;
    int br = blockDim.y;
    int bc = blockDim.x;
    int outer_steps = (seq_len + bc - 1) / bc;
    int inner_steps = (seq_len + br - 1) / br;

    int stride_batch = nhead * seq_len * head_dim;
    int stride_head = seq_len * head_dim;
    int stride_seq = head_dim;
    // typedef cub::BlockLoad<T, block_dim, ele_per_thread,
    //                      cub::BLOCK_LOAD_VECTORIZE>
    //     BlockLoad;
    // __shared__ typename BlockLoad::TempStorage ts_load;
    // typedef cub::BlockStore<T, block_dim, ele_per_thread,
    //                         cub::BLOCK_STORE_VECTORIZE>
    //     BlockStore;
    // __shared__ typename BlockStore::TempStorage ts_store;

    __shared__ T shared_k[bc][head_dim];
    __shared__ T shared_v[bc][head_dim];
    __shared__ T shared_q[br][head_dim];
    __shared__ T shared_o[br][head_dim];
    __shared__ T shared_l[br];
    __shared__ T shared_m[br];
    __shared__ T shared_s[br][bc];
    
    // __shared__ T shared_q[br][head_dim];
    // __shared__ T shared_o[br][bc];
    for (int j=0;j<outer_steps;++j){
        // load KV to on-chip memory
        int kv_per_thread = (head_dim + br -1) / br;
        for (int num = 0; num < kv_per_thread; ++num){
            int ele_idx = threadIdx.y * num_per_thread + num;
            shared_k[threadIdx.x][ele_idx] = K[batch_id * stride_batch + head_id * stride_head + (j * bc + threadIdx.x) * stride_seq + ele_idx];
            shared_v[threadIdx.x][ele_idx] = V[batch_id * stride_batch + head_id * stride_head + (j * bc + threadIdx.x) * stride_seq + ele_idx];
        }
        __syncthreads();
        // inner loop
        for (int i = 0;i<inner_steps;++i){
            //l load Q to on-chip memory
            int q_per_thread = (head_dim + br -1) / br;
            for (int num = 0; num < q_per_thread; ++num){
                int ele_idx = threadIdx.x * num_per_thread + num;
                shared_q[threadIdx.y][ele_idx] = Q[batch_id * stride_batch + head_id * stride_head + (i * br + threadIdx.y) * stride_seq + ele_idx];
                if (threadIdx.x == 0){
                    shared_l[threadIdx.y] = L[batch_id * n_head * seq_len + head_id * seq_len + (i * br + threadIdx.y)]; 
                    shared_m[threadIdx.y] = M[batch_id * n_head * seq_len + head_id * seq_len + (i * br + threadIdx.y)];
                }
                __syncthreads();
            }
            // compute attention
            T sum_ = 0;
            for (int k = 0; k < head_dim; ++k){
                sum_ += shared_q[threadIdx.y][k] * shared_k[threadIdx.x][k];
            }
            shared_s[threadIdx.y][threadIdx.x] = sum_; 
            __syncthreads();
            // row max
            if (threadIdx.x == 0){
                T m_ij = 0;
                for (int k = 0; k < bc; ++k){
                    m_ij = max(max_, shared_s[threadIdx.y][k]);
                }
                // softmax
                T l_ij = 0;
                for (int k = 0; k < bc; ++k){
                    shared_s[threadIdx.y][k] = exp(shared_s[threadIdx.y][k] - max_);
                    l_ij += shared_s[threadIdx.y][k];
                }
                // compute l and m
                T m_new =  max(shared_m[threadIdx.y], m_ij);
                shared_l[threadIdx.y] = exp(shared_m[threadIdx.y] - m_new) * shared_l[threadIdx.y] + exp(m_ij - m_new) * l_ij;
                shared_m[threadIdx.y] = m_new;
            }

            
            




        }

        

    }


}
    

extern "C" {
void launch_flash_attn_fw(const float *Q, const float* K, const float * V, float * O, 
                                int batch_size, int nhead, int seq_len, int head_dim,
                                bool is_causal,
                                cudaStream_t stream) {

  int float_size = sizeof(float);
  int qkv_size = batch_size * nhead * seq_len * head_dim * float_size;

  float *d_q, *d_k, *d_v, *d_o;
  cudaMalloc((void **)&d_q, qkv_size);
  cudaMalloc((void **)&d_k, qkv_size);
  cudaMalloc((void **)&d_v, qkv_size);
  cudaMalloc((void **)&d_o, qkv_size);


  cudaMemcpy(d_q, Q, qkv_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_k, K, qkv_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, V, qkv_size, cudaMemcpyHostToDevice);


  dim3 grid_dim(1, batch_size, nhead);

  // get shared memory size M
  cudaDeviceProp prop;
  int deviceId;
  cudaGetDevice(&deviceId);
  cudaGetDeviceProperties(&prop, deviceId);
  int M = prop.sharedMemPerBlock;

  // calculate block size
  int bc, br;
  bc = min(M/(4*head_dim), 32);
  br = min(M/(4*head_dim), 32);

  dim3 block_dim(bc, br);

  // launch kernel
  flash_attn_fw<float><<<grid_dim, block_dim, 0, stream>>>(d_q, d_k, d_v, seq_len, head_dim,nullptr, is_causal);



  
  // Copy back to the host
  cudaMemcpy(inp, d_inp, inp_size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  // Check CUDA execution
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_attn_softmax Error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Free memory on device
  cudaFree(d_inp);
  cudaFree(d_attn_mask);

}}


    
}
}