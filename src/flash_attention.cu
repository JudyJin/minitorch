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

// called from global, use pointer to modify the bias in-place
template <typename T>
__device__ T* GetSharedPtr(T * shared_mem, int * ptr_bias, int mem_size) {
    T * cur_shared_mem = shared_mem + *ptr_bias;
    *ptr_bias += mem_size;
    return cur_shared_mem;
}
// remember that br/bc might divide seq_len


template <typename T>
__global__ void flash_attn_fw(const T *Q, const T* K, const T* V, T* O, T* L, T* M, T* attn_mask, int seq_len, int head_dim,const T * masks, bool is_causal) {
    int batch_id = blockIdx.y;
    int head_id = blockIdx.x;
    int batch_size = gridDim.y;
    int nhead = gridDim.x;
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

    extern __shared__ T shared_mem[];
    T *shared_mem_start =  reinterpret_cast<T*>(shared_mem);
    int ptr_bias = 0;
    T* shared_q = GetSharedPtr<T>(shared_mem_start, &ptr_bias, br * head_dim); // size of (br * head_dim)
    T* shared_k = GetSharedPtr<T>(shared_mem_start, &ptr_bias, bc * head_dim); // size of (bc * head_dim)
    T* shared_v = GetSharedPtr<T>(shared_mem_start, &ptr_bias, bc * head_dim); // size of (bc * head_dim)
    T* shared_o = GetSharedPtr<T>(shared_mem_start, &ptr_bias, br * head_dim); // size of (br * head_dim)
    T* shared_l = GetSharedPtr<T>(shared_mem_start, &ptr_bias, br);
    T* shared_l_ij = GetSharedPtr<T>(shared_mem_start, &ptr_bias, br);
    T* shared_l_new = GetSharedPtr<T>(shared_mem_start, &ptr_bias, br);
    T* shared_m = GetSharedPtr<T>(shared_mem_start, &ptr_bias, br);
    T* shared_m_ij = GetSharedPtr<T>(shared_mem_start, &ptr_bias, br);
    T* shared_m_new = GetSharedPtr<T>(shared_mem_start, &ptr_bias, br);
    T* shared_s = GetSharedPtr<T>(shared_mem_start, &ptr_bias, br * bc);
    T* shared_mask = GetSharedPtr<T>(shared_mem_start, &ptr_bias, br * bc);
    for (int j=0;j<outer_steps;++j){
        int row_KV = (j * bc + threadIdx.x);
        // load KV to on-chip memory
        int kv_per_thread = (head_dim + br -1) / br;
        for (int col_idx = 0; col_idx < kv_per_thread; ++col_idx){
            int ele_idx = threadIdx.y * kv_per_thread + col_idx;
            if (row_KV < seq_len && ele_idx < head_dim){
                shared_k[threadIdx.x * head_dim + ele_idx] = K[batch_id * stride_batch + head_id * stride_head + (j * bc + threadIdx.x) * stride_seq + ele_idx];
                shared_v[threadIdx.x * head_dim + ele_idx] = V[batch_id * stride_batch + head_id * stride_head + (j * bc + threadIdx.x) * stride_seq + ele_idx];
            }
        }
        __syncthreads();
        // inner loop
        for (int i = 0;i<inner_steps;++i){
            int row_QO = (i * br + threadIdx.y);
            //load QO to on-chip memory
            int qo_per_thread = (head_dim + bc -1) / bc;
            for (int col_idx = 0; col_idx < qo_per_thread; ++col_idx){
                int ele_idx = threadIdx.x * qo_per_thread + col_idx;
                if (row_QO < seq_len && ele_idx < head_dim){
                    shared_q[threadIdx.y * head_dim + ele_idx] = Q[batch_id * stride_batch + head_id * stride_head + (i * br + threadIdx.y) * stride_seq + ele_idx];
                    shared_o[threadIdx.y * head_dim + ele_idx] = O[batch_id * stride_batch + head_id * stride_head + (i * br + threadIdx.y) * stride_seq + ele_idx];
                }
            }
            // always true for threadIdx.y < bc
            // load l and m to on-chip memory only when j > 0
            if (row_QO < seq_len && threadIdx.x == 0 && j > 0){
                shared_l[threadIdx.y] = L[batch_id * nhead * seq_len + head_id * seq_len + (i * br + threadIdx.y)]; 
                shared_m[threadIdx.y] = M[batch_id * nhead * seq_len + head_id * seq_len + (i * br + threadIdx.y)];
            }
            __syncthreads();
            // compute attention
            if (row_KV < seq_len && row_QO < seq_len){
                T sum_ = 0;
                for (int k = 0; k < head_dim; ++k){
                    sum_ += shared_q[threadIdx.y * head_dim + k] * shared_k[threadIdx.x * head_dim + k];
                }
                shared_s[threadIdx.y * bc + threadIdx.x] = sum_ * rsqrtf(head_dim) ;
            }
            __syncthreads();

            // row-wise computation for the normalization factors
            if (row_QO < seq_len && threadIdx.x == 0){
                T m_ij_ = -FLT_MAX; // need to be -inf
                for (int k = 0; k < bc; ++k){
                   if ((j * bc + k) < seq_len){
                        m_ij_ = fmaxf(m_ij_, shared_s[threadIdx.y * bc + k]);
                   }
                }
                shared_m_ij[threadIdx.y] = m_ij_;
                // softmax
                T l_ij_ = 0;
                for (int k = 0; k < bc; ++k){
                    if ((j * bc + k) < seq_len){
                        shared_s[threadIdx.y * bc + k] = __expf(shared_s[threadIdx.y * bc + k] - m_ij_);
                        l_ij_ += shared_s[threadIdx.y * bc + k];
                    }
                }
                shared_l_ij[threadIdx.y] = l_ij_; //Todo: the shared memory for lij is is actually not needed

                // compute l_new and m_new
                T m_new_, l_new_;
                if (j==0){
                    m_new_ = m_ij_;
                    l_new_ = l_ij_;
                }
                else{
                    m_new_ = fmaxf(shared_m[threadIdx.y], m_ij_);
                    l_new_ = __expf(shared_m[threadIdx.y] - m_new_) * shared_l[threadIdx.y] + __expf(m_ij_ - m_new_) * l_ij_;
                }
                
                // need to write back to shared_memory for row sharing
                shared_l_new[threadIdx.y] = l_new_;
                shared_m_new[threadIdx.y] = m_new_;
            }
            __syncthreads();

            /*
            Compute O on-chip and write back to HBM
            */
            // step 1: compute the shared_o
            T factor_o = shared_l[threadIdx.y] * __expf(shared_m[threadIdx.y] - shared_m_new[threadIdx.y]) / shared_l_new[threadIdx.y];
            T factor_pv = __expf(shared_m_ij[threadIdx.y] - shared_m_new[threadIdx.y]) / shared_l_new[threadIdx.y];
            if (row_QO < seq_len){
                for (int col_idx = 0;col_idx<qo_per_thread; ++col_idx){
                    int ele_idx = threadIdx.x * qo_per_thread + col_idx;
                    if (ele_idx < head_dim){
                        T sum_pv = 0;
                        for (int k = 0; k<bc; ++k){
                            if ((j * bc + k) < seq_len){
                                sum_pv += shared_s[threadIdx.y * bc + k] * shared_v[k * head_dim + ele_idx];
                            }
                        }
                        sum_pv *= factor_pv;
                        if (j==0){
                            shared_o[threadIdx.y * head_dim + ele_idx] = sum_pv; // no previous O and normalization for pv
                        }
                        else{
                            T sum_o = shared_o[threadIdx.y * head_dim + ele_idx] * factor_o + sum_pv;
                            shared_o[threadIdx.y * head_dim + ele_idx] = sum_o;
                        }
                    }
                }
            }
            
            __syncthreads();

            // step 2: write O back to HBM
            if (row_QO < seq_len){
                for (int col_idx = 0; col_idx < qo_per_thread; ++col_idx){
                    int ele_idx = threadIdx.x * qo_per_thread + col_idx;
                    if (ele_idx < head_dim){
                        O[batch_id * stride_batch + head_id * stride_head + (i * br + threadIdx.y) * stride_seq + ele_idx] = shared_o[threadIdx.y * head_dim + ele_idx];
                    }
                }
            }
            // step 3: write l, m back to HBM
            if (row_QO < seq_len && threadIdx.x==0){
                L[batch_id * nhead * seq_len + head_id * seq_len + (i * br + threadIdx.y)] = shared_l_new[threadIdx.y];
                M[batch_id * nhead * seq_len + head_id * seq_len + (i * br + threadIdx.y)] = shared_m_new[threadIdx.y];
            }
            __syncthreads();

        }
    }

}

extern "C" {
void launch_flash_attn_fw(const float *Q, const float* K, const float * V, float * O, 
                                float *L, float *M,
                                int batch_size, int nhead, int seq_len, int head_dim,
                                bool is_causal,
                                cudaStream_t stream) {

  int float_size = sizeof(float);
  int qkv_size = batch_size * nhead * seq_len * head_dim * float_size;
  int lm_size = batch_size * nhead * seq_len * float_size;

  float *d_q, *d_k, *d_v, *d_o;
  float *d_l, *d_m;
  cudaMalloc((void **)&d_q, qkv_size);
  cudaMalloc((void **)&d_k, qkv_size);
  cudaMalloc((void **)&d_v, qkv_size);
  cudaMalloc((void **)&d_o, qkv_size);
  cudaMalloc((void **)&d_l, lm_size);
  cudaMalloc((void **)&d_m, lm_size);
//   cudaMemset(d_l, 0, lm_size);
//   cudaMemset(d_m, 0, lm_size);


  cudaMemcpy(d_q, Q, qkv_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_k, K, qkv_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, V, qkv_size, cudaMemcpyHostToDevice);


  dim3 grid_dim(nhead, batch_size);

  // get shared memory size M
  cudaDeviceProp prop;
  int deviceId;
  cudaGetDevice(&deviceId);
  cudaGetDeviceProperties(&prop, deviceId);
  int Mem = prop.sharedMemPerBlock;

  // calculate block size
  int bc, br;
  bc = min(Mem/(4*head_dim), 16);
  br = min(Mem/(4*head_dim), 16);


  dim3 block_dim(bc, br);

  // launch kernel
  int total_shared_mem_size = ((br * 2 + bc * 2) * head_dim + br * 6 + br * bc) * float_size;
  printf("here get the M size of %d with br size %d, head_dim %d and shared memory size %d\n", Mem, br,head_dim, total_shared_mem_size);

//   flash_attn_fw<float><<<grid_dim, block_dim, total_shared_mem_size, stream>>>(d_q, d_k, d_v, d_o, d_l, d_m, seq_len,head_dim, nullptr, is_causal);
  flash_attn_fw<float><<<grid_dim, block_dim, total_shared_mem_size, stream>>>(d_q, d_k, d_v, d_o, d_l, d_m, nullptr, seq_len,head_dim, nullptr, is_causal);
  
  // Synchronize and check for errors
  cudaDeviceSynchronize();
  // Check CUDA execution
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_attn_softmax Error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  // Copy back to the host
  cudaMemcpy(O, d_o, qkv_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(L, d_l, lm_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(M, d_m, lm_size, cudaMemcpyDeviceToHost);

  // Free memory on device
  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_o);
  cudaFree(d_l);
  cudaFree(d_m);

  
}}


template <typename T>
__global__ void flash_attn_bw(T* dQ, T* dK, T* dV, const T* dO, const T *Q, const T* K, const T* V, const T* O, const T* L, const T* M, int seq_len, int head_dim,const T * masks, bool is_causal) {
    // flash attention bw function
    int batch_id = blockIdx.y;
    int head_id = blockIdx.x;
    int batch_size = gridDim.y;
    int nhead = gridDim.x;
    int br = blockDim.y;
    int bc = blockDim.x;
    int outer_steps = (seq_len + bc - 1) / bc;
    int inner_steps = (seq_len + br - 1) / br;
    
    int stride_batch = nhead * seq_len * head_dim;
    int stride_head = seq_len * head_dim;
    int stride_seq = head_dim;
    // initialize dQ, dK, dV to 0
    // T *dQ = new T[batch_size * nhead * seq_len * head_dim];
    // T *dK = new T[batch_size * nhead * seq_len * head_dim];
    // T *dV = new T[batch_size * nhead * seq_len * head_dim];

    extern __shared__ T shared_mem[];
    T *shared_mem_start =  reinterpret_cast<T*>(shared_mem);
    int ptr_bias = 0;
    T* shared_q = GetSharedPtr<T>(shared_mem_start, &ptr_bias, br * head_dim); // size of (br * head_dim)
    T* shared_k = GetSharedPtr<T>(shared_mem_start, &ptr_bias, bc * head_dim); // size of (bc * head_dim)
    T* shared_v = GetSharedPtr<T>(shared_mem_start, &ptr_bias, bc * head_dim); // size of (bc * head_dim)
    T* shared_o = GetSharedPtr<T>(shared_mem_start, &ptr_bias, br * head_dim); // size of (br * head_dim)
    T* shared_dq = GetSharedPtr<T>(shared_mem_start, &ptr_bias, br * head_dim); // size of (br * head_dim)
    T* shared_dk = GetSharedPtr<T>(shared_mem_start, &ptr_bias, bc * head_dim); // size of (bc * head_dim)
    T* shared_dv = GetSharedPtr<T>(shared_mem_start, &ptr_bias, bc * head_dim); // size of (bc * head_dim)
    T* shared_do = GetSharedPtr<T>(shared_mem_start, &ptr_bias, br * head_dim); // size of (br * head_dim)
    T* shared_l = GetSharedPtr<T>(shared_mem_start, &ptr_bias, br);
    T* shared_m = GetSharedPtr<T>(shared_mem_start, &ptr_bias, br);
    T* shared_s = GetSharedPtr<T>(shared_mem_start, &ptr_bias, br * bc);
    T* shared_p = GetSharedPtr<T>(shared_mem_start, &ptr_bias, br * bc);
    T* shared_ds = GetSharedPtr<T>(shared_mem_start, &ptr_bias, br * bc);
    T* shared_dp = GetSharedPtr<T>(shared_mem_start, &ptr_bias, br * bc);
    T* shared_d = GetSharedPtr<T>(shared_mem_start, &ptr_bias, br);
    // printf("finish initializing shared memory\n");
    for (int j=0;j<outer_steps;++j){
        // load KV to on-chip memory
        // initialize dk, dv = 0 in shared memory
        
        int kv_per_thread = (head_dim + br -1) / br;
        for (int col_idx = 0; col_idx < kv_per_thread; ++col_idx){
            int ele_idx = threadIdx.y * kv_per_thread + col_idx;
            if (ele_idx < head_dim){
                shared_k[threadIdx.x * head_dim + ele_idx] = K[batch_id * stride_batch + head_id * stride_head + (j * bc + threadIdx.x) * stride_seq + ele_idx];
                shared_v[threadIdx.x * head_dim + ele_idx] = V[batch_id * stride_batch + head_id * stride_head + (j * bc + threadIdx.x) * stride_seq + ele_idx];
                // printf("loading kv to on-chip memory\n");
                shared_dk[threadIdx.x * head_dim + ele_idx] = 0;
                // printf("loading dk to on-chip memory\n");
                shared_dv[threadIdx.x * head_dim + ele_idx] = 0;
                // printf("loading dv to on-chip memory\n");
            }
        }
        // printf("finish loading kv to on-chip memory\n");
        __syncthreads();
        // inner loop
        for (int i = 0;i<inner_steps;++i){
            //load Q to on-chip memory

            int qo_per_thread = (head_dim + bc -1) / bc;
            for (int col_idx = 0; col_idx < qo_per_thread; ++col_idx){
                int ele_idx = threadIdx.x * qo_per_thread + col_idx;
                if (ele_idx < head_dim){
                    shared_q[threadIdx.y * head_dim + ele_idx] = Q[batch_id * stride_batch + head_id * stride_head + (i * br + threadIdx.y) * stride_seq + ele_idx];
                    shared_o[threadIdx.y * head_dim + ele_idx] = O[batch_id * stride_batch + head_id * stride_head + (i * br + threadIdx.y) * stride_seq + ele_idx];
                    // printf("loading q to on-chip memory\n");
                    shared_dq[threadIdx.y * head_dim + ele_idx] = dQ[batch_id * stride_batch + head_id * stride_head + (i * br + threadIdx.y) * stride_seq + ele_idx];
                    // printf("loading dq to on-chip memory\n");
                    shared_do[threadIdx.y * head_dim + ele_idx] = dO[batch_id * stride_batch + head_id * stride_head + (i * br + threadIdx.y) * stride_seq + ele_idx];
                    // printf("loading do to on-chip memory\n");
                }
            }
            // printf("finish loading q to on-chip memory\n");
            // always true for threadIdx.y < bc
            // load l and m to on-chip memory
            if (threadIdx.x == 0){
                shared_l[threadIdx.y] = L[batch_id * nhead * seq_len + head_id * seq_len + (i * br + threadIdx.y)]; 
                shared_m[threadIdx.y] = M[batch_id * nhead * seq_len + head_id * seq_len + (i * br + threadIdx.y)];
            }
            // printf("finish loading l and m to on-chip memory\n");
            __syncthreads();
            // compute attention
            T sum_ = 0;
            for (int k = 0; k < head_dim; ++k){
                sum_ += shared_q[threadIdx.y * head_dim + k] * shared_k[threadIdx.x * head_dim + k];
            }
            shared_s[threadIdx.y * bc + threadIdx.x] = sum_ * rsqrtf(head_dim) ;
            // printf("finish computing attention\n");
            __syncthreads();

            // TODO: Add mask

            // calculate p
            shared_p[threadIdx.y * bc + threadIdx.x] = __expf(shared_s[threadIdx.y * bc + threadIdx.x] - shared_m[threadIdx.y]) / shared_l[threadIdx.y];
            // printf("finish computing p\n");
            // TODO: Add dropout
            
            // calcluate dv
            for (int col_idx = 0; col_idx < kv_per_thread; ++col_idx){
                int ele_idx = threadIdx.y * kv_per_thread + col_idx;
                if (ele_idx < head_dim){
                    sum_ = 0;
                    for (int k = 0; k < br; ++k){
                        sum_ += shared_p[k * bc + threadIdx.x] * shared_do[k * head_dim + ele_idx];
                    }
                    shared_dv[threadIdx.x * head_dim + ele_idx] += sum_;
                }
            }
            // printf("finish computing dv\n");
            // calcluate dp
            sum_ = 0;
            for(int k=0;k<head_dim; ++k){
                sum_ += shared_do[threadIdx.y * head_dim + k] * shared_v[threadIdx.x * head_dim + k];
            }
            shared_dp[threadIdx.y * bc + threadIdx.x] = sum_;
            __syncthreads();
            // printf("finish computing dp\n");
            // calculate d, rowsum do*o
            if (threadIdx.x == 0){
                sum_ = 0;
                for (int k = 0; k < head_dim; ++k){
                    sum_ += shared_do[threadIdx.y * head_dim + k] * shared_o[threadIdx.y * head_dim + k];
                }
                shared_d[threadIdx.y] = sum_;
            }
            // printf("finish computing d\n");
            __syncthreads();
            // calculate ds
            shared_ds[threadIdx.y * bc + threadIdx.x] = shared_p[threadIdx.y * bc + threadIdx.x] * (shared_dp[threadIdx.y * bc + threadIdx.x] - shared_d[threadIdx.y]);
            // printf("finish computing ds\n");
            __syncthreads();
            // calculate dq
            for (int col_idx = 0; col_idx < qo_per_thread; ++col_idx){
                int ele_idx = threadIdx.x * qo_per_thread + col_idx;
                if (ele_idx < head_dim){
                    T sum_ = 0;
                    for (int k = 0; k < bc; ++k){
                        sum_ += shared_ds[threadIdx.y * bc + k] * shared_k[k * head_dim + ele_idx];
                    }
                    shared_dq[threadIdx.y * head_dim + ele_idx] += sum_ * rsqrtf(head_dim);
                }
            }
            __syncthreads();
            // printf("finish computing dq\n");
            // write back to HBM dq
            for (int col_idx = 0; col_idx < qo_per_thread; ++col_idx){
                int ele_idx = threadIdx.x * qo_per_thread + col_idx;
                if (ele_idx < head_dim){
                    dQ[batch_id * stride_batch + head_id * stride_head + (i * br + threadIdx.y) * stride_seq + ele_idx] = shared_dq[threadIdx.y * head_dim + ele_idx];
                }
            }
            __syncthreads();
            // printf("finish writing dq back to HBM\n");
            // calculate dk
            for (int col_idx = 0; col_idx < kv_per_thread; ++col_idx){
                int ele_idx = threadIdx.y * kv_per_thread + col_idx;
                if (ele_idx < head_dim){
                    T sum_ = 0;
                    for (int k = 0; k < br; ++k) {
                        sum_ += shared_ds[k * bc + threadIdx.x] * shared_q[k * head_dim + ele_idx];
                    }
                    shared_dk[threadIdx.x * head_dim + ele_idx] += sum_ * rsqrtf(head_dim);
                }
            }
            // printf("finish computing dk\n");
            __syncthreads();
        } // end of inner loop
        // write back to HBM, dv, dk
        for (int col_idx = 0; col_idx < kv_per_thread; ++col_idx){
            int ele_idx = threadIdx.y * kv_per_thread + col_idx;
            if (ele_idx < head_dim){
                dV[batch_id * stride_batch + head_id * stride_head + (j * bc + threadIdx.x) * stride_seq + ele_idx] = shared_dv[threadIdx.x * head_dim + ele_idx];
                dK[batch_id * stride_batch + head_id * stride_head + (j * bc + threadIdx.x) * stride_seq + ele_idx] = shared_dk[threadIdx.x * head_dim + ele_idx];
            }
        }
        __syncthreads(); 
    } // end of outer loop
} // flash_attn_bw

extern "C" {
void launch_flash_attn_bw(float *dQ, float *dK, float *dV, const float *dO, const float *Q, const float* K, const float * V, const float * O,
                                const float *L, const float *M,
                                int batch_size, int nhead, int seq_len, int head_dim,
                                bool is_causal,
                                cudaStream_t stream) {

    int float_size = sizeof(float);
    int qkv_size = batch_size * nhead * seq_len * head_dim * float_size;
    int lm_size = batch_size * nhead * seq_len * float_size;

    float *d_q, *d_k, *d_v, *d_o;
    float *grad_q, *grad_k, *grad_v, *grad_o;
    float * d_l, *d_m;
    cudaMalloc((void **)&d_q, qkv_size);
    cudaMalloc((void **)&d_k, qkv_size);
    cudaMalloc((void **)&d_v, qkv_size);
    cudaMalloc((void **)&d_o, qkv_size);
    cudaMalloc((void **)&d_l, lm_size);
    cudaMalloc((void **)&d_m, lm_size);
    cudaMalloc((void **)&grad_q, qkv_size);
    cudaMalloc((void **)&grad_k, qkv_size);
    cudaMalloc((void **)&grad_v, qkv_size);
    cudaMalloc((void **)&grad_o, qkv_size);
    // cudaMemset(d_l, 0, lm_size);
    // cudaMemset(d_m, 0, lm_size);

    cudaMemcpy(d_q, Q, qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, K, qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, V, qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_o, O, qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_l, L, lm_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, M, lm_size, cudaMemcpyHostToDevice);
    cudaMemcpy(grad_o, dO, qkv_size, cudaMemcpyHostToDevice);

    dim3 grid_dim(nhead, batch_size);

    // get shared memory size M
    cudaDeviceProp prop;
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaGetDeviceProperties(&prop, deviceId);
    int Mem = prop.sharedMemPerBlock;

    // calculate block size
    int bc, br;
    bc = min(Mem/(4*head_dim), 16);
    br = min(Mem/(4*head_dim), 16);

    dim3 block_dim(bc, br);

    // launch kernel
    int total_shared_mem_size = ((br * 4 + bc * 4) * head_dim + br * 3 + 4 * (br * bc)) * float_size;
    printf("here get the M size of %d with br size %d, head_dim %d and shared memory size %d\n", Mem, br,head_dim, total_shared_mem_size);

    flash_attn_bw<float><<<grid_dim, block_dim, total_shared_mem_size, stream>>>(grad_q, grad_k, grad_v, grad_o, d_q, d_k, d_v, d_o, d_l, d_m, seq_len,head_dim, nullptr, is_causal);
    // Synchronize and check for errors
    cudaDeviceSynchronize();
    // Check CUDA execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "launch_attn_softmax_bw Error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Copy back to the host
    cudaMemcpy(dQ, grad_q, qkv_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(dK, grad_k, qkv_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(dV, grad_v, qkv_size, cudaMemcpyDeviceToHost);

    // Free memory on device
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
    cudaFree(d_l);
    cudaFree(d_m);
    cudaFree(grad_q);
    cudaFree(grad_k);
    cudaFree(grad_v);
    cudaFree(grad_o);

} // launch_attn_softmax_bw
} // extern "C"
    
    } // namespace cuda
    } // namespace flash_attn
