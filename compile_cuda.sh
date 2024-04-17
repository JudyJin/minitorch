mkdir -p minitorch/cuda_kernels
nvcc -o minitorch/cuda_kernels/combine.so --shared src/combine.cu -Xcompiler -fPIC -G -g
nvcc -o minitorch/cuda_kernels/softmax_kernel.so --shared  src/softmax_kernel.cu -Xcompiler -fPIC -G -g
nvcc -o minitorch/cuda_kernels/layernorm_kernel.so --shared src/layernorm_kernel.cu -Xcompiler -fPIC -G -g
nvcc -o minitorch/cuda_kernels/flash_attention.so --shared src/flash_attention.cu -Xcompiler -fPIC -G -g

