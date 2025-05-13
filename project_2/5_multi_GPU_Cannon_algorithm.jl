using AMDGPU # or CUDA
using AMDGPU: device_id!, device
using GPUArrays
atype = ROCArray # CuArray

# Core function for GPU version of Cannon's algorithm
function gpu_cannon_multiply(A_global, B_global, p::Int)
    n = size(A_global, 1)
    block_size = div(n, p)
    C = atype(zeros(Float64, n, n))
    
    # Using AMDGPU kernel or broadcast multiplication
    @sync for i in 1:p, j in 1:p
        @async begin
            device_id = mod1((j-1) * p + i, 8)
            @show device_id
            device_id!(device_id)
            C_block_temp = atype(zeros(block_size,block_size))
            cache = GPUArrays.AllocCache()
            for k in 1:p
                # GPUArrays.@cached cache begin
                    # Calculate block positions
                    A_col = mod1(j-1 - (i-1) + k, p) 
                    B_row = mod1(i-1 - (j-1) + k, p) 

                    # Extract blocks
                    device_id!(1)
                    A_block = A_global[(i-1)*block_size+1:i*block_size, 
                                        (A_col-1)*block_size+1:A_col*block_size]
                    B_block = B_global[(B_row-1)*block_size+1:B_row*block_size,
                                        (j-1)*block_size+1:j*block_size]
                    device_id!(device_id)
                    A_block = ROCArray(A_block)
                    B_block = ROCArray(B_block)
                    C_block_temp += A_block * B_block
                # end
            end
            device_id!(1)
            C[(i-1)*block_size+1:i*block_size,
              (j-1)*block_size+1:j*block_size] .= ROCArray(C_block_temp)
            # GPUArrays.unsafe_free!(cache)
        end
    end

    return C
end

# Main function: coordinate GPU computation
function gpu_cannon(A, B, p)
    n = size(A, 1)
    @assert n % p == 0 "Matrix dimension must be multiple of process grid dimension"
    
    # Transfer data to GPU
    device_id!(1)
    A_gpu = atype(A)
    B_gpu = atype(B)
    
    # Execute GPU version of Cannon's algorithm
    C_gpu = gpu_cannon_multiply(A_gpu, B_gpu, p)
    
    # Transfer result back to CPU
    return Array(C_gpu)
end

# Test case
n = 10 # Matrix dimension
p = 2  # Process grid dimension (2x2)

# Create test matrices
A = rand(n,n)
B = rand(n,n)

# Run GPU algorithm
C_gpu = gpu_cannon(A, B, p)

# Standard result
C_standard = A * B

println("\nResults match: ", isapprox(C_gpu, C_standard))