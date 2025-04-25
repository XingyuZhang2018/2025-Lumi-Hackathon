using KernelAbstractions
using Test
using Random
using BenchmarkTools

using AMDGPU # or CUDA
atype = ROCArray # or CuArray

# Simple kernel for matrix multiplication
@kernel function matmul_kernel!(output, @Const(a), @Const(b))
    i, j = @index(Global, NTuple)

    # creating a temporary sum variable for matrix multiplication
    tmp_sum = zero(eltype(output))
    @inbounds @fastmath for k in 1:size(a)[2]
        tmp_sum += a[i, k] * b[k, j]
    end

    output[i, j] = tmp_sum
end

# Creating a wrapper kernel for launching with error checks
function matmul!(output, a, b)
    if size(a)[2] != size(b)[1]
        println("Matrix size mismatch!")
        return nothing
    end
    backend = KernelAbstractions.get_backend(a)
    kernel! = matmul_kernel!(backend)
    kernel!(output, a, b, ndrange = size(output))
    return
end

@testset "dims for $N, $R, $M" for (N,R,M) in [rand(500:1000,3) for _ in 1:5]
    A = atype(rand(ComplexF64, N, R))
    B = atype(rand(ComplexF64, R, M))
    C = atype(zeros(ComplexF64, N, M))

    backend = KernelAbstractions.get_backend(A)
    matmul!(C, A, B)
    KernelAbstractions.synchronize(backend)
    @test isapprox(C, A * B)
end