using BenchmarkTools
using LinearAlgebra

using CUDA # or AMDGPU
atype = CuArray # or ROCArray

for N in 2 .^(1:10)
    println("N = $N")
    A = atype(rand(ComplexF64, N, N))
    B = atype(rand(ComplexF64, N, N))
    C = atype(zeros(ComplexF64, N, N))
    @btime CUDA.@sync mul!(C, A, B)
end
