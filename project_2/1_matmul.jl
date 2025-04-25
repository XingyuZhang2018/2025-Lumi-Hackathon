using BenchmarkTools
using LinearAlgebra

using AMDGPU # or CUDA
atype = ROCArray # or CuArray

for N in 2 .^(5:10)
    println("N = $N")
    A = atype(rand(ComplexF64, N, N))
    B = atype(rand(ComplexF64, N, N))
    C = atype(zeros(ComplexF64, N, N))
    @btime AMDGPU.@sync mul!($C, $A, $B)
end
