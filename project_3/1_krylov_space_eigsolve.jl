using KrylovKit
using LinearAlgebra
using BenchmarkTools
using Test
using Random

using CUDA # or AMDGPU
atype = CuArray # or ROCArray

@testset "eigsolve $atype" for atype in [Array, atype]
    Random.seed!(100)
    N = 10^3
    A = atype(rand(ComplexF64, N, N))
    v0 = atype(rand(ComplexF64, N))
    linearmap(v) = A * v
    println("dot product for $atype:")
    @btime CUDA.@sync dot($v0, $v0)
    println("linearmap for $atype:")
    @btime CUDA.@sync $linearmap($v0)
    println("eigsolve for $atype:")
    @btime CUDA.@sync λs, vs = eigsolve(v -> $linearmap(v), $v0, 1, :LM)
end
