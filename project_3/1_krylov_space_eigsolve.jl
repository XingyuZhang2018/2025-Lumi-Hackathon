using KrylovKit
using LinearAlgebra
using BenchmarkTools
using Test
using Random

using AMDGPU # or CUDA
atype = ROCArray # or CuArray

@testset "eigsolve $atype" for atype in [Array, atype]
    Random.seed!(100)
    N = 10^3
    A = atype(rand(ComplexF64, N, N))
    v0 = atype(rand(ComplexF64, N))
    linearmap(v) = A * v
    println("dot product for $atype:")
    @btime AMDGPU.@sync dot($v0, $v0)
    println("linearmap for $atype:")
    @btime AMDGPU.@sync $linearmap($v0)
    println("eigsolve for $atype:")
    @btime AMDGPU.@sync λs, vs = eigsolve(v -> $linearmap(v), $v0, 1, :LM)
end
