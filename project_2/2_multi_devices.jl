using BenchmarkTools
using LinearAlgebra
using Test
using AMDGPU # or CUDA
atype = ROCArray # or CuArray

@testset for N in 2 .^(5:10)
    println("N = $N")
    AMDGPU.device_id!(1)
    A1 = atype(rand(ComplexF64, N, N))
    B1 = atype(rand(ComplexF64, N, N))
    C1 = atype(zeros(ComplexF64, N, N))
    AMDGPU.device_id!(2)
    A2 = atype(rand(ComplexF64, N, N))
    B2 = atype(rand(ComplexF64, N, N))
    C2 = atype(zeros(ComplexF64, N, N))
    function multi_matmul!(C1, C2, A1, A2, B1, B2)
        @sync begin
            @async begin 
                AMDGPU.device_id!(1)
                mul!(C1, A1, B1)
            end
            @async begin
                AMDGPU.device_id!(2)
                mul!(C2, A2, B2)
            end
        end
        return C1, C2
    end

    multi_matmul!(C1, C2, A1, A2, B1, B2)
    AMDGPU.device_id!(1)
    @test C1 ≈ A1*B1

    AMDGPU.device_id!(2)
    @test C2 ≈ A2*B2
    @btime AMDGPU.@sync $multi_matmul!($C1, $C2, $A1, $A2, $B1, $B2)
end
