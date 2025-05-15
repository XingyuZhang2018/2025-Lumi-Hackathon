using AMDGPU
using AMDGPU: device_id!, device
using Test
using BenchmarkTools
using LinearAlgebra
using Printf

atype = ROCArray
N = 2^10
device_id!(1)
A = atype(rand(ComplexF64, N,N))
@show device(A).device_id
device_id!(2)
B = atype(A)
@show device(B).device_id 
C = B*B
@show device(C).device_id 

function transfer_data_GPU_CPU_GPU(A; id=2)
    A_array = Array(A)
    device_id!(id)
    B = atype(A_array)
    return B
end

function transfer_GPU_GPU(A; id=2)
    device_id!(id)
    B = atype(A)
    return B
end

function multi_transfer(A, ids)
    for id in ids
        begin
            B = transfer_GPU_GPU(A; id=id)
        end
    end
end

t = []
for N in 2 .^ (2:2)
    device_id!(1)
    A = atype(rand(ComplexF64, N,N))
    # @btime AMDGPU.@sync transfer_GPU_GPU(A; id=2)
    # @btime AMDGPU.@sync multi_transfer(A, [2])
    t1 = @belapsed AMDGPU.@sync B = transfer_data_GPU_CPU_GPU($A)
    t2 = @belapsed AMDGPU.@sync B = transfer_GPU_GPU($A)
    push!(t,[N,t1,t2])
    @printf "size: %i × %i\n" N N
    @printf "GPU → CPU → GPU %.5f ms\n" t1*1000
    @printf "GPU → GPU %.5f ms\n" t2*1000
end

# for i in t
#     @printf "{%i,%.8f,%.8f}," i[1] i[2] i[3]
# end