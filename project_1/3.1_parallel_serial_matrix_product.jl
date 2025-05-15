using LinearAlgebra
using BenchmarkTools
using KernelAbstractions
using Random
using Test
using AMDGPU # or CUDA
using AMDGPU: device_id!, device
atype = ROCArray # or CuArray

# Test cases
Random.seed!(1234)
matrix_sizes = Tuple([Tuple(rand(500:1000,3)) for _ in 1:100])
Adim = sum(map(m->prod(m[[1,2]]), matrix_sizes))
Bdim = sum(map(m->prod(m[[2,3]]), matrix_sizes))
Cdim = sum(map(m->prod(m[[1,3]]), matrix_sizes))
device_id!(1)
a = atype(rand(ComplexF64, Adim));
b = atype(rand(ComplexF64, Bdim));
c_1 = atype(zeros(ComplexF64, Cdim));
c_2 = atype(zeros(ComplexF64, Cdim));

# serial 
function serial_matrix_product(A, B, C, matrix_sizes)
    N = length(matrix_sizes)
    prefix_sumA = [0; cumsum([prod(d[[1,2]]) for d in matrix_sizes])]
    prefix_sumB = [0; cumsum([prod(d[[2,3]]) for d in matrix_sizes])]
    prefix_sumC = [0; cumsum([prod(d[[1,3]]) for d in matrix_sizes])]

    for i in 1:N
        mul!(reshape(view(C, prefix_sumC[i]+1:prefix_sumC[i+1]), matrix_sizes[i][[1,3]]), 
             reshape(view(A, prefix_sumA[i]+1:prefix_sumA[i+1]), matrix_sizes[i][[1,2]]), 
             reshape(view(B, prefix_sumB[i]+1:prefix_sumB[i+1]), matrix_sizes[i][[2,3]]) )
    end
    return C
end

# parallel
function parallel_serial_matrix_product(A1, B1, C1, A2, B2, C2, matrix_sizes; N_device=2)
    N = length(matrix_sizes)
    prefix_sumA = [0; cumsum([prod(d[[1,2]]) for d in matrix_sizes])]
    prefix_sumB = [0; cumsum([prod(d[[2,3]]) for d in matrix_sizes])]
    prefix_sumC = [0; cumsum([prod(d[[1,3]]) for d in matrix_sizes])]

    @sync begin
        Threads.@spawn begin
            device_id!(1)
            for i in 1:div(N,2)
                mul!(reshape(view(C1, prefix_sumC[i]+1:prefix_sumC[i+1]), matrix_sizes[i][[1,3]]), 
                     reshape(view(A1, prefix_sumA[i]+1:prefix_sumA[i+1]), matrix_sizes[i][[1,2]]), 
                     reshape(view(B1, prefix_sumB[i]+1:prefix_sumB[i+1]), matrix_sizes[i][[2,3]]) )
            end
        end
        Threads.@spawn begin
            device_id!(2)
            for i in (div(N,2)+1):N
                mul!(reshape(view(C2, prefix_sumC[i]+1:prefix_sumC[i+1]), matrix_sizes[i][[1,3]]), 
                     reshape(view(A2, prefix_sumA[i]+1:prefix_sumA[i+1]), matrix_sizes[i][[1,2]]), 
                     reshape(view(B2, prefix_sumB[i]+1:prefix_sumB[i+1]), matrix_sizes[i][[2,3]]) )
            end
        end
    end
    device_id!(1)
    C1 .+= atype(C2)
    return C1
end

serial_matrix_product(a, b, c_1, matrix_sizes);

device_id!(2)
c2 = AMDGPU.zeros(ComplexF64, Cdim);
a2 = atype(a)
b2 = atype(b)
parallel_serial_matrix_product(a, b, c_2, a2, b2, c2, matrix_sizes; N_device=2);
# @test c_1 ≈ c_2

println("serial_matrix_product (GPU):")
@btime AMDGPU.@sync serial_matrix_product($a, $b, $c_1, $matrix_sizes);
println("parallel_serial_matrix_product (GPU):")
@btime AMDGPU.@sync parallel_serial_matrix_product($a, $b, $c_2, $a2, $b2, $c2, $matrix_sizes; N_device=2);