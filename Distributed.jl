using Distributed, AMDGPU
addprocs(length(AMDGPU.devices()))
@everywhere using AMDGPU

# assign devices
asyncmap((zip(workers(), AMDGPU.devices()))) do (p, d)
    remotecall_wait(p) do
        @info "Worker $p uses $d"
        AMDGPU.device!(d)
        A = AMDGPU.rand(10,10)
        B = AMDGPU.rand(10,10)
        C = A*B
        @show AMDGPU.device(C)
    end
end
