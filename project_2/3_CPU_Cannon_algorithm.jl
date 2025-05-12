using Distributed

# Initialize process grid (assuming p x p processes)
p = 2  # Grid dimension (can be modified)
if nprocs() < p^2
    addprocs(p^2 - nprocs())  # Add needed worker processes
end
println("Started $(nprocs()) processes")
@everywhere using LinearAlgebra

# Define matrix block type
@everywhere struct MatrixBlock
    data::Matrix{Float64}
    i::Int
    j::Int
end

# Define Cannon's algorithm core function
@everywhere function cannon_multiply(A_global, B_global, p, proc_id)
    # Calculate process position in the grid
    i, j = divrem(proc_id - 1, p)
    i += 1  # Adjust to 1-based indexing
    j += 1
    
    # Print debug information
    println("Process $(myid()) handling grid position ($i, $j), proc_id = $proc_id")
    
    # Matrix block division
    n = size(A_global, 1)
    block_size = div(n, p)
    
    # Safely initialize and print block size
    println("Matrix size = $n, block size = $block_size")
    
    # Calculate initial block positions according to Cannon's algorithm
    # A matrix's each row shifts left by (i-1) steps
    # B matrix's each column shifts up by (j-1) steps
    A_col = mod(j-1 - (i-1), p) + 1
    B_row = mod(i-1 - (j-1), p) + 1
    
    # Extract initial blocks
    A_block = A_global[(i-1)*block_size+1:i*block_size, 
                       (A_col-1)*block_size+1:A_col*block_size]
    B_block = B_global[(B_row-1)*block_size+1:B_row*block_size, 
                       (j-1)*block_size+1:j*block_size]
    
    # Initialize result block
    C_block = zeros(block_size, block_size)
    
    # Main loop: p multiply-add operations
    for k in 1:p
        # Print current position information
        println("Process $(myid()): Iteration $k, using A[$i,$(A_col)] × B[$(B_row),$j]")
        
        # Local matrix multiplication
        C_block += A_block * B_block
        
        # Update block positions (circular left shift for A block, circular up shift for B block)
        A_col = mod(A_col, p) + 1
        B_row = mod(B_row, p) + 1
        
        # Get next blocks
        A_block = A_global[(i-1)*block_size+1:i*block_size, 
                          (A_col-1)*block_size+1:A_col*block_size]
        B_block = B_global[(B_row-1)*block_size+1:B_row*block_size, 
                          (j-1)*block_size+1:j*block_size]
    end
    
    return MatrixBlock(C_block, i, j)
end

# Main function: Coordinate parallel computation
function parallel_cannon(A, B, p)
    n = size(A, 1)
    
    # Ensure matrix size is a multiple of p
    @assert n % p == 0 "Matrix dimension must be a multiple of the process grid dimension"
    
    println("Starting parallel Cannon's algorithm")
    
    # Distribute data to all processes (not explicitly needed, as Julia's parallel model handles it automatically)
    # But we ensure all data is of basic type
    A_full = convert(Matrix{Float64}, A)  # Ensure standard matrix, avoid ReshapedArray issues
    B_full = convert(Matrix{Float64}, B)
    
    # Debug output
    println("A matrix $(size(A_full)):")
    display(A_full)
    println("\nB matrix $(size(B_full)):")
    display(B_full)
    
    # Execute Cannon's algorithm in parallel
    results = Vector{MatrixBlock}(undef, p^2)
    
    # Ensure we have enough workers
    available_workers = workers()
    if length(available_workers) < p^2
        @warn "Warning: Available worker count ($(length(available_workers))) less than required process count ($(p^2))"
    end
    
    for proc_id in 1:p^2
        # For even load distribution, may cycle through available workers
        worker = available_workers[mod(proc_id-1, length(available_workers)) + 1]
        println("Assigning position $proc_id to worker $worker")
        results[proc_id] = remotecall_fetch(cannon_multiply, worker, A_full, B_full, p, proc_id)
    end
    
    # Combine result blocks
    C = zeros(size(A))
    block_size = div(n, p)
    for block in results
        i, j = block.i, block.j
        C[(i-1)*block_size+1:i*block_size, (j-1)*block_size+1:j*block_size] = block.data
    end
    
    return C
end

# Test case
n = 4  # Matrix dimension
p = 2  # Process grid dimension (2x2)

# Use standard matrices instead of ReshapedArray
A = Matrix{Float64}(reshape(1.0:n^2, (n,n)))
B = ones(Float64, n, n)

# Run parallel algorithm
C_parallel = parallel_cannon(A, B, p)

# Standard result
C_standard = A * B

println("Final parallel result:")
display(C_parallel)
println("\nStandard result:")
display(C_standard)
println("\nResults match: ", isapprox(C_parallel, C_standard))