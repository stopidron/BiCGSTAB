# test/runtests.jl
using Test
using LinearAlgebra
using ..bicgstab
using PartitionedArrays
using Distributed
using SparseArrays

addprocs(2)

@everywhere using Pkg; Pkg.activate(".")
@everywhere using PartitionedArrays, LinearAlgebra, SparseArrays

using ..bicgstab

PartitionedArrays.with_debug() do distribute
    ranks = distribute(LinearIndices((2,)))

    A_global = sparse([1,1,2,2], [1,2,1,2], [4.0, 1.0, 2.0, 3.0], 2, 2)
    b_global = [1.0, 2.0]

    row_partition = uniform_partition(ranks, size(A_global, 1))
    col_partition = uniform_partition(ranks, size(A_global, 2))

    A = psparse(row_partition, col_partition, assembled=false) do my_rows, my_cols
        A_global[my_rows, my_cols]
    end

    b = pvector(row_partition) do my_rows
        @show typeof(b_global[my_rows])
        (b_global[my_rows])  #TODO
    end
    @show eltype(b)

    state = initialize_state(A, b; tol=1e-9, max_iter=10)
    x, iters = solve_bicgstab!(state)

    if i_am_main(x)
        println("Solution: ", collect(x))
        println("Iterations: ", iters)
        @test LinearAlgebra.norm(A_global * collect(x) - b_global) < 1e-8
    end
end
