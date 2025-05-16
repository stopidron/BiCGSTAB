# test/runtests.jl
using Test
using LinearAlgebra
using PartitionedArrays
using Distributed
using SparseArrays
using .bicgstab

import PartitionedArrays as PA

np_test=3
n_matrix_dim = 2
n=2
b_global=Float64[1, 2]

#---------make spare matrix -------------------------------------------------

A_global_data = Float64[4  1;
                        2  3]
A_global_csc = SparseArrays.sparse(A_global_data) 

AI, AJ, AV = findnz(A_global_csc) # AI, AJ are global row/column indices
num_global_rows = size(A_global_csc, 1)
num_global_cols = size(A_global_csc, 2)
@assert n_matrix_dim == num_global_rows
#-----------------------------------------------------------------------------

#-------pvector construct-----------------------------------------------------

function create_pvector(distribute, b_global, np)
    ranks = distribute(LinearIndices((np,)))
    row_partition = uniform_partition(ranks, length(b_global))

    pb = pvector(row_partition) do local_indices_for_part
        global_indices_in_local_view = local_to_global(local_indices_for_part)
        return b_global[global_indices_in_local_view]
    end

    return pb
end

#---------------------------------------------------------------------------------

@testset "bicgstab solver test" begin

    ranks = PA.DebugArray(LinearIndices((np_test,)))
    row_partition = uniform_partition(ranks, num_global_rows)
    col_partition = uniform_partition(ranks, num_global_cols)

    vector_partition_layout = row_partition

    per_part_coo_triplets = map(row_partition) do current_part_row_localindices
        I_part_global = Int[]
        J_part_global = Int[]
        V_part = eltype(A_global_data)[]

        for g_row in PA.own_to_global(current_part_row_localindices)
            for g_col in 1:num_global_cols
                val = A_global_data[g_row, g_col]
                if !iszero(val)
                    push!(I_part_global, g_row)
                    push!(J_part_global, g_col)
                    push!(V_part, val)
                end
            end
        end
        return (I_part_global, J_part_global, V_part)
    end

    AI, AJ, AV = PA.tuple_of_arrays(per_part_coo_triplets)
    t = PA.psparse(AI, AJ, AV,
                                row_partition, col_partition,
                                assembled=false, indices=:global)

    A = fetch(t)
    #PA.centralize(A)

    #----------debug------------------------------------

    if PA.i_am_main(ranks)
        println("DEBUG: Centralized A for inspection = ", PA.centralize(A)) # If available and meaningful
    end
    # map(ranks, A.matrix_partition) do rank, local_mat
    #     println("DEBUG: Rank $rank, local matrix: ", local_mat)
    # end

    # println("HELLOOOO")
    # display(A.matrix_partition)
    # display(local_values(A))
    # display(own_own_values(A))
    # display(ghost_own_values(A))
    # display(ghost_ghost_values(A))

    final_A_col_partition = A.col_partition # Or partition(axes(A,2))

    #----------debug------------------------------------

    @assert length(b_global) == num_global_rows 

    b = pvector(final_A_col_partition) do local_indices_for_part #TODO changed row_partitioned to final_A_col_partition
        global_indices_in_local_view = local_to_global(local_indices_for_part)
        return b_global[global_indices_in_local_view]
    end
    #collect(b)
    #----------debug------------------------------------
    # b_collected = PartitionedArrays.collect(b)
    # println("DEBUG: Collected PVector b = ", b_collected)
    #----------debug------------------------------------
    
    state = initialize_state(A,b,tol=1e-8,max_iter=30)

    #----------debug------------------------------------
    # p_initial_collected = PartitionedArrays.collect(state.p)
    # r_initial_collected = PartitionedArrays.collect(state.r)
    # r_hat_initial_collected = PartitionedArrays.collect(state.r_hat)

    # println("DEBUG: Initial state.p (after initialize_state) = ", p_initial_collected)
    # println("DEBUG: Initial state.r (after initialize_state) = ", r_initial_collected)
    # println("DEBUG: Initial state.r_hat (after initialize_state) = ", r_hat_initial_collected)  
    #----------debug------------------------------------
    x, iter = solve_bicgstab!(state)

    tol=1e-8

    x_vec = collect(x)
    expected = A_global_data \ b_global
    @test isapprox(x_vec, expected; atol=tol, rtol=0)
end
#--------------------------------------------------------------------------------------