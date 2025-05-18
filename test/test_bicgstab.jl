# test/runtests.jl
using Test
using LinearAlgebra
using PartitionedArrays
using Distributed
using SparseArrays
using .bicgstab

import PartitionedArrays as PA

function create_distr_matrix(A::Matrix{Float64}, ranks; assembled)
    num_global_rows = size(A, 1)
    num_global_cols = size(A, 2)

    row_partition = PA.uniform_partition(ranks, num_global_rows)
    col_partition = PA.uniform_partition(ranks, num_global_cols)

    display(row_partition)

    per_part_coo_triplets = map(row_partition) do row_indices
        I = Int[]
        J = Int[]
        V = eltype(A)[]

        for global_row in PA.own_to_global(row_indices)
            for global_col in 1:num_global_cols
                val = A[global_row, global_col]
                if !iszero(val)
                    push!(I, global_row)
                    push!(J, global_col)
                    push!(V, val)
                end
            end
        end
        return (I, J, V)
    end

    AI, AJ, AV = PA.tuple_of_arrays(per_part_coo_triplets)

    t = PA.psparse(AI, AJ, AV,
        row_partition, col_partition,
        assembled=assembled)

    A = fetch(t)
    return A
end

function create_pvector(b::Vector{Float64}, A::PA.PSparseMatrix)
    col_A = A.col_partition

    pv = pvector(col_A) do local_indices_for_part
        global_indices_in_local_view = PA.local_to_global(local_indices_for_part)
        return b[global_indices_in_local_view]
    end
    return pv
end

@testset "bicgstab solver" begin
    np_test = 2
    ranks = PA.DebugArray(LinearIndices((np_test,)))
    tol = 1e-8
    max_iter = 30

    @testset "Test 1: 2x2 system" begin
        b_global = Float64[1, 2]
        A_global = Float64[4 1; 2 3]

        A = create_distr_matrix(A_global, ranks; assembled=false)
        b = create_pvector(b_global, A)

        @assert size(A_global, 2) == length(b_global)

        state = initialize_state(A, b, tol=1e-8, max_iter=30)
        x, iter = solve_bicgstab!(state)
        tol = 1e-8
        x_vec = collect(x)
        expected = A_global \ b_global
        @test isapprox(x_vec, expected; atol=tol, rtol=0)
    end

    @testset "Test 2: 2x2 system" begin
        b_global = Float64[1, 1]
        A_global = Float64[3 1; 1 2]

        A = create_distr_matrix(A_global, ranks; assembled=false)
        b = create_pvector(b_global, A)

        @assert size(A_global, 2) == length(b_global)

        state = initialize_state(A, b, tol=1e-8, max_iter=30)
        x, iter = solve_bicgstab!(state)
        tol = 1e-8
        x_vec = collect(x)
        expected = A_global \ b_global
        @test isapprox(x_vec, expected; atol=tol, rtol=0)

    end

    @testset "Test 2: singular" begin
        b_global = Float64[1, 1]
        A_global = Float64[1 1; 1 1]

        A = create_distr_matrix(A_global, ranks; assembled=false)
        b = create_pvector(b_global, A)

        @assert size(A_global, 2) == length(b_global)

        state = initialize_state(A, b, tol=1e-8, max_iter=30)
        x, iter = solve_bicgstab!(state)
        tol = 1e-8
        x_vec = collect(x)
        @test norm(A_global * x_vec - b_global) <= tol
    end

    @testset "Test 5: a 3x3 system" begin
        A_global = [3.0 2.0 -1.0;
            2.0 -2.0 4.0;
            -1.0 0.5 -1.0]
        b_global = [1.0, -2.0, 0.0]

        A = create_distr_matrix(A_global, ranks; assembled=false)
        b= create_pvector(b_global, A)

        state = initialize_state(A, b, tol=tol, max_iter=max_iter)
        x, iters = solve_bicgstab!(state)
        x_expected = A_global \ b_global

        x_vec = collect(x)
        @test isapprox(x_vec, x_expected; atol=tol, rtol=0)
        @test iters < max_iter
        @test norm(A_global * x_vec - b_global) < tol
    end
end