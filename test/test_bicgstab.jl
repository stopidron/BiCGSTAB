# test/runtests.jl
using Test
using LinearAlgebra
using PartitionedArrays
using .bicgstab

function create_sparsematrix(A::Matrix{Float64}, ranks; assembled)
    num_of_rows = size(A, 1)
    num_of_cols = size(A, 2)

    row_partition = uniform_partition(ranks, num_of_rows)
    col_partition = uniform_partition(ranks, num_of_cols)

    # display(row_partition)
    IJV = map(row_partition) do row_indices
        I = Int[]
        J = Int[]
        V = eltype(A)[]

        for row in row_indices
            for col in 1:num_of_cols
                val = A[row, col]
                if !iszero(val)
                    push!(I, row)
                    push!(J, col)
                    push!(V, val)
                end
            end
        end
        return (I, J, V)
    end

    AI, AJ, AV = tuple_of_arrays(IJV)

    t = psparse(AI, AJ, AV,
        row_partition, col_partition,
        assembled=assembled)

    A = fetch(t)
    # println("HELLOOOO")
    # display(A.row_partition)
    # display(A.col_partition)
    # display(local_values(A))
    # display(own_own_values(A))
    # display(ghost_own_values(A))
    # display(ghost_ghost_values(A))
    # display(own_ghost_values(A))
    return A
end

function create_pvector(b::Vector{Float64}, A::PSparseMatrix)
    A_col_partition = A.col_partition

    pv = pvector(A_col_partition) do local_indices
        global_indices = local_to_global(local_indices)
        return b[global_indices]
    end
    return pv
end

@testset "bicgstab solver" begin
    np_test = 2
    ranks = DebugArray(LinearIndices((np_test,)))
    tol = 1e-8
    max_iter = 30

    @testset "Test 1: 2x2 system" begin
        b_global = Float64[1, 2]
        A_global = Float64[4 1; 2 3]

        A = create_sparsematrix(A_global, ranks; assembled=false)
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

        A = create_sparsematrix(A_global, ranks; assembled=false)
        b = create_pvector(b_global, A)

        @assert size(A_global, 2) == length(b_global)

        state = initialize_state(A, b, tol=1e-8, max_iter=30)
        x, iter = solve_bicgstab!(state)
        tol = 1e-8
        x_vec = collect(x)
        expected = A_global \ b_global
        @test isapprox(x_vec, expected; atol=tol, rtol=0)

    end

    @testset "Test 3: singular" begin
        b_global = Float64[1, 1]
        A_global = Float64[1 1; 1 1]

        A = create_sparsematrix(A_global, ranks; assembled=false)
        b = create_pvector(b_global, A)

        @assert size(A_global, 2) == length(b_global)

        state = initialize_state(A, b, tol=1e-8, max_iter=30)
        x, iter = solve_bicgstab!(state)
        tol = 1e-8
        x_vec = collect(x)
        @test norm(A_global * x_vec - b_global) <= tol
    end

    @testset "Test 4: a 3x3 system" begin
        A_global = [3.0 2.0 -1.0;
            2.0 -2.0 4.0;
            -1.0 0.5 -1.0]
        b_global = [1.0, -2.0, 0.0]

        A = create_sparsematrix(A_global, ranks; assembled=false)
        b = create_pvector(b_global, A)

        state = initialize_state(A, b, tol=tol, max_iter=max_iter)
        x, iters = solve_bicgstab!(state)
        x_expected = A_global \ b_global

        x_vec = collect(x)
        @test isapprox(x_vec, x_expected; atol=tol, rtol=0)
        @test iters < max_iter
        @test norm(A_global * x_vec - b_global) < tol
    end

    @testset "Test 5: Larger System (5x5, Symmetric)" begin
        A_global = Float64[5 -1 0 0 0;
            -1 5 -1 0 0;
            0 -1 5 -1 0;
            0 0 -1 5 -1;
            0 0 0 -1 5]

        b_global = Float64[1, 2, 3, 4, 5]

        A = create_sparsematrix(A_global, ranks; assembled=false)
        b = create_pvector(b_global, A)

        state = initialize_state(A, b, tol=tol, max_iter=max_iter)
        x, iter = solve_bicgstab!(state)
        x_expected = A_global \ b_global
        x_vec = collect(x)

        @test isapprox(x_vec, x_expected; atol=tol, rtol=0)
        @test norm(A_global * x_vec - b_global) < tol 
    end
end