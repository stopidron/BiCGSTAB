# test/runtests.jl
using Test
using LinearAlgebra

using PartitionedArrays
using PartitionedArrays: distribute_with_mpi, get_diag
using MPI

using .bicgstab
using SparseArrays
using Random
using MAT


MPI.Init()

comm = MPI.COMM_WORLD
num_mpi_procs = MPI.Comm_size(comm)
mpi_rank = MPI.Comm_rank(comm)

original_stdout = stdout
if mpi_rank != 0
    redirect_stdout(devnull) # Suppress output from other ranks
    redirect_stderr(devnull) # Optionally suppress errors too, or log them to files
end


# function create_psparse(A::Matrix{Float64}, ranks; assembled)
#     num_global_rows = size(A, 1)
#     num_global_cols = size(A, 2)

#     row_partition = uniform_partition(ranks, num_global_rows)
#     col_partition = uniform_partition(ranks, num_global_cols)

#     per_part_coo_triplets = map(row_partition) do row_indices
#         I = Int[]
#         J = Int[]
#         V = eltype(A)[]

#         for global_row in row_indices
#             for global_col in 1:num_global_cols
#                 val = A[global_row, global_col]
#                 if !iszero(val)
#                     push!(I, global_row)
#                     push!(J, global_col)
#                     push!(V, val)
#                 end
#             end
#         end
#         return (I, J, V)
#     end

#     AI, AJ, AV = tuple_of_arrays(per_part_coo_triplets)

#     t = psparse(AI, AJ, AV,
#         row_partition, col_partition,
#         assembled=assembled)

#     A = fetch(t)

#     return A
# end

function create_diagonal_preconditioner(A::PSparseMatrix)
    d = get_diag(A)
    map!(val -> iszero(val) ? one(val) : val, d, d)

    return d
end

function create_psparse(A::SparseMatrixCSC{Float64,Int64}, ranks; assembled)
    num_global_rows = size(A, 1)
    num_global_cols = size(A, 2)

    row_partition = uniform_partition(ranks, num_global_rows)
    col_partition = uniform_partition(ranks, num_global_cols)

    per_part_coo_triplets = map(row_partition) do row_indices
        I = Int[]
        J = Int[]
        V = eltype(A)[]

        for global_row in row_indices
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

    AI, AJ, AV = tuple_of_arrays(per_part_coo_triplets)

    t = psparse(AI, AJ, AV,
        row_partition, col_partition,
        assembled=assembled)

    A_ps = fetch(t)

    return A_ps
end

function create_pvector(b::Vector{Float64}, A::PSparseMatrix)
    col_A = A.col_partition

    pv = pvector(col_A) do local_indices_for_part
        return b[local_indices_for_part]
    end
    return pv
end

# @testset "bicgstab solver" begin
#     ranks = distribute_with_mpi(LinearIndices((num_mpi_procs,)))

#     if mpi_rank == 0
#         println("Running BiCGSTAB tests on $num_mpi_procs MPI processes.\n")
#     end

#     tol = 1e-8
#     max_iter = 200

#     # @testset "Test 2: 2x2 system" begin
#     #     b_global = Float64[1, 1]
#     #     A_global = Float64[3 1; 1 2]

#     #     A = create_psparse(A_global, ranks; assembled=false)
#     #     b = create_pvector(b_global, A)

#     #     state = initialize_state(A, b, tol=1e-8, max_iter=30)
#     #     x, iter = solve_bicgstab!(state)
#     #     tol = 1e-8
#     #     x_vec = collect(x)
#     #     expected = A_global \ b_global
#     #     @test isapprox(x_vec, expected; atol=tol, rtol=0)

#     # end

#     # @testset "Test 3: singular" begin
#     #     b_global = Float64[1, 1]
#     #     A_global = Float64[1 1; 1 1]

#     #     A = create_psparse(A_global, ranks; assembled=false)
#     #     b = create_pvector(b_global, A)

#     #     state = initialize_state(A, b, tol=1e-8, max_iter=30)
#     #     x, iter = solve_bicgstab!(state)
#     #     tol = 1e-8
#     #     x_vec = collect(x)
#     #     @test norm(A_global * x_vec - b_global) <= tol
#     # end

#     # @testset "Test 4: a 3x3 system" begin
#     #     A_global = [3.0 2.0 -1.0;
#     #         2.0 -2.0 4.0;
#     #         -1.0 0.5 -1.0]
#     #     b_global = [1.0, -2.0, 0.0]

#     #     A = create_psparse(A_global, ranks; assembled=false)
#     #     b = create_pvector(b_global, A)

#     #     state = initialize_state(A, b, tol=tol, max_iter=max_iter)
#     #     x, iters = solve_bicgstab!(state)
#     #     x_expected = A_global \ b_global

#     #     x_vec = collect(x)
#     #     @test isapprox(x_vec, x_expected; atol=tol, rtol=0)
#     #     @test iters < max_iter
#     #     @test norm(A_global * x_vec - b_global) < tol
#     # end

#     # @testset "Test 5: Larger System (5x5, Symmetric)" begin
#     #     A_global = Float64[5 -1 0 0 0;
#     #         -1 5 -1 0 0;
#     #         0 -1 5 -1 0;
#     #         0 0 -1 5 -1;
#     #         0 0 0 -1 5]

#     #     b_global = Float64[1, 2, 3, 4, 5]

#     #     A = create_psparse(A_global, ranks; assembled=false)
#     #     b = create_pvector(b_global, A)

#     #     state = initialize_state(A, b, tol=tol, max_iter=max_iter)
#     #     x, iter = solve_bicgstab!(state)
#     #     x_expected = A_global \ b_global
#     #     x_vec = collect(x)

#     #     @test isapprox(x_vec, x_expected; atol=tol, rtol=0)
#     #     @test norm(A_global * x_vec - b_global) < tol
#     # end

#     # @testset "Test 6: 14x14 system: too ill condition doesn't work" begin
#     #     A_global_singular = Float64[
#     #         1 0 0 0 0 0 1 0 0 0 0 0 0 0;
#     #         0 1 0 0 0 0 0 1 1 0 0 0 0 0;
#     #         0 0 1 0 0 0 0 0 0 1 0 0 0 0;
#     #         0 0 0 1 0 0 0 0 0 0 -1 0 0 0;
#     #         0 0 0 0 1 0 0 0 0 0 0 -1 -1 0;
#     #         0 0 0 0 0 1 0 0 0 0 0 0 0 -1;
#     #         1 0 0 0 0 0 2 1 0 0 -1 -1 0 0;
#     #         0 1 0 0 0 0 1 2 1 0 -1 -1 0 0;
#     #         0 1 0 0 0 0 0 1 2 1 0 0 -1 -1;
#     #         0 0 1 0 0 0 0 0 1 2 0 0 -1 -1;
#     #         0 0 0 -1 0 0 -1 -1 0 0 2 1 0 0;
#     #         0 0 0 0 -1 0 -1 -1 0 0 1 2 1 0;
#     #         0 0 0 0 -1 0 0 0 -1 -1 0 1 2 1;
#     #         0 0 0 0 0 -1 0 0 -1 -1 0 0 1 2
#     #     ]

#     #     epsilon = 1e-8
#     #     A_global = A_global_singular + epsilon * I

#     #     if mpi_rank == 0
#     #         println("Condition number of A_global: ", cond(A_global))
#     #     end

#     #     b_global = Float64[20, 20, 20, -10, -20, -20, 20, 20, 20, 20, 10, 20, 20, 30]

#     #     A = create_psparse(A_global, ranks; assembled=false)
#     #     b = create_pvector(b_global, A)

#     #     state = initialize_state(A, b, tol=1e-8, max_iter=max_iter)
#     #     x, iter = solve_bicgstab!(state)
#     #     tol = 1e-8
#     #     x_vec = collect(x)
#     #     current_residual_norm = norm(A_global * x_vec - b_global)
#     #     @test current_residual_norm < tol

#     # end
# end

#**************SUIT SPARSE MATRIX PROBLEMS******************************
tol = 1e-8
ranks = distribute_with_mpi(LinearIndices((num_mpi_procs,)))

if mpi_rank == 0
    println("Running BiCGSTAB tests on $num_mpi_procs MPI processes.\n")
end

# @testset "Stranke 10x10" begin
#     max_iter = 30
#     mat_name = "test/problems/Stranke94.mat"
#     dict = MAT.matread(mat_name)
#     A_global = dict["Problem"]["A"]
#     dims = size(A_global, 1)

#     # x_true = ones(Float64, dims)
#     x_true = Float64.(1:dims)
#     b_global = A_global * x_true

#     A = create_psparse(A_global, ranks; assembled=false)
#     b = create_pvector(b_global, A)

#     state = initialize_state(A, b, tol=tol, max_iter=max_iter)
#     x, i = solve_bicgstab!(state)

#     x_vec = collect(x)
#     current_residual_norm = norm(A_global * x_vec - b_global)
#     @test current_residual_norm < tol

# end

# @testset "mycielskian5 23x23" begin
#     max_iter = 100
#     mat_name = "test/problems/mycielskian5.mat"
#     dict = MAT.matread(mat_name)
#     A_global = dict["Problem"]["A"]
#     dims = size(A_global, 1)

#     # x_true = ones(Float64, dims)
#     x_true = Float64.(1:dims)
#     b_global = A_global * x_true

#     A = create_psparse(A_global, ranks; assembled=false)
#     b = create_pvector(b_global, A)

#     state = initialize_state(A, b, tol=tol, max_iter=max_iter)
#     x, i = solve_bicgstab!(state)

#     x_vec = collect(x)
#     current_residual_norm = norm(A_global * x_vec - b_global)
#     @test current_residual_norm < tol
# end

# @testset "dolphins 62x62" begin
#     max_iter = 200
#     mat_name = "test/problems/dolphins.mat"
#     dict = MAT.matread(mat_name)
#     A_global = dict["Problem"]["A"]
#     dims = size(A_global, 1)

#     # x_true = ones(Float64, dims)
#     x_true = Float64.(1:dims)
#     b_global = A_global * x_true

#     A = create_psparse(A_global, ranks; assembled=false)
#     b = create_pvector(b_global, A)

#     state = initialize_state(A, b, tol=tol, max_iter=max_iter)
#     x, i = solve_bicgstab!(state)

#     x_vec = collect(x)
#     current_residual_norm = norm(A_global * x_vec - b_global)
#     @test current_residual_norm < tol
# end

# @testset " bcsstm07 420x420: Ill conditiond" begin
#     max_iter = 1000
#     mat_name = "test/problems/bcsstm07.mat"
#     dict = MAT.matread(mat_name)
#     A_global = dict["Problem"]["A"]
#     dims = size(A_global, 1)

#     # x_true = ones(Float64, dims)
#     x_true = Float64.(1:dims)
#     b_global = A_global * x_true

#     A = create_psparse(A_global, ranks; assembled=false)
#     b = create_pvector(b_global, A)

#     state = initialize_state(A, b, tol=tol, max_iter=max_iter)
#     x, i = solve_bicgstab!(state)

#     x_vec = collect(x)
#     current_residual_norm = norm(A_global * x_vec - b_global)
#     @test current_residual_norm < tol
# end

# @testset " rdb 450x450" begin
#     max_iter = 1000
#     mat_name = "test/problems/rdb450l.mat"
#     dict = MAT.matread(mat_name)
#     A_global = dict["Problem"]["A"]
#     dims = size(A_global, 1)

#     x_true = ones(Float64, dims)
#     # x_true = Float64.(1:dims)
#     b_global = A_global * x_true

#     A = create_psparse(A_global, ranks; assembled=false)
#     b = create_pvector(b_global, A)

#     state = initialize_state(A, b, tol=tol, max_iter=max_iter)
#     x, i = solve_bicgstab!(state)

#     x_vec = collect(x)
#     current_residual_norm = norm(A_global * x_vec - b_global)
#     @test current_residual_norm < tol
# end


#**************SUIT SPARSE MATRIX PROBLEMS******************************

if mpi_rank != 0
    redirect_stdout(original_stdout)
    redirect_stderr(original_stdout)
end
MPI.Finalize()