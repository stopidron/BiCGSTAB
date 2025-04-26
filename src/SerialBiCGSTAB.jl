module SerialBiCGSTAB

export solve_bicgstab!, initialize_state

using LinearAlgebra

mutable struct BiCGSTAB{T}
  A::AbstractMatrix{T}
  b::Vector{T}
  x::Vector{T}
  r::Vector{T}
  r_hat::Vector{T}
  p::Vector{T}
  v::Vector{T}
  h::Vector{T}
  s::Vector{T}
  t::Vector{T}
  alpha::T
  omega::T
  rho_old::T
  tol::T
  max_iter::Int
end

function initialize_state(A, b; tol=1e-8, max_iter=30)
  T = eltype(b)
  n = length(b)
  x = zeros(T, n) #similar(b) maybe 
  r = b - A * x #mul!()
  return BiCGSTAB(
      A, b, x, r, rand(T,n), #maybe do similar(b) here if it saves memory idk 
      zeros(T, n), zeros(T, n),
      zeros(T, n), zeros(T, n), zeros(T, n),
      one(T), one(T), one(T),
      tol, max_iter
  )
end

function update_p!(state, rho_new, iter)
  if iter == 1
      state.p .= state.r
  else
      beta = (rho_new / state.rho_old) * (state.alpha / state.omega)
      state.p .= state.r .+ beta .* (state.p .- state.omega .* state.v)
  end
end

function step!(state, iter)
  A = state.A
  r_hat = state.r_hat
  r = state.r

  rho_new = dot(r_hat, r)
  if abs(rho_new) < 1e-14
      error("Breakdown: rho_new == 0")
  end

  update_p!(state, rho_new, iter) 
  state.v .= A * state.p
  state.alpha = rho_new / dot(r_hat, state.v)

  state.h .= state.x .+ state.alpha .* state.p
  state.s .= state.r .- state.alpha .* state.v

  if norm(state.s) < state.tol
      state.x .= state.h
      return true 
  end

  state.t .= A * state.s
  denom = dot(state.t, state.t)
  if denom == 0.0
      error("Breakdown: t ⋅ t == 0")
  end

  state.omega = dot(state.t, state.s) / denom
  state.x .= state.h .+ state.omega .* state.s
  state.r .= state.s .- state.omega .* state.t
  state.rho_old = rho_new

  return norm(state.r) < state.tol
end

function solve_bicgstab!(state)
  if norm(state.r) < state.tol
    println("Initial guess is already within tolerance.")
    return state.x, 0 
  end
  for i in 1:state.max_iter #aaaaaaa noo iteration split each iteration to a proccess idek
      if step!(state, i)
          println("Converged at step $i")
          return state.x, i
      end
  end
  @warn "Did not converge."
  return state.x, state.max_iter
end
end

# parallel 
# Extract sxs of the matrix, expected ranks
# give ghost alues to x using A
# b doesn't need ghost values
# homework: try to run in parallel using mpi: look at documentation
# implement solver interface bc how user friendly is it, highly recommended 😱