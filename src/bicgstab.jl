module bicgstab

export solve_bicgstab!, initialize_state

using PartitionedArrays
using LinearAlgebra

mutable struct BiCGSTAB{Tval, TA <: PSparseMatrix, TVec <: PVector}
  A::TA
  b::TVec
  x::TVec
  r::TVec
  r_hat::TVec
  p::TVec
  v::TVec
  h::TVec
  s::TVec
  t::TVec
  alpha::Tval
  omega::Tval
  rho_old::Tval
  tol::Tval
  max_iter::Int
end

function initialize_state(A::PSparseMatrix, b::PVector; tol=1e-8, max_iter=30) # where T

  T = eltype(A) 
  @assert eltype(b) == T 
  x = similar(b, T); x .= zero(T) #TODO might not work
  r = copy(b)

  #r .-= A * x :not needed

  #r_hat = prand(b)
  partition_info = axes(b, 1) ;r_hat = prand(T,partition_info)


  p     = similar(b, T); p .= zero(T)
  v     = similar(b, T); v .= zero(T)
  h     = similar(b, T); h .= zero(T)
  s     = similar(b, T); s .= zero(T)
  t     = similar(b, T); t .= zero(T)
  

  _one = one(T)
  _tol = T(tol) 


  return BiCGSTAB(
      A, b, x, r, r_hat,
      p, v, h, s, t,
      _one, _one, _one,
      _tol, max_iter
  )
end

function update_p!(state, rho_new, iter)
  if iter == 1
    copyto!(state.p, state.r)

    #state.p .= state.r
  else
      beta = (rho_new / state.rho_old) * (state.alpha / state.omega)

      tmp = copy(state.p) #temp = step.p 
      LinearAlgebra.axpy!(-state.omega, state.v, tmp) #tmp = -o*v+p
      copyto!(state.p, state.r) #p=r
      LinearAlgebra.axpy!(beta, tmp, state.p) #p (aka r) =beta*tmp + p 

     #state.p .= state.r .+ beta .* (state.p .- state.omega .* state.v)
  end
end

function step!(state, iter)
  A = state.A
  r_hat = state.r_hat
  r = state.r

  #rho_new = dot(r_hat, r)
  rho_new = LinearAlgebra.dot(r_hat, r) #TODO maybe not needed
  if abs(rho_new) < 1e-14
      error("rho_new == 0")
  end

  update_p!(state, rho_new, iter)
  #state.v .= A * state.p
  mul!(state.v, A, state.p)
  
  # dot_rhat_v = LinearAlgebra.dot(r_hat, state.v) TODO add
  # if abs(dot_rhat_v) < 1e-14 # Add a safety check
  #   error("Breakdown: dot(r_hat, v) is near zero.")
  # end
  # state.alpha = rho_new / dot_rhat_v
  
  state.alpha = rho_new / LinearAlgebra.dot(r_hat, state.v)#TODO maybe not needed
  #state.alpha = rho_new / dot(r_hat, state.v) TODO 

  copyto!(state.h, state.x)
  LinearAlgebra.axpy!(state.alpha, state.p, state.h)
  #state.h .= state.x .+ state.alpha .* state.p TODO check 

  copyto!(state.s, state.r)
  LinearAlgebra.axpy!(-state.alpha, state.v, state.s)
  #state.s .= state.r .- state.alpha .* state.v TODO

  norm_s = LinearAlgebra.norm(state.s)
  if norm_s < state.tol
    state.x .= state.h
    return true 
  end

  mul!(state.t, A, state.s)
  #state.t .= A * state.s

  denom = LinearAlgebra.dot(state.t, state.t) #TODO
  if denom == 0.0
      error("Breakdown: t ⋅ t == 0")
  end

  state.omega = LinearAlgebra.dot(state.t, state.s) / denom #TODO

  copyto!(state.x, state.h)
  LinearAlgebra.axpy!(state.omega, state.s, state.x)
  #state.x .= state.h .+ state.omega .* state.s

  copyto!(state.r, state.s)
  LinearAlgebra.axpy!(-state.omega, state.t, state.r)
  #state.r .= state.s .- state.omega .* state.t
  state.rho_old = rho_new

  return LinearAlgebra.norm(state.r) < state.tol #TODO
end

function solve_bicgstab!(state)
  norm_r = LinearAlgebra.norm(state.r)

  if i_am_main(state.r) 
     if norm_r < state.tol
       println("Initial guess is already within tolerance.")
     end
  end
  if norm_r < state.tol
    return state.x, 0
  end

  converged_iter = 0
  converged = false
  for i in 1:state.max_iter
      if step!(state, i)
          converged_iter = i
          converged = true
          if i_am_main(state.r)
              println("Converged at step $i")
          end
          break
      end
  end

  if !converged && i_am_main(state.r)
      @warn "Did not converge after $(state.max_iter) iterations."
  end

  return state.x, (converged ? converged_iter : state.max_iter)
end
end