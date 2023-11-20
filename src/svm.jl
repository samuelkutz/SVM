export SVC 

using LinearAlgebra

mutable struct SVC 
    options :: Dict{Symbol, Any}
end

const svc_options = [(:C, 1.0), (:kernel, :rbf), (:degree, 3), (:gamma, :scale), (:coef0, 0.0)]

function SVC(; kwargs...)
    options = Dict{Symbol, Any}(k => get(kwargs, k, y) for (k, v) in svc_options)

    for k in keys(kwargs)
        if !(k in getfield.(svc_options, 1))
            @warn "Keyword argument $k ignored" 
        end
    end

    return SVC(options) # returns the struct SVC
end

function fit!(model :: SVC,
              X :: Matrix, 
              y :: Vector)

  kernel = model.options[:kernel]
  n, p = size(X)

  Ker = (X :: Matrix, y :: Vector) =>
  if kernel == :rbf
    γ = model.options[:gamma]
    if !(γ isa Number)
      γ = (γ == :scale ? 1 / n / p : 1 / n)
    end
    (x1, x2) -> exp(-γ * norm(x1 - x2)^2)
  elseif kernel == :poly
    d = model.options[:degree]
    c₀ = model.options[:coef0]
    (x1, x2) -> (c₀ + dot(x1, x2))^d
  elseif kernel == :sigmoid
    γ = model.options[:gamma]
    if !(γ isa Number)
      γ = (γ == :scale ? 1 / n / p : 1 / n)
    end
    c₀ = model.options[:coef0]
    (x1, x2) -> tanh(c₀ + γ * dot(x1, x2))
  elseif kernel == :linear
    (x1, x2) -> dot(x1, x2)
  end

  Kmat = [Ker(X[i,:], X[j,:]) for i = 1:n, j = 1:n] # calculates the Kernel chosen in all points X

  C = model.options[:C]

  # implementation of SMO here

  end

function predict(model :: SVC, X :: Matrix; prob = false)
end