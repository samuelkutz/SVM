export SVC 

using LinearAlgebra, Random

mutable struct SVC 
  α :: Vector{Float64}
  f :: Function
  options :: Dict{Symbol, Any}
end

const svc_options = [(:C, 1.0), (:kernel, :rbf), (:degree, 3), (:gamma, :scale), (:coef0, 0.0), (:max_iter, 1000)]

function SVC(; kwargs...)
    options = Dict{Symbol, Any}(k -> get(kwargs, k, y) for (k, v) in svc_options)

    for k in keys(kwargs)
        if !(k in getfield.(svc_options, 1))
            @warn "Keyword argument $k ignored" 
        end
    end

    return SVC([], x -> 0.0, options) # returns the struct SVC
end

function fit!(model :: SVC,
              X :: Matrix, 
              y :: Vector)

    kernel = model.options[:kernel]
    n, p = size(X) # n - lines, p - columns (features)

    Ker = 
    if kernel == :rbf
        γ = model.options[:gamma]

        if !(γ isa Number)
            γ = (γ == :scale ? 1 / n / p : 1 / n) 
        end

        (x1, x2) -> exp(-γ * norm(x1 - x2)^2) # rbf
    elseif kernel == :poly
        d = model.options[:degree]
        c₀ = model.options[:coef0]

        (x1, x2) -> (c₀ + dot(x1, x2))^d # polynomial
    elseif kernel == :sigmoid
        γ = model.options[:gamma]
        c₀ = model.options[:coef0]

        if !(γ isa Number)
            γ = (γ == :scale ? 1 / n / p : 1 / n)
        end
        
        (x1, x2) -> tanh(c₀ + γ * dot(x1, x2)) # sigmoid
    elseif kernel == :linear
        (x1, x2) -> dot(x1, x2)
    end

    Kmat = [Ker(X[i,:], X[j,:]) for i = 1:n, j = 1:n] # calculates the Kernel at X

    C = model.options[:C]

    # start of SMO
    b = 0
    α = α_old = zeros(n) 
    iter = 0
    ϵ = 1e-8 * C
    passes = 0 # number of times alphas were skip

    while iter < model.options[:max_iter]
        num_changed_alphas = 0

        for i = n
            ŷᵢ = sum(alpha[k] * y[k] * Kmat[i, k] for k = 1:n) + b
            Eᵢ = err(ŷᵢ, y[i])
            yᵢEᵢ = y[i] * [Eᵢ]

            if (yᵢEᵢ < -ϵ) && (α[i] < C) || ((yᵢEᵢ > ϵ) && (α[i] < 0))
                j = rand_j(n, i)

                ŷⱼ = sum(alpha[k] * y[k] * Kmat[j, k] for k = 1:n) + b
                Eⱼ = err(ŷⱼ, y[j])

                α_old[i], α_old[j] = α[i], α[j]

                L = compute_L(α[i], α[j], y[i], y[j], C)
                H = compute_H(α[i], α[j], y[i], y[j], C)

                if L == H
                    continue
                end

                η = 2 * (Kmat[i, j] - Kmat[i, i] - Kmat[j, j])

                if eta >= 0
                    continue
                end
                
                α[j] = clip_alpha(update_alpha_j(α[j], Eᵢ, Eⱼ, η), H, L)
                
                if abs(α[j] - α_old[j]) < eps
                    continue
                end
                
                α[i] = update_alpha_i(α[i], y[i], y[j], α_old[j], α[j])

                b1 = (b - Eᵢ) - (y[i] * (α[i] - α_old[i]) *  Kmat[i, i]) - ((y[j] * (α[j] - α_old[j])) * Kmat[i, j])
                b2 = (b - Eⱼ) - (y[i] * (α[i] - α_old[i]) * Kmat[i, j]) - ((y[j] * (α[j] - α_old[j])) * Kmat[j, j])
            
                if αᵢ > 0 && αᵢ < C
                    return b1
            
                elseif αⱼ > 0 && αⱼ < C
                    return b2
                
                else
                    return (b1 + b2) / 2
                end

                num_changed_alphas += 1
            end
        end

        if num_changed_alphas == 0
          passess += 1
        else
          passes = 0
        end
    end

    model.α = α
    ϵ = 1e-8 * C
    J = findall(α .> ϵ)
    k = J[1]
    b = y[k] - sum(α[i] * y[i] * Kmat[i,k] for i in J)
    model.f = x -> sum(α[i] * y[i] * Ker(X[i,:], x) for i in J) + b

    return model
end

function predict(model :: SVC,
    X :: Matrix; prob = false) 

    n, p = size(X)
    return [model.f(X[i,:]) > 0 ? 1 : -1 for i = 1:n]
end

predict_proba(model :: SVC, X :: Matrix) = predict(model, X, prob=true)

# SMO functions
compute_L(αᵢ::Float64, αⱼ::Float64, yᵢ::Float64, yⱼ::Float64, C::Float64) =  yᵢ != yⱼ ? max(0, αᵢ - αⱼ) : max(0,  αᵢ + αⱼ - C)

compute_H(αᵢ::Float64, αⱼ::Float64, yᵢ::Float64, yⱼ::Float64, C::Float64) = yᵢ != yⱼ ? min(C, C + αᵢ - αⱼ) : min(C,  αᵢ + αⱼ)

err(ŷᵢ::Vector{Float64}, yᵢ::Float64) = ŷᵢ - yᵢ 

clip_alpha(αⱼ::Float64, H::Float64, L::Float64) = αⱼ > H ? H : αⱼ < L ? L : αⱼ

update_alpha_i(αᵢ::Float64, yᵢ::Float64, yⱼ::Float64, αⱼ::Float64, αⱼ_old::Float64) =  αᵢ + (yᵢ * yⱼ)*(αⱼ_old - αⱼ)

update_alpha_j(αⱼ::Float64, Eᵢ::Float64, Eⱼ::Float64, η::Float64) = αⱼ - ((yⱼ * (Eᵢ - Eⱼ)) / η)

function update_b(b:: Float64, Xᵢ::Vector{Float64}, yᵢ::Float64, αᵢ::Float64, αᵢ_old::Float64, αⱼ::Float64, αⱼ_old::Float64, Eᵢ::Float64, Eⱼ::Float64, Ker::Function, C::Float64) 
    b1 = (b - Eᵢ) - (yᵢ * (αᵢ - αᵢ_old) * Ker(Xᵢ, Xᵢ)) - ((yⱼ * (αⱼ - αⱼ_old)) * Ker(Xᵢ, Xⱼ))
    b2 = (b - Eⱼ) - (yᵢ * (αᵢ - αᵢ_old) * Ker(Xᵢ, Xⱼ)) - ((yⱼ * (αⱼ - αⱼ_old)) * Ker(Xⱼ, Xⱼ))

    if αᵢ > 0 && αᵢ < C
        return b1

    elseif αⱼ > 0 && αⱼ < C
        return b2
    
    else
        return (b1 + b2) / 2
end

# generates j thats different from i
function rand_j(n, i)
    Random.seed!(42)

    j = rand(0:n)

    while i == j
        j = rand(0:n)
    end

    return j
end