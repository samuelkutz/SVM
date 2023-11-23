export SVM

using LinearAlgebra, Statistics, Random

mutable struct SVM
    max_iter::Int
    kernel::AbstractString
    C::Float64
    epsilon::Float64
    w::Vector{Float64}
    b::Float64
end

function SVM(; max_iter=100, kernel="linear", C=1.0, epsilon=1e-3)
    w = zeros(0)  # Initialize w as an empty vector
    b = 0.0
    return SVM(max_iter, kernel, C, epsilon, w, b)
end

function fit!(svm::SVM, X::Matrix, y::Vector)
    n, p = size(X)
    C = svm.C
    Kmat = [Ker(svm, X[i, :], X[j, :]) for i = 1:n, j = 1:n]

    α = zeros(n)

    iter = 0

    while true
        iter += 1
        α_old = copy(α)

        for j = 1:n
            i = rand_i(n, j)
            xᵢ, xⱼ, yᵢ, yⱼ = X[i, :], X[j, :], y[i], y[j]

            η = Kmat[i, i] + Kmat[j, j] - 2 * Kmat[i, j]

            if η == 0
                continue
            end

            α_prime_j, α_prime_i = α[j], α[i]

            L = yᵢ != yⱼ ? max(0, α[j] - α[i]) : max(0, α[j] + α[i] - C)
            H = yᵢ != yⱼ ? min(C, C + α[j] - α[i]) : min(C, α[j] + α[i])
            
            svm.w = calc_w(X, y, α)
            svm.b = calc_b(X, y, svm.w)

            E_i = E(svm, xᵢ, yᵢ)
            E_j = E(svm, xⱼ, yⱼ)
            
            α[j] = α_prime_j + yⱼ * (E_i - E_j) / η
            α[j] = max(α[j], L)
            α[j] = min(α[j], H)
            
            α[i] = α_prime_i + yᵢ * yⱼ * (α_prime_j - α[j])
        end

        diff = norm(α - α_old)
        if diff < svm.epsilon
            break
        end

        if iter >= svm.max_iter
            println("Iteration number exceeded the max of $(svm.max_iter) iterations")
            return
        end

        svm.w = calc_w(X, y, α)
        svm.b = calc_b(X, y, svm.w)
    end

    # Get support vectors
    alpha_idx = findall(α .> 0)  # Use α .> 0 to find support vectors
    support_vectors = X[alpha_idx, :]
    
    return support_vectors, iter
end

function predict(svm::SVM, X)
    return h(svm, X)
end

# Generate j that's different from i
function rand_i(n, j)
    Random.seed!(42)

    i = rand(1:n)

    while i == j
        i = rand(1:n)
    end

    return i
end

function Ker(svm::SVM, xᵢ, xⱼ)
    if svm.kernel == "linear"
        return dot(xᵢ, xⱼ)
    elseif svm.kernel == "quadratic"
        return (dot(xᵢ, xⱼ) ^ 2)
    elseif svm.kernel == "rbf"
        return exp(-norm(xᵢ - xⱼ) ^ 2)
    else
        error("Invalid kernel type")
    end
end

function h(svm::SVM, X)
    return sign(dot(svm.w, X) + svm.b)
end

function E(svm::SVM, xₖ, yₖ)
    return h(svm, xₖ) - yₖ
end

function calc_b(X, y, w)
    return mean(y - X * w)
end

function calc_w(X, y, α)
    result = zeros(size(X, 2)) 

    for i in eachindex(α)
        result .+= α[i] * y[i] * X[i, :]
    end

    return result
end