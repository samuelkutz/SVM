export SVM

include("../src/smo.jl")

using LinearAlgebra

mutable struct SVM
    X::Matrix
    y::Vector

    C::Float64
    tol::Float64
    kernel::Function
    use_linear_optim::Bool
    
    gamma::Float64
    degree::Float64
    coef0::Float64

    errors::Vector
    eps::Float64
    
    alphas::Vector
    w::Vector
    b::Float64
end

# TODO: parameters validation
function SVM(; C::Float64=1.0, tol::Float64=1e-3, kernel_type::Symbol=:linear, gamma::Float64=0.5, degree::Float64=3.0, coef0::Float64=0.0)
    use_linear_optim=false
    
    X = Matrix{Float64}(undef, 0, 0)
    y = Vector{Float64}(undef, 0)
    
    m = size(X, 1)

    errors = zeros(m)
    eps = 1e-3 
    alphas = zeros(m) 

    b = 0.0
    w = zeros(m)

    if kernel_type == :linear
        use_linear_optim=true
        kernel = (x1::Vector{Float64}, x2::Vector{Float64}) -> dot(x1, x2)
    elseif kernel_type == :rbf
        kernel = (x1::Vector{Float64}, x2::Vector{Float64}) -> exp(-gamma * norm(x1 - x2)^2)
    elseif kernel_type == :poly
        kernel = (x1::Vector{Float64}, x2::Vector{Float64}) -> (coef0 + dot(x1, x2) ^ degree)
    elseif kernel_type == :sigmoid
        kernel = (x1::Vector{Float64}, x2::Vector{Float64}) -> tanh(coef0 + gamma * dot(x1, x2))
    else
        error("Unsupported kernel type: $kernel_type")
    end

    return SVM(X, y, C, tol, kernel, use_linear_optim, gamma, degree, coef0, errors, eps, alphas, w, b)
end

function fit!(svm::SVM, X::Matrix{Float64}, y::Vector{Float64})
    svm.X = X
    svm.y = y
    m = size(X, 1)
    svm.alphas = zeros(m)
    svm.errors = zeros(m)

    smo = SMO(X, y, svm.C, svm.tol, svm.kernel, svm.use_linear_optim)

    optimize(smo)

    # Update SVM parameters based on the SMO results
    svm.alphas = smo.alphas
    svm.errors = smo.Es
    svm.w = smo.w
    svm.b = smo.b
end

function predict(svm::SVM, x_new::Vector{Float64})
    if svm.use_linear_optim
        return dot(svm.w, x_new) - svm.b
    else
        return sum(svm.alphas[j] * svm.y[j] * svm.kernel(svm.X[:, j], x_new) for j in 1:svm.m) - svm.b
    end
end