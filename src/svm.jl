module SVM

include("../src/SMO.jl")

using LinearAlgebra

mutable struct SVC
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
function SVC(; C::Float64=1.0, tol::Float64=1e-3, kernel_type::Symbol=:linear, gamma::Float64=0.5, degree::Float64=3.0, coef0::Float64=0.0)
    use_linear_optim=true
    
    X = Matrix{Float64}(undef, 0, 0)
    y = Vector{Float64}(undef, 0)
    
    m = size(X, 1)

    errors = zeros(m)
    eps = 1e-3 
    alphas = zeros(m) 

    b = 0.0
    w = zeros(m)

    if kernel_type == :linear
        kernel = (x1::Vector{Float64}, x2::Vector{Float64}) -> dot(x1, x2)
    # elseif kernel_type == :rbf
    #     kernel = (x1::Vector{Float64}, x2::Vector{Float64}) -> exp(-gamma * norm(x1 - x2)^2)
    # elseif kernel_type == :poly
    #     kernel = (x1::Vector{Float64}, x2::Vector{Float64}) -> (coef0 + dot(x1, x2) ^ degree)
    # elseif kernel_type == :sigmoid
    #     kernel = (x1::Vector{Float64}, x2::Vector{Float64}) -> tanh(coef0 + gamma * dot(x1, x2))
    else
        error("Unsupported kernel type: $kernel_type")
    end

    return SVC(X, y, C, tol, kernel, use_linear_optim, gamma, degree, coef0, errors, eps, alphas, w, b)
end

function fit!(SVC::SVC, X::Matrix{Float64}, y::Vector{Float64})
    SVC.X = X
    SVC.y = y
    m = size(X, 1)
    SVC.alphas = zeros(m)
    SVC.errors = zeros(m)

    smo = SMO(X, y, SVC.C, SVC.tol, SVC.kernel, SVC.use_linear_optim)

    optimize(smo)

    # Update SVC parameters based on the SMO results
    SVC.alphas = smo.alphas
    SVC.errors = smo.Es
    SVC.w = smo.w
    SVC.b = smo.b
end

function predict(SVC::SVC, x_new::Vector{Float64})
    if SVC.use_linear_optim
        return dot(SVC.w, x_new) - SVC.b
    else
        return sum(SVC.alphas[j] * SVC.y[j] * SVC.kernel(SVC.X[:, j], x_new) for j in 1:SVC.m) - SVC.b
    end
end

export SVC, fit!, predict

end # SVM