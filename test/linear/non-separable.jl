include("../../src/svm.jl")

using Random, LinearAlgebra, Plots

Random.seed!(0)

n = 100
X = randn(n, 2) * 2
w = [0.5, 0.4]
b = -0.8
y = [dot(w, X[i, :]) + b > 0.5 * randn()  ? 1.0 : -1.0 for i = 1:n]

model = SVM(C=1000.0, kernel_type=:linear)

fit!(model, X, y)

plot(leg=false)

I = findall(0 .< model.alphas .< model.C)
scatter!(X[I,1],X[I,2], c=:pink, m=(:white, stroke(1, :pink), 10))

I = findall(y.==1)  
scatter!(X[I,1], X[I,2], c=:blue, m=:square)

I = findall(y.==-1)
scatter!(X[I,1],X[I,2], c=:red, m=:circle)

plot!(x -> (-(model.w[1] * x - model.b) / model.w[2]), extrema(X[:, 1])..., c=:magenta)
plot!(x -> (-(model.w[1] * x - model.b - 1) / model.w[2]), extrema(X[:, 1])..., c=:orange, l=:dash)
plot!(x -> (-(model.w[1] * x - model.b + 1) / model.w[2]), extrema(X[:, 1])..., c=:orange,l=:dash)

xlims!(extrema(X[:, 1])...)
ylims!(extrema(X[:, 2])...)

fig_path = "docs/images/linear-non-separable.png"
savefig(fig_path)