include("../../src/svm.jl")

using Random, LinearAlgebra, Plots

Random.seed!(0)

n = 100
X = randn(n, 2) * 2
y = [X[i, 1]^2 + X[i, 2]^2 > 2  ? 1.0 : -1.0 for i = 1:n]

# we need to create a new dimension for X

𝜙(x) = [x[1] x[2] (x[1]^2 + x[2]^2)]

𝜙X = zeros(n, 3)

for i in 1:n
    x = X[i, :]
    𝜙X[i, :] = 𝜙(x)
end

model = SVM(C=10.0)

fit!(model, 𝜙X, y)

gr()
plot(leg=false)

I = findall(0 .< model.alphas .< model.C)
scatter!(X[I,1],X[I,2], c=:pink, m=(:white, stroke(1, :black), 10))

I = findall(y.==1)
scatter!(X[I,1], X[I,2], c=:blue, m=:square)

I = findall(y.==-1)
scatter!(X[I,1], X[I,2], c=:red, m=:circle)

x1g = range(extrema(X[:, 1])..., length=100)
x2g = range(extrema(X[:, 2])..., length=100)

contour!(x1g, x2g, (x1, x2) -> dot(model.w, 𝜙([x1; x2])) + model.b, levels=25)

# plot!(x -> (-(model.w[1] * x - model.b) / model.w[2]), extrema(X[:, 1])..., c=:magenta)
# plot!(x -> (-(model.w[1] * x - model.b - 1) / model.w[2]), extrema(X[:, 1])..., c=:orange, l=:dash)
# plot!(x -> (-(model.w[1] * x - model.b + 1) / model.w[2]), extrema(X[:, 1])..., c=:orange,l=:dash)

xlims!(extrema(X[:, 1])...)
ylims!(extrema(X[:, 2])...)

fig_path = "docs/images/non-linear-separable.png"
savefig(fig_path)