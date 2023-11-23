include("../src/svm.jl")

using Random, LinearAlgebra, Plots

Random.seed!(0)

n = 1000
X = rand(n, 2)
w = [0.5, 0.7]
b = -0.5
y = [dot(w, X[i, :]) + b > 0.01 * randn()  ? 1.0 : -1.0 for i = 1:n]

model = SVM(C=100, kernel="linear", max_iter=1000)

support_vectors, num_iterations = fit!(model, X, y)

plot(leg=false)

I = findall(y.==1)
scatter!(X[I,1],X[I,2], c=:blue, m=:square)

I = findall(y.==-1)
scatter!(X[I,1],X[I,2], c=:red, m=:circle)


println(w ./ model.w)
println(b ./ model.b)

plot!(x -> (-(model.w[1] * x + model.b) / model.w[2]), 0, 1, c=:magenta)

grafico_pontos_svm = "test/grafico_pontos_svm.png"

savefig(grafico_pontos_svm)

