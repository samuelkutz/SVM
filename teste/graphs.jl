using Random, LinearAlgebra, Plots

Random.seed!(0)

n = 50 
X = rand(n, 2) 
w = [0.3, 0.7]
b = -0.5
y = [dot(w, X[i,:]) + b > 0 ? 1 : -1 for i = 1:n]

plot(leg=false)
I = findall(y.==1)
scatter!(X[I,1],X[I,2], c=:blue, m=:square)
I = findall(y.==-1)
scatter!(X[I,1],X[I,2], c=:red, m=:circle)

grafico_pontos_svm = "teste/grafico_pontos_svm.png"
savefig(grafico_pontos_svm)
savefig(grafico_pontos_svm)