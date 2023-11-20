export SVC 

using LinearAlgebra, Logging

mutable struct SVC 
    options :: Dict{Symbols, Any}
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

function fit!(model::SVC, X::Matrix, y::Vector)

end