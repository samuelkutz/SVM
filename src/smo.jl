using Random

mutable struct SMO
    X::Matrix
    y::Vector

    C::Float64
    tol::Float64
    kernel::Function
    use_linear_optim::Bool
    
    eps::Float64

    Es::Vector
    b::Float64
    w::Vector
    
    alphas::Vector
    m::Int
    n::Int
    y2::Float64
    a2::Float64
    X2::Vector
    E2::Float64
end


# TODO: Kernel CHACHE
function SMO(X::Matrix{Float64}, y::Vector{Float64}, C::Float64, tol::Float64, kernel::Function, use_linear_optim::Bool)
    m, n = size(X)
    
    alphas = zeros(m)
    eps = 1e-3 
    
    errors = zeros(m)
    b = 0.0
    w = zeros(n)

    return SMO(X, y, C, tol, kernel, use_linear_optim, eps, errors, b, w, alphas, m, n, 0.0, 0.0, zeros(n), 0.0)
end

# Function to compute the SVM output for example i
function output(smo::SMO, i)
    if smo.use_linear_optim
        return dot(smo.w', smo.X[i, :]) - smo.b
    else
        return sum([smo.alphas[j] * smo.y[j] * smo.kernel(smo.X[j, :], smo.X[i, :]) for j in 1:smo.m]) - smo.b
    end
end

# Function to try to solve the problem analytically
function take_step(smo::SMO, i1, i2)
    if i1 == i2
        return false
    end

    a1 = smo.alphas[i1]
    y1 = smo.y[i1]
    X1 = smo.X[i1, :]
    X2 = smo.X[i2, :]

    E1 = get_E(smo, i1)

    s = y1 * smo.y[i2]

    if y1 != smo.y[i2]
        L = max(0, smo.alphas[i2] - a1)
        H = min(smo.C, smo.C + smo.alphas[i2] - a1)
    else
        L = max(0, smo.alphas[i2] + a1 - smo.C)
        H = min(smo.C, smo.alphas[i2] + a1)
    end

    if L == H
        return false
    end

    k11 = smo.kernel(X1, X1)
    k12 = smo.kernel(X1, X2)
    k22 = smo.kernel(X2, X2)

    eta = k11 + k22 - 2 * k12

    if eta > 0
        a2_new = smo.alphas[i2] + smo.y[i2] * (E1 - get_E(smo, i2)) / eta

        if a2_new < L
            a2_new = L
        elseif a2_new > H
            a2_new = H
        end
    else
        f1 = y1 * (E1 + smo.b) - a1 * k11 - s * smo.alphas[i2] * k12
        f2 = smo.y[i2] * (get_E(smo, i2) + smo.b) - s * a1 * k12 - smo.alphas[i2] * k22
        L1 = a1 + s * (smo.alphas[i2] - L)
        H1 = a1 + s * (smo.alphas[i2] - H)
        Lobj = L1 * f1 + L * f2 + 0.5 * L1^2 * k11 + 0.5 * L^2 * k22 + s * L * L1 * k12
        Hobj = H1 * f1 + H * f2 + 0.5 * H1^2 * k11 + 0.5 * H^2 * k22 + s * H * H1 * k12

        if Lobj < Hobj - smo.eps
            a2_new = L
        elseif Lobj > Hobj + smo.eps
            a2_new = H
        else
            a2_new = smo.alphas[i2]
        end
    end

    if abs(a2_new - smo.alphas[i2]) < smo.eps * (a2_new + smo.alphas[i2] + smo.eps)
        return false
    end

    a1_new = a1 + s * (smo.alphas[i2] - a2_new)

    new_b = compute_b(smo, E1, a1, a1_new, a2_new, k11, k12, k22, y1, i2)

    delta_b = new_b - smo.b

    smo.b = new_b

    if smo.use_linear_optim
        smo.w += y1 * (a1_new - a1) * X1 + smo.y[i2] * (a2_new - smo.alphas[i2]) * smo.X[i2, :]
    else
        # if not linear. This need to be revised


        # OBS: THE FOLLOWING FOR LOOP IS IMPROVISED AND IS NOT CERTAIN THAT THIS WORKS

        smo.w .= 0.0  

        for i in 1:smo.m
            # TODO: MAKE THE ACUMULATION NOT BASED ON SIMPLE X, IT SHOULD BE BASED ON ALL 𝝓(X) INSTEAD

            smo.w .+= smo.alphas[i] * smo.y[i] * smo.X[i, :]  # Accumulate based on Lagrange multipliers
        end
    end

    delta1 = y1 * (a1_new - a1)
    delta2 = smo.y[i2] * (a2_new - smo.alphas[i2])

    for i in 1:smo.m
        if 0 < smo.alphas[i] < smo.C
            smo.Es[i] += delta1 * smo.kernel(X1, smo.X[i, :]) + delta2 * smo.kernel(smo.X[i2, :], smo.X[i, :]) - delta_b
        end
    end

    smo.Es[i1] = 0
    smo.Es[i2] = 0

    smo.alphas[i1] = a1_new
    smo.alphas[i2] = a2_new

    return true
end

function compute_b(smo::SMO, E1, a1, a1_new, a2_new, k11, k12, k22, y1, i2)
    b1 = E1 + y1 * (a1_new - a1) * k11 + smo.y[i2] * (a2_new - smo.alphas[i2]) * k12 + smo.b
    b2 = get_E(smo, i2) + y1 * (a1_new - a1) * k12 + smo.y[i2] * (a2_new - smo.alphas[i2]) * k22 + smo.b

    if (0 < a1_new < smo.C)
        new_b = b1
    elseif (0 < a2_new < smo.C)
        new_b = b2
    else
        new_b = (b1 + b2) / 2.0
    end

    return new_b
end

function get_E(smo::SMO, i1)
    if 0 < smo.alphas[i1] < smo.C
        return smo.Es[i1]
    else
        return output(smo, i1) - smo.y[i1]
    end
end

function second_heuristic(smo::SMO, non_bound_indices)
    i1 = -1
    if length(non_bound_indices) > 1
        max_step = 0

        for j in non_bound_indices
            E1 = smo.Es[j] - smo.y[j]
            step = abs(E1 - get_E(smo, j))  # approximation
            if step > max_step
                max_step = step
                i1 = j
            end
        end
    end
    return i1
end

function examine_example(smo::SMO, i2)
    smo.y2 = smo.y[i2]
    smo.a2 = smo.alphas[i2]
    smo.X2 = smo.X[i2, :]
    smo.E2 = get_E(smo, i2)

    r2 = smo.E2 * smo.y2

    if !((r2 < -smo.tol && smo.a2 < smo.C) || (r2 > smo.tol && smo.a2 > 0))
        return 0
    end

    non_bound_idx = get_non_bound_indexes(smo)
    i1 = second_heuristic(smo, non_bound_idx)

    if i1 >= 0 && take_step(smo, i1, i2)
        return 1
    end

    rand_i = rand(1:length(non_bound_idx))

    for i1 in non_bound_idx[rand_i:end] ∪ non_bound_idx[1:rand_i-1]
        if take_step(smo, i1, i2)
            return 1
        end
    end
    
    rand_i = rand(1:smo.m)
    all_indices = 1:smo.m
    for i1 in all_indices[rand_i:end] ∪ all_indices[1:rand_i-1]
        if take_step(smo, i1, i2)
            return 1
        end
    end

    return 0
end

function E(smo::SMO, i2)
    return output(smo, i2) - smo.y2
end

function get_non_bound_indexes(smo::SMO)
    non_bound_idx = Int[]

    for i in 1:smo.m
        if 0 <= smo.alphas[i] < smo.C # needs to be <= in order to start (alphas are initialized as zeros)
            push!(non_bound_idx, i)
        end
    end

    return non_bound_idx
end

function first_heuristic(smo::SMO)
    num_changed = 0
    non_bound_idx = get_non_bound_indexes(smo)

    for i in non_bound_idx
        num_changed += examine_example(smo, i)
    end
    return num_changed
end

function optimize(smo::SMO)
    num_changed = 0
    examine_all = true
    iter = 0

    while num_changed > 0 || examine_all
        num_changed = 0

        if examine_all
            for i in 1:smo.m
                num_changed += examine_example(smo, i)
            end
        else
            num_changed += first_heuristic(smo)
        end

        if examine_all
            examine_all = false
        elseif num_changed == 0
            examine_all = true
        end

        iter += 1

    end
end