#observables.jl
using BinningAnalysis
using LinearAlgebra

mutable struct Observables
    energy::ErrorPropagator{Float64,32}
    magnetization::ErrorPropagator{Float64,32}
    Observables() = new(ErrorPropagator(Float64), ErrorPropagator(Float64))
end

function get_magnetization(lattice::Lattice)::Float64
    m = (0.0, 0.0, 0.0)
    for i in 1:lattice.size
        m = m .+ get_spin(lattice.spins, i)
    end
    return norm(m)
end

function update_observables!(mc, energy::Float64, magnetization::Float64)
    #measure energy and energy^2
    push!(mc.observables.energy, energy, energy^2 )
    #measure magnetization and magnetization^2 
    push!(mc.observables.magnetization, magnetization, magnetization^2)
end

function std_error_tweak(ep::ErrorPropagator, gradient, lvl = BinningAnalysis._reliable_level(ep))
    sqrt(abs(var(ep, gradient, lvl) / ep.count[lvl]))
end

function specific_heat(mc) 
    ep = mc.observables.energy
    temp = mc.T
    lat = mc.lattice 

    #compute specific heat 
    c(e) = 1/temp^2 * (e[2]-e[1]*e[1]) / lat.size
    ∇c(e) = [-2.0 * 1/temp^2 * e[1] / lat.size, 1/temp^2 / lat.size] 

    heat = mean(ep, c)
    dheat = std_error_tweak(ep, ∇c)

    return heat, dheat
end 

function susceptibility(mc)
    m = mc.observables.magnetization 
    temp = mc.T 
    lat = mc.lattice 

    #compute susceptibility
    x(m) = 1/temp * (m[2] - m[1]*m[1]) / lat.size 
    ∇x(m) = [-2 * 1/temp * m[1] / lat.size, 1/temp / lat.size ] 
    chi = mean(m, x)
    dchi = std_error_tweak(m, ∇x)

    return chi, dchi 
end 

function fourier_transform_S(mc, k::Array{Float64,1})::Array{Complex{Float64},1}
    lat = mc.lattice
    spins = lat.spins
    N = lat.size
    f = zeros(Complex{Float64}, 3)
    for i in 1:N
        f += exp(im*dot(k, lat.site_positions[i]))*spins[:,i]
    end
    return f/sqrt(N)
end


function static_spin_structure_factor(mc, k::Array{Float64,1})::Float64
    A = fourier_transform_S(mc, k)
    S = zeros(Complex{Float64}, 3, 3)
    for i in range(1,3)
        for j in range(1,3)
            S[i,j] = A[i]*conj(A[j])
        end
    end
end

function SSSF(mc, N::Int64)
    kx = ky = kz = range(-2*π, 2*π, length=N)
    S = zeros(Float64, N, N)
    for i in 1:N
        for j in 1:N
            k = [kx[i], ky[i], kz[j]]
            S[i,j] = static_spin_structure_factor(mc, k)
        end
    end
    return S
end

function plot_SSSF(mc, N::Int64)
    S = SSSF(mc, N)
    kx = kz = range(-2*π, 2*π, length=N)
    p = heatmap(kx, kz, S, aspect_ratio=1, color=:viridis, xlabel="(h,h,0)", ylabel="(0,0,l)", title="Static Spin Structure Factor")
    return p
end

repBasisPyrochlore = 2*π*[-1 1 1; 1 -1 1; 1 1 -1]

function meshgrid(x, y, z)
    X = [i for i in x, j in 1:length(y), k in 1:length(z)]
    Y = [j for i in 1:length(x), j in y, k in 1:length(z)]
    Z = [k for i in 1:length(x), j in 1:length(y), k in z]
    return hcat(vec(X), vec(Y), vec(Z))
end

function gen_brillouin_zone_points(N::Int64)
    nx = ny = nz = range(0, 1, length=N)
    A = meshgrid(nx, ny, nz)
    K = zeros(Float64, N^3, 3)
    for i in range(1, N^3)
        K[i,:] = A[i,1]*repBasisPyrochlore[1,:] + A[i,2]*repBasisPyrochlore[2,:] + A[i,3]*repBasisPyrochlore[3,:]
    end
    return K
end


function find_SSSF_peaks(mc, N::Int64)
    K = gen_brillouin_zone_points(N)
    S = zeros(Float64, N^3, 3, 3)
    for i in 1:N^3
        S[i] = static_spin_structure_factor(mc, K[i])
    end
    max = findmax(S, 1)
    return [K[max[2][1][1]], K[max[2][1][2]], K[max[2][1][3]],
            K[max[2][2][1]], K[max[2][2][2]], K[max[2][2][3]],
            K[max[2][3][1]], K[max[2][3][2]], K[max[2][3][3]]]
end