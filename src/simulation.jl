module Simulation

using DrWatson
using Statistics
using SparseArrays
using LinearAlgebra
using Random
using SpinSymmetry
using SpinModels
using SpinTruncatedWigner
using OhMyThreads
using OrdinaryDiffEq

include("numerics.jl")
using .Numerics

function positions_for_chunkID(; chunkID, N, filling, chunksize, _...)
    config = (; chunkID, N, filling, chunksize)
    @info "Loading positions for $(config)"
    data, file = produce_or_load(config, datadir("positions"); tag=false) do config
        @info "Position file not found. Generating!"
        geom = geometry_from_filling(N, filling)
        positions = generate_positions(geom, chunkID, chunksize)
        data = merge(tostringdict(ntuple2dict(config)), Dict("positions"=>positions))
        data
    end
    @info "Loaded positions from $file"
    return data["positions"]
end

function run(params::Dict)
    @info params
    param_tuple = dict2ntuple(params)
    params["results"] = run(param_tuple)
    wsave(datadir("simulations", savename(params, "jld2")), params)
end

function run(params::NamedTuple)
    runfunction = if params.alg == "ed"
        ed_run
    elseif params.alg == "dTWA"
        dtwa_run
    elseif params.alg == "gcTWA" || params.alg == "dcTWA"
        ctwa_run
    else
        @warn "Unknown algorithm: $alg"
        return
    end
    positions = positions_for_chunkID(; params...)
    geom = geometry_from_filling(params.N, params.filling)
    interaction = PowerLaw(params.α)
    ψ0 = NeelState(params.N, :z)
    results = []
    rng = Xoshiro(params.chunkID)
    for (shot, pos) in enumerate(positions)
        @info "Shot $shot/$(length(positions))"
        J = interaction_matrix(interaction, geom, pos)
        push!(results, runfunction(J, ψ0; rng, params...))
    end
    return results
end

function ed_run(J, ψ0; tlist, N, Δ, _...)
    t1 = time()
    H = hamiltonian(J, Δ)
    psi0 = SpinTruncatedWigner.quantum(ψ0)
    Hsymmetries = iseven(N) ? symmetrized_basis.(N, div(N,2), Ref(Flip(N)), 0:1) : [symmetrized_basis(N, div(N,2))]
    Hsymm_P = transformationmatrix.(Hsymmetries)
    Hfull = sparse(H)

    Osymmetry = symmetrized_basis(N, div(N,2))
    OsymmP = transformationmatrix(Osymmetry)
    O = OsymmP*staggered_magnetization_operator(N)*OsymmP'
    state_to_O_space_Ps = [OsymmP*P' for P in Hsymm_P]

    t2 = time()
    eigens = [eigen(Hermitian(Matrix(P*Hfull*P'))) for P in Hsymm_P]
    t3 = time()
    psi0_UP = [eig.vectors'*(P*psi0) for (eig, P) in zip(eigens, Hsymm_P)]

    ψt = zeros(ComplexF64, size(OsymmP, 1))
    magnetization_results = zeros(length(tlist))
    pair_renyi2 = zeros(length(tlist))
    for (i,t) in enumerate(tlist)
        fill!(ψt, 0)
        for (P, eig, UPψ) in zip(state_to_O_space_Ps, eigens, psi0_UP)
            ψt .+= P * (eig.vectors * (cis(Diagonal(eig.values)*(-t)) * UPψ))
        end
        magnetization_results[i] = real(dot(ψt, O, ψt))
        pair_renyi2[i] = all_pair_renyi_ed(ψt, N, N÷2)
    end
    t4 = time()
    @info "tsetup=$(round(t2-t1; sigdigits=4)), tdiag=$(round(t3-t2; sigdigits=4)), tcompute=$(round(t4-t3; sigdigits=4))"
    return (; magnetization_mean=magnetization_results,
              magnetization_eom=zeros(size(magnetization_results)),
              pair_renyi2,)
end

function ctwa_run(J, ψ0; tlist, N, Δ, rng, trajectories, clustering, clustersize, alg, _...)
    H = hamiltonian(J, Δ)
    clusters = if clustering == "RG"
        clustering_rsrg(J)
    elseif clustering == "naive"
        clustering_naive(J, clustersize)
    else
        error("Unknown clustering method: $clustering")
    end
    cb = ClusterBasis(clusters)
    cTWA_state = if alg == "dcTWA"
        cTWADiscreteState(cb, ψ0)
    elseif alg == "gcTWA"
        cTWAGaussianState(cb, ψ0)
    else
        error("Unknown sampling method: $alg")
    end

    prob = TWAProblem(cb, H, cTWA_state, tlist; rng)
    t1 = time()
    sol = solve(prob, Vern8(); trajectories, abstol=1e-9, reltol=1e-9)
    t2 = time()
    all_magnetization_data = stack(staggered_magnetization.(Ref(cb), sol.(tlist));dims=1) #[time, shot]
    t3 = time()
    #pair_renyi2 = all_pair_renyi_ctwa.(Ref(cb), sol.(tlist)) #[time]
    pair_renyi2 = tmap(t->all_pair_renyi_ctwa(cb, sol(t)), Float64, tlist) #[time]
    t4 = time()
    @info "tsolve=$(round(t2-t1; sigdigits=4)), tmag=$(round(t3-t2; sigdigits=4)), trenyi2=$(round(t4-t3; sigdigits=4))"
    return (;   magnetization_mean=meandrop(all_magnetization_data; dims=2),
                magnetization_eom=eomdrop(all_magnetization_data; dims=2),
                pair_renyi2)
end

function dtwa_run(J, ψ0; tlist, N, Δ, rng, trajectories, _...)
    H = hamiltonian(J, Δ)
    prob = TWAProblem(H, ψ0, tlist; rng)
    t1 = time()
    sol = solve(prob, Vern8(); trajectories, abstol=1e-9, reltol=1e-9)
    t2 = time()
    all_magnetization_data = stack(staggered_magnetization.(N, sol.(tlist));dims=1) #[time, shot]
    t3 = time()
    #pair_renyi2 = all_pair_renyi_dtwa.(sol.(tlist)) #[time]
    pair_renyi2 = tmap(t->all_pair_renyi_dtwa(sol(t)), Float64, tlist) #[time]
    t4 = time()
    @info "tsolve=$(round(t2-t1; sigdigits=4)), tmag=$(round(t3-t2; sigdigits=4)), trenyi2=$(round(t4-t3; sigdigits=4))"
    return (;   magnetization_mean=meandrop(all_magnetization_data; dims=2),
                magnetization_eom=eomdrop(all_magnetization_data; dims=2),
                pair_renyi2)
end

end # module
