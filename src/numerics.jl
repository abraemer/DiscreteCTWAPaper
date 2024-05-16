module Numerics

using LinearAlgebra
using Random
using SparseArrays
using SpinSymmetry
using SpinModels
using SpinTruncatedWigner
using Statistics
using StaticArrays: @SVector, @SArray

export meandrop, stddrop, vardrop, eomdrop
export clustering_naive, clustering_rsrg
export generate_positions, geometry_from_filling, hamiltonian
export all_pair_renyi_ed, staggered_magnetization_operator
export all_pair_renyi_dtwa, all_pair_renyi_ctwa, staggered_magnetization

#################### SOME STATISTICS CONVENIENCE

meandrop(A; dims) = dropdims(mean(A; dims); dims)
stddrop(A; dims) = dropdims(std(A; dims); dims)
vardrop(A; dims) = dropdims(var(A; dims); dims)
eomdrop(A; dims) = stddrop(A; dims) ./ sqrt(reduce(*, size(A,d) for d in dims))

#################### CLUSTERING

"""
    clustering_rsrg(J)
Compute the RSRG inspired clustering of the coupling matrix `J`.
Return a vector of 2-element vectors of the paired indices.
"""
function clustering_rsrg(J)
    N = size(J, 1)
    iseven(N) || error("Pairing only works for even number of spins. Got $N spins.")
    sortedinds = sort!([(i,j) for i in 1:N for j in i+1:N]; by=x->J[x...], rev=true)
	seen = falses(N)
	pairs = Vector{Int}[]
	for (i,j) in sortedinds
		if !seen[i] && !seen[j]
			seen[i] = seen[j] = true
			push!(pairs, [i,j])
			2length(pairs) == N && break
		end
	end
	return pairs
end

"""
    clustering_naive(J, clustersize)
Compute the naive clustering of the coupling matrix `J` with clusters of size `clustersize`.
Return a vector of 2-element vectors of the paired indices.
"""
function clustering_naive(J, clustersize)
    N = size(J,1)
    N % clustersize == 0 || @warn "Cluster size $clustersize does not divide systemsize N=$N!"
    return [collect(i:min(N, i+clustersize-1)) for i in 1:clustersize:N]
end

#################### GENERATE HAMILTONIANS

function geometry_from_filling(N, fillingfraction)
    numsites = round(Int, N/fillingfraction)
    return PartiallyFilledChain(N, numsites)
end

function generate_positions(geom, chunkID, chunksize=100)
    rng = Xoshiro(chunkID)
    return [positions(geom; rng) for _ in 1:chunksize]
end

function hamiltonian(J, Δ)
    # 1/4 to convert to Pauli
    return J*Hopp(1/4) + J*Δ*ZZ(1/4)
end

#################### ED DATA EVALUATION

"""
    _twobytwo_eigenvalues(A,B,C)
Compute eigenvalues of matrix `[A C; conj(C) B]`
"""
function _twobytwo_eigenvalues(A,B,C)
    Δ = 0.5sqrt((A-B)^2 + 4abs2(C)) # discriminant
    s = 0.5(A+B)
    return s+Δ, s-Δ
end

function _isbitset(val, k)
    # first bit is k=0
    return ((val >>> k) & 1) == 1
end

"""
    all_pair_renyi_ed(ψ, numspins, nspinsup)
Compute the Renyi entropy of all 2-spin subsystems from the given quantum state `ψ` of
`numspins` spins which lives in the sector with `nspinsup` spins up.
Return the averaged Renyi entropy (a single number).
"""
function all_pair_renyi_ed(ψ, numspins, nspinsup)
    # Idea:
    # Since the full state lives in a single magnetization sector K
    # the Schmidt decomposition for a pair is almost trivial:
    # |ψ> = c1|↑↑⟩⊗|K-2⟩ + c2|↓↓⟩⊗|K⟩ + c3|↑↓⟩⊗|ψ1⟩ + c4|↓↑⟩⊗|ψ2⟩
    # where |K-2⟩ and |K⟩ are orthogonal to all other states since they live in different sectors.
    # So the reduced density matrix reads simply:
    # Tr_B |ψ⟩⟨ψ| = |c1|^2 |↑↑⟩⟨↑↑| + |c2|^2 |↓↓⟩⟨↓↓| + |c3|^2|↑↓⟩⟨↑↓| + |c4|^2|↓↑⟩⟨↓↑|
    #               + c3 ̄c4 ⟨ψ2|ψ1⟩ |↑↓⟩⟨↓↑| + ̄̄c3 c4 ⟨ψ1|ψ2⟩ |↓↑⟩⟨↑↓|
    # so the eigenvalues of the reduced density matrix are:
    # |c1|^2, |c2|^2, 0.5(A+B ± √((A-B)^2 + 4 |C|^2))
    # where A = |c3|^2, B = |c4|^4, C = c3 ̄c4 ⟨ψ2|ψ1⟩
    # Actually we only need Tr ρ^2 which is even easier to compute:
    # Tr ρ^2 = |c1|^4 + |c2|^4 + A^2 + B^2 + 2|C|^2
    # These things are easy to compute if we think of it like this:
    # |ψ> = |↑↑⟩⊗(c1|K-2⟩) + |↓↓⟩⊗(c2|K⟩) + |↑↓⟩⊗(c3|ψ1⟩) + |↓↑⟩⊗(c4|ψ2⟩)
    # because that means, we can just take all basis elements and sort them by their configuration of the pair
    # This gives 4 new vectors. To find the coefficients, we can then just use the normal
    # scalar product.
    basisstates = SpinSymmetry._indices(zbasis(numspins, nspinsup))
    basisstates .-= 1 # to convert from index space to binary representation
    spinups = [BitVector(_isbitset(state, k-1) for state in basisstates) for k in 1:numspins]
    total_renyi = 0.0
    for i in 1:numspins
        up1 = spinups[i]
        down1 = .!(up1)
        for j in i+1:numspins
            up2 = spinups[j]
            down2 = .!(up2)
            ψupup = ψ[up1 .& up2]
            ψdowndown = ψ[down1 .& down2]
            trρsquared = abs2(dot(ψupup,ψupup)) + abs2(dot(ψdowndown,ψdowndown))

            ψ1 = ψ[up1 .& down2]
            ψ2 = ψ[down1 .& up2]
            A = dot(ψ1, ψ1)
            B = dot(ψ2, ψ2)
            C = dot(ψ1, ψ2)
            trρsquared += abs2(A)+abs2(B)+2abs2(C)

            total_renyi += log2(trρsquared)
        end
    end
    return -total_renyi / binomial(numspins, 2)
end

function staggered_magnetization_operator(N)
    return sparse(Z((-1) .^ (0:N-1))/N)
end

#################### TWA DATA EVALUATION

# data is a vector of states (different shots)
# indsA, indsB are the indices where the operators from different clusters live
# indsA = (X_1, Y_1, Z_1)
function renyi2_differentcluster(data, indsA, indsB)
	correlators = zeros(length(indsA)+1,length(indsB)+1)
	for shot in data
		for i in 1:3
			correlators[i+1,1] += shot[indsA[i]]
			correlators[1,i+1] += shot[indsB[i]]
			for j in 1:3
				correlators[i+1,j+1] += shot[indsA[i]]*shot[indsB[j]]
			end
		end
	end
	correlators ./= length(data)
	correlators[1,1] = 1
	return 2 - log2(sum(abs2, correlators))
end

function renyi2_samecluster(data, indsA)
	correlators = zeros(length(indsA))
	for shot in data
		correlators .+= @view shot[indsA]
	end
	correlators ./= length(data)
	return 2 - log2(1+sum(abs2, correlators))
end

"""
    all_pair_renyi_dtwa(data)
Compute the Renyi entropy of all 2-spin subsystems from the given dTWA data.
`data` should be a list of dTWA state vectors.
Return the averaged Renyi entropy (a single number).
"""
function all_pair_renyi_dtwa(data)
    # data[shot][operator]
    N = length(data[1]) ÷ 3
	all_renyi = 0.0
	for i in 1:N
		indsA = (3i-2):3i
		for j in i+1:N
			indsB = (3j-2):3j
			all_renyi += renyi2_differentcluster(data, indsA, indsB)
		end
	end
    return all_renyi/binomial(N,2)
end

"""
    all_pair_renyi_ctwa(clusterbasis, data)
Compute the Renyi entropy of all 2-spin subsystems from the given cTWA data.
`data` should be a list of cTWA state vectors.
Return the averaged Renyi entropy (a single number).
"""
function all_pair_renyi_ctwa(clusterbasis, data)
    # data[shot][operator]
	N = maximum(maximum, clusterbasis.clustering)
	all_renyi = 0.0
	for i in 1:N
		for j in i+1:N
			if SpinTruncatedWigner.sameCluster(clusterbasis, i, j)
				let inds1 = @SVector([lookupClusterOp(clusterbasis, (i, d1)) for d1 in 1:3]),
					inds2 = @SVector([lookupClusterOp(clusterbasis, (j, d1)) for d1 in 1:3]),
					inds3 = @SArray([lookupClusterOp(clusterbasis, (i, d1), (j, d2)) for d1 in 1:3, d2 in 1:3]),
					inds = vcat(inds1,inds2,vec(inds3))
					all_renyi += renyi2_samecluster(data, inds)
				end
			else
				let indsA = @SVector[lookupClusterOp(clusterbasis, (i, d1)) for d1 in 1:3],
					indsB = @SVector[lookupClusterOp(clusterbasis, (j, d1)) for d1 in 1:3]
					all_renyi += renyi2_differentcluster(data, indsA, indsB)
				end
			end
		end
	end
	#return mean(results)
	return all_renyi / binomial(N,2)
end

#################### STAGGERED MAGNETIZATION

function _staggered_magnetization_indices(clusterbasis::ClusterBasis)
    N = length(clusterbasis.clusters) # number of spins
    return [lookupClusterOp(clusterbasis, (i,3)) for i in 1:N]
end

function _staggered_magnetization_indices(numspins::Int)
    return (1:numspins)*3
end

_sum_alternate(x) = sum(x .* (-1) .^ (0:length(x)-1))/length(x)

"""
    staggered_magnetization(numspins, data)
    staggered_magnetization(clusterbasis, data)
Compute the staggered magnetization for each shot.
Needs either `numspins` to be the number of spins and data to be dTWA data or a `ClusterBasis`
if the data was generated with cTWA. `data` should be a list of TWA state vectors.
Return an array of numbers.
"""
function staggered_magnetization(clusterbasis_or_numspins, data)
    inds = _staggered_magnetization_indices(clusterbasis_or_numspins)
    return [_sum_alternate(@view shots[inds]) for shots in data]
end


end # module
