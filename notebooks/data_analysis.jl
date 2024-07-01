### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# ╔═╡ 5a33306e-1906-11ef-3591-e322fc561c62
# ╠═╡ show_logs = false
begin
	import Pkg
	Pkg.activate(@__DIR__)
	using Statistics, JLD2, SpinTruncatedWigner
	include("../src/numerics.jl")
	import .Numerics
end

# ╔═╡ 7fb38461-36f3-4909-b297-10ba47c7d6a7
md"""
# Another implementation of Pair renyi entropy
This implementation is based on the idea to first extract all of the required correlators from every cTWA state vector and then use these to compute the Renyi entropy. This has the following advantages:
1. Each correlator is computed only once. The current implementation computes the single-spin expectation values repeatedly for every pair (so a total of $\binom{N}{2}$ times)
2. Flexibility: If we just want to compute the Renyi entropy, we don't need to materialize all of the intermediate vector of correlators simultaneously and reuse the memory instead. On the other hand, if we wanted to do statistical bootstrapping, we can compute all correlators once and thus drastically reduce the overhead for each subsequent computation of Renyi entropies.
3. We found another good speedup by forcing inlining of the 3-arg `lookupClusterOp`. which Julia did not do automatically. This reduced the number of allocations to just the ones for the return vector. With this optimization this code is faster than the one in `numerics.jl`
"""

# ╔═╡ 5074848a-6fd5-4a6c-a345-fc7fc1d930dc
md"""
## Implementation
"""

# ╔═╡ 3054177c-8972-4e92-b11d-0be0560a5179
const sameCluster = SpinTruncatedWigner.sameCluster

# ╔═╡ a8de906a-f748-4325-90a8-08d33765310d
begin
function prepare_vector(cb, state)
	N = length(cb.clusters)
	total_length = 3*N + binomial(N,2)*9 # ∑_k (N choose k) 3^k
	ret = zeros(total_length)
	return prepare_vector!(ret, cb, state)
end
function prepare_vector!(ret, cb, state)
	N = length(cb.clusters)
	# ret stores all single and two spin values
	# -> [1:3N] contains the single spin values,
	#	 where [1:3] are [X,Y,Z] of spin 1, [4:6] of spin 2 and so on
	# -> [3N+1:end] contains the correlators of spins (x,y) in lexicographic order
	#    so the correlators of spins (i,j) are offset by Δ=(i-1)N - i(i+1)/2 + j at
	#    [3N+9Δ-8:3N+9Δ]
	# note that the code below stores the 2 spin correlators in different orders
	# within the respective blocks depending on whether the spins form a cluster.
	# For computing the renyi entropy this does not matter.
	for i in 1:N
		ret[3i-2] = state[lookupClusterOp(cb, (i, 1))]
		ret[3i-1] = state[lookupClusterOp(cb, (i, 2))]
		ret[3i-0] = state[lookupClusterOp(cb, (i, 3))]
	end
	offset = 3N+1
	for i in 1:N
		for j in i+1:N
			# Δ = (i-1)*N - binomial(i+1,2) + j
			# offset = 3N+9Δ-8
			if sameCluster(cb, i, j)
				for d1 in 1:3
					for d2 in 1:3
						ret[offset] = state[@inline lookupClusterOp(cb, (i, d1), (j, d2))]
						offset += 1
					end
				end
			else
				kron!(@view(ret[offset:offset+8]),
					@view(ret[3i-2:3i]),
					@view(ret[3j-2:3j]))
				offset += 9
			end
		end
	end
	return ret
end
end

# ╔═╡ da72a5e0-7a09-401c-a52b-1093838e75a6
begin
	function all_pair_renyi_prepared(states, N)
		all_pair_renyi_prepared!(similar(states[1]), states, N)
	end
	function all_pair_renyi_prepared!(means, states, N)
		# means = mean(states)
		fill!(means, zero(eltype(means)))
		for state in states
			means .+= state
		end
		means ./= length(states)
		means .^= 2
		renyi = 0
		for i in 1:N
			for j in i+1:N
				temp = 1
				temp += means[3i-2]
				temp += means[3i-1]
				temp += means[3i-0]
				temp += means[3j-2]
				temp += means[3j-1]
				temp += means[3j-0]

				Δ = (i-1)*N - binomial(i+1,2) + j
				offset = 3N+9Δ-8
				temp += sum(@view means[offset:offset+8])
				renyi += log2(temp)
			end
		end
		return 2 - renyi/binomial(N,2)
	end
end

# ╔═╡ c54c4cc9-609c-4cb8-98a9-f2b07e8ff078
function all_pair_renyi_ctwa(clusterbasis, states)
	tmp1 = prepare_vector(clusterbasis, states[1])
	tmp2 = similar(tmp1)
	# note that this looks a bit like dangerous reuse of memory
	# it works reliably because all_pair_renyi_prepared!
	# only accesses the prepared states sequentially
	return all_pair_renyi_prepared!(
		tmp1,
		(prepare_vector!(tmp2,clusterbasis,s) for s in states),
		length(clusterbasis.clusters))
end

# ╔═╡ ff848156-762e-482d-8694-16afbbeb539b
md"""
## Consistency checks
"""

# ╔═╡ 33f5599d-6b9f-4182-840f-7243c92b36a6
# only execute tests in notebook mode or if requested
shouldtest = @isdefined(PlutoRunner) || in("--perform-test", ARGS)

# ╔═╡ d1347295-d800-4d57-b57a-e3ee8f9d4d8a
if shouldtest
	testdata = JLD2.load(readdir(joinpath(@__DIR__, "../data/fulldata-simulations"), join=true)[1])
end

# ╔═╡ 7aa0275d-dcd7-4795-87e7-335ae900506a
@info "" shouldtest

# ╔═╡ 76c79aeb-7a25-4874-82d4-4f7ebd761b4e
shouldtest && let results = testdata["results"][1],
	cb = ClusterBasis(results.clusters)
	states = results.fulldata[1]

	# precompile
	all_pair_renyi_ctwa(cb, states[1:2])
	Numerics.all_pair_renyi_ctwa(cb, states[1:2])

	# run to compare
	@info @time "New Implementation" all_pair_renyi_ctwa(cb, states)
	@info @time "From numerics.jl  " Numerics.all_pair_renyi_ctwa(cb, states)
	@info @time "New Implementation" all_pair_renyi_ctwa(cb, results.fulldata[5])
	@info @time "From numerics.jl  " Numerics.all_pair_renyi_ctwa(cb, results.fulldata[5])
end

# ╔═╡ 7a140d37-3052-48eb-bb3c-833664143fd9
md"""
# Data Evaluation
We perform the following statistical analysis:

## Staggered Magentization
We just estimate the Monte-Carlo shot noise by computing the standard deviation for each time point over all trajectories.
"""

# ╔═╡ 73c50c5b-019d-4e2f-b8f5-c5225302d10c
function analyze_magnetization(results)
	cb = ClusterBasis(results.clusters)
	mags = Numerics.staggered_magnetization.(Ref(cb), results.fulldata)
	(; magnetization_std = std.(mags), magnetization_mean = mean.(mags))
end

# ╔═╡ 92f21464-4c4a-4122-bb82-9b519aaf9156
md"""
## Pair Renyi entropy
We particition the 10,000 shots into chunks and compute the Renyi entropy for each of the chunks. From this we estimate a mean and a variance.
We repeat this for different sizes of the chunks.
"""

# ╔═╡ bdddb3b3-cde4-47c4-b712-68b2249ae918
function analyze_renyi(results, chunksize)
	cb = ClusterBasis(results.clusters)
	nshots = length(results.fulldata[1])
	!iszero(nshots % chunksize) && @warn "Chunksize $chunksize does not evenly divide the number of shots $nshots"
	chunks = Iterators.partition(1:nshots, chunksize)
	entropies = [[all_pair_renyi_ctwa(cb, states[chunk]) for chunk in chunks] for states in results.fulldata] # [t][chunk]
	(; renyi_std = std.(entropies), renyi_mean = mean.(entropies))
end

# ╔═╡ 0e0c756f-1ceb-46ff-8b41-c2642e4ad7d6
md"""
## Analysis function
"""

# ╔═╡ 13354d48-d654-4085-bc1b-01d7bc437339
function analyze(file, renyi_chunksizes=[50,100,200,250,500,1000])
	data = JLD2.load(file)
	results = data["results"][1]

	(; magnetization_std, magnetization_mean) = analyze_magnetization(results)
	renyi_stds = Vector{Float64}[]
	renyi_means = Vector{Float64}[]
	for chunksize in renyi_chunksizes
		(; renyi_std, renyi_mean) = analyze_renyi(results, chunksize)
		push!(renyi_stds, renyi_std)
		push!(renyi_means, renyi_mean)
	end
	new_results = (; renyi_stds, renyi_means, renyi_chunksizes,
					magnetization_std, magnetization_mean)
	data["results"] = [new_results]

	filename = basename(file)
	dir = dirname(file)
	savedir = joinpath(dir, "..", basename(dir)*"-avg")
	mkpath(savedir)
	JLD2.save(joinpath(savedir, filename), data)
end

# ╔═╡ 1e084018-9527-4e47-a702-aa5216bb9024
md"""
## Perform the analysis
"""

# ╔═╡ 2c1fc902-2bea-4c6f-b735-f026d962f1bd
let files = readdir(joinpath(@__DIR__, "../data/fulldata-simulations"), join=true)
	@info "Found $(length(files)) files"
	for file in files
		@info "Doing $(basename(file))"
		@time analyze(file)
	end
end

# ╔═╡ a152d149-941b-46b0-9f42-c1b613ed0a02
# unused function for statistical bootstrapping of renyi
# function pair_renyi_error2(cb, data; samples=1000)
# 	vecs = prepare_vector.(Ref(cb), data)
# 	N = length(cb.clusters)
# 	tmp = similar(vecs[1])
# 	stderror(bootstrap(vecs, d->all_pair_renyi_prepared!(tmp, d, N), BasicSampling(samples)))[1]
# end

# ╔═╡ Cell order:
# ╠═5a33306e-1906-11ef-3591-e322fc561c62
# ╟─7fb38461-36f3-4909-b297-10ba47c7d6a7
# ╟─5074848a-6fd5-4a6c-a345-fc7fc1d930dc
# ╠═a8de906a-f748-4325-90a8-08d33765310d
# ╠═3054177c-8972-4e92-b11d-0be0560a5179
# ╠═da72a5e0-7a09-401c-a52b-1093838e75a6
# ╠═c54c4cc9-609c-4cb8-98a9-f2b07e8ff078
# ╟─ff848156-762e-482d-8694-16afbbeb539b
# ╠═33f5599d-6b9f-4182-840f-7243c92b36a6
# ╠═d1347295-d800-4d57-b57a-e3ee8f9d4d8a
# ╠═7aa0275d-dcd7-4795-87e7-335ae900506a
# ╠═76c79aeb-7a25-4874-82d4-4f7ebd761b4e
# ╟─7a140d37-3052-48eb-bb3c-833664143fd9
# ╠═73c50c5b-019d-4e2f-b8f5-c5225302d10c
# ╟─92f21464-4c4a-4122-bb82-9b519aaf9156
# ╠═bdddb3b3-cde4-47c4-b712-68b2249ae918
# ╟─0e0c756f-1ceb-46ff-8b41-c2642e4ad7d6
# ╠═13354d48-d654-4085-bc1b-01d7bc437339
# ╟─1e084018-9527-4e47-a702-aa5216bb9024
# ╠═2c1fc902-2bea-4c6f-b735-f026d962f1bd
# ╟─a152d149-941b-46b0-9f42-c1b613ed0a02
