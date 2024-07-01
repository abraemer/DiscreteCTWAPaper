### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ e78c212a-222f-11ef-09d0-0971c3e6d9ce
# ╠═╡ show_logs = false
begin
	import Pkg
	Pkg.activate(".")
	using SpinModels, SpinTruncatedWigner, OrdinaryDiffEq, Statistics, CairoMakie, LinearAlgebra,SparseArrays
end

# ╔═╡ 6aaf7256-0dea-4379-8abc-108db9b027a9
md"""
# Setup
"""

# ╔═╡ 5f2bf23e-a6d2-46d6-9aa8-96b31faf8865
function staggered_magnetization(state; indices)
    odd = 1:2:length(indices)
    even = 2:2:length(indices)
    return (sum(state[indices[odd]]) - sum(state[indices[even]]))/length(indices)
end

# ╔═╡ 7fcce890-58d8-4574-bde8-2de69d97538e
N = 2

# ╔═╡ d7d8623c-850d-4cf1-bdaf-35e8aa5aeb00
hamiltonian(Δ;N=N) = NN(Chain(2)) * XXZ(Δ)

# ╔═╡ b7e8b8f2-ebbf-48fb-9e10-6128dfd72a4b
psi0 = NeelState(N, :z)

# ╔═╡ a40b7b2a-4a7f-43c3-b1da-b574d4ffa624
times = range(0, 4; length=400)

# ╔═╡ b386892e-4e54-4728-80d6-2e2af0602bf3
md"""
# dTWA
"""

# ╔═╡ 32e74aa9-9a89-4b10-afed-ca698425174e
# exact in the sense that the sampling is exact
# i.e. all possible initial states are sampled once
function dtwa_exactsolution(H, times, initial_states; int=Vern8(), abstol=1e-10,reltol=1e-10)
	prob = ODEProblem{true, SciMLBase.FullSpecialize}( 
		SpinTruncatedWigner.twaUpdate!, 
		zeros(3N), 
		(0, maximum(times)), 
		TWAParameters(H); saveat=sort(times))
	ensemble = EnsembleProblem(prob;
	    prob_func = (prob, i, repeat) -> remake(prob; u0 = initial_states[i]))
	solve(ensemble, int; abstol, reltol, trajectories = length(initial_states))
end

# ╔═╡ 34ef48f6-c01f-4e69-8f3c-7227a2f303fa
dtwa_state_up_z = [[x,y,1.0] for x in (-1,1) for y in (-1,1)]

# ╔═╡ 9d057a93-ccfc-4dfc-ac13-2ead9503f575
dtwa_state_down_z = [[x,y,-1] for x in (-1,1) for y in (-1,1)]

# ╔═╡ dc5aac2d-82ca-49bb-85f9-7409e02b5f18
dtwa_initial_states = [vcat(s1,s2) for s1 in dtwa_state_up_z for s2 in dtwa_state_down_z]
# we sample the states exactly i.e. each phase-point vector exactly once

# ╔═╡ 56c55b88-d1fc-4828-b7a1-55d16fd658e6
Δlist = [0,2,4,6]

# ╔═╡ 2fc4dee0-1737-4f1e-b3a4-cfdb7d31ca85
dtwa_sols = [dtwa_exactsolution(hamiltonian(Δ), times, dtwa_initial_states) for Δ in Δlist]

# ╔═╡ b7443a86-3f5a-475f-b9a1-f59993841261
dtwa_magnetization_indices = 3:3:3N

# ╔═╡ 31f6a794-72e2-46d5-a67f-7f02c87e9659
dtwa_results = let f(sol) = mean(s->staggered_magnetization(s; indices=dtwa_magnetization_indices), sol)
	[f.(sol.(times)) for sol in dtwa_sols]
end

# ╔═╡ 574059ca-c72b-4a1a-9a92-456e0f7cd7c0
md"""
# ED
"""

# ╔═╡ f8f9af8a-ff82-4d5d-b6b5-663fdeb79381
ed_result = let (evals, U) = eigen(Hermitian(Matrix(hamiltonian(2)))),
	D = Diagonal(evals),
	O = sparse(Z((-1) .^ (0:N-1))/2),
	# O = sparse(X(ones(N))),
	Uψ0 = U'*SpinTruncatedWigner.quantum(psi0),
	res = zeros(length(times))
	for (i,t) in enumerate(times)
		ψt = U*(cis(-D*t)*Uψ0)
		res[i] = real(dot(ψt, O, ψt))
	end
	res
end

# ╔═╡ bfd939a1-8177-465a-af96-fb1d195f2605
md"""
# Plot
"""

# ╔═╡ 6eb6ec32-9ba8-40f5-ab8b-0c666f82c164
begin
	function save_and_display(name, folder="../plots")
		return fig -> save_and_display(name, fig, folder)
	end
	function save_and_display(name, fig, folder)
		mkpath(folder)
		mkpath(joinpath(folder, "png"))
		Makie.save(joinpath(folder, name*".pdf"), fig)
		Makie.save(joinpath(folder, "png", name*".png"), fig)
		fig
	end
end

# ╔═╡ 3d962620-e923-4d46-9485-c5a506e284c2
HEIGHT = 250

# ╔═╡ d75da2cd-9a05-4d14-a863-469b072c4270
SCALE = 2

# ╔═╡ effaf020-63be-447e-9178-f3398f9f7ebb
THEME = merge(theme_latexfonts(),
	Theme(fontsize=9*SCALE, size=(246*SCALE,HEIGHT*SCALE), pt_per_unit=1/SCALE,
		figure_padding=(1,7,1,1),
		Axis=(; xtickalign=1, ytickalign=1, xscale=identity),
		Label=(; font=:bold,
			halign=:left, valign=:top)))

# ╔═╡ 42ec211f-c787-49be-bfe9-d029ba460676
with_theme(THEME) do
	let fig = Figure(;size=(246*SCALE,170*SCALE)),
		ax = Axis(fig[1,1]; ylabel=L"\langle M^{st}(t)\rangle", xlabel=L"Time $t$ [$J^{-1}$]")
		for (Δ, res) in zip(Δlist, dtwa_results)
			lines!(ax, times, res; label=L"\Delta=%$(Δ)")
		end
		# lines!(ax, times, gctwa_result; label="gcTWA")
		# lines!(ax, times, dctwa_result; label="dcTWA")
		lines!(ax, times, ed_result; label="ED", color=:black, linestyle=:dash)
		axislegend(ax; orientation=:horizontal, position=:cb)
		#xlims!(ax, extrema(times)...)
		ylims!(ax, -1.5,1.2)
		ax.yticks = -1:0.5:1
		fig
	end
end |> save_and_display("appendix-two-spin-dynamics")

# ╔═╡ Cell order:
# ╠═e78c212a-222f-11ef-09d0-0971c3e6d9ce
# ╟─6aaf7256-0dea-4379-8abc-108db9b027a9
# ╠═5f2bf23e-a6d2-46d6-9aa8-96b31faf8865
# ╠═7fcce890-58d8-4574-bde8-2de69d97538e
# ╠═d7d8623c-850d-4cf1-bdaf-35e8aa5aeb00
# ╠═b7e8b8f2-ebbf-48fb-9e10-6128dfd72a4b
# ╠═a40b7b2a-4a7f-43c3-b1da-b574d4ffa624
# ╟─b386892e-4e54-4728-80d6-2e2af0602bf3
# ╠═32e74aa9-9a89-4b10-afed-ca698425174e
# ╠═34ef48f6-c01f-4e69-8f3c-7227a2f303fa
# ╠═9d057a93-ccfc-4dfc-ac13-2ead9503f575
# ╠═dc5aac2d-82ca-49bb-85f9-7409e02b5f18
# ╠═56c55b88-d1fc-4828-b7a1-55d16fd658e6
# ╠═2fc4dee0-1737-4f1e-b3a4-cfdb7d31ca85
# ╠═b7443a86-3f5a-475f-b9a1-f59993841261
# ╠═31f6a794-72e2-46d5-a67f-7f02c87e9659
# ╟─574059ca-c72b-4a1a-9a92-456e0f7cd7c0
# ╠═f8f9af8a-ff82-4d5d-b6b5-663fdeb79381
# ╟─bfd939a1-8177-465a-af96-fb1d195f2605
# ╠═6eb6ec32-9ba8-40f5-ab8b-0c666f82c164
# ╠═3d962620-e923-4d46-9485-c5a506e284c2
# ╠═d75da2cd-9a05-4d14-a863-469b072c4270
# ╠═effaf020-63be-447e-9178-f3398f9f7ebb
# ╠═42ec211f-c787-49be-bfe9-d029ba460676
