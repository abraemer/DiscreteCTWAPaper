### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ 7b9b98d2-12c3-11ef-04f4-6dee8d5b1d75
begin
	import Pkg
	Pkg.activate(".")
	using CairoMakie, DrWatson, DataFramesMeta, Statistics, PlutoUI, Colors
end

# ╔═╡ 6470aee0-2ea9-4a68-b4dc-ddfac4f36253
html"""
<style>
main { max-width: 60%}
</style>
"""

# ╔═╡ 47cdbc4e-c9b7-4f58-bced-1c39100ed48d
TableOfContents()

# ╔═╡ 6b154251-1959-424d-8898-e60dc0e46007
begin
	eom(A;dims) = std(A;dims) ./ √(reduce(*, size(A,d) for d in dims))
	meandrop(A;dims) = dropdims(mean(A;dims);dims)
	stddrop(A;dims) = dropdims(std(A;dims);dims)
	eomdrop(A;dims) = dropdims(eom(A;dims);dims)
end

# ╔═╡ afb34dd5-db43-47f9-be89-681e214bd9e0
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

# ╔═╡ 3783c6b2-e545-4ead-8cd2-c47881997671
md"""
# Style
"""

# ╔═╡ c44c732b-e237-4641-a2a8-828f1cd1657e
HEIGHT = 250

# ╔═╡ ecd73c4e-9694-4816-8dc7-55a2a926c79d
SCALE = 2

# ╔═╡ ca77d862-09ca-45f9-a0f1-227628d30e6d
bgcolor = RGBf(1,1,1)

# ╔═╡ f780449e-523c-4dcf-805f-aa2a34f7c05a
THEME = merge(theme_latexfonts(),
	Theme(fontsize=9*SCALE, size=(246*SCALE,HEIGHT*SCALE), pt_per_unit=1/SCALE,
		figure_padding=(1,7,1,1),
		Axis=(; xtickalign=1, ytickalign=1, xscale=identity,
			backgroundcolor=bgcolor),
		Legend=(; backgroundcolor=bgcolor),
		Label=(; font=:bold,
			halign=:left, valign=:top)))

# ╔═╡ 6ea4ae92-60f6-456e-b503-306dcdde6649
to_color(r,g,b) = RGBf(r/255,g/255,b/255)

# ╔═╡ ebfdf40e-e026-44aa-8df2-116227c47dcb
brini = [to_color(29, 59, 181), to_color(220, 171, 67), to_color(51, 51, 51), to_color(119, 188, 101), to_color(135, 78, 189)]

# ╔═╡ a729fc18-4b28-44f1-a589-4c56715f2402
# section1_palette = (;
# 	dTWA = Makie.wong_colors()[1],
# 	rg_dcTWA = Makie.wong_colors()[7],
# 	ed = RGBAf(0,0,0),
# 	rg_gcTWA = Makie.wong_colors()[3],
# 	naive_gcTWA = Makie.RGBf(0.9,0.3,0.7))
section1_palette = (;
	dTWA = brini[2],
	rg_dcTWA = brini[5],
	ed = brini[3],
	rg_gcTWA = brini[4],
	naive_gcTWA = brini[1])

# ╔═╡ 0619e6fc-cf83-48f3-90d9-ccb5daea0ee9
md"""
# Section 1&2
## Load data
"""

# ╔═╡ 3df43d03-8e46-4a0a-88ac-86aa3f3ccc95
df_raw = collect_results("../data/simulations");

# ╔═╡ 53f862e6-8a2b-474a-81d3-2ce6e0a8c5d8
df_avg = @chain df_raw begin
	groupby([:clustersize, :N, :α, :Δ, :tlist, :alg, :filling, :clustering])
	@combine(
		:magnetization_mean = Ref(mapreduce(x->getproperty.(x, Ref(Symbol("magnetization_mean"))), vcat, :results)),
		:magnetization_eom = Ref(mapreduce(x->getproperty.(x, Ref(Symbol("magnetization_eom"))), vcat, :results)),
		:pair_renyi2 = Ref(mapreduce(x->getproperty.(x, Ref(Symbol("pair_renyi2"))), vcat, :results)))
	@rtransform(
		:nshots = length(:magnetization_mean),
		:magnetization_mean = meandrop(stack(:magnetization_mean); dims=2),
		:magnetization_eom_stat = eomdrop(stack(:magnetization_mean); dims=2),
		:magnetization_eom_syst = sqrt.(meandrop(stack(:magnetization_eom) .^ 2; dims=2)),
		:pair_renyi2_mean = meandrop(stack(:pair_renyi2); dims=2),
		:pair_renyi2_eom = eomdrop(stack(:pair_renyi2); dims=2))
	@rtransform(
		:magnetization_eom_full = hypot.(:magnetization_eom_stat, :magnetization_eom_syst))
	select!(Not([:pair_renyi2,:magnetization_eom]))
end

# ╔═╡ 4886c159-c4a3-4dbd-830e-57b203775bdc
function make_plot_section12!(ax, data, mean, error; colors=section1_palette)
	dTWA, rg_dcTWA, naive_dcTWA, ed, rg_gcTWA, naive_gcTWA = eachrow(sort(data, [:alg, :clustering]))
	lines!(ax, ed.tlist, ed[mean]; color=colors.ed, linewidth=2, label="ED")
	lines!(ax, dTWA.tlist, dTWA[mean]; color=colors.dTWA, label="dTWA")
	lines!(ax, rg_dcTWA.tlist, rg_dcTWA[mean]; color=colors.rg_dcTWA, label="dcTWA (RG)", linewidth=2)
	lines!(ax, rg_gcTWA.tlist, rg_gcTWA[mean]; color=colors.rg_gcTWA, linestyle=:dash, label="gcTWA (RG)", linewidth=1.8)
	lines!(ax, naive_gcTWA.tlist, naive_gcTWA[mean]; color=colors.naive_gcTWA, linestyle=Linestyle([0.0,1.0,2.0]), label="gcTWA (naive)")
end

# ╔═╡ 7a738d24-a0f4-49f4-b143-1a3568d4ebf4
md"""
## Fig. 3 Magnetization f=10%
"""

# ╔═╡ b1ee0c7b-df63-41b7-a2c9-f1d92f047aa8
with_theme(THEME) do
	let fig = Figure(),
		ax1 = Axis(fig[1,1]; ylabel=L"\langle M^{st}(t)\rangle")
		make_plot_section12!(ax1,
			@rsubset(df_avg, :α == 1, :filling == 0.1, :Δ == 0),
			:magnetization_mean, :magnetization_eom_full)
		ax2 = Axis(fig[2,1]; xlabel=L"Time $t$ $[J_0^{-1}]$", ylabel=L"\langle M^{st}(t)\rangle")
		make_plot_section12!(ax2,
			@rsubset(df_avg, :α == 3, :filling == 0.1, :Δ == 0),
			:magnetization_mean, :magnetization_eom_full)
		xlims!.([ax1, ax2],-2, 102)
		ylims!(ax1,-0.17, 1.05)
		ax1.xticks = 0:20:100
		ax2.xticks = 0:20:100
		ax1.yticks = -1:0.2:1
		ax2.yticks = -1:0.2:1
		ax1.xticklabelsvisible = false
		#Legend(fig[3,1], ax; )
		axislegend(ax1; position=:rt)
		text!(ax1, 30.0, 1.0; text=L"\alpha=1.0,\ f=10%", align=(:center,:top))
		text!(ax2, 30.0, 1.0; text=L"\alpha=3.0,\ f=10%", align=(:center,:top))

		Label(fig[1,1]; text="(a)", tellwidth=false, tellheight=false, alignmode=Mixed(;left=-55, bottom=6))
		Label(fig[2,1]; text="(b)", tellwidth=false, tellheight=false, alignmode=Mixed(;left=-55, bottom=6))

		fig
	end
end |> save_and_display("fig3")

# ╔═╡ 2d86e00c-04d2-43e2-a2ed-2cedecbf084f
md"""
## Fig. 4 Magnetization f=50%
"""

# ╔═╡ c00ca17a-24d8-4855-8406-109b1452f6b0
with_theme(THEME) do
	let fig = Figure(),
		ax1 = Axis(fig[1,1]; ylabel=L"\langle M^{st}(t)\rangle")
		make_plot_section12!(ax1,
			@rsubset(df_avg, :α == 1, :filling == 0.5, :Δ == 0),
			:magnetization_mean, :magnetization_eom_full)
		ax2 = Axis(fig[2,1]; xlabel=L"Time $t$ $[J_0^{-1}]$", ylabel=L"\langle M^{st}(t)\rangle")
		make_plot_section12!(ax2,
			@rsubset(df_avg, :α == 3, :filling == 0.5, :Δ == 0),
			:magnetization_mean, :magnetization_eom_full)
		xlims!(ax1, -0.25, 15)
		ax1.xticks = 0:4:20
		ax1.yticks = -0.2:0.2:1

		xlims!(ax2,-2, 102)
		ax2.xticks = 0:20:100
		ax2.yticks = -0.2:0.2:1
		linkyaxes!(ax1,ax2)

		axislegend(ax1; position=:rt)
		text!(ax1, 4.5, 1.0; text=L"\alpha=1.0,\ f=50%", align=(:center,:top))
		text!(ax2, 30.0, 1.0; text=L"\alpha=3.0,\ f=50%", align=(:center,:top))

		Label(fig[1,1]; text="(a)", tellwidth=false, tellheight=false, alignmode=Mixed(;left=-55, bottom=6))
		Label(fig[2,1]; text="(b)", tellwidth=false, tellheight=false, alignmode=Mixed(;left=-55, bottom=6))

		fig
	end
end |> save_and_display("fig4")

# ╔═╡ 34931634-392e-41ea-a894-80946701df64
md"""
## Fig. 6: Renyi2 f=10%
"""

# ╔═╡ 65094c08-9581-45b8-992e-fd8465bd168b
with_theme(THEME) do
	let fig = Figure(),
		ax1 = Axis(fig[1,1]; ylabel=L"S_{2}(t)")
		make_plot_section12!(ax1,
			@rsubset(df_avg, :α == 1, :filling == 0.1, :Δ == 0),
			:pair_renyi2_mean, :pair_renyi2_eom)
		ax2 = Axis(fig[2,1]; xlabel=L"Time $t$ $[J_0^{-1}]$", ylabel=L"S_{2}(t)")
		make_plot_section12!(ax2,
			@rsubset(df_avg, :α == 3, :filling == 0.1, :Δ == 0),
			:pair_renyi2_mean, :pair_renyi2_eom)
		xlims!.([ax1, ax2],-2, 102)
		#ylims!(ax1,-0.17, 1.05)
		ax1.xticks = 0:20:100
		ax2.xticks = 0:20:100
		ax1.yticks = -0:0.4:2
		ax2.yticks = -1:0.2:2
		ax1.xticklabelsvisible = false
		#Legend(fig[3,1], ax; )
		axislegend(ax1; position=:rb)
		text!(ax1, 30.0, 0.2; text=L"\alpha=1.0,\ f=10%", align=(:center,:top))
		text!(ax2, 30.0, 0.1; text=L"\alpha=3.0,\ f=10%", align=(:center,:top))

		Label(fig[1,1]; text="(a)", tellwidth=false, tellheight=false, alignmode=Mixed(;left=-55, bottom=6))
		Label(fig[2,1]; text="(b)", tellwidth=false, tellheight=false, alignmode=Mixed(;left=-55, bottom=6))

		fig
	end
end |> save_and_display("fig6")

# ╔═╡ 5547f002-6f28-423e-b0e5-34f0605d3ab4
md"""
## Fig. 7 Renyi2 f=50%
"""

# ╔═╡ 53234bb5-3bc4-4f40-acda-6d8ad48f5a45
with_theme(THEME) do
	let fig = Figure(),
		ax1 = Axis(fig[1,1]; ylabel=L"S_{2}(t)")
		make_plot_section12!(ax1,
			@rsubset(df_avg, :α == 1, :filling == 0.5, :Δ == 0),
			:pair_renyi2_mean, :pair_renyi2_eom)
		ax2 = Axis(fig[2,1]; xlabel=L"Time $t$ $[J_0^{-1}]$", ylabel=L"S_{2}(t)")
		make_plot_section12!(ax2,
			@rsubset(df_avg, :α == 3, :filling == 0.5, :Δ == 0),
			:pair_renyi2_mean, :pair_renyi2_eom)

		xlims!(ax1, -0.25, 15)
		ax1.xticks = 0:4:20
		ax1.yticks = 0:0.4:2

		xlims!(ax2,-2, 102)
		ax2.xticks = 0:20:100
		ax2.yticks = 0:0.4:2
		linkyaxes!(ax1,ax2)

		#ax1.xticklabelsvisible = false

		axislegend(ax1; position=:rb)
		text!(ax1, 4.0, 0.2; text=L"\alpha=1.0,\ f=50%", align=(:center,:top))
		text!(ax2, 30.0, 0.2; text=L"\alpha=3.0,\ f=50%", align=(:center,:top))

		Label(fig[1,1]; text="(a)", tellwidth=false, tellheight=false, alignmode=Mixed(;left=-55, bottom=6))
		Label(fig[2,1]; text="(b)", tellwidth=false, tellheight=false, alignmode=Mixed(;left=-55, bottom=6))

		fig
	end
end |> save_and_display("fig7")

# ╔═╡ 9ac3b5bd-6aab-4b2f-8f2c-3699e359b3ff
md"""
## Fig. 8: XXZ Magnetization f=10%, α=0.5
"""

# ╔═╡ 30f31979-ef62-4452-960c-d5b796f57ab4
with_theme(THEME) do
	let fig = Figure(),
		ax1 = Axis(fig[1,1]; ylabel=L"\langle M^{st}(t)\rangle")
		make_plot_section12!(ax1,
			@rsubset(df_avg, :α == 0.5, :filling == 0.1, :Δ == 0),
			:magnetization_mean, :magnetization_eom_full)
		ax2 = Axis(fig[2,1]; ylabel=L"\langle M^{st}(t)\rangle")
		make_plot_section12!(ax2,
			@rsubset(df_avg, :α == 0.5, :filling == 0.1, :Δ == 2),
			:magnetization_mean, :magnetization_eom_full)
		ax3 = Axis(fig[3,1]; xlabel=L"Time $t$ $[J_0^{-1}]$", ylabel=L"\langle M^{st}(t)\rangle")
		make_plot_section12!(ax3,
			@rsubset(df_avg, :α == 0.5, :filling == 0.1, :Δ == 4),
			:magnetization_mean, :magnetization_eom_full)
		xlims!.([ax1, ax2,ax3],-2, 52)
		#ylims!(ax1,-0.17, 1.05)
		ax1.xticks = 0:10:100
		ax2.xticks = 0:10:100
		ax3.xticks = 0:10:100
		ax1.yticks = -1:0.25:1
		ax2.yticks = -1:0.25:1
		ax3.yticks = -1:0.25:1
		ax1.xticklabelsvisible = false
		ax2.xticklabelsvisible = false

		Legend(fig[0,1], ax1; orientation=:horizontal, nbanks=2, width=Relative(1))
		text!(ax1, 30.0, 1.0; text=L"\Delta=0,\ \alpha=0.5,\ f=10%", align=(:center,:top))
		text!(ax2, 30.0, 1.0; text=L"\Delta=2,\ \alpha=0.5,\ f=10%", align=(:center,:top))
		text!(ax3, 30.0, 1.0; text=L"\Delta=4,\ \alpha=0.5,\ f=10%", align=(:center,:top))

		Label(fig[1,1]; text="(a)", tellwidth=false, tellheight=false, alignmode=Mixed(;left=-70, bottom=6))
		Label(fig[2,1]; text="(b)", tellwidth=false, tellheight=false, alignmode=Mixed(;left=-70, bottom=6))
		Label(fig[3,1]; text="(c)", tellwidth=false, tellheight=false, alignmode=Mixed(;left=-70, bottom=6))

		fig
	end
end |> save_and_display("fig8")

# ╔═╡ df836ff5-d809-4243-be61-e7187fdadba6
md"""
## Fig. 9: XXZ Magnetization f=10%, α=6
"""

# ╔═╡ f185475e-acab-4ba4-8767-fa0955626f7e
with_theme(THEME) do
	let fig = Figure(),
		ax1 = Axis(fig[1,1]; ylabel=L"\langle M^{st}(t)\rangle")
		make_plot_section12!(ax1,
			@rsubset(df_avg, :α == 6, :filling == 0.1, :Δ == 0),
			:magnetization_mean, :magnetization_eom_full)
		ax2 = Axis(fig[2,1]; ylabel=L"\langle M^{st}(t)\rangle")
		make_plot_section12!(ax2,
			@rsubset(df_avg, :α == 6, :filling == 0.1, :Δ == 2),
			:magnetization_mean, :magnetization_eom_full)
		ax3 = Axis(fig[3,1]; xlabel=L"Time $t$ $[J_0^{-1}]$", ylabel=L"\langle M^{st}(t)\rangle")
		make_plot_section12!(ax3,
			@rsubset(df_avg, :α == 6, :filling == 0.1, :Δ == 4),
			:magnetization_mean, :magnetization_eom_full)
		xlims!.([ax1, ax2, ax3], -2, 102)
		ylims!.([ax1, ax2, ax3], 0.5, 1.02)
		ax1.xticks = 0:20:100
		ax2.xticks = 0:20:100
		ax3.xticks = 0:20:100
		ax1.yticks = -1:0.2:1
		ax2.yticks = -1:0.2:1
		ax3.yticks = -1:0.2:1
		ax1.xticklabelsvisible = false
		ax2.xticklabelsvisible = false
		#Legend(fig[3,1], ax; )
		#axislegend(ax1; position=:lb, orientation=:horizontal, nbanks=2)
		Legend(fig[0,1], ax1; orientation=:horizontal, nbanks=2, width=Relative(1))
		text!(ax1, 0.0, 0.57; text=L"\Delta=0,\ \alpha=6,\ f=10%", align=(:left,:center))
		text!(ax2, 0.0, 0.57; text=L"\Delta=2,\ \alpha=6,\ f=10%", align=(:left,:center))
		text!(ax3, 0.0, 0.57; text=L"\Delta=4,\ \alpha=6,\ f=10%", align=(:left,:center))

		Label(fig[1,1]; text="(a)", tellwidth=false, tellheight=false, alignmode=Mixed(;left=-55, bottom=6))
		Label(fig[2,1]; text="(b)", tellwidth=false, tellheight=false, alignmode=Mixed(;left=-55, bottom=6))
		Label(fig[3,1]; text="(c)", tellwidth=false, tellheight=false, alignmode=Mixed(;left=-55, bottom=6))

		fig
	end
end |> save_and_display("fig9")

# ╔═╡ 1c241b2e-cf5a-40a7-b4c6-43b78a76a241
md"""
## Fig. 10: XXZ Renyi f=10%, α=0.5
"""

# ╔═╡ 37b85827-8dbb-4386-a073-748e9fe2a24f
with_theme(THEME) do
	let fig = Figure(),
		ax1 = Axis(fig[1,1]; ylabel=L"S_{2}(t)")
		make_plot_section12!(ax1,
			@rsubset(df_avg, :α == 0.5, :filling == 0.1, :Δ == 0),
			:pair_renyi2_mean, :pair_renyi2_eom)
		ax2 = Axis(fig[2,1]; ylabel=L"S_{2}(t)")
		make_plot_section12!(ax2,
			@rsubset(df_avg, :α == 0.5, :filling == 0.1, :Δ == 2),
			:pair_renyi2_mean, :pair_renyi2_eom)
		ax3 = Axis(fig[3,1]; xlabel=L"Time $t$ $[J_0^{-1}]$", ylabel=L"S_{2}(t)")
		make_plot_section12!(ax3,
			@rsubset(df_avg, :α == 0.5, :filling == 0.1, :Δ == 4),
			:pair_renyi2_mean, :pair_renyi2_eom)
		xlims!.([ax1, ax2,ax3],-2, 52)
		ylims!(ax1; high=2.1) # ensure ytick at 2.0 does not clip
		ax1.xticks = 0:10:100
		ax2.xticks = 0:10:100
		ax3.xticks = 0:10:100
		ax1.yticks = -0:0.4:2
		ax2.yticks = -0:0.4:2
		ax3.yticks = -0:0.4:2
		ax1.xticklabelsvisible = false
		ax2.xticklabelsvisible = false
		#Legend(fig[3,1], ax; )
		axislegend(ax1; position=:rb, rowgap=0)
		text!(ax1, 5.0, 0.02; text=L"\Delta=0,\ \alpha=0.5,\ f=10%", align=(:left,:bottom))
		text!(ax2, 5.0, 0.02; text=L"\Delta=2,\ \alpha=0.5,\ f=10%", align=(:left,:bottom))
		text!(ax3, 5.0, 0.02; text=L"\Delta=4,\ \alpha=0.5,\ f=10%", align=(:left,:bottom))

		Label(fig[1,1]; text="(a)", tellwidth=false, tellheight=false, alignmode=Mixed(;left=-55, bottom=6))
		Label(fig[2,1]; text="(b)", tellwidth=false, tellheight=false, alignmode=Mixed(;left=-55, bottom=6))
		Label(fig[3,1]; text="(c)", tellwidth=false, tellheight=false, alignmode=Mixed(;left=-55, bottom=6))

		fig
	end
end |> save_and_display("fig10")

# ╔═╡ 6d29ac93-63f4-4ae1-810a-a18929136a8a
md"""
## Fig. 11: XXZ Renyi f=10%, α=6
"""

# ╔═╡ 86248bac-bf97-4a12-bafe-5078acb30634
with_theme(THEME) do
	let fig = Figure()

		ax1 = Axis(fig[1,1]; ylabel=L"S_{2}(t)")
		make_plot_section12!(ax1,
			@rsubset(df_avg, :α == 6, :filling == 0.1, :Δ == 0),
			:pair_renyi2_mean, :pair_renyi2_eom)
		ax2 = Axis(fig[2,1]; ylabel=L"S_{2}(t)")
		make_plot_section12!(ax2,
			@rsubset(df_avg, :α == 6, :filling == 0.1, :Δ == 2),
			:pair_renyi2_mean, :pair_renyi2_eom)
		ax3 = Axis(fig[3,1]; xlabel=L"Time $t$ $[J_0^{-1}]$", ylabel=L"S_{2}(t)")
		make_plot_section12!(ax3,
			@rsubset(df_avg, :α == 6, :filling == 0.1, :Δ == 4),
			:pair_renyi2_mean, :pair_renyi2_eom)

		xlims!.([ax1, ax2,ax3],-2, 102)
		ylims!.([ax1, ax2,ax3],-0.02, 0.65)

		ax1.xticks = 0:20:100
		ax2.xticks = 0:20:100
		ax3.xticks = 0:20:100
		ax1.yticks = -1:0.2:1
		ax2.yticks = -1:0.2:1
		ax3.yticks = -1:0.2:1
		ax1.xticklabelsvisible = false
		ax2.xticklabelsvisible = false

		Legend(fig[0,1], ax1; orientation=:horizontal, nbanks=2, width=Relative(1))

		text!(ax1, 0.0, 0.45; text=L"\Delta=0,\ \alpha=6,\ f=10%", align=(:left,:bottom))
		text!(ax2, 0.0, 0.45; text=L"\Delta=2,\ \alpha=6,\ f=10%", align=(:left,:bottom))
		text!(ax3, 0.0, 0.45; text=L"\Delta=4,\ \alpha=6,\ f=10%", align=(:left,:bottom))

		Label(fig[1,1]; text="(a)", tellwidth=false, tellheight=false, alignmode=Mixed(;left=-55, bottom=6))
		Label(fig[2,1]; text="(b)", tellwidth=false, tellheight=false, alignmode=Mixed(;left=-55, bottom=6))
		Label(fig[3,1]; text="(c)", tellwidth=false, tellheight=false, alignmode=Mixed(;left=-55, bottom=6))

		fig
	end
end |> save_and_display("fig11")

# ╔═╡ 2e958e44-c753-412a-a74f-ef924f0e1087
md"""
# Section 1: ordered
"""

# ╔═╡ 4869330d-8d87-41d7-9a8f-70da5a6ba2d2
sort(@rsubset(df_avg, :α == 1, :filling == 1, :Δ == 0), [:alg, :clustersize])

# ╔═╡ e373101f-f7af-4443-bd76-e7292bbdf399
md"""
## Fig. 5 Magnetization/Renyi ordered
"""

# ╔═╡ 38588cee-b474-4987-bed8-174739edc636
function lines_with_band!(ax, x, y, err; color, label=nothing, linestyle=:solid, line_kwargs=(;), band_kwargs=(;))
	band!(ax, x, y .- err, y .+ err; alpha=0.5, color, band_kwargs...)
	lines!(ax, x, y; color, label, linestyle, line_kwargs...)
end

# ╔═╡ 0a6d4ab8-2a04-4e49-a4e0-45c54cd70dd0
with_theme(THEME) do
	let fig = Figure()

		ax1 = Axis(fig[1,1]; ylabel=L"\langle M^{st}(t)\rangle")
		_, d2, d4, ed, g2, g4 = eachrow(sort(@rsubset(df_avg, :α == 1, :filling == 1, :Δ == 0), [:alg, :clustersize]))
		lines!(ax1, ed.tlist, ed.magnetization_mean;
			color=section1_palette.ed,
			label="ED")
		lines!(ax1, g2.tlist, g2.magnetization_mean;
			color=section1_palette.dTWA,
			label="gcTWA (size 2)")
		lines!(ax1, d2.tlist, d2.magnetization_mean;
			color=section1_palette.rg_dcTWA,
			label="dcTWA (size 2)")
		lines!(ax1, g4.tlist, g4.magnetization_mean;
			color=section1_palette.rg_gcTWA,
			label="gcTWA (size 4)",
			linestyle=:dash)
		lines!(ax1, d4.tlist, d4.magnetization_mean;
			color=section1_palette.naive_gcTWA,
			label="dcTWA (size 4)",
			linestyle=:dash)

		ax2 = Axis(fig[2,1]; xlabel=L"Time $t$ $[J_0^{-1}]$", ylabel=L"\langle M^{st}(t)\rangle")
		_, d2, d4, ed, g2, g4 = eachrow(sort(@rsubset(df_avg, :α == 3, :filling == 1, :Δ == 0), [:alg, :clustersize]))
		lines!(ax2, ed.tlist, ed.magnetization_mean;
			color=section1_palette.ed,
			label="ED")
		lines!(ax2, g2.tlist, g2.magnetization_mean;
			color=section1_palette.dTWA,
			label="gcTWA (size 2)")
		lines!(ax2, d2.tlist, d2.magnetization_mean;
			color=section1_palette.rg_dcTWA,
			label="dcTWA (size 2)")
		lines!(ax2, g4.tlist, g4.magnetization_mean;
			color=section1_palette.rg_gcTWA,
			label="gcTWA (size 4)",
			linestyle=:dash)
		lines!(ax2, d4.tlist, d4.magnetization_mean;
			color=section1_palette.naive_gcTWA,
			label="dcTWA (size 4)",
			linestyle=:dash)

		ax1.xticklabelsvisible = false
		ax1.xticks = 0:2:10
		ax2.xticks = 0:2:10
		ax1.yticks = -0.4:0.2:1
		ax2.yticks = -0.4:0.2:1

		axislegend(ax1; position=:rt)

		text!(ax1, 2.0, 0.9; text=L"\alpha=1,\ f=100%", align=(:left,:center))
		text!(ax2, 2.0, 0.9; text=L"\alpha=3,\ f=100%", align=(:left,:center))

		Label(fig[1,1]; text="(a)", tellwidth=false, tellheight=false, alignmode=Mixed(;left=-60, bottom=6))
		Label(fig[2,1]; text="(b)", tellwidth=false, tellheight=false, alignmode=Mixed(;left=-60, bottom=6))
		fig
	end
end |> save_and_display("fig5")

# ╔═╡ 585f6dfb-a313-419a-8656-779041953c0b
md"""
With statistical error as ribbon:
"""

# ╔═╡ 946dcae3-77fa-4b25-be79-382b85779cc4
with_theme(THEME) do
	let fig = Figure(),
		ax1 = Axis(fig[1,1]; ylabel=L"\langle M^{st}(t)\rangle")
		_, d2, d4, ed, g2, g4 = eachrow(sort(@rsubset(df_avg, :α == 1, :filling == 1, :Δ == 0), [:alg, :clustersize]))
		lines!(ax1, ed.tlist, ed.magnetization_mean;
			color=section1_palette.ed,
			label="ED")
		lines_with_band!(ax1, g2.tlist, g2.magnetization_mean, g2.magnetization_eom_syst;
			color=section1_palette.dTWA,
			label="gcTWA (size 2)")
		lines_with_band!(ax1, d2.tlist, d2.magnetization_mean, d2.magnetization_eom_syst;
			color=section1_palette.rg_dcTWA,
			label="dcTWA (size 2)")
		lines_with_band!(ax1, g4.tlist, g4.magnetization_mean, g4.magnetization_eom_syst;
			color=section1_palette.rg_gcTWA,
			label="gcTWA (size 4)",
			linestyle=:dash)
		lines_with_band!(ax1, d4.tlist, d4.magnetization_mean, d4.magnetization_eom_syst;
			color=section1_palette.naive_gcTWA,
			label="dcTWA (size 4)",
			linestyle=:dash)

		ax2 = Axis(fig[2,1]; xlabel=L"Time $t$ $[J_0^{-1}]$",ylabel=L"\langle M^{st}(t)\rangle")
		_, d2, d4, ed, g2, g4 = eachrow(sort(@rsubset(df_avg, :α == 3, :filling == 1, :Δ == 0), [:alg, :clustersize]))
		lines!(ax2, ed.tlist, ed.magnetization_mean;
			color=section1_palette.ed,
			label="ED")
		lines_with_band!(ax2, g2.tlist, g2.magnetization_mean, g2.magnetization_eom_syst;
			color=section1_palette.dTWA,
			label="gcTWA (size 2)")
		lines_with_band!(ax2, d2.tlist, d2.magnetization_mean, d2.magnetization_eom_syst;
			color=section1_palette.rg_dcTWA,
			label="dcTWA (size 2)")
		lines_with_band!(ax2, g4.tlist, g4.magnetization_mean, g4.magnetization_eom_syst;
			color=section1_palette.rg_gcTWA,
			label="gcTWA (size 4)",
			linestyle=:dash)
		lines_with_band!(ax2, d4.tlist, d4.magnetization_mean, d4.magnetization_eom_syst;
			color=section1_palette.naive_gcTWA,
			label="dcTWA (size 4)",
			linestyle=:dash)

		ax1.xticklabelsvisible = false
		ax1.xticks = 0:2:10
		ax2.xticks = 0:2:10
		ax1.yticks = -0.4:0.2:1
		ax2.yticks = -0.4:0.2:1

		axislegend(ax1; position=:rt)

		text!(ax1, 2.0, 0.9; text=L"\alpha=1,\ f=100%", align=(:left,:center))
		text!(ax2, 2.0, 0.9; text=L"\alpha=3,\ f=100%", align=(:left,:center))
		fig
	end
end |> save_and_display("fig5")

# ╔═╡ da8a82d2-1e65-46af-8eb8-d31c619b39f4
with_theme(THEME) do
	let fig = Figure(),
		ax1 = Axis(fig[1,1]; ylabel=L"S_{2}(t)")
		_, d2, d4, ed, g2, g4 = eachrow(sort(@rsubset(df_avg, :α == 1, :filling == 1, :Δ == 0), [:alg, :clustersize]))
		lines!(ax1, ed.tlist, ed.pair_renyi2_mean;
			color=section1_palette.ed,
			label="ED")
		lines_with_band!(ax1, g2.tlist, g2.pair_renyi2_mean, g2.magnetization_eom_syst;
			color=section1_palette.dTWA,
			label="gcTWA (size 2)")
		lines!(ax1, d2.tlist, d2.pair_renyi2_mean;
			color=section1_palette.rg_dcTWA,
			label="dcTWA (size 2)")
		lines_with_band!(ax1, g4.tlist, g4.pair_renyi2_mean, g4.magnetization_eom_syst;
			color=section1_palette.rg_gcTWA,
			label="gcTWA (size 4)",
			linestyle=:dash)
		lines_with_band!(ax1, d4.tlist, d4.pair_renyi2_mean, d4.magnetization_eom_syst;
			color=section1_palette.naive_gcTWA,
			label="dcTWA (size 4)",
			linestyle=:dash)

		ax2 = Axis(fig[2,1]; xlabel=L"Time $t$ $[J_0^{-1}]$", ylabel=L"S_{2}(t)")
		_, d2, d4, ed, g2, g4 = eachrow(sort(@rsubset(df_avg, :α == 3, :filling == 1, :Δ == 0), [:alg, :clustersize]))
		lines!(ax2, ed.tlist, ed.pair_renyi2_mean;
			color=section1_palette.ed,
			label="ED")
		lines!(ax2, g2.tlist, g2.pair_renyi2_mean;
			color=section1_palette.dTWA,
			label="gcTWA (size 2)")
		lines!(ax2, d2.tlist, d2.pair_renyi2_mean;
			color=section1_palette.rg_dcTWA,
			label="dcTWA (size 2)")
		lines!(ax2, g4.tlist, g4.pair_renyi2_mean;
			color=section1_palette.rg_gcTWA,
			label="gcTWA (size 4)",
			linestyle=:dash)
		lines!(ax2, d4.tlist, d4.pair_renyi2_mean;
			color=section1_palette.naive_gcTWA,
			label="dcTWA (size 4)",
			linestyle=:dash)

		ax1.xticklabelsvisible = false
		ax1.xticks = 0:2:10
		ax2.xticks = 0:2:10
		ax1.yticks = -0.4:0.4:2
		ax2.yticks = -0.4:0.4:2

		axislegend(ax1; position=:rb)

		text!(ax1, 2.0, 0.9; text=L"\alpha=1,\ f=100%", align=(:left,:center))
		text!(ax2, 2.0, 0.9; text=L"\alpha=3,\ f=100%", align=(:left,:center))
		fig
	end
end

# ╔═╡ 588c4b11-8640-4a58-871e-8d5fc5994eda
md"""
# Section 3: Statistics
"""

# ╔═╡ d2c571a0-c7a4-4e0d-a15b-2e538e6da1b2
data_full_raw = @chain collect_results("../data/fulldata-simulations-avg") begin
	@rtransform(
		:magnetization_mean = :results[1].magnetization_mean,
		:magnetization_std = :results[1].magnetization_std,
		:renyi_means = :results[1].renyi_means,
		:renyi_stds = :results[1].renyi_stds,
		:renyi_chunksizes = :results[1].renyi_chunksizes)
	select!(Not([:path, :results, :chunkID, :fulldata]))
    @rsubset(:α == 1)
	sort!([:alg, :clustersize])
end

# ╔═╡ 944c511c-b11c-44ab-a905-aadeb0cde079
8/binomial(16,2)

# ╔═╡ 5d83874d-9f8a-479c-bcca-173aa3e77663
md"""
## Fig. 12: Stddev of magnetization
"""

# ╔═╡ 54b1348a-dd78-4af0-97ea-f06156dd89b7
let (d2,d4,g2,g4) = eachrow(data_full_raw)
	f2 = sum(d2.magnetization_std)/sum(g2.magnetization_std)
	println("1-D2/G2: ", 1-f2, " -> ", 1-f2^2)
	f4 = sum(d4.magnetization_std)/sum(g4.magnetization_std)
	println("1-D4/G4: ", 1-f4, " -> ", 1-f4^2)
end

# ╔═╡ 01c14141-6e58-4053-b2e5-2b16bdaa2336
let (d2,d4,g2,g4) = eachrow(data_full_raw)
	f2 = sum(d2.renyi_stds[2])/sum(g2.renyi_stds[2])
	println("1-D2/G2: ", 1-f2, " -> ", 1-f2^2)
	f4 = sum(d4.renyi_stds[2])/sum(g4.renyi_stds[2])
	println("1-D4/G4: ", 1-f4, " -> ", 1-f4^2)
end

# ╔═╡ ee883650-6321-467c-a8e9-dbfad5594bd3
with_theme(THEME) do
	let fig = Figure()

		ax1 = Axis(fig[1,1]; ylabel=L"\sigma\langle M^{st}(t)\rangle")
		d2,d4,g2,g4 = eachrow(data_full_raw)
		lines!(ax1, g2.tlist, g2.magnetization_std;
			color=section1_palette.rg_gcTWA,
			label="gcTWA (size 2)",
			linestyle=:dash)
		lines!(ax1, d2.tlist, d2.magnetization_std;
			color=section1_palette.rg_gcTWA,
			label="dcTWA (size 2)")
		lines!(ax1, g4.tlist, g4.magnetization_std;
			color=section1_palette.rg_dcTWA,
			label="gcTWA (size 4)",
			linestyle=:dash)
		lines!(ax1, d4.tlist, d4.magnetization_std;
			color=section1_palette.rg_dcTWA,
			label="dcTWA (size 4)")

		ax2 = Axis(fig[2,1]; xlabel=L"Time $t$ $[J_0^{-1}]$",ylabel=L"\sigma S_2(t)")
		lines!(ax2, g2.tlist, g2.renyi_stds[2];
			color=section1_palette.rg_gcTWA,
			label="gcTWA (size 2)",
			linestyle=:dash)
		lines!(ax2, d2.tlist, d2.renyi_stds[2];
			color=section1_palette.rg_gcTWA,
			label="dcTWA (size 2)")
		lines!(ax2, g4.tlist, g4.renyi_stds[2];
			color=section1_palette.rg_dcTWA,
			label="gcTWA (size 4)",
			linestyle=:dash)
		lines!(ax2, d4.tlist, d4.renyi_stds[2];
			color=section1_palette.rg_dcTWA,
			label="dcTWA (size 4)")


		ax1.xticklabelsvisible = false
		ax1.xticks = 0:2:10
		ax2.xticks = 0:2:10

		axislegend(ax1; position=:rb)

		Label(fig[1,1]; text="(a)", tellwidth=false, tellheight=false, alignmode=Mixed(;left=-55, bottom=6))
		Label(fig[2,1]; text="(b)", tellwidth=false, tellheight=false, alignmode=Mixed(;left=-55, bottom=26))
		fig
	end
end |> save_and_display("fig12")

# ╔═╡ 481a1793-68e8-49cd-bd21-f46f4e2e44d3
with_theme(THEME) do
	let fig = Figure(; size=(246*SCALE,2*HEIGHT*SCALE)),
		(d2,d4,g2,g4) = eachrow(data_full_raw),
		axes = []
		for (i, chunksize) in enumerate(d2.renyi_chunksizes)
			ax = Axis(fig[i,1]; ylabel=L"\Delta\langle M^{st}(t)\rangle")
			push!(axes, ax)
			ax.xticklabelsvisible = false
			ax.xticks = 0:2:10

			lines!(ax, g2.tlist, g2.renyi_stds[i];
			color=section1_palette.dTWA,
			label="gcTWA (size 2)")
			lines!(ax, d2.tlist, d2.renyi_stds[i];
				color=section1_palette.rg_dcTWA,
				label="dcTWA (size 2)")
			lines!(ax, g4.tlist, g4.renyi_stds[i];
				color=section1_palette.rg_gcTWA,
				label="gcTWA (size 4)",
				linestyle=:dash)
			lines!(ax, d4.tlist, d4.renyi_stds[i];
				color=section1_palette.naive_gcTWA,
				label="dcTWA (size 4)",
				linestyle=:dash)
			text!(ax, 4, 0; text=L"c=%$chunksize", align=(:center,:bottom))
		end
		axes[end].xticklabelsvisible = true

		axislegend(axes[1]; position=:rb, rowgap=0)
		fig
	end
end# |> save_and_display("fig12")

# ╔═╡ 25a1d382-ee99-41aa-bacf-1c6f62d3c7c3
with_theme(THEME) do
	let fig = Figure(),
		ax1 = Axis(fig[1,1]; ylabel=L"\Delta\langle M^{st}(t)\rangle")
		d2,d4,g2,g4 = eachrow(data_full_raw)
		lines!(ax1, g2.tlist, g2.magnetization_mean;
			color=section1_palette.dTWA,
			label="gcTWA (size 2)")
		lines!(ax1, d2.tlist, d2.magnetization_mean;
			color=section1_palette.rg_dcTWA,
			label="dcTWA (size 2)")
		lines!(ax1, g4.tlist, g4.magnetization_mean;
			color=section1_palette.rg_gcTWA,
			label="gcTWA (size 4)",
			linestyle=:dash)
		lines!(ax1, d4.tlist, d4.magnetization_mean;
			color=section1_palette.naive_gcTWA,
			label="dcTWA (size 4)",
			linestyle=:dash)

		ax2 = Axis(fig[2,1]; ylabel=L"\Delta S_2(t)")
		lines!(ax2, g2.tlist, g2.renyi_means[2];
			color=section1_palette.dTWA,
			label="gcTWA (size 2)")
		lines!(ax2, d2.tlist, d2.renyi_means[2];
			color=section1_palette.rg_dcTWA,
			label="dcTWA (size 2)")
		lines!(ax2, g4.tlist, g4.renyi_means[2];
			color=section1_palette.rg_gcTWA,
			label="gcTWA (size 4)",
			linestyle=:dash)
		lines!(ax2, d4.tlist, d4.renyi_means[2];
			color=section1_palette.naive_gcTWA,
			label="dcTWA (size 4)",
			linestyle=:dash)

		ax1.xticklabelsvisible = false
		ax1.xticks = 0:2:10
		ax2.xticks = 0:2:10
		# ax1.yticks = -0.4:0.2:1
		# ax2.yticks = -0.4:0.2:1

		axislegend(ax1; position=:rt)

		# text!(ax1, 2.0, 0.9; text=L"\alpha=1,\ f=100%", align=(:left,:center))
		# text!(ax2, 2.0, 0.9; text=L"\alpha=3,\ f=100%", align=(:left,:center))
		fig
	end
end# |> save_and_display("fig12")

# ╔═╡ 7afe0db8-eef1-4470-8e33-2c6aa5290de6
with_theme(THEME) do
	let fig = Figure(),
		ax1 = Axis(fig[1,1]; ylabel="Total uncertainty", xlabel="Chunksize c", xscale=log10, yscale=log10)
		d2,d4,g2,g4 = eachrow(data_full_raw)
		scatter!(ax1, g2.renyi_chunksizes, sum.(g2.renyi_stds);
			color=section1_palette.dTWA,
			label="gcTWA (size 2)")
		scatter!(ax1, d2.renyi_chunksizes, sum.(d2.renyi_stds);
			color=section1_palette.rg_dcTWA,
			label="dcTWA (size 2)", marker='X')
		scatter!(ax1, g4.renyi_chunksizes, sum.(g4.renyi_stds);
			color=section1_palette.rg_gcTWA,
			label="gcTWA (size 4)")
		scatter!(ax1, d4.renyi_chunksizes, sum.(d4.renyi_stds);
			color=section1_palette.naive_gcTWA,
			label="dcTWA (size 4)", marker='X')
		lines!(ax1, d4.renyi_chunksizes, x->200/√(x); label=L"\propto c^{-0.5}", color=section1_palette.ed, linestyle=:dot)


		# ax1.xticklabelsvisible = false
		# ax1.xticks = 0:2:10
		ax1.yticks = (10 .^ (0.6:0.2:1.4), [L"10^{%$x}" for x in (0.6:0.2:1.4)])
		ax1.xticks = (10 .^ (1.6:0.2:3), [L"10^{%$x}" for x in (1.6:0.2:3)])
		# ax2.yticks = -0.4:0.2:1

		axislegend(ax1; position=:rt)

		# text!(ax1, 2.0, 0.9; text=L"\alpha=1,\ f=100%", align=(:left,:center))
		# text!(ax2, 2.0, 0.9; text=L"\alpha=3,\ f=100%", align=(:left,:center))
		fig
	end
end# |> save_and_display("fig12")

# ╔═╡ Cell order:
# ╠═7b9b98d2-12c3-11ef-04f4-6dee8d5b1d75
# ╠═6470aee0-2ea9-4a68-b4dc-ddfac4f36253
# ╟─47cdbc4e-c9b7-4f58-bced-1c39100ed48d
# ╠═6b154251-1959-424d-8898-e60dc0e46007
# ╠═afb34dd5-db43-47f9-be89-681e214bd9e0
# ╟─3783c6b2-e545-4ead-8cd2-c47881997671
# ╠═c44c732b-e237-4641-a2a8-828f1cd1657e
# ╠═ecd73c4e-9694-4816-8dc7-55a2a926c79d
# ╠═ca77d862-09ca-45f9-a0f1-227628d30e6d
# ╠═f780449e-523c-4dcf-805f-aa2a34f7c05a
# ╠═a729fc18-4b28-44f1-a589-4c56715f2402
# ╠═6ea4ae92-60f6-456e-b503-306dcdde6649
# ╠═ebfdf40e-e026-44aa-8df2-116227c47dcb
# ╟─0619e6fc-cf83-48f3-90d9-ccb5daea0ee9
# ╠═3df43d03-8e46-4a0a-88ac-86aa3f3ccc95
# ╠═53f862e6-8a2b-474a-81d3-2ce6e0a8c5d8
# ╠═4886c159-c4a3-4dbd-830e-57b203775bdc
# ╟─7a738d24-a0f4-49f4-b143-1a3568d4ebf4
# ╠═b1ee0c7b-df63-41b7-a2c9-f1d92f047aa8
# ╟─2d86e00c-04d2-43e2-a2ed-2cedecbf084f
# ╠═c00ca17a-24d8-4855-8406-109b1452f6b0
# ╟─34931634-392e-41ea-a894-80946701df64
# ╠═65094c08-9581-45b8-992e-fd8465bd168b
# ╟─5547f002-6f28-423e-b0e5-34f0605d3ab4
# ╠═53234bb5-3bc4-4f40-acda-6d8ad48f5a45
# ╟─9ac3b5bd-6aab-4b2f-8f2c-3699e359b3ff
# ╠═30f31979-ef62-4452-960c-d5b796f57ab4
# ╟─df836ff5-d809-4243-be61-e7187fdadba6
# ╠═f185475e-acab-4ba4-8767-fa0955626f7e
# ╟─1c241b2e-cf5a-40a7-b4c6-43b78a76a241
# ╠═37b85827-8dbb-4386-a073-748e9fe2a24f
# ╟─6d29ac93-63f4-4ae1-810a-a18929136a8a
# ╠═86248bac-bf97-4a12-bafe-5078acb30634
# ╟─2e958e44-c753-412a-a74f-ef924f0e1087
# ╠═4869330d-8d87-41d7-9a8f-70da5a6ba2d2
# ╟─e373101f-f7af-4443-bd76-e7292bbdf399
# ╠═38588cee-b474-4987-bed8-174739edc636
# ╠═0a6d4ab8-2a04-4e49-a4e0-45c54cd70dd0
# ╟─585f6dfb-a313-419a-8656-779041953c0b
# ╟─946dcae3-77fa-4b25-be79-382b85779cc4
# ╠═da8a82d2-1e65-46af-8eb8-d31c619b39f4
# ╟─588c4b11-8640-4a58-871e-8d5fc5994eda
# ╠═d2c571a0-c7a4-4e0d-a15b-2e538e6da1b2
# ╠═944c511c-b11c-44ab-a905-aadeb0cde079
# ╟─5d83874d-9f8a-479c-bcca-173aa3e77663
# ╠═54b1348a-dd78-4af0-97ea-f06156dd89b7
# ╠═01c14141-6e58-4053-b2e5-2b16bdaa2336
# ╠═ee883650-6321-467c-a8e9-dbfad5594bd3
# ╠═481a1793-68e8-49cd-bd21-f46f4e2e44d3
# ╠═25a1d382-ee99-41aa-bacf-1c6f62d3c7c3
# ╠═7afe0db8-eef1-4470-8e33-2c6aa5290de6
