#!/bin/sh
# ########## Begin Slurm header ##########
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=16:00:00
#SBATCH --mem=180gb
#SBATCH --cpus-per-task=96
#SBATCH --job-name=TWA
#SBATCH --output="logs/TWA-N_16-%j.out"
########### End Slurm header ##########
#=
export ON_CLUSTER=1
export MKL_DYNAMIC=false
exec julia --heap-size-hint=200G --color=no --threads=96 --startup-file=no scripts/run_twa.jl "$@"
=#
using DrWatson
@quickactivate "DiscreteCTWAPaper"

# Here you may include files from the source directory
include(srcdir("setup.jl"))
using .Setup
@setup

include(srcdir("simulation.jl"))
using .Simulation

CHUNKIDS = length(ARGS) > 0 ? parse.(Int, ARGS) : collect(1:10)

params_section1_ctwa = Dict(
    "alg" => ["gcTWA", "dcTWA"],
    "N" => 16,
    "α" => [1,3],
    "Δ" => 0,
    "filling" => [0.1,0.5],
    "chunkID" => CHUNKIDS,
    "chunksize" => 100,
    "tlist" => range(0,100;length=501),
    "trajectories" => 1000,
    "clustering" => ["RG", "naive"],
    "clustersize" => 2,
)

params_section1_dtwa = Dict(
    "alg" => "dTWA",
    "N" => 16,
    "α" => [1,3],
    "Δ" => 0,
    "filling" => [0.1,0.5],
    "chunkID" => CHUNKIDS,
    "chunksize" => 100,
    "tlist" => range(0,100;length=501),
    "trajectories" => 1000,
)

params_section2_ctwa = Dict(
    "alg" => ["gcTWA", "dcTWA"],
    "N" => 16,
    "α" => [0.5,6],
    "Δ" => [0,2,4],
    "filling" => [0.1,0.5],
    "chunkID" => CHUNKIDS,
    "chunksize" => 100,
    "tlist" => range(0,100;length=501),
    "trajectories" => 1000,
    "clustering" => ["RG", "naive"],
    "clustersize" => 2,
)

params_section2_dtwa = Dict(
    "alg" => "dTWA",
    "N" => 16,
    "α" => [0.5,6],
    "Δ" => [0,2,4],
    "filling" => [0.1,0.5],
    "chunkID" => CHUNKIDS,
    "chunksize" => 100,
    "tlist" => range(0,100;length=501),
    "trajectories" => 1000,
)

paramset = mapreduce(remove_done∘dict_list, vcat, (params_section1_ctwa, params_section1_dtwa, params_section2_ctwa, params_section2_dtwa))
sort!(paramset; by=x->x["chunkID"])
total = length(paramset)
@info "TODO" total
for (i, params) in enumerate(paramset)
    @info "Doing $i/$total" params
    flush(stdout)
    with_logger(logger_for_params(params)) do
        @info "Job id" get(ENV, "SLURM_JOB_ID", "") # for the param log file
        @time Simulation.run(params)
    end
    flush(stdout)
end
exit(0)
