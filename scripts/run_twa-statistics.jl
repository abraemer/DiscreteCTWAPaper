#!/bin/sh
# ########## Begin Slurm header ##########
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=50gb
#SBATCH --cpus-per-task=96
#SBATCH --job-name=TWA-statistics
#SBATCH --output="logs/TWA-statistics-N_16-%j.out"
########### End Slurm header ##########
#=
export ON_CLUSTER=1
export MKL_DYNAMIC=false
exec julia --heap-size-hint=50G --color=no --threads=96 --startup-file=no scripts/run_twa-statistics.jl
=#
using DrWatson
@quickactivate "DiscreteCTWAPaper"

# Here you may include files from the source directory
include(srcdir("setup.jl"))
using .Setup
@setup

include(srcdir("simulation.jl"))
using .Simulation

params_section3 = Dict(
    "alg" => ["gcTWA", "dcTWA"],
    "N" => 16,
    "α" => 1,
    "Δ" => 0,
    "filling" => 0.1,
    "chunkID" => 1,
    "chunksize" => 1,
    "tlist" => range(0,10;length=501),
    "trajectories" => 10000,
    "clustering" => "naive",
    "clustersize" => [2,4],
    "fulldata" => true
)

PREFIX = "fulldata-"

paramset = remove_done(dict_list(params_section3); prefix=PREFIX)
total = length(paramset)
@info "TODO" total
for (i, params) in enumerate(paramset)
    @info "Doing $i/$total" params
    with_logger(logger_for_params(params; prefix=PREFIX)) do
        @info "Job id" get(ENV, "SLURM_JOB_ID", "") # for the param log file
        @time Simulation.run(params; prefix=PREFIX)
    end
    flush(stdout)
end
include("../notebooks/data_analysis.jl")
exit(0)
