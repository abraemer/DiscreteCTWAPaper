#!/bin/sh
# ########## Begin Slurm header ##########
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=16:00:00
#SBATCH --mem=180gb
#SBATCH --cpus-per-task=48
#SBATCH --job-name=TWA-ctwa-rg2_N16
#SBATCH --output="logs/ctwa-rg2-N_16-%j.out"
########### End Slurm header ##########
#=
# load modules
# not needed - julia installed locally

# export JULIA_DEPOT_PATH=$SCRATCH
export ON_CLUSTER=1
export MKL_DYNAMIC=false
exec julia1.9 --heap-size-hint=100G --color=no --threads=96 --startup-file=no scripts/run_ctwa_rg2.jl
=#
using DrWatson
@quickactivate "DiscreteCTWAPaper"

# Here you may include files from the source directory
include(srcdir("setup.jl"))
using .Setup
@setup

include(srcdir("simulation.jl"))
using .Simulation
using OrdinaryDiffEq

BLAS.set_num_threads(Threads.nthreads())

CHUNKIDS = length(ARGS) > 0 ? parse.(Int, ARGS) : collect(1:10)

params_section1_ed = Dict(
    "alg" => "ed",
    "N" => 8,
    "α" => [1,3],
    "Δ" => 0,
    "filling" => [0.1,0.5],
    "chunkID" => CHUNKIDS,
    "chunksize" => 5,
    "tlist" => range(0,100;length=101),
)

params_section1_ordered_ed = Dict(
    "alg" => "ed",
    "N" => 8,
    "α" => [1,3],
    "Δ" => 0,
    "filling" => 1.0,
    "chunkID" => 1,
    "chunksize" => 5,
    "tlist" => range(0,10;length=101),
)

#paramset = remove_done(dict_list(params_section1_ed))
paramset = mapreduce(remove_done∘dict_list, vcat, (params_section1_ed, params_section1_ordered_ed))
total = length(paramset)
@info "TODO" total
for (i, params) in enumerate(paramset)
    @info "Doing $i/$total" params
    with_logger(logger_for_params(params)) do
        @info "Job id" get(ENV, "SLURM_JOB_ID", "") # for the param log file
        @time Simulation.run(params)
    end
end
exit(0)
