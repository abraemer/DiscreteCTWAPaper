#!/bin/sh
# ########## Begin Slurm header ##########
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH --mem=20gb
#SBATCH --cpus-per-task=96
#SBATCH --job-name=ED-ordered
#SBATCH --output="logs/ED-ordered-N_16-%j.out"
########### End Slurm header ##########
#=
export ON_CLUSTER=1
export MKL_DYNAMIC=false
exec julia --heap-size-hint=300G --color=no --threads=96 --startup-file=no scripts/run_ED_ordered.jl "$@"
=#
using DrWatson
@quickactivate "DiscreteCTWAPaper"

# Here you may include files from the source directory
include(srcdir("setup.jl"))
using .Setup
@setup

include(srcdir("simulation.jl"))
using .Simulation

BLAS.set_num_threads(Threads.nthreads())

params_section1_ordered_ed = Dict(
    "alg" => "ed",
    "N" => 16,
    "α" => [1,3],
    "Δ" => 0,
    "filling" => 1.0,
    "chunkID" => 1,
    "chunksize" => 1,
    "tlist" => range(0,10;length=501),
)

paramset = remove_done(dict_list(params_section1_ordered_ed))
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
