module Setup

export @setup, logger_for_params, save_file_missing, remove_done

using LoggingExtras
using DrWatson
using Dates
# general setup of Julia

macro setup()
    esc(quote
        const ONCLUSTER = haskey(ENV, "ON_CLUSTER")

        using LoggingExtras

        Setup.global_logger(Setup.add_datetime_logger(ConsoleLogger(stdout)))

        if haskey(ENV, "ON_CLUSTER")
            @eval using MKL
            println("Added MKL.jl")
        end

        using LinearAlgebra
        BLAS.set_num_threads(1)

        println("Working Directory:          $(pwd())" )
        println("Running on host:            $(gethostname())" )
        println("Job id:                     $(get(ENV, "SLURM_JOB_ID", ""))" )
        println("Job name:                   $(get(ENV, "SLURM_JOB_NAME", ""))" )
        println("Number of nodes allocated:  $(get(ENV, "SLURM_JOB_NUM_MODES", ""))" )
        println("Number of cores allocated:  $(get(ENV, "SLURM_NTASKS", ""))" )
        println("#threads of Julia:          $(Threads.nthreads())")
        println("#threads of BLAS:           $(BLAS.get_num_threads())")
        println("#BLAS config:               $(BLAS.get_config())")
        @info "ARGS" ARGS

        using ThreadPinning
        pinthreads(:cores)
        threadinfo(stdout; hints=true, color=false, blas=true)
        flush(stdout)

        mkpath(datadir("logs"))
    end)

end

function add_datetime_logger(logger)
    return TransformerLogger(logger) do log
        merge(log, (; message = "[$(Dates.now())]-#$(Threads.threadid()): $(log.message)"))
    end
end

function logger_for_params(params, minlevel = Logging.Info)
    logfile = datadir("logs", savename(params, "txt"))
    return MinLevelLogger(
        add_datetime_logger(
            TeeLogger(
                MinLevelLogger(ConsoleLogger(stdout), Logging.Warn),
                FileLogger(logfile))),
        minlevel)
end

save_file_missing(param) = !isfile(datadir("simulations", savename(param, "jld2")))
remove_done(paramset) = filter(save_file_missing, paramset)

end# module
