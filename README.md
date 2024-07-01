# DiscreteCTWAPaper

This repo contains all of the code used in the process of writing the Paper ["Cluster truncated Wigner approximation for bond-disordered Heisenberg spin models"](). Comments welcome. If you find anything useful in this repo, feel free to use it for your own projects with appropriate credit. For citing our paper see `CITATION.bib` (which will be created once the paper is published).

This code base is using the [Julia Language](https://julialang.org/) and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> DiscreteCTWAPaper

It is authored by Adrian Braemer and Javad Vahedi.

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

## Folders

### `src`
`src` contains code used for the simulation code:
 - `src/setup.jl`: Setup code for each of the simulations, i.e. loading libraries and preparing logging
 - `src/simulation.jl`: driver for the simulations: Saving/loading of data, building/solving the Hamiltonian
 - `src/numerics.jl`: defines how observables are computed

 ### `scripts`
 Contains the SLURM scripts with the parameters used to generate all the data. Each of these can take `chunkID`s as CLI parameters. In the paper 10 chunks where used. Each script produces multiple output files and can (and needs to) simply be rerun until everything is done.

 ### `notebooks`
 [`Pluto.jl` notebooks](https://plutojl.org/) for data analysis and plotting. `data_analysis.jl` can be run as a Julia script to process the large samples used for evaluation of statistical errors into averaged ones. (this is a bit historic and not necessary if set up differently...)
 Pdf versions of the notebooks are provided for convenience.

 ### `data`
Folder containing the data. Currently, only averages for the statistical analysis part of the paper are included due to file size limitations. Full data is available by reasonable request.

# Citing
See `cite.bib` (not yet created - will update once paper is published).

# License
DiscreteCTWAPaper © 2024 by Adrian Braemer and Javad Vahedi is licensed under [MIT license](https://opensource.org/license/mit).