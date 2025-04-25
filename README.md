This repository provides minimal working examples created for the [Hackathon: Optimizing for AMD GPUs](https://www.lumi-supercomputer.eu/events/lumi-hackathon-spring2025/) hosted by the LUMI supercomputer.

- project 1: Multiple matrix multiplication
- project 2: Large size matrix multiplication using multiple GPUs
- project 3: The Krylov method in GPU

## install environment
```shell
> git clone https://github.com/XingyuZhang2018/2025-Lumi-Hackathon
```
move to the file and run `julia REPL`, press `]` into `Pkg REPL`
```julia
(@v1.11) pkg> activate .
Activating environment at `..\2025-Lumi-Hackathon\Project.toml`

(2025-Lumi-Hackathon) pkg> instantiate
```