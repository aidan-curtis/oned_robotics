using oned_robotics
using POMDPSimulators
using Distributions
using DataStructures
using Dates
using JSON
using Plots
using POMDPModels
using POMDPs
import POMDPs:solve
using ParticleFilters
using POMDPPolicies
using Random
using D3Trees
using POMCPOW
using BasicPOMCP

max_steps = 10

# solver = POMCPOWSolver(criterion=MaxUCB(20.0))
# pomdp = BabyPOMDP() # from POMDPModels
# planner = solve(solver, pomdp)

struct POMDPProblem
    pomdp::Any
    discrete::Bool
    num_steps::Int64
end

struct POMDPAlgo
    solver::Any
    discrete::Bool
end

struct POMDPPolicySolver
    policy::Any
end

function solve(solver::POMDPPolicySolver, pomdp::POMDPs.POMDP)
    return solver.policy(pomdp)
end

problem = POMDPProblem(Env1DGen(), false, 10)
algo = POMDPAlgo(POMCPOWSolver(criterion=MaxUCB(20.0)), false)
pomdp = problem.pomdp()
planner = solve(algo.solver, pomdp)
hr = HistoryRecorder(max_steps=max_steps)
pomcpow_history = simulate(hr, pomdp, planner)
rollout = []

for (s, a, sp, o, r) in pomcpow_hist
    push!(rollout, Dict("action"=>a, "state"=>s, "state_prime"=>sp, "obs"=>o, "reward"=>r))
    @show s, a, sp, o, r
end