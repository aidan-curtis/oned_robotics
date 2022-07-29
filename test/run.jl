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
using Images
using BasicPOMCP

max_steps = 1000

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
algo = POMDPAlgo(POMCPOWSolver(criterion=MaxUCB(20.0), tree_queries=10), false)
pomdp = problem.pomdp()
planner = solve(algo.solver, pomdp)
hr = HistoryRecorder(max_steps=max_steps)
pomcpow_hist = simulate(hr, pomdp, planner)
rollout = []

tmp_dir = "temp_images"
if (isdir(tmp_dir))
    rm(tmp_dir, recursive=true)
end
mkdir(tmp_dir)

for (index, (s, a, sp, o, r)) in enumerate(pomcpow_hist)
    push!(rollout, Dict("action"=>a, "state"=>s, "state_prime"=>sp, "obs"=>o, "reward"=>r))
    println("==============")
    println("State: $(s)")
    println("Action: $(a)")
    println("Obs: $(o)")
    println("Reward: $(r)")
    println("State Prime: $(sp)")
    image_obs = generate_image_obs(pomdp, s)
    save("temp_images/robotics1d-$(index).png", colorview(RGB, image_obs))
end