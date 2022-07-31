using oned_robotics
using POMDPSimulators
using POMDPModelTools
using POMDPPolicies
using POMDPModels
using POMDPs
using Distributions
using DataStructures
using Dates
using JSON
using Plots
import POMDPs:solve
using ParticleFilters
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

function main()
    problem = POMDPProblem(Env1DGen(), false, 100)
    algo = POMDPAlgo(POMCPOWSolver(criterion=MaxUCB(20.0), tree_queries=10), false)
    pomdp = problem.pomdp()
    planner = solve(algo.solver, pomdp)

    # hr = HistoryRecorder(max_steps=max_steps)
    # pomcpow_hist = simulate(hr, pomdp, planner)

    rollout = []
    tmp_dir = "temp_images"
    if (isdir(tmp_dir))
        rm(tmp_dir, recursive=true)
    end
    mkdir(tmp_dir)

    up = updater(planner)
    b = initialstate(pomdp)
    s = rand(planner.solver.rng, initialstate(pomdp))
    particles = initialize_belief(up, b)
    is = [1, s, particles]
    vis_tree = false

    while  is[1] < problem.num_steps
        t, s, b = is
        a, ai = action_info(planner, b, tree_in_info=true) # Planner
        if (vis_tree)
            inchrome(D3Tree(ai[:tree], init_expand=3))
        end
        sp, o, r, _ = @gen(:sp, :o, :r, :info)(pomdp, s, a, planner.solver.rng) # Env step
        bp, _ = update_info(up, b, a, o) # Particle filter update
        is = [t+1, sp, bp]

        println("==============")
        println("Step: $(t)")
        println("State: $(s)")
        println("Action: $(a)")
        println("Obs: $(o)")
        println("Reward: $(r)")
        println("State Prime: $(sp)")
        println("==============")
    end
end

main()