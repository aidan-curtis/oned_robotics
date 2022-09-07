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
using Images
using D3Trees
using POMCPOW
using Images
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

function resize_image(image_matrix, scale)
    matrix_zeros = zeros(3, scale*size(image_matrix, 2), scale*size(image_matrix, 3))
    for i in 0:size(image_matrix, 2)-1
        for j in 0:size(image_matrix, 3)-1
            matrix_zeros[:, i*scale+1:i*scale+scale, j*scale+1:j*scale+scale] .= image_matrix[:, i+1, j+1]
        end
    end
    return matrix_zeros
end


mutable struct ConstrainedParticleFilter{PM,RM,RS,RNG<:AbstractRNG,PMEM} <: Updater
    predict_model::PM
    reweight_model::RM
    resampler::RS
    n_init::Int
    rng::RNG
    _particle_memory::PMEM
    _weight_memory::Vector{Float64}
end

## Constructors ##
function ConstrainedParticleFilter(model, resampler, n::Integer, rng::AbstractRNG=Random.GLOBAL_RNG)
    return ConstrainedParticleFilter(model, model, resampler, n, rng)
end

function ConstrainedParticleFilter(pmodel, rmodel, resampler, n::Integer, rng::AbstractRNG=Random.GLOBAL_RNG)
    return ConstrainedParticleFilter(pmodel,
                               rmodel,
                               resampler,
                               n,
                               rng,
                               particle_memory(pmodel),
                               Float64[]
                              )
end

function initialize_belief(up::ConstrainedParticleFilter, d::D) where D
    return HistoryParticleCollection(collect(rand(up.rng, d) for i in 1:up.n_init), [])
end

struct HistoryParticleCollection{T} <: AbstractParticleBelief{T}
    particles::Vector{T}
    h::Vector{Any}
    # _probs::Union{Nothing, Dict{T,Float64}} # a cache for the probabilities
end

function main()
    problem = POMDPProblem(Env1DGen(0.0), false, max_steps)
    algo = POMDPAlgo(POMCPOWSolver(criterion=MaxUCB(20.0), tree_queries=10, alpha_action=0.2, k_action=10, k_observation=10, alpha_observation=0.2), false)
    pomdp = problem.pomdp()
    planner = solve(algo.solver, pomdp)

    # hr = HistoryRecorder(max_steps=max_steps)
    # pomcpow_hist = simulate(hr, pomdp, planner)

    tmp_dir = "temp_images"
    if (isdir(tmp_dir))
        rm(tmp_dir, recursive=true)
    end

    mkdir(tmp_dir)

    rng = MersenneTwister(rand(planner.solver.rng, UInt32))
    n = 10*planner.solver.tree_queries
    up = ConstrainedParticleFilter(planner.problem, JitterResampler(n), n, rng)
   
    b = initialstate(pomdp)
    s = rand(planner.solver.rng, initialstate(pomdp))
    particles = initialize_belief(up, b)
    is = [1, s, particles]
    println("Num particles: $(length(particles.particles))")
    vis_tree = false

    while  is[1] < problem.num_steps
        t, s, b = is
        a, ai = action_info(planner, b, tree_in_info=true) # Planner
        if (vis_tree)
            inchrome(D3Tree(ai[:tree], init_expand=3))
        end
        println("==============")
        println("Step: $(t)")
        println("State: $(s)")
        println("Action: $(a)")
        
        sp, o, r, _ = @gen(:sp, :o, :r, :info)(pomdp, s, a, planner.solver.rng) # Env step
        bp, _ = update_info(up, b, a, o) # Particle filter update
        is = [t+1, sp, bp]

        resized = resize_image(generate_state_image(pomdp, sp)/255.0, 20)
        state_image = colorview(RGB, resized) 
        save("$(tmp_dir)/state$(t).jpg", state_image)
        if (size(o.val, 2) > 0)
            save("$(tmp_dir)/observation$(t).jpg", colorview(RGB, resize_image(o.val/255.0, 20)))
        end
        
        for particle_index in collect(1:10)
            particle = bp.particles[particle_index]
            save("$(tmp_dir)/belief$(t)_particle$(particle_index).jpg", colorview(RGB, resize_image(generate_state_image(pomdp, particle)/255.0, 20)))
        end


        println("Reward: $(r)")
        println("State Prime: $(sp)")
        println("==============")
    end
end

main()