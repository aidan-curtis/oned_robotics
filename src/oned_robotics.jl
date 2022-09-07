module oned_robotics

using POMDPs
using POMDPModelTools
using Distributions
import Plots
import Distributions: pdf
import Random:rand, rand!
import POMDPs: action, solve, updater, actions, initialstate, gen, isterminal
import POMDPModelTools: action_info, UnderlyingMDP, BoolDistribution, transition, observation, reward, discount
using Random
using BeliefUpdaters
using ParticleFilters
import ParticleFilters: resample
import BasicPOMCP: extract_belief
import Base: (==), hash

include("environment.jl")
include("jitter_resampler.jl")

export 
    Env1D,
    Env1DGen,
    generate_image_obs,
    Env1DAction,
    Env1DState,
    Env1DObject,
    Interval,
    Color,
    RED,
    GREEN,
    BLUE,
    generate_state_image,
    JitterResampler,
    resample,
    jitter

end # module
