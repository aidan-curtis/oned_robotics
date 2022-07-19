module oned_robotics

using POMDPs
using POMDPModelTools
using Distributions

import POMDPs: action, solve, updater, actions, initialstate, gen, isterminal
import POMDPModelTools: action_info, UnderlyingMDP, BoolDistribution, transition, observation, reward, discount

include("environment.jl")

export 
    Env1D,
    Env1DGen


end # module
