import Base: ==, +, *, -


struct Env1DObject
    pos::Float64
    side::Float64
end

struct Env1DState
    robot_loc::Float64
    grasped::Union{Nothing, Int64} # Object index or nothing
    objects::Vector{Env1DObject}
end

Base.copy(s::Env1DState) = Env1DState(s.robot_loc, s.grasped, copy(s.objects))

function lo(o::Env1DObject)
    return o.pos-o.size
end
function hi(o::Env1DObject)
    return o.pos+o.size
end
function range(o::Env1DObject)
    return [o.pos, o.size]
end

struct ObjectProperties
    color::String
    lo::Float64
    lo_occluded::Float64
    hi::Float64
    hi_occluded::Float64
end

struct Env1DObs
    properties::Vector{ObjectProperties}
    success::Bool
end

function EmptyEnv1DObs()
    return Env1DObs([], false)
end

struct Env1DAction
    type::Symbol
    target_loc::Float64
end


mutable struct Env1D <: POMDPs.POMDP{Env1DState, Env1DAction, Env1DObs}
    discount_factor::Float64
end

MIN_OBJECT_DETECT_SIZE = 1
OBJ_FRAGMENT_PROB = 0.3
WORKSPACE = [-10, 10]
FOV = 2

function Env1D() 
    return Env1D(0.95)
end

function Env1DGen()
    return () -> Env1D()
end

discount(p::Env1D) = p.discount_factor
isterminal(::Env1D, s::Env1DState) = false


function USparseCat(values)
    return SparseCat(values, [1/length(values) for _ in 1:length(values)])
end

function within(loc1, loc2, tolerance)
    return loc1<=loc2+tolerance && loc1>=loc2-tolerance
end

function actions(::Env1D) 
    return ImplicitDistribution() do rng
        action_type = rand(rng, USparseCat([:look, :look_obj_hi, :look_obj_lo, :look_region, :move, :pick, :place]))
        target_loc = rand(rng, Distributions.Uniform(WORKSPACE[1], WORKSPACE[2]))
        return Env1DAction(action_type, target_loc)
    end
end

function initialstate(p::Env1D)
    return ImplicitDistribution() do rng
        return Env1DState(0, false, [])
    end
end

observation(p::Env1D, sp::Env1DState) = Normal(sp.robot_loc, 0.1)

function is_look_action(act::Env1DAction)
    return (act.type == :look || 
           act.type == :look_obj_hi ||
           act.type == :look_obj_lo || 
           act.type == :look_region)
end

function is_push_action(act::Env1DAction)
    return act.type == :push_from_top ||
           act.type == :push_and_look1 ||
           act.type == :push_and_look2
end

function overlaps_interval(reg, range)
    return !isnothing(interval_intersection(reg, range))
end

function collision_free(s)
    for o in s.objects 
        if !isnothing(o.pos)
            if (overlaps_interval([s.robot_loc, s.grasped.size], range(o)))
                return false
            end
        end
    end
    return true
end

function obj_at(s::Env1DState)
    for o in self.objects
        if point_in_interval(pos, range(o))
            return o
        end
    end
    return None
end

function generate_obj_obs(o, overlap)
    return ObjectProperties(o.color,
                    interval_lo(overlap),
                    interval_lo(o.range()) < interval_lo(obs_field) - 1e-10,
                    interval_hi(overlap),
                    interval_hi(o.range()) > interval_hi(obs_field) + 1e-10)
end

function touching_objects(s, o)
    left = nothing
    right = nothing
    for to in s.objects
        if to == o
             continue
        end
        if o.lo() > to.hi()
            left = to
        end
        if o.hi() > to.lo()
            right = to
        end
    end
    return left, right
end

function point_in_interval(pos, range)
    println(pos)
    println(range)
end

function obj_at(s, pos)
    for o in s.objects
        if point_in_interval(pos, o.range())
            return o
        end
    end
    return nothing
end

function interval_intersection(i1, i2)
    if (first(i2)<first(i1))
        i1, i2 = i2, i1
    end
    if (last(i1) < first(i2))
        return nothing
    elseif (last(i1) > last(i2))
        return i2
    else
        return [first(i2), last(i2)]
    end
end

function generate_observation(s::Env1DState)
    view_field = [s.robot_loc, FOV/2]
    obs_field = interval_intersection(WORKSPACE, view_field)
    obs_obj_props = []
    fail = false
    for o in s.objects
        if isnothing(o.pos)
            continue  # Object is being held
        end
        overlap = interval_intersection(o.range(), obs_field)

        if interval_size(overlap) < MIN_OBJECT_DETECT_SIZE 
            continue
        end
        push!(obs.properties, generate_obj_obs(o, overlap))
        # Get objects to the left and right of me and make extra detections
        ool, oor = touching_objects(s, o)
        if ool && ool.color == o.color
            l_overlap = interval_intersection(interval_union(ool.range(), o.range()), obs_field)
            if (interval_size(l_overlap) >= MIN_OBJECT_DETECT_SIZE)
                push!(obs.properties, generate_obj_obs(o, l_overlap))
            end
        end
        if !isnothing(oor) && oor.color == o.color
            r_overlap = interval_intersection(interval_union(o.range(),  oor.range()), obs_field)
            if (interval_size(r_overlap) >= MIN_OBJECT_DETECT_SIZE)
                push!(obs.properties, generate_obj_obs(o, r_overlap))
            end
        end
        if (rand(rng, Distributions.Uniform(0, 1)) < OBJ_FRAGMENT_PROB)
            # Add two new detections corresponding to this object
            # Could also consider removing the original detection, but not doing that for now
            p = rand(rng, Distributions.Uniform(interval_lo(overlap), interval_hi(overlap)))
            push!(obs.properties, generate_obj_obs(o, lohi_to_interval(interval_lo(overlap), p)))
            push!(obs.properties, generate_obj_obs(o, lohi_to_interval(p, interval_hi(overlap))))
        end
    end
    return obs_obj_props, fail
end

function POMDPs.gen(p::Env1D, s::Env1DState, act::Env1DAction, rng) 

    println("Gen State $(s)")
    println("Gen Action $(act)")

    # Next state default values
    next_grasped = s.grasped
    sp_robot_loc = s.robot_loc
    next_objects = s.objects

    # Observation default values
    fail = false
    obs_obj_props = []

    # Default reward
    reward = 0

    if (is_look_action(act))
        obs_obj_props, fail = generate_observation(s)
    elseif act.type == :move
        sp_robot_loc = act.target_loc
    elseif act.type == :pick
        if isnothing(s.grasped)
            # If we are already holding something, then fail
            fail = true
        else
            # If we are within eps of the center of an object, then we pick otherwise we leave it where it is, or somewhat perturbed
            # Assume there are not two such objects!
            fail = true
            for (oi, o) in enumerate(s.objects)
                if within(o.pos, s.robot_loc, PICK_TOLERANCE)
                    next_grasped = oi
                    next_objects[oi].pos = nothing
                    fail = false
                end
            end
            # if obs is fail, Attempted to pick but not near center of any obj')
        end
    elseif act.type == :place
        obs = EmptyEnv1DObs()
        # No arguments
        # If there is free space under it, then it is placed whp, else it either stays in the gripper or slides sideways into a spot
        if isnothing(s.grasped) || !collision_free(s)
            obs.success = false
            # print("Attempted to place, but not holding anything or Attempted to place, but space not clear")
        else
            sp.grasped.pos = s.robot_loc
            sp.grasped = nothing
            obs = "succeed"
        end
    elseif (is_push_action(act))
        sp_robot_loc = act.target_loc
        touched_o = obj_at(s, sp_robot_loc)
        if isnothing(touched_o)
            # If we are not touching something, then fail
            fail = true
            print("Attempted to push, but not touching anything.")
        else
            self.push_obj(touched_o, p2-p1)
            fail = false
        end
        if act.type == :push_and_look1 || act.type == :push_and_look2
            obs_obj_props, fail = generate_observation(s)
        end
    end

    obs = Env1DObs(obs_obj_props, !fail)
    sp = Env1DState(sp_robot_loc, next_grasped, s.objects)

    return (sp=sp, o=obs, r=reward)
end
