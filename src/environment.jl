import Base: ==, +, *, -

struct Color
    r::Float64
    g::Float64
    b::Float64
end

struct Interval
    lo::Float64
    hi::Float64
end

RED = Color(1.0, 0, 0)
GREEN = Color(0, 1.0, 0)
BLUE = Color(0, 0, 1.0)

function sample_color(rng)
    return rand(rng, USparseCat([RED, GREEN, BLUE]))
end

struct Env1DObject
    pos::Float64
    size::Float64
    color::Color
end

struct Env1DState
    robot_loc::Float64
    grasped::Union{Nothing, Int64} # Object index or nothing
    objects::Vector{Env1DObject}
end

Base.copy(s::Env1DState) = Env1DState(s.robot_loc, s.grasped, copy(s.objects))

function lo(o::Env1DObject)
    return o.pos-o.size/2.0
end

function hi(o::Env1DObject)
    return o.pos+o.size/2.0
end

function range(o::Env1DObject)
    return Interval(lo(o), hi(o))
end

struct ObjectProperties
    color::Color
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
WORKSPACE = Interval(-10, 10)
FOV = 2
MAX_N_OBJS = 3
MIN_SIZE = 1
MAX_SIZE = 3
ABUT_PROB = 0.5

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

function close(loc1::Float64, loc2::Float64, tolerance::Float64)
    return loc1<=loc2+tolerance && loc1>=loc2-tolerance
end

function actions(::Env1D) 
    return ImplicitDistribution() do rng
        action_type = rand(rng, USparseCat([:look, :look_obj_hi, :look_obj_lo, :look_region, :move, :pick, :place]))
        target_loc = rand(rng, Distributions.Uniform(WORKSPACE.lo, WORKSPACE.hi))
        return Env1DAction(action_type, target_loc)
    end
end

# Find a position for this object in this interval
function get_placement(size::Float64, interval::Interval, abut::Bool, rng)
    min_pos = interval.lo + size/2.0
    max_pos = interval.hi - size/2.0
    if abut
        return rand(rng, USparseCat([min_pos, max_pos]))
    else
        return rand(rng, Distributions.Uniform(min_pos, max_pos))
    end
end

# Do bookkeeping to remove this interval and add replacement(s)
function update_free_intervals(interval::Interval, pos::Float64, size::Float64, free_intervals::Vector{Interval})
    free_intervals = collect(filter!(x->x!=interval, free_intervals))
    
    obj_int = Interval(pos-size/2.0, pos+size/2.0)
    abuts_lo = close(interval.lo, obj_int.lo, 0.001)
    abuts_hi = close(interval.hi, obj_int.hi, 0.001)

    if !abuts_hi
        push!(free_intervals, Interval(obj_int.hi, interval.hi))
    end
    if !abuts_lo
        push!(free_intervals, Interval(interval.lo, obj_int.lo))
    end
    return free_intervals
end

# Place an object of this color
function attempt_place(color::Color, objects::Vector{Env1DObject}, free_intervals::Vector{Interval}, rng)
    size = rand(rng, Distributions.Uniform(MIN_SIZE, MAX_SIZE))
    abut = rand(rng, Bernoulli(ABUT_PROB))
    free_intervals = shuffle(free_intervals)
    for interval in free_intervals
        if (size <= interval_size(interval))
            pos = get_placement(size, interval, abut, rng)
            push!(objects, Env1DObject(pos, size, color))
            update_free_intervals(interval, pos, size, free_intervals)
            return objects, free_intervals
        end
    end
    return objects, free_intervals
end

function initialstate(p::Env1D)
    return ImplicitDistribution() do rng
        num_objects = rand(rng, USparseCat(collect(1:MAX_N_OBJS)))        
        objects = Vector{Env1DObject}()
        free_intervals = [WORKSPACE]

        for _ in 1:num_objects
            color = sample_color(rng)
            objects, free_intervals = attempt_place(color, objects, free_intervals, rng)
        end

        agent_pos = rand(rng, Distributions.Uniform(WORKSPACE.lo, WORKSPACE.hi))
        return Env1DState(agent_pos, nothing, objects)
    end
end


function color_vector(c::Color, alpha::Float64=1.0)
    return [min(c.r+1-alpha, 1.0), min(c.g+1-alpha, 1.0), min(c.b+1-alpha, 1.0)]
end


function bound(v, minv, maxv)
    return max(min(v, maxv), minv)
end

function generate_image_obs(p::Env1D, s::Env1DState)

    SCALE = 10
    HEIGHT = 40
    AGENT_SIZE = 5

    state_im = ones(3, HEIGHT, floor(Int64, interval_size(WORKSPACE) * SCALE))
    max_coord = size(state_im)[3]

    function tf(v::Float64)
        return convert(Int64, bound(floor((v-WORKSPACE.lo)*SCALE), 1, max_coord))
    end

    # Objects
    for o in s.objects
        cv = color_vector(o.color, 0.3)
        state_im[:, :, tf(lo(o)):tf(hi(o))] .= repeat(cv, 1, HEIGHT)
    end

    # Agent
    cv = color_vector(RED, 1.0)
    rcoord = tf(s.robot_loc)
    ahlow = convert(Int64, bound(HEIGHT/2.0-AGENT_SIZE, 1, max_coord))
    ahhigh = convert(Int64, bound(HEIGHT/2.0+AGENT_SIZE, 1, max_coord))
    chlow = convert(Int64, bound(rcoord-AGENT_SIZE, 1, max_coord))
    chhigh = convert(Int64, bound(rcoord+AGENT_SIZE, 1, max_coord))

  
    state_im[:, ahlow:ahhigh, chlow:chhigh] .= repeat(cv, 1, 1, chhigh-chlow+1)

    return state_im
    
end

function pdf(d::Normal{Float64}, o::Env1DObs)
    pdf(d, 0)
end

function observation(p::Env1D, a::Env1DAction, sp::Env1DState)
    return Normal(sp.robot_loc, 0.1)
end

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
            if (overlaps_interval(Interval(s.robot_loc-s.grasped.size/2.0, s.robot_loc+s.grasped.size/2.0), range(o)))
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



function generate_obj_obs(o::Env1DObject, overlap::Interval, obs_field::Interval)
    return ObjectProperties(o.color,
                    overlap.lo,
                    lo(o) < obs_field.lo - 1e-10, # Why?
                    overlap.hi,
                    hi(o) > obs_field.hi + 1e-10)
end

function touching_objects(s::Env1DState, o::Env1DObject)
    left = nothing
    right = nothing
    for to in s.objects
        if to == o
             continue
        end
        if lo(o) > hi(to)
            left = to
        end
        if hi(o) > lo(to)
            right = to
        end
    end
    return left, right
end

function point_in_interval(pos::Float64, range::Interval)
    return pos>=range.lo && pos<=range.hi
end

function obj_at(s::Env1DState, pos::Float64)
    for o in s.objects
        if point_in_interval(pos, range(o))
            return o
        end
    end
    return nothing
end

function interval_intersection(i1::Interval, i2::Interval)

    if (i2.lo<i1.lo)
        i1, i2 = i2, i1
    end

    if (i1.hi < i2.lo)
        return Interval(0, 0)
    elseif (i1.hi > i2.hi)
        return i2
    else
        return Interval(i2.lo, i2.hi)
    end
end

function interval_size(interval::Interval)
    return interval.hi-interval.lo
end

function interval_union(interval1::Interval, interval2::Interval)
    return [min(interval1.lo, interval2.lo), 
            max(interval1.hi, interval2.hi)]
end

function generate_observation(s::Env1DState, rng)
    view_field = Interval(s.robot_loc-FOV/2, s.robot_loc+FOV/2)
    obs_field = interval_intersection(WORKSPACE, view_field)
    obs_obj_props = []
    fail = false
    for o in s.objects
        if isnothing(o.pos)
            continue  # Object is being held
        end
        overlap = interval_intersection(range(o), obs_field)
        if interval_size(overlap) < MIN_OBJECT_DETECT_SIZE 
            continue
        end
        push!(obs_obj_props, generate_obj_obs(o, overlap, obs_field))
        # Get objects to the left and right of me and make extra detections
        ool, oor = touching_objects(s, o)
        if !isnothing(ool) && ool.color == o.color
            l_overlap = interval_intersection(interval_union(range(ool), range(o)), obs_field)
            if (interval_size(l_overlap) >= MIN_OBJECT_DETECT_SIZE)
                push!(obs_obj_props, generate_obj_obs(o, l_overlap, obs_field))
            end
        end
        if !isnothing(oor) && oor.color == o.color
            r_overlap = interval_intersection(interval_union(range(o),  range(oor)), obs_field)
            if (interval_size(r_overlap) >= MIN_OBJECT_DETECT_SIZE)
                push!(obs_obj_props, generate_obj_obs(o, r_overlap, obs_field))
            end
        end
        if (rand(rng, Distributions.Uniform(0, 1)) < OBJ_FRAGMENT_PROB)
            # Add two new detections corresponding to this object
            # Could also consider removing the original detection, but not doing that for now
            p = rand(rng, Distributions.Uniform(overlap.lo, overlap.hi))
            push!(obs_obj_props, generate_obj_obs(o, Interval(overlap.lo, p), obs_field))
            push!(obs_obj_props, generate_obj_obs(o, Interval(p, overlap.hi), obs_field))
        end
    end
    return obs_obj_props, fail
end

function POMDPs.gen(p::Env1D, s::Env1DState, act::Env1DAction, rng) 

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
        obs_obj_props, fail = generate_observation(s, rng)
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
                if close(o.pos, s.robot_loc, PICK_TOLERANCE)
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
            fail = true
            # print("Attempted to place, but not holding anything or Attempted to place, but space not clear")
        else
            next_objects[s.grasped].pos = s.robot_loc
            next_grasped = nothing
            fail = false
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
            obs_obj_props, fail = generate_observation(s, rng)
        end
    end

    obs = Env1DObs(obs_obj_props, !fail)
    sp = Env1DState(sp_robot_loc, next_grasped, next_objects)

    # The goal is placing an object of a certain color at a certain location within an interval
    return (sp=sp, o=obs, r=reward)
end
