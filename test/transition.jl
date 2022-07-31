using oned_robotics
using Test
using POMDPSimulators
using POMDPModelTools
using POMDPPolicies
using POMDPModels
using POMDPs
using Random


GLOBAL_RNG = MersenneTwister(0)

function gen(s::Env1DState, a::Env1DAction)
    env = Env1D(0.95)
    sp, o, r = @gen(:sp, :o, :r)(env, s, a, GLOBAL_RNG)
    return sp, o, r
end

NO_OBJECTS = Vector{Env1DObject}()
SOME_OBJECTS = [
    Env1DObject(5, 2, RED),
    Env1DObject(0, 1, GREEN)
]



@testset "Move" begin
    @testset "Move while not holding" begin
        s = Env1DState(0.0, nothing, NO_OBJECTS)
        a = Env1DAction(:move, 5.0)
        sp, _, _ = gen(s, a)
        @test sp.robot_loc == 5.0
        @test sp.objects == []
        @test isnothing(sp.grasped)
    end

    @testset "Move while holding" begin
        s = Env1DState(0.0, 1, SOME_OBJECTS)
        a = Env1DAction(:move, 2.0)
        sp, _, _ = gen(s, a)
        @test sp.robot_loc == 2.0
        @test sp.objects[1].pos == sp.robot_loc
        @test sp.grasped == 1
    end
end

@testset "Pick" begin
    @testset "Pick while holding" begin
        s = Env1DState(0.0, 1, SOME_OBJECTS)
        a = Env1DAction(:pick, 0.0)
        sp, o, _ = gen(s, a)
        @test isterminal(sp)
        @test sp.grasped == 1
        @test sp.robot_loc == 0
    end

    @testset "Pick while near object" begin
        s = Env1DState(0.0, nothing, SOME_OBJECTS)
        a = Env1DAction(:pick, 0.0)
        sp, o, _ = gen(s, a)
        @test !isterminal(sp)
        @test sp.grasped == 2
        @test sp.objects[sp.grasped].pos == sp.robot_loc
        @test sp.robot_loc == 0.0
    end

    @testset "Pick while not near object" begin
        s = Env1DState(-10.0, nothing, SOME_OBJECTS)
        a = Env1DAction(:pick, 0)
        sp, o, _ = gen(s, a)
        @test isterminal(sp)
        @test isnothing(sp.grasped)
        @test sp.robot_loc == -10.0
    end 
end

@testset "Place" begin
    @testset "Place while holding" begin
        s = Env1DState(8.0, 1, SOME_OBJECTS)
        a = Env1DAction(:place, 0.0)
        sp, o, _ = gen(s, a)
        @test !isterminal(sp)
        @test isnothing(sp.grasped)
    end

    @testset "Place not holding" begin
        s = Env1DState(7.0, nothing, SOME_OBJECTS)
        a = Env1DAction(:place, 0.0)
        sp, o, _ = gen(s, a)
        @test isterminal(sp)
    end

    @testset "Place on other object" begin
        s = Env1DState(0.0, nothing, SOME_OBJECTS)
        a = Env1DAction(:place, 0.0)
        sp, o, _ = gen(s, a)
        @test isterminal(sp)
    end
end

@testset "Push" begin

    @testset "Push while holding" begin
        s = Env1DState(-10.0, 1, SOME_OBJECTS)
        a = Env1DAction(:push, 0.0)
        sp, o, _ = gen(s, a)
        @test isterminal(sp)
    end


    @testset "Push from top not touching" begin
        s = Env1DState(-10.0, nothing, SOME_OBJECTS)
        a = Env1DAction(:push, 0.0)
        sp, o, _ = gen(s, a)
        @test isterminal(sp)
    end

    @testset "Push from top touching no contact +" begin
        s = Env1DState(0.0, nothing, SOME_OBJECTS)
        a = Env1DAction(:push, 1.5)
        sp, o, _ = gen(s, a)
        @test sp.objects[2].pos == 1.5
        @test sp.objects[1].pos == 5.0
        @test !isterminal(sp)
    end

    @testset "Push from top touching no contact -" begin
        s = Env1DState(0.0, nothing, SOME_OBJECTS)
        a = Env1DAction(:push, -1.5)
        sp, o, _ = gen(s, a)
        @test sp.objects[2].pos == -1.5
        @test sp.objects[1].pos == 5.0
        @test !isterminal(sp)
    end

    @testset "Push from top touching one object abut +" begin
        OBJS = [
            Env1DObject(1.5, 2, RED),
            Env1DObject(0,   1, GREEN)
        ]
        s = Env1DState(0.0, nothing, OBJS)
        a = Env1DAction(:push, 1.5)
        sp, o, _ = gen(s, a)
        @test sp.objects[1].pos == 3.0
        @test sp.objects[2].pos == 1.5
        @test !isterminal(sp)
    end

    @testset "Push from top touching one object abut -" begin
        OBJS = [
            Env1DObject(-1.5, 2, RED),
            Env1DObject(0,   1, GREEN)
        ]
        s = Env1DState(0.0, nothing, OBJS)
        a = Env1DAction(:push, -1.5)
        sp, o, _ = gen(s, a)
        @test sp.objects[1].pos == -3.0
        @test sp.objects[2].pos == -1.5
        @test !isterminal(sp)
    end

    @testset "Push from top touching two objects abut +" begin
        OBJS = [
            Env1DObject(1.5, 2, RED),
            Env1DObject(0,   1, GREEN),
            Env1DObject(3.6, 2, RED),
        ]
        s = Env1DState(0.0, nothing, OBJS)
        a = Env1DAction(:push, 1.5)
        sp, o, _ = gen(s, a)
        @test sp.objects[1].pos == 3.0
        @test sp.objects[2].pos == 1.5
        @test sp.objects[3].pos == 5.0
        @test !isterminal(sp)
    end   

    @testset "Push from top touching two objects abut -" begin
        OBJS = [
            Env1DObject(-1.5, 2, RED),
            Env1DObject(-0.0, 1, GREEN),
            Env1DObject(-3.6, 2, RED),
        ]
        s = Env1DState(0.0, nothing, OBJS)
        a = Env1DAction(:push, -1.5)
        sp, o, _ = gen(s, a)
        @test sp.objects[1].pos == -3.0
        @test sp.objects[2].pos == -1.5
        @test sp.objects[3].pos == -5.0
        @test !isterminal(sp)
    end

    @testset "Push and look not touching" begin
        s = Env1DState(-10.0, nothing, SOME_OBJECTS)
        a = Env1DAction(:push_and_look, 0.0)
        sp, o, _ = gen(s, a)
        @test isterminal(sp)
    end

end

@testset "Push and look" begin
    @testset "Push from top touching one object abut +" begin
        OBJS = [
            Env1DObject(1.5, 2, RED),
            Env1DObject(0,   1, GREEN)
        ]
        s = Env1DState(0.0, nothing, OBJS)
        a = Env1DAction(:push_and_look, 1.5)
        sp, o, _ = gen(s, a)
        @test sp.objects[1].pos == 3.0
        @test sp.objects[2].pos == 1.5
        @test !isterminal(sp)
    end
end