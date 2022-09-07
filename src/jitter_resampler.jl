struct JitterResampler
    n::Int
end

function resample(re::JitterResampler, b::AbstractParticleBelief{S}, rng::AbstractRNG) where {S}
    ps = Array{S}(undef, re.n)
    r = rand(rng)*weight_sum(b)/re.n
    c = weight(b,1)
    i = 1
    U = r
    for m in 1:re.n
        while U > c && i < n_particles(b)
            i += 1
            println(weight(b, i))
            c += weight(b, i)
        end
        U += weight_sum(b)/re.n
        ps[m] = jitter(particles(b)[i])
    end
    return ParticleCollection(ps)
end

function resample(re::JitterResampler, b::ParticleCollection{S}, rng::AbstractRNG) where {S}
    r = rand(rng)*n_particles(b)/re.n
    chunk = n_particles(b)/re.n
    inds = ceil.(Int, chunk*(0:re.n-1).+r)
    ps = particles(b)[inds]
    return ParticleCollection(ps)
end