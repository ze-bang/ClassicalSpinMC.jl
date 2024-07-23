function landau_lifshitz_step(mc::MonteCarlo, dt::Float64, T::Float64)
    lat = mc.lattice
    spins = lat.spins
    # Create an array to store the updated spin configuration
    new_spins = similar(spins)
    out = length(mc.outpath) > 0

    # Loop over each spin in the array
    for i in 1:n, j in 1:3
        # Get the current spin value
        spin = spins[i]
        local_field = get_local_field(mc.lattice, point)
        new_spin = spin + dt * cross(spin, local_field) + noise
        new_spin /= norm(new_spin)
        new_spins[i] = new_spin
    end

    # Return the updated spin configuration
    mc.latttice.spins = copy(new_spins)
    if out
        write_MC_checkpoint_t(mc, T)
    end
end

function time_evolve(mc::MonteCarlo, T::Float64, dt::Float64)
    nsteps = int(T / dt)
    for i in 1:nsteps
        landau_lifshitz_step(mc, dt, T)
    end
end
