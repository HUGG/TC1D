# Import needed libraries
import numpy as np
from scipy.interpolate import interp1d


def madtrax_zircon(time_history, temp_history, kinetic_model, out_flag):
    """Calculates a zircon fission-track age from an input time-temperature history.

    This subroutine is based on the subroutine "ftmod.pas" provided by
    Peter van der Beek in December 1995. The algorithm is explained in
    Peter's PhD thesis and is based on the work by Lutz and Omar (1991)
    This adaptation of the program for zircon fission-track annealing was
    written by Peter in August/September 2006 and is based on algorithms
    given by Galbraith & Laslett (1997), Tagami et al. (1998) and
    Rahn et al. (2004)

    References:

    van der Beek, P., 1995.Tectonic evolution of continental rifts, PhD Thesis,
          Faculty of Earth Sicences, Free University, Amsterdam.

    Lutz, T.M. and Omar, G.I., 1991. An inverse method of modeling thermal
          histories from apatite fission-track data. EPSL, 104, 181-195.

    Galbraith, R. F., and G. M. Laslett (1997), Statistical modelling of thermal
          annealing of fission tracks in zircon, Chemical Geology, 140, 123-135.

    Tagami, T., et al. (1998), Revised annealing kinetics of fission tracks in
           zircon and geological implications, in Advances in Fission-Track Geochronology,
           edited by P. Van den haute and F. De Corte, pp. 99-112, Kluwer Academic
           Publishers, Dordrecht, Netherlands.

    Rahn, M. K., et al. (2004), A zero-damage model for fission track annealing in zircon,
           American Mineralogist, 89, 473-484.
    """

    # Set ZFT kinetic model parameters
    if kinetic_model == 1:
        # Alpha-damaged zircon (Tagami et al., 1998)
        a = -10.77
        b = 2.599e-4
        c = 1.026e-2
    elif kinetic_model == 2:
        # Zero-damage zircon (Rahn et al., 2004)
        a = -11.57
        b = 2.755e-4
        c = 1.075e-2
    else:
        raise ValueError("kinetic_model must be 1 or 2.")

    # Set unannealed fission track length
    xind = 10.8

    # Set mean length of spontaneous tracks in standards
    xfct = 10.8

    # Create probability array if used
    if out_flag == 1:
        prob = np.zeros(101)

    # Create fission track length array
    ft_length_dist = np.zeros(17)

    # Make dummy outputs if out_flag is 0
    if out_flag == 0:
        mean_ft_length = 0.0
        std_dev_ft_length = 0.0

    # Set number of time steps assuming
    # - 1 Myr time step length if run time exceeds 100 Myr
    # - 100 steps otherwise
    nstep = int(time_history[0])
    if nstep > 1000:
        print(
            f"WARNING: Zircon fission-track ages cooling over 1 Gyr may be inaccurate."
        )
    time_interval = 1.0
    if nstep < 100:
        nstep = 100
        time_interval = time_history[0] / 100.0
    delta_t = time_interval * 1.0e6 * 365.25 * 24.0 * 3600.0

    # Create array for length reductions
    length_reduction = np.zeros(nstep)

    # Calculate final temperature
    tt_interp = interp1d(time_history, temp_history)
    temp_prev = tt_interp(0.0)
    temp_prev += 273.15
    length_reduction_prev = 0.5

    # Begin time stepping loop
    for i in range(nstep):
        time = float(i) * time_interval

        # Calculate temperature by linear interpolation
        temperature = tt_interp(time)
        temperature += 273.15

        # Calculate mean temperature over the time step
        temp_mean = (temperature + temp_prev) / 2.0

        # Calculate equivalent time
        if i == 0:
            time_eq = 0.0
        else:
            time_eq = np.exp(
                (np.log(1 - length_reduction_prev) - a - (c * temp_mean))
                / (b * temp_mean)
            )

        # Calculate length_reduction over the time step
        dt = time_eq + delta_t
        gr = a + ((b * temp_mean) * np.log(dt)) + (c * temp_mean)
        length_reduction[i] = 1 - np.exp(gr)

        # Stop the calculation if length_reduction <= 0.4
        if length_reduction[i] <= 0.4:
            nstep = i
            break

        # Update variables for next time step
        temp_prev = temperature
        length_reduction_prev = length_reduction[i]

    # All reduction factors have now been calculated
    # Estimate FT age by simple summation
    sumdj = 0.0
    for i in range(nstep):
        if length_reduction[i] <= 0.4:
            dj = 0.0
        elif length_reduction[i] <= 0.66:
            dj = 2.15 * length_reduction[i] - 0.76
        else:
            dj = length_reduction[i]
        sumdj += dj

    age = (xind / xfct) * sumdj * time_interval

    if out_flag == 1:
        # Calculate PDF using Lutz and Omar (1991)
        sumprob = 0.0
        for j in range(101):
            rr = float(j) / 100.0
            if rr <= 0.43:
                h = 2.53
            elif rr <= 0.67:
                h = 5.08 - 5.93 * rr
            else:
                h = 1.39 - 0.61 * rr

            fr = 0.0
            for i in range(nstep):
                x = (rr - length_reduction[i]) * xind / h
                dfr = 0.39894228 * np.exp(-(x**2) / 2.0) / h
                fr += dfr

            prob[j] = fr / nstep
            sumprob += prob[j]

        # Rescale track length dist, its mean, and standard deviation
        ft_length_dist[10] = 100.0
        imin = 0

        for l in range(10):
            imax = int(l * 100.0 / xind)
            ft_length_dist[l] = 0.0
            for i in range(imin, imax + 1):
                ft_length_dist[l] += prob[i]

            ft_length_dist[l] = ft_length_dist[l] * 100.0 / sumprob
            ft_length_dist[10] -= ft_length_dist[l]

            imin = imax + 1

        sum_ft_length_dist = 0.0
        for l in range(10):
            sum_ft_length_dist += ft_length_dist[l] * (float(l) - 0.5)
        mean_ft_length = sum_ft_length_dist / 100.0

        dev_ft_length = 0.0
        for l in range(10):
            dev_ft_length += (
                ft_length_dist[l] * (float(l) - 0.5 - mean_ft_length) ** 2.0
            )
        std_dev_ft_length = np.sqrt(dev_ft_length / 100.0)

    return age, ft_length_dist, mean_ft_length, std_dev_ft_length
