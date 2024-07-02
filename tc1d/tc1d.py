#!/usr/bin/env python3

import csv
import os
import subprocess

# Import libaries we need
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.linalg import solve
from sklearn.model_selection import ParameterGrid
from neighpy import NASearcher, NAAppraiser

# Import user functions
from madtrax.madtrax_apatite import madtrax_apatite
from madtrax.madtrax_zircon import madtrax_zircon


# Exceptions
class UnstableSolutionException(Exception):
    pass


class MissingOption(Exception):
    pass


# Unit conversions
def yr2sec(time):
    """Converts time from years to seconds."""
    return time * 60.0 * 60.0 * 24.0 * 365.25


def myr2sec(time):
    """Converts time from million years to seconds."""
    return yr2sec(time) * 1.0e6


def kilo2base(value):
    """Converts value from kilo-units to the base unit."""
    return value * 1000.0


def milli2base(value):
    """Converts value from milli-units to the base unit."""
    return value / 1.0e3


def micro2base(value):
    """Converts value from micro-units to the base unit."""
    return value / 1.0e6


def mmyr2ms(rate):
    """Converts rate from mm/yr to m/s."""
    return milli2base(rate) / yr2sec(1)


def echo_model_info(
    dx,
    nt,
    dt,
    t_total,
    implicit,
    ero_type,
    exhumation_magnitude,
    cond_stab,
    adv_stab,
    cond_crit,
    adv_crit,
):
    print("")
    print("--- General model information ---")
    print("")
    print(f"- Node spacing: {dx} m")
    print(f"- Total simulation time: {t_total / myr2sec(1):.1f} million years")
    print(f"- Time steps: {nt} @ {dt / yr2sec(1):.1f} years each")

    if implicit:
        print("- Solution type: Implicit")
    else:
        print("- Solution type: Explicit")

    # Check stability conditions
    if not implicit:
        print(
            f"- Conductive stability: {(cond_stab < cond_crit)} ({cond_stab:.3f} < {cond_crit:.4f})"
        )
        print(
            f"- Advective stability: {(adv_stab < adv_crit)} ({adv_stab:.3f} < {adv_crit:.4f})"
        )

    # Output erosion model
    ero_models = {
        1: "Constant",
        2: "Step-function",
        3: "Exponential decay",
        4: "Thrust sheet emplacement/erosion",
        5: "Tectonic exhumation and erosion",
        6: "Linear rate change",
    }
    print(f"- Erosion model: {ero_models[ero_type]}")
    print(f"- Total erosional exhumation: {exhumation_magnitude:.1f} km")


# Explicit solution stability criteria calculation
def calculate_explicit_stability(
    vx,
    k_crust,
    rho_crust,
    cp_crust,
    k_mantle,
    rho_mantle,
    cp_mantle,
    k_a,
    dt,
    dx,
    cond_crit,
    adv_crit,
):
    # Check stability conditions
    kappa_crust = k_crust / (rho_crust * cp_crust)
    kappa_mantle = k_mantle / (rho_mantle * cp_mantle)
    kappa_a = k_a / (rho_mantle * cp_mantle)
    kappa = max(kappa_crust, kappa_mantle, kappa_a)
    cond_stab = kappa * dt / dx**2
    if cond_stab >= cond_crit:
        raise UnstableSolutionException(
            f"Heat conduction solution unstable: {cond_stab:.3f} > {cond_crit:.4f}. Decrease nx or dt."
        )
    adv_stab = vx * dt / dx
    if adv_stab >= adv_crit:
        raise UnstableSolutionException(
            f"Heat advection solution unstable: {adv_stab:.3f} > {adv_crit:.4f}. Decrease nx, dt, or vx."
        )

    return cond_stab, adv_stab


# Mantle adiabat from Turcotte and Schubert (eqn 4.254)
def adiabat(alphav, temp, cp):
    """Calculates a mantle adiabat in degress / m."""
    grav = 9.81
    return alphav * grav * temp / cp


# Conductive steady-state heat transfer
def temp_ss_implicit(nx, dx, temp_surf, temp_base, vx, rho, cp, k, heat_prod):
    """Calculates a steady-state thermal solution."""
    # Create the empty (zero) coefficient and right hand side arrays
    a_matrix = np.zeros((nx, nx))  # 2-dimensional array, ny rows, ny columns
    b = np.zeros(nx)

    # Set B.C. values in the coefficient array and in the r.h.s. array
    a_matrix[0, 0] = 1
    b[0] = temp_surf
    a_matrix[nx - 1, nx - 1] = 1
    b[nx - 1] = temp_base

    # Matrix loop
    for ix in range(1, nx - 1):
        a_matrix[ix, ix - 1] = (-(rho[ix] * cp[ix] * -vx[ix]) / (2 * dx)) - k[
            ix - 1
        ] / dx**2
        a_matrix[ix, ix] = k[ix] / dx**2 + k[ix - 1] / dx**2
        a_matrix[ix, ix + 1] = (rho[ix] * cp[ix] * -vx[ix]) / (2 * dx) - k[ix] / dx**2
        b[ix] = heat_prod[ix]

    temp = solve(a_matrix, b)
    return temp


def update_materials(
    x,
    xstag,
    moho_depth,
    rho_crust,
    rho_mantle,
    rho,
    cp_crust,
    cp_mantle,
    cp,
    k_crust,
    k_mantle,
    k,
    heat_prod_crust,
    heat_prod_mantle,
    heat_prod,
    temp_adiabat,
    temp_prev,
    k_a,
    removal_fraction,
):
    """Updates arrays of material properties."""
    rho[:] = rho_crust
    rho[x > moho_depth] = rho_mantle
    cp[:] = cp_crust
    cp[x > moho_depth] = cp_mantle
    k[:] = k_crust
    k[xstag > moho_depth] = k_mantle

    interp_temp_prev = interp1d(x, temp_prev)
    temp_stag = interp_temp_prev(xstag)
    k[temp_stag >= temp_adiabat] = k_a
    if removal_fraction > 0.0:
        # Handle cases where delamination has not yet occurred, so no temps exceed adiabat

        # DAVE: THIS DOES NOT WORK PROPERLY!!!

        # if (temp_stag - temp_adiabat).min() <= 0.0:
        #    lab_depth = x.max()
        #    print("No delamination yet!")
        # else:
        lab_depth = xstag[temp_stag >= temp_adiabat].min()
    else:
        lab_depth = x.max()

    heat_prod[:] = heat_prod_crust
    heat_prod[x > moho_depth] = heat_prod_mantle
    return rho, cp, k, heat_prod, lab_depth


def init_ero_types(params, x, xstag, temp_prev, moho_depth):
    """Defines temperatures and material properties for ero_types 4 and 5."""

    # Find index where depth reaches or exceeds thrust sheet thickness
    ref_index = np.min(np.where(x >= kilo2base(params["ero_option1"])))

    # Make copy of initial temperatures
    initial_temps = temp_prev.copy()

    # Adjust temperatures depending on erosion model type
    if params["ero_type"] == 4:
        # Reassign temperatures
        for ix in range(params["nx"]):
            if ix >= ref_index:
                temp_prev[ix] = initial_temps[ix - ref_index]
        moho_depth += kilo2base(params["ero_option1"])

    elif params["ero_type"] == 5:
        # Reassign temperatures
        for ix in range(1, params["nx"]):
            if ix < (params["nx"] - ref_index):
                temp_prev[ix] = initial_temps[ix + ref_index]
            else:
                temp_prev[ix] = temp_prev[-1]
        moho_depth -= kilo2base(params["ero_option1"])

    # Modify material property arrays
    rho = np.ones(len(x)) * params["rho_crust"]
    rho[x > moho_depth] = params["rho_mantle"]
    cp = np.ones(len(x)) * params["cp_crust"]
    cp[x > moho_depth] = params["cp_mantle"]
    k = np.ones(len(xstag)) * params["k_crust"]
    k[xstag > moho_depth] = params["k_mantle"]
    heat_prod = np.ones(len(x)) * micro2base(params["heat_prod_crust"])
    heat_prod[x > moho_depth] = micro2base(params["heat_prod_mantle"])
    alphav = np.ones(len(x)) * params["alphav_crust"]
    alphav[x > moho_depth] = params["alphav_mantle"]

    return temp_prev, moho_depth, rho, cp, k, heat_prod, alphav


def temp_transient_explicit(
    temp_prev, temp_new, temp_surf, temp_base, nx, dx, vx, dt, rho, cp, k, heat_prod
):
    """Updates a transient thermal solution."""
    # Set boundary conditions
    temp_new[0] = temp_surf
    temp_new[nx - 1] = temp_base

    # Calculate internal grid point temperatures
    # Use upwinding
    if vx[0] > 0:
        for ix in range(1, nx - 1):
            temp_new[ix] = (
                (1 / (rho[ix] * cp[ix]))
                * (
                    k[ix] * (temp_prev[ix + 1] - temp_prev[ix])
                    - k[ix - 1] * (temp_prev[ix] - temp_prev[ix - 1])
                )
                / dx**2
                + heat_prod[ix] / (rho[ix] * cp[ix])
                + vx[ix] * (temp_prev[ix] - temp_prev[ix - 1]) / (dx)
            ) * dt + temp_prev[ix]
    else:
        for ix in range(1, nx - 1):
            temp_new[ix] = (
                (1 / (rho[ix] * cp[ix]))
                * (
                    k[ix] * (temp_prev[ix + 1] - temp_prev[ix])
                    - k[ix - 1] * (temp_prev[ix] - temp_prev[ix - 1])
                )
                / dx**2
                + heat_prod[ix] / (rho[ix] * cp[ix])
                + vx[ix] * (temp_prev[ix + 1] - temp_prev[ix]) / (dx)
            ) * dt + temp_prev[ix]

    return temp_new


# Conductive steady-state heat transfer
def temp_transient_implicit(
    nx, dx, dt, temp_prev, temp_surf, temp_base, vx, rho, cp, k, heat_prod
):
    """Calculates a steady-state thermal solution."""
    # Create the empty (zero) coefficient and right hand side arrays
    a_matrix = np.zeros((nx, nx))  # 2-dimensional array, ny rows, ny columns
    b = np.zeros(nx)

    # Set B.C. values in the coefficient array and in the r.h.s. array
    a_matrix[0, 0] = 1
    b[0] = temp_surf
    a_matrix[nx - 1, nx - 1] = 1
    b[nx - 1] = temp_base

    # Matrix loop
    for ix in range(1, nx - 1):
        a_matrix[ix, ix - 1] = (
            -(rho[ix] * cp[ix] * -vx[ix]) / (2 * dx) - k[ix - 1] / dx**2
        )
        a_matrix[ix, ix] = (
            (rho[ix] * cp[ix]) / dt + k[ix] / dx**2 + k[ix - 1] / dx**2
        )
        a_matrix[ix, ix + 1] = (rho[ix] * cp[ix] * -vx[ix]) / (2 * dx) - k[ix] / dx**2
        b[ix] = heat_prod[ix] + ((rho[ix] * cp[ix]) / dt) * temp_prev[ix]

    temp = solve(a_matrix, b)
    return temp


def he_ages(
    file,
    ap_rad=45.0,
    ap_uranium=10.0,
    ap_thorium=40.0,
    zr_rad=60.0,
    zr_uranium=100.0,
    zr_thorium=40.0,
):
    """Calculates (U-Th)/He ages."""

    # Define filepath to find executable
    fp = os.path.dirname(os.path.realpath(__file__))

    # Run executable to calculate age
    command = (
        os.path.join(fp, "..", "bin", "RDAAM_He")
        + " "
        + file
        + " "
        + str(ap_rad)
        + " "
        + str(ap_uranium)
        + " "
        + str(ap_thorium)
        + " "
        + str(zr_rad)
        + " "
        + str(zr_uranium)
        + " "
        + str(zr_thorium)
    )
    p = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )

    stdout = p.stdout.readlines()

    ahe_age = stdout[0].split()[3][:-1].decode("UTF-8")
    corr_ahe_age = stdout[0].split()[7].decode("UTF-8")
    zhe_age = stdout[1].split()[3][:-1].decode("UTF-8")
    corr_zhe_age = stdout[1].split()[7].decode("UTF-8")

    retval = p.wait()
    return ahe_age, corr_ahe_age, zhe_age, corr_zhe_age


def ft_ages(file):
    """Calculates AFT ages."""

    # Define filepath to find executable
    fp = os.path.dirname(os.path.realpath(__file__))

    command = os.path.join(fp, "..", "bin", "ketch_aft") + " " + file
    p = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )

    stdout = p.stdout.readlines()
    aft_age = stdout[0].split()[4][:-1].decode("UTF-8")
    mean_ft_length = stdout[0].split()[9][:-1].decode("UTF-8")

    retval = p.wait()
    return aft_age, mean_ft_length


def calculate_ages_and_tcs(
    params, time_history, temp_history, depth_history, pressure_history
):
    """Calculates thermochronometer ages and closure temperatures"""
    if params["debug"]:
        print("")
        print(
            f"Calculating ages and closure temperatures for {len(time_history)} thermal history points."
        )
        print(f"- Max time: {time_history.max() / myr2sec(1)} Ma")
        print(f"- Max temperature: {temp_history.max()} °C")
        print(f"- Max depth: {depth_history.max() / kilo2base(1)} km")
        print(f"- Max pressure: {pressure_history.max() * micro2base(1)} MPa")

    # Convert time since model start to time before end of simulation
    current_max_time = time_history.max()
    time_ma = current_max_time - time_history
    time_ma = time_ma / myr2sec(1)

    # Calculate AFT age using MadTrax
    if params["madtrax_aft"]:
        aft_age, _, _, _ = madtrax_apatite(
            time_ma, temp_history, len(time_ma), 1, params["madtrax_aft_kinetic_model"]
        )

    # Calculate ZFT age using MadTrax
    zft_age, _, _, _ = madtrax_zircon(
        time_ma, temp_history, params["madtrax_zft_kinetic_model"], 1
    )

    # Write time-temperature history to file for (U-Th)/He age prediction
    with open("time_temp_hist.csv", "w") as csvfile:
        writer = csv.writer(csvfile, delimiter=",", lineterminator="\n")
        # Write time-temperature history in reverse order!
        if len(time_ma) > 1000.0:
            write_increment = int(round(len(time_ma) / 100, 0))
        elif len(time_ma) > 100.0:
            write_increment = int(round(len(time_ma) / 10, 0))
        else:
            write_increment = 2
        for i in range(-1, -(len(time_ma) + 1), -write_increment):
            writer.writerow([time_ma[i], temp_history[i]])

        # Write fake times if time history padding is enabled
        if params["pad_thist"]:
            if params["pad_time"] > 0.0:
                # Make array of pad times with 1.0 Myr time increments
                pad_times = np.arange(
                    current_max_time / myr2sec(1),
                    current_max_time / myr2sec(1) + params["pad_time"] + 0.1,
                    1.0,
                )
                for pad_time in pad_times:
                    writer.writerow([pad_time, temp_history[i]])

    # Write pressure-time-temperature-depth history to file for reference
    with open("time_temp_depth_pressure_hist.csv", "w") as csvfile:
        writer = csv.writer(csvfile, delimiter=",", lineterminator="\n")
        # Write header
        writer.writerow(["Time (Ma)", "Temperature (C)", "Depth (m)", "Pressure (MPa)"])
        # Write time-temperature history in reverse order!
        for i in range(-1, -(len(time_ma) + 1), -write_increment):
            writer.writerow(
                [
                    time_ma[i],
                    temp_history[i],
                    depth_history[i],
                    pressure_history[i] * micro2base(1),
                ]
            )

    ahe_age, corr_ahe_age, zhe_age, corr_zhe_age = he_ages(
        file="time_temp_hist.csv",
        ap_rad=params["ap_rad"],
        ap_uranium=params["ap_uranium"],
        ap_thorium=params["ap_thorium"],
        zr_rad=params["zr_rad"],
        zr_uranium=params["zr_uranium"],
        zr_thorium=params["zr_thorium"],
    )
    if params["ketch_aft"]:
        aft_age, aft_mean_ftl = ft_ages("time_temp_hist.csv")

    # Find effective closure temperatures
    ahe_temp = np.interp(float(corr_ahe_age), np.flip(time_ma), np.flip(temp_history))
    aft_temp = np.interp(float(aft_age), np.flip(time_ma), np.flip(temp_history))
    zhe_temp = np.interp(float(corr_zhe_age), np.flip(time_ma), np.flip(temp_history))
    zft_temp = np.interp(float(zft_age), np.flip(time_ma), np.flip(temp_history))

    return (
        corr_ahe_age,
        ahe_age,
        ahe_temp,
        aft_age,
        aft_mean_ftl,
        aft_temp,
        corr_zhe_age,
        zhe_age,
        zhe_temp,
        zft_age,
        zft_temp,
    )


def calculate_erosion_rate(
    t_total,
    current_time,
    ero_type,
    ero_option1,
    ero_option2,
    ero_option3,
    ero_option4,
    ero_option5,
):
    """Defines the way in which erosion should be applied."""

    # Split the code below into separate functions?
    # Could have tests integrated more easily that way.

    # Constant erosion rate
    # Convert to inputting rate directly?
    if ero_type == 1:
        vx = kilo2base(ero_option1) / t_total
        vx_max = vx

    # Constant erosion rate with a step-function change at a specified time
    # Convert to inputting rates directly?
    elif ero_type == 2:
        interval1 = myr2sec(ero_option2)
        interval2 = myr2sec(ero_option4 - ero_option2)
        interval3 = t_total - myr2sec(ero_option4)
        rate1 = kilo2base(ero_option1) / interval1
        rate2 = kilo2base(ero_option3) / interval2
        rate3 = kilo2base(ero_option5) / interval3
        # First stage of erosion
        if current_time < myr2sec(ero_option2):
            vx = rate1
        # Second stage of erosion
        elif current_time < myr2sec(ero_option4):
            vx = rate2
        # Third stage of erosion
        else:
            vx = rate3
        vx_max = max(rate1, rate2, rate3)

    # Exponential erosion rate decay with a set characteristic time
    # Convert to inputting rate directly?
    elif ero_type == 3:
        erosion_magnitude = kilo2base(ero_option1)
        decay_time = myr2sec(ero_option2)
        max_rate = erosion_magnitude / (
            decay_time * (np.exp(0.0 / decay_time) - np.exp(-t_total / decay_time))
        )
        vx = max_rate * np.exp(-current_time / decay_time)
        vx_max = max_rate

    # Emplacement and erosional removal of a thrust sheet
    elif ero_type == 4:
        # Calculate erosion magnitude
        erosion_magnitude = kilo2base(ero_option1 + ero_option2)
        ero_start = myr2sec(ero_option4)
        if current_time < ero_start:
            vx = 0.0
        else:
            vx = erosion_magnitude / (t_total - ero_start)
        vx_max = erosion_magnitude / (t_total - ero_start)

    # Emplacement and erosional removal of a thrust sheet
    elif ero_type == 5:
        # Calculate erosion magnitude
        erosion_magnitude = kilo2base(ero_option2)
        vx = erosion_magnitude / t_total
        vx_max = vx

    # Linear increase in erosion rate from a starting rate/time until an ending time
    # TODO: Make this work for negative erosion rate initial phase
    elif ero_type == 6:
        init_rate = mmyr2ms(ero_option1)
        rate_change_start = myr2sec(ero_option2)
        final_rate = mmyr2ms(ero_option3)
        if abs(ero_option4) <= 1.0e-8:
            rate_change_end = t_total
        else:
            rate_change_end = myr2sec(ero_option4)
        if current_time < rate_change_start:
            vx = init_rate
        elif current_time < rate_change_end:
            vx = init_rate + (current_time - rate_change_start) / (
                rate_change_end - rate_change_start
            ) * (final_rate - init_rate)
        else:
            vx = final_rate
        vx_max = max(init_rate, final_rate)

    # Catch bad cases
    else:
        raise MissingOption("Bad erosion type. Type should be between 1 and 6.")

    return vx, vx_max


def calculate_exhumation_magnitude(
    ero_type, ero_option1, ero_option2, ero_option3, ero_option4, ero_option5, t_total
):
    """Calculates erosion magnitude in kilometers."""

    # Constant erosion rate
    if ero_type == 1:
        magnitude = ero_option1

    elif ero_type == 2:
        magnitude = ero_option1 + ero_option3 + ero_option5

    elif ero_type == 3:
        magnitude = ero_option1

    elif ero_type == 4:
        magnitude = ero_option1 + ero_option2

    elif ero_type == 5:
        magnitude = ero_option1

    elif ero_type == 6:
        magnitude = myr2sec(ero_option2) * mmyr2ms(ero_option1)
        # Handle case that ero_option4 is not specified (i.e., linear increase ends at end of simulation)
        if abs(ero_option4) <= 1.0e-8:
            rate_change_end = t_total
        else:
            rate_change_end = myr2sec(ero_option4)
        magnitude += (rate_change_end - myr2sec(ero_option2)) * (
            0.5 * (mmyr2ms(ero_option3) - mmyr2ms(ero_option1)) + mmyr2ms(ero_option1)
        )
        magnitude += (t_total - rate_change_end) * mmyr2ms(ero_option3)
        magnitude /= 1000.0

    else:
        raise MissingOption("Bad erosion type. Type should be between 1 and 6.")

    return magnitude


def calculate_pressure(density, dx, g=9.81):
    """Calculates lithostatic pressure"""
    pressure = np.zeros(len(density))
    for i in range(1, len(density)):
        pressure[i] = pressure[i - 1] + density[i] * g * dx

    return pressure


def calculate_crust_solidus(composition, crustal_pressure):
    """Reads in data from MELTS for different compositions and returns a solidus"""

    # Composition options
    compositions = {
        "wet_felsic": "wetFelsic4.csv",
        "wet_intermediate": "wetImed3.csv",
        "wet_basalt": "wetBasalt2.csv",
        "dry_felsic": "dryFelsic.csv",
        "dry_basalt": "dryBasalt.csv",
    }

    # Read composition data file
    fp = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "melts_data",
        compositions[composition],
    )
    comp_data = np.genfromtxt(fp, delimiter=",")

    # Create interpolation function for composition
    crust_interp = RectBivariateSpline(
        comp_data[2:, 0], comp_data[0, 1:], comp_data[2:, 1:], kx=1, ky=1
    )

    # Creating Pressure vs Melts fraction grid which gives the values for temperatures
    Tn = np.linspace(0, 1, 121)  # Last number defines the number of melt fraction steps
    Tn[-1] = 0.99999999  # Highest temperature value
    interp_list = []  # lists of melt fractions at interpolated pressure ranges

    for i in range(len(comp_data[0, 1:])):
        interp_list.append(
            np.interp(Tn, comp_data[2:, i + 1], comp_data[2:, 0])
        )  # x(melt), y(T)

    # Creating 2D array of Pressure (x axis) vs melt fraction (y axis)
    interp_list = np.transpose(np.array(interp_list))

    # Interpolating the melt fraction vs pressure data (gives temperature)
    interp_temp = RectBivariateSpline(Tn, comp_data[0, 1:], interp_list, kx=1, ky=1)

    # Melting curve plots for the materials
    # melt_fractions = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    melt_fractions = np.array([0.05])

    for i in range(len(melt_fractions)):
        crust_solidus = interp_temp(melt_fractions[i], crustal_pressure)

    return crust_solidus[0, :]


def calculate_mantle_solidus(pressure, xoh=0.0):
    """Calculates the solidus for the mantle.
    Uses the equation from Hirschmann, 2000 as modified by Sarafian et al. (2017)."""

    # Convert ppm to parts
    xoh *= 1.0e-6

    # Hirschmann (2000) constants
    a = -5.104
    b = 132.899
    c = 1120.661

    # Hirschmann (2000) solidus
    solidus = a * pressure**2 + b * pressure + c

    # Sarafian modifications
    gas_constant = 8.314
    silicate_mols = 59.0
    entropy = 0.4
    solidus = solidus / (
        1 - (gas_constant / (silicate_mols * entropy)) * np.log(1 - xoh)
    )

    return solidus


def calculate_misfit(
    predicted_ages, measured_ages, measured_stdev, misfit_type=1, num_params=0
):
    """
    Calculates misfit value between measured and predicted thermochronometer ages

    type 1 = Braun et al. (2012) equation 8 (Default)

    type 2 = Braun et al. (2012) equation 9

    type 3 = Braun et al. (2012) equation 10

    Braun, J., Van Der Beek, P., Valla, P., Robert, X., Herman, F., Glotzbach, C., Pedersen, V., Perry, C.,
    Simon-Labric, T. and Prigent, C., 2012. Quantifying rates of landscape evolution and tectonic processes
    by thermochronology and numerical modeling of crustal heat transport using PECUBE. Tectonophysics, 524,
    pp.1-28.

    Parameters
    ----------
    predicted_ages : numpy array
        Array of predicted thermochronometer ages.
    measured_ages : numpy arrray
        Array of measured thermochronometer ages.
    measured_stdev : numpy arrray
        Array of standard deviations for the measured thermochronometer ages.
    misfit_type : int, default=1
        Misfit type to calculate. See equations 8-10 of Braun et al. (2012).
    num_params : int, default=0
        Number of model parameters if using misfit type 2.

    Returns
    -------
    misfit : float
        Calculated misfit value.
    """
    # Caclulate general misfit, modify for types 1 and 2 as needed
    misfit = ((predicted_ages - measured_ages) ** 2 / measured_stdev**2).sum()

    if misfit_type == 1:
        misfit = np.sqrt(misfit) / len(predicted_ages)

    if misfit_type == 2:
        misfit = misfit / (len(predicted_ages) - num_params - 1)

    return misfit


def init_params(
    echo_inputs=False,
    echo_info=True,
    echo_thermal_info=True,
    echo_ages=True,
    debug=False,
    length=125.0,
    nx=251,
    time=50.0,
    dt=5000.0,
    init_moho_depth=50.0,
    crustal_uplift=False,
    fixed_moho=False,
    removal_fraction=0.0,
    removal_time=0.0,
    rho_crust=2850.0,
    cp_crust=800.0,
    k_crust=2.75,
    heat_prod_crust=0.5,
    alphav_crust=3.0e-5,
    rho_mantle=3250.0,
    cp_mantle=1000.0,
    k_mantle=2.5,
    heat_prod_mantle=0.0,
    alphav_mantle=3.0e-5,
    rho_a=3250.0,
    k_a=20.0,
    implicit=True,
    temp_surf=0.0,
    temp_base=1300.0,
    mantle_adiabat=True,
    vx_init=0.0,
    ero_type=1,
    ero_option1=0.0,
    ero_option2=0.0,
    ero_option3=0.0,
    ero_option4=0.0,
    ero_option5=0.0,
    calc_ages=True,
    ketch_aft=True,
    madtrax_aft=False,
    madtrax_aft_kinetic_model=1,
    madtrax_zft_kinetic_model=1,
    ap_rad=45.0,
    ap_uranium=10.0,
    ap_thorium=40.0,
    zr_rad=60.0,
    zr_uranium=100.0,
    zr_thorium=40.0,
    pad_thist=False,
    pad_time=0.0,
    past_age_increment=0.0,
    obs_ahe=[],
    obs_ahe_stdev=[],
    obs_aft=[],
    obs_aft_stdev=[],
    obs_zhe=[],
    obs_zhe_stdev=[],
    obs_zft=[],
    obs_zft_stdev=[],
    misfit_num_params=0,
    misfit_type=1,
    plot_results=True,
    display_plots=True,
    t_plots=[0.1, 1, 5, 10, 20, 30, 50],
    crust_solidus=False,
    crust_solidus_comp="wet_intermediate",
    mantle_solidus=False,
    mantle_solidus_xoh=0.0,
    solidus_ranges=False,
    log_output=False,
    log_file="",
    model_id="",
    write_temps=False,
    write_past_ages=False,
    save_plots=False,
    read_temps=False,
    compare_temps=False,
):
    """
    Define the model parameters.

    Parameters
    ----------
    echo_inputs : bool, default=False
        Print input values to the screen.
    echo_info : bool, default=True
        Print basic model info to the screen.
    echo_thermal_info : bool, default=True
        Print thermal model info to the screen.
    echo_ages : bool, default=True
        Print calculated thermochronometer age(s) to the screen.
    debug : bool, default=False
        Enable debug output.
    length : float or int, default=125.0
        Model depth extent in km.
    nx : int, default=251
        Number of grid points for temperature calculation.
    time : float or int, default=50.0
        Total simulation time in Myr.
    dt : float or int, default=5000.0
        Time step in years.
    init_moho_depth : float or int, default=50.0
        Initial depth of Moho in km.
    crustal_uplift : bool, default=False
        Uplift only the crust in the thermal model.
    fixed_moho : bool, default=False
        Prevent changes in Moho depth (e.g., due to erosion).
    removal_fraction : float or int, default=0.0
        Fraction of lithospheric mantle to remove due to delamination. 0 = none, 1 = all.
    removal_time : float or int, default=0.0
        Timing of removal of lithospheric mantle in Myr from start of simulation.
    rho_crust : float or int, default=2850.0
        Crustal density in kg/m^3.
    cp_crust : float or int, default=800.0
        Crustal heat capacity in J/kg/K.
    k_crust : float or int, default=2.75
        Crustal thermal conductivity in W/m/K.
    heat_prod_crust : float or int, default=0.5
        Crustal heat production in uW/m^3.
    alphav_crust : float or int, default=3.0e-5
        Crustal coefficient of thermal expansion in 1/K.
    rho_mantle : float or int, default=3250.0
        Mantle lithosphere density in kg/m^3.
    cp_mantle : float or int, default=1000.0
        Mantle lithosphere heat capacity in J/kg/K.
    k_mantle : float or int, default=2.5
        Mantle lithosphere thermal conductivity in W/m/K.
    heat_prod_mantle : float or int, default=0.0
        Mantle lithosphere heat production in uW/m^3.
    alphav_mantle : float or int, default=3.0e-5
        Mantle lithosphere coefficient of thermal expansion in 1/K.
    rho_a : float or int, default=3250.0
        Mantle asthenosphere density in kg/m^3.
    k_a : float or int, default=20.0
        Mantle asthenosphere thermal conductivity in W/m/K.
    implicit : bool, default=True
        Use implicit instead of explicit finite-difference calculation.
    temp_surf : float or int, default=0.0
        Surface boundary condition temperature in °C.
    temp_base : float or int, default=1300.0
        Basal boundary condition temperature in °C.
    mantle_adiabat : bool, default=True
        Use adiabat for asthenosphere temperature.
    vx_init : float or int, default=0.0
        Initial steady-state advection velocity in mm/yr.
    ero_type : int, default=1
        Type of erosion model (1, 2, 3, 4, 5, 6 - see https://tc1d.readthedocs.io/en/latest/erosion-models.html).
    ero_option1 : float or int, default=0.0
        Erosion model option 1 (see https://tc1d.readthedocs.io/en/latest/erosion-models.html).
    ero_option2 : float or int, default=0.0
        Erosion model option 2 (see https://tc1d.readthedocs.io/en/latest/erosion-models.html).
    ero_option3 : float or int, default=0.0
        Erosion model option 3 (see https://tc1d.readthedocs.io/en/latest/erosion-models.html).
    ero_option4 : float or int, default=0.0
        Erosion model option 4 (see https://tc1d.readthedocs.io/en/latest/erosion-models.html).
    ero_option5 : float or int, default=0.0
        Erosion model option 5 (see https://tc1d.readthedocs.io/en/latest/erosion-models.html).
    calc_ages : bool, default=True
        Enable calculation of thermochronometer ages.
    ketch_aft : bool, default=True
        Use the Ketcham et al. (2007) model for predicting FT ages.
    madtrax_aft : bool, default=False
        Use the MadTrax algorithm for predicting apatite FT ages.
    madtrax_aft_kinetic_model : int, default=1
        Kinetic model to use for AFT age prediction with MadTrax (see https://tc1d.readthedocs.io).
    madtrax_zft_kinetic_model : int, default=1
        Kinetic model to use for ZFT age prediction with MadTrax (see https://tc1d.readthedocs.io).
    ap_rad : float or int, default=45.0
        Apatite grain radius in um.
    ap_uranium : float or int, default=10.0
        Apatite U concentration in ppm.
    ap_thorium : float or int, default=40.0
        Apatite Th concentration radius in ppm.
    zr_rad : float or int, default=60.0
        Zircon grain radius in um.
    zr_uranium : float or int, default=100.0
        Zircon U concentration in ppm.
    zr_thorium : float or int, default=40.0
        Zircon Th concentration radius in ppm.
    pad_thist : bool, default=False
        Add time at the starting temperature in t-T history.
    pad_time : float or int, default=0.0
        Additional time at starting temperature in t-T history in Myr.
    past_age_increment : float or int, default=0.0
        Time increment in past (in Myr) at which ages should be calculated. Works only if greater than 0.0.
    obs_ahe : list of float or int, default=[]
        Measured apatite (U-Th)/He age(s) in Ma.
    obs_ahe_stdev : list of float or int, default=[]
        Measured apatite (U-Th)/He age standard deviation(s) in Ma.
    obs_aft : list of float or int, default=[]
        Measured apatite fission-track age(s) in Ma.
    obs_aft_stdev : list of float or int, default=[]
        Measured apatite fission-track age standard deviation(s) in Ma.
    obs_zhe : list of float or int, default=[]
        Measured zircon (U-Th)/He age(s) in Ma.
    obs_zhe_stdev : list of float or int, default=[]
        Measured zircon (U-Th)/He age standard deviation(s) in Ma.
    obs_zft : list of float or int, default=[]
        Measured zircon fission-track age(s) in Ma.
    obs_zft_stdev : list of float or int, default=[]
        Measured zircon fission-track age standard deviation(s) in Ma.
    misfit_num_params : int, default=0
        Number of model parameters to use in misfit calculation. Only applies to misfit type 2.
    misfit_type : int, default=1
        Misfit type for misfit calculation.
    plot_results : bool, default=True
        Plot calculated results.
    display_plots : bool, default=True
        Display plots on screen.
    t_plots : list of float or int, default=[0.1, 1, 5, 10, 20, 30, 50]
        Output times for temperature plotting in Myr. Treated as increment if only one value given.
    crust_solidus : bool, default=False
        Calculate and plot a crustal solidus.
    crust_solidus_comp : str, default="wet_intermediate"
        Crustal composition for solidus.
    mantle_solidus : bool, default=False
        Calculate and plot a mantle solidus.
    mantle_solidus_xoh : float or int, default=0.0
        Water content for mantle solidus calculation in ppm.
    solidus_ranges : bool, default=False
        Plot ranges for the crustal and mantle solidii.
    log_output : bool, default=False
        Write model summary info to a csv file.
    log_file : str, default=""
        CSV filename for log output.
    model_id : str, default=""
        Model identification character string.
    write_temps : bool, default=False
        Save model temperatures to a file.
    write_past_ages : bool, default=False
        Write out incremental past ages to csv file.
    save_plots : bool, default=False
        Save plots to a file.
    read_temps : bool, default=False
        Read temperatures from a file.
    compare_temps : bool, default=False
        Compare model temperatures to those from a file.
    inverse_mode : bool, default=False
        Activate inverse mode.

    Returns
    -------
    params : dict
        Dictionary of model parameter values.
    """
    params = {
        "cmd_line_call": False,
        "echo_inputs": echo_inputs,
        "echo_info": echo_info,
        "echo_thermal_info": echo_thermal_info,
        "calc_ages": calc_ages,
        "echo_ages": echo_ages,
        "plot_results": plot_results,
        "save_plots": save_plots,
        "display_plots": display_plots,
        # Batch mode not supported when called as a function
        "batch_mode": False,
        "mantle_adiabat": mantle_adiabat,
        "implicit": implicit,
        "read_temps": read_temps,
        "compare_temps": compare_temps,
        "write_temps": write_temps,
        "debug": debug,
        "madtrax_aft": madtrax_aft,
        "madtrax_aft_kinetic_model": madtrax_aft_kinetic_model,
        "madtrax_zft_kinetic_model": madtrax_zft_kinetic_model,
        "ketch_aft": ketch_aft,
        "t_plots": t_plots,
        "max_depth": length,
        "nx": nx,
        "init_moho_depth": init_moho_depth,
        "removal_fraction": removal_fraction,
        "removal_time": removal_time,
        "crustal_uplift": crustal_uplift,
        "fixed_moho": fixed_moho,
        "ero_type": ero_type,
        "ero_option1": ero_option1,
        "ero_option2": ero_option2,
        "ero_option3": ero_option3,
        "ero_option4": ero_option4,
        "ero_option5": ero_option5,
        "temp_surf": temp_surf,
        "temp_base": temp_base,
        "t_total": time,
        "dt": dt,
        "vx_init": vx_init,
        "rho_crust": rho_crust,
        "cp_crust": cp_crust,
        "k_crust": k_crust,
        "heat_prod_crust": heat_prod_crust,
        "alphav_crust": alphav_crust,
        "rho_mantle": rho_mantle,
        "cp_mantle": cp_mantle,
        "k_mantle": k_mantle,
        "heat_prod_mantle": heat_prod_mantle,
        "alphav_mantle": alphav_mantle,
        "rho_a": rho_a,
        "k_a": k_a,
        "ap_rad": ap_rad,
        "ap_uranium": ap_uranium,
        "ap_thorium": ap_thorium,
        "zr_rad": zr_rad,
        "zr_uranium": zr_uranium,
        "zr_thorium": zr_thorium,
        "pad_thist": pad_thist,
        "pad_time": pad_time,
        "past_age_increment": past_age_increment,
        "write_past_ages": write_past_ages,
        "crust_solidus": crust_solidus,
        "crust_solidus_comp": crust_solidus_comp,
        "mantle_solidus": mantle_solidus,
        "mantle_solidus_xoh": mantle_solidus_xoh,
        "solidus_ranges": solidus_ranges,
        "obs_ahe": obs_ahe,
        "obs_aft": obs_aft,
        "obs_zhe": obs_zhe,
        "obs_zft": obs_zft,
        "obs_ahe_stdev": obs_ahe_stdev,
        "obs_aft_stdev": obs_aft_stdev,
        "obs_zhe_stdev": obs_zhe_stdev,
        "obs_zft_stdev": obs_zft_stdev,
        "misfit_num_params": misfit_num_params,
        "misfit_type": misfit_type,
        "log_output": log_output,
        "log_file": log_file,
        "model_id": model_id,
        #Inverse mode
        "inverse_mode": False
    }

    return params


def prep_model(params):
    """Prepares models to be run as single models or in batch mode"""

    batch_keys = [
        "max_depth",
        "nx",
        "temp_surf",
        "temp_base",
        "t_total",
        "dt",
        "vx_init",
        "init_moho_depth",
        "removal_fraction",
        "removal_time",
        "ero_type",
        "ero_option1",
        "ero_option2",
        "ero_option3",
        "ero_option4",
        "ero_option5",
        "mantle_adiabat",
        "rho_crust",
        "cp_crust",
        "k_crust",
        "heat_prod_crust",
        "alphav_crust",
        "rho_mantle",
        "cp_mantle",
        "k_mantle",
        "heat_prod_mantle",
        "alphav_mantle",
        "rho_a",
        "k_a",
        "ap_rad",
        "ap_uranium",
        "ap_thorium",
        "zr_rad",
        "zr_uranium",
        "zr_thorium",
        "pad_thist",
        "pad_time"
    ]

    # Create empty dictionary for batch model parameters, if any
    batch_params = {}

    # Check that prep_model was called from the command line
    if params["cmd_line_call"]:
        # We know all batch model values are lists, check their lengths
        # If all are 1 then run in single-model mode
        params["batch_mode"] = False

        for key in batch_keys:
            if len(params[key]) != 1:
                params["batch_mode"] = True
            batch_params[key] = params[key]

        # Now we see what to do for running the model(s)
        if not params["batch_mode"]:
            # Convert list values, run single model
            for key in batch_keys:
                params[key] = params[key][0]
            run_model(params)
        else:
            # Run in batch mode
            batch_run(params, batch_params)

    else:
        # If called as a function, check for lists and their lengths
        params["batch_mode"] = False

        for key in batch_keys:
            if isinstance(params[key], list):
                if len(params[key]) != 1:
                    params["batch_mode"] = True
                batch_params[key] = params[key]

        # Now we see what to do for running the model(s)
        if not params["batch_mode"]:
            # Convert list values, run single model
            run_model(params)
        else:
            # Run in batch mode
            batch_run(params, batch_params)


def log_output(params, batch_mode=False):
    """Writes model summary output to a csv file"""
    # Define file for csv output
    if batch_mode:
        if params["log_file"] == "":
            params["log_file"] = "../csv/TC1D_batch_log.csv"
    outfile = params["log_file"]

    # Check number of past models and write header if needed
    model_count = 0
    try:
        with open(outfile) as f:
            write_header = False
            infile = f.readlines()
            if len(infile) < 1:
                write_header = True
            else:
                model_count = len(infile) - 1
    except IOError:
        write_header = True

    # Define model id if using batch mode
    if batch_mode:
        model_count += 1
        model_id = f"M{str(model_count).zfill(4)}"
        params["model_id"] = model_id

    # Open file for writing header
    with open(outfile, "a+") as f:
        if write_header:
            f.write(
                "Model ID,Simulation time (Myr),Time step (yr),Model thickness (km),Node points,"
                "Surface temperature (C),Basal temperature (C),Mantle adiabat,"
                "Crustal density (kg m^-3),Mantle removal fraction,Mantle removal time (Ma),"
                "Erosion model type,Erosion model option 1,"
                "Erosion model option 2,Erosion model option 3,Initial Moho depth (km),Initial Moho temperature (C),"
                "Initial surface heat flow (mW m^-2),Initial surface elevation (km),"
                "Final Moho depth (km),Final Moho temperature (C),Final surface heat flow (mW m^-2),"
                "Final surface elevation (km),Total exhumation (km),Apatite grain radius (um),Apatite U "
                "concentration (ppm), Apatite Th concentration (ppm),Zircon grain radius (um),Zircon U "
                "concentration (ppm), Zircon Th concentration (ppm),Predicted apatite (U-Th)/He age (Ma),"
                "Predicted apatite (U-Th)/He closure temperature (C),Measured apatite (U-Th)/He age (Ma),"
                "Measured apatite (U-Th)/He standard deviation (Ma),Predicted apatite fission-track age (Ma),"
                "Predicted apatite fission-track closure temperature (C),Measured apatite fission-track age (Ma),"
                "Measured apatite fission-track standard deviation (Ma),Predicted zircon (U-Th)/He age (Ma),"
                "Predicted zircon (U-Th)/He closure temperature (C),Measured zircon (U-Th)/He age (Ma),"
                "Measured zircon (U-Th)/He standard deviation (Ma),Predicted zircon fission-track age (Ma),"
                "Predicted zircon fission-track closure temperature (C),Measured zircon fission-track age (Ma),"
                "Measured zircon fission-track standard deviation (Ma),Misfit,Misfit type,Number of ages for misfit\n"
            )
            write_header = False
        f.write(f"{params['model_id']},")

    # Print warning(s) if more than one observed age of a given type was provided
    if (
        (len(params["obs_ahe"]) > 1)
        or (len(params["obs_aft"]) > 1)
        or (len(params["obs_zhe"]) > 1)
        or (len(params["obs_zft"]) > 1)
    ):
        print("")
        if len(params["obs_ahe"]) > 1:
            print(
                "WARNING: More than one measured AHe age supplied, only the first was written to the output file!"
            )
        if len(params["obs_aft"]) > 1:
            print(
                "WARNING: More than one measured AFT age supplied, only the first was written to the output file!"
            )
        if len(params["obs_zhe"]) > 1:
            print(
                "WARNING: More than one measured ZHe age supplied, only the first was written to the output file!"
            )
        if len(params["obs_zft"]) > 1:
            print(
                "WARNING: More than one measured ZFT age supplied, only the first was written to the output file!"
            )

    return None


def batch_run(params, batch_params):
    """Runs TC1D in batch mode"""
    param_list = list(ParameterGrid(batch_params))

    print(f"--- Starting batch processor for {len(param_list)} models ---\n")

    # Check number of past models and write header as needed
    success = 0
    failed = 0

    # Define file for csv output
    outfile = params["log_file"]

    #If inverse mode is enabled, run with the neighbourhood algorithm
    if params["inverse_mode"] == True:
        
        print(f"--- Starting inverse mode ---\n")
        log_output(params, batch_mode=True)

        #Batch params only for testing
        #batch_params = {'max_depth': [125.0], 'nx': [251], 'temp_surf': [0.0], 'temp_base': [1300.0], 't_total': [50.0], 'dt': [5000.0], 'vx_init': [0.0], 'init_moho_depth': [50.0, 60], 'removal_fraction': [0.0], 'removal_time': [0.0], 'ero_type': [1], 'ero_option1': [10.0, 15.0], 'ero_option2': [0.0], 'ero_option3': [0.0], 'ero_option4': [0.0], 'ero_option5': [0.0], 'mantle_adiabat': [True], 'rho_crust': [2850.0], 'cp_crust': [800.0], 'k_crust': [2.75], 'heat_prod_crust': [0.5], 'alphav_crust': [3e-05], 'rho_mantle': [3250.0], 'cp_mantle': [1000.0], 'k_mantle': [2.5], 'heat_prod_mantle': [0.0], 'alphav_mantle': [3e-05], 'rho_a': [3250.0], 'k_a': [20.0], 'ap_rad': [45.0], 'ap_uranium': [10.0], 'ap_thorium': [40.0], 'zr_rad': [60.0], 'zr_uranium': [100.0], 'zr_thorium': [40.0], 'pad_thist': [False], 'pad_time': [0.0]}
        
        #Starting model
        model = param_list[0]
        for key in batch_params:
                params[key] = model[key]
        
        #Filter params for multiple supplied values, use these as bounds
        #To do: filter out certain params
        filtered_params = {}
        for key, value in batch_params.items():
            if len(value) > 1:
                filtered_params[key] = value
                
        #Bounds of the parameter space
        bounds = list(filtered_params.values())
        print(f" The bounds are: {filtered_params}")
                
        # Objective function to be minimised, run for misfit
        def objective(x):
            #Update bounds
            for key, value in zip(filtered_params, x):
                filtered_params[key] = value
            #Add bounds to parameters
            params.update(filtered_params)
            print(f" The current bounds are: {x}")
            return run_model(params)
            #return x[0]*2 + x[1]*2 + 100*2 #lighter test function
                
        #Initialize NA searcher
        searcher = NASearcher(
            objective,
            ns= 10, #100, # number of samples per iteration #10
            nr= 1, #10, # number of cells to resample #1
            ni= 1, #100, # size of initial random search #1
            n= 1, #20, # number of iterations #1
            bounds=bounds
            )
        # Run the direct search phase
        searcher.run() # results stored in searcher.samples and searcher.objectives

        appraiser = NAAppraiser(
            initial_ensemble=searcher.samples, # points of parameter space already sampled
            log_ppd=-searcher.objectives, # objective function values
            bounds=bounds,
            n_resample=2000, # number of desired new samples #100
            n_walkers=5, # number of parallel walkers #1
        )

        appraiser.run() # Results stored in appraiser.samples

        #Best param
        best = searcher.samples[np.argmin(searcher.objectives)]
        print(f" The best parameters are: {best}")

        #Plot for 2 params
        if len(bounds) == 2:
            #Other params
            x_searcher = searcher.samples[:,0]
            y_searcher = searcher.samples[:,1]
            #Appraiser params
            x_appraiser = appraiser.samples[:,0]
            y_appraiser = appraiser.samples[:,1]

            #Plot
            plt.scatter(x_searcher, y_searcher, color="grey", marker="x", label="Searcher samples")
            plt.scatter(x_appraiser, y_appraiser, color="red", marker="x", label="Appraiser samples")
            plt.scatter(best[0], best[1], color="green", marker="x", label="Best sample")
        
            #
            plt.xlabel(list(filtered_params.keys())[0])
            plt.ylabel(list(filtered_params.keys())[1])
            plt.legend(loc="upper right")
            plt.title("Neighbourhood Algorithm samples")
            plt.show()
        
                
        print("Inverse mode complete")
        success += 1
    
    ###
    else:
        for i in range(len(param_list)):
            log_output(params, batch_mode=True)
            model = param_list[i]
            print(f"Iteration {i + 1}", end="", flush=True)
            # Update model parameters
            for key in batch_params:
                params[key] = model[key]

            try:
                run_model(params)
                print("Complete")
                success += 1
            except:
                print("FAILED!")
                with open(outfile, "a+") as f:
                    # TODO: Add ero_options 4 and 5 to output here
                    f.write(
                        f'{params["t_total"]:.4f},{params["dt"]:.4f},{params["max_depth"]:.4f},{params["nx"]},'
                        f'{params["temp_surf"]:.4f},{params["temp_base"]:.4f},{params["mantle_adiabat"]},'
                        f'{params["rho_crust"]:.4f},{params["removal_fraction"]:.4f},{params["removal_time"]:.4f},'
                        f'{params["ero_type"]},{params["ero_option1"]:.4f},'
                        f'{params["ero_option2"]:.4f},{params["ero_option3"]:.4f},{params["init_moho_depth"]:.4f},,,,,,,,,{params["ap_rad"]:.4f},{params["ap_uranium"]:.4f},'
                        f'{params["ap_thorium"]:.4f},{params["zr_rad"]:.4f},{params["zr_uranium"]:.4f},{params["zr_thorium"]:.4f},,,,,,,,,,,,,,,\n'
                    )
                failed += 1

    print(f"\n--- Execution complete ({success} succeeded, {failed} failed) ---")

def run_model(params):
    # Say hello
    if not params["batch_mode"]:
        print("")
        print(30 * "-" + " Execution started " + 31 * "-")

    # Ensure relative paths work by setting working dir to dir containing this script file
    # wd_orig = os.getcwd()
    # script_path = os.path.abspath(__file__)
    # dir_name = os.path.dirname(script_path)
    # os.chdir(dir_name)

    # Set flags if using batch mode
    if params["batch_mode"]:
        params["echo_info"] = False
        params["echo_thermal_info"] = False
        params["echo_ages"] = False
        params["plot_results"] = False

    # Conversion factors and unit conversions
    max_depth = kilo2base(params["max_depth"])
    moho_depth_init = kilo2base(params["init_moho_depth"])
    moho_depth = moho_depth_init
    t_total = myr2sec(params["t_total"])
    dt = yr2sec(params["dt"])

    # Calculate node spacing
    dx = max_depth / (params["nx"] - 1)  # m

    # Calculate time step
    nt = int(np.floor(t_total / dt))  # -

    # Create arrays to hold temperature fields
    temp_new = np.zeros(params["nx"])
    temp_prev = np.zeros(params["nx"])

    # Create coordinates of the grid points
    x = np.linspace(0, max_depth, params["nx"])
    xstag = x[:-1] + dx / 2
    vx_hist = np.zeros(nt)

    # Calculate erosion magnitude
    exhumation_magnitude = calculate_exhumation_magnitude(
        params["ero_type"],
        params["ero_option1"],
        params["ero_option2"],
        params["ero_option3"],
        params["ero_option4"],
        params["ero_option5"],
        t_total,
    )

    # Create velocity array for heat transfer
    vx_array = np.zeros(len(x))

    # Set initial exhumation velocity
    vx_init = mmyr2ms(params["vx_init"])

    # Fill velocity array to be able to have crust-only uplift
    if params["crustal_uplift"]:
        vx_array[x <= moho_depth] = vx_init
        vx_array[x > moho_depth] = 0.0
    else:
        vx_array[:] = vx_init

    vx, vx_max = calculate_erosion_rate(
        t_total,
        0.0,
        params["ero_type"],
        params["ero_option1"],
        params["ero_option2"],
        params["ero_option3"],
        params["ero_option4"],
        params["ero_option5"],
    )

    # Set number of passes needed based on erosion model type
    # Types 1-5 need only 1 pass
    if params["ero_type"] < 7:
        num_pass = 1

    # Create array of plot times
    t_plots = myr2sec(np.array(params["t_plots"]))
    t_plots.sort()

    # If populate t_plots array if only one value given (treated like a plot increment)
    if len(t_plots) == 1:
        t_plots = np.arange(t_plots[0], t_total, t_plots[0])

    # Set flag if more than one plot to produce
    if len(t_plots) > 0:
        more_plots = True
    else:
        more_plots = False

    # Determine thickness of mantle to remove
    mantle_lith_thickness = max_depth - moho_depth
    removal_thickness = params["removal_fraction"] * mantle_lith_thickness

    # Calculate explicit model stability conditions
    cond_stab = 0.0
    adv_stab = 0.0
    if not params["implicit"]:
        cond_stab, adv_stab = calculate_explicit_stability(
            vx_max,
            params["k_crust"],
            params["rho_crust"],
            params["cp_crust"],
            params["k_mantle"],
            params["rho_mantle"],
            params["cp_mantle"],
            params["k_a"],
            dt,
            dx,
            cond_crit=0.5,
            adv_crit=0.5,
        )

    # Echo model info if requested
    if params["echo_info"]:
        echo_model_info(
            dx,
            nt,
            dt,
            t_total,
            params["implicit"],
            params["ero_type"],
            exhumation_magnitude,
            cond_stab,
            adv_stab,
            cond_crit=0.5,
            adv_crit=0.5,
        )

    # Create array of past ages at which ages should be calculated, if not zero
    if params["past_age_increment"] > 0.0:
        surface_times_ma = np.arange(
            0.0, params["t_total"], params["past_age_increment"]
        )
        surface_times_ma = np.flip(surface_times_ma)
    else:
        surface_times_ma = np.array([0.0])

    # Create lists for storing depth, pressure, temperature, and time histories
    depth_hists = []
    pressure_hists = []
    temp_hists = []
    time_hists = []
    depths = np.zeros(len(surface_times_ma))

    # Create empty numpy arrays for depth, temperature, and time histories
    for i in range(len(surface_times_ma)):
        time_inc_now = myr2sec(params["t_total"] - surface_times_ma[i])
        nt_now = int(np.floor(time_inc_now / dt))
        depth_hists.append(np.zeros(nt_now))
        pressure_hists.append(np.zeros(nt_now))
        temp_hists.append(np.zeros(nt_now))
        time_hists.append(np.zeros(nt_now))

    if params["mantle_adiabat"]:
        adiabat_m = adiabat(
            alphav=params["alphav_mantle"],
            temp=params["temp_base"] + 273.15,
            cp=params["cp_mantle"],
        )
        temp_adiabat = params["temp_base"] + (xstag - max_depth) * adiabat_m
    else:
        adiabat_m = 0.0
        temp_adiabat = params["temp_base"]

    # Create material property arrays
    rho = np.ones(len(x)) * params["rho_crust"]
    rho[x > moho_depth] = params["rho_mantle"]
    cp = np.ones(len(x)) * params["cp_crust"]
    cp[x > moho_depth] = params["cp_mantle"]
    k = np.ones(len(xstag)) * params["k_crust"]
    k[xstag > moho_depth] = params["k_mantle"]
    heat_prod = np.ones(len(x)) * micro2base(params["heat_prod_crust"])
    heat_prod[x > moho_depth] = micro2base(params["heat_prod_mantle"])
    alphav = np.ones(len(x)) * params["alphav_crust"]
    alphav[x > moho_depth] = params["alphav_mantle"]

    # Generate initial temperature field
    if not params["batch_mode"]:
        print("")
        print("--- Calculating initial thermal model ---")
        print("")
    temp_init = temp_ss_implicit(
        params["nx"],
        dx,
        params["temp_surf"],
        params["temp_base"],
        vx_array,
        rho,
        cp,
        k,
        heat_prod,
    )
    interp_temp_init = interp1d(x, temp_init)
    init_moho_temp = interp_temp_init(moho_depth)
    init_heat_flow = kilo2base((k[0] + k[1]) / 2 * (temp_init[1] - temp_init[0]) / dx)
    if params["echo_thermal_info"]:
        print(f"- Initial surface heat flow: {init_heat_flow:.1f} mW/m^2")
        print(f"- Initial Moho temperature: {init_moho_temp:.1f}°C")
        print(f"- Initial Moho depth: {params['init_moho_depth']:.1f} km")
        print(
            f"- Initial LAB depth: {(max_depth - removal_thickness) / kilo2base(1):.1f} km"
        )

    # Create arrays to store elevation history
    elev_list = []
    time_list = []
    elev_list.append(0.0)
    time_list.append(0.0)

    # Create list to store LAB depths
    lab_depths = []

    # Calculate initial densities
    rho_prime = -rho * alphav * temp_init
    rho_inc_temp = rho + rho_prime

    # --- Set temperatures at 0 Ma ---
    temp_prev = temp_init.copy()

    # Modify temperatures and material properties for ero_types 4 and 5
    # TODO: Remove this?
    fault_activated = False
    if (params["ero_type"] == 4 or params["ero_type"] == 5) and (
        params["ero_option3"] < 1.0e-6
    ):
        temp_prev, moho_depth, rho, cp, k, heat_prod, alphav = init_ero_types(
            params, x, xstag, temp_prev, moho_depth
        )
        fault_activated = True

    # TODO: Remove this?
    delaminated = False
    if (params["removal_fraction"] > 0.0) and (params["removal_time"] < 1e-6):
        for ix in range(params["nx"]):
            if x[ix] > (max_depth - removal_thickness):
                temp_prev[ix] = params["temp_base"] + (x[ix] - max_depth) * adiabat_m
        delaminated = True

    if params["plot_results"]:
        # Set plot style
        plt.style.use("seaborn-v0_8-darkgrid")

        # Plot initial temperature field
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        if t_plots.max() < t_total - 1.0:
            # Add an extra color for the final temperature if it is not in the
            # list of times for plotting
            colors = plt.cm.viridis_r(np.linspace(0, 1, len(t_plots) + 1))
        else:
            colors = plt.cm.viridis_r(np.linspace(0, 1, len(t_plots)))
        ax1.plot(temp_init, -x / 1000, "k:", label="Initial")
        ax1.plot(temp_prev, -x / 1000, "k-", label="0 Myr")
        ax2.plot(rho_inc_temp, -x / 1000, "k-", label="0 Myr")

    # Loop over number of required passes
    for j in range(num_pass):
        # Start the loop over time steps
        curtime = 0.0
        idx = 0
        if j == num_pass - 1:
            plotidx = 0

        # Restore values if using more than 1 pass
        if num_pass > 1:
            # Reset moho depth
            moho_depth = moho_depth_init
            # Reset initial temperatures
            for ix in range(params["nx"]):
                if x[ix] > (max_depth - removal_thickness):
                    temp_prev[ix] = (
                        params["temp_base"] + (x[ix] - max_depth) * adiabat_m
                    )
                else:
                    temp_prev[ix] = temp_init[ix]

        # Reset erosion rate
        vx, _ = calculate_erosion_rate(
            t_total,
            curtime,
            params["ero_type"],
            params["ero_option1"],
            params["ero_option2"],
            params["ero_option3"],
            params["ero_option4"],
            params["ero_option5"],
        )

        # Calculate initial densities
        rho, cp, k, heat_prod, lab_depth = update_materials(
            x,
            xstag,
            moho_depth,
            params["rho_crust"],
            params["rho_mantle"],
            rho,
            params["cp_crust"],
            params["cp_mantle"],
            cp,
            params["k_crust"],
            params["k_mantle"],
            k,
            micro2base(params["heat_prod_crust"]),
            micro2base(params["heat_prod_mantle"]),
            heat_prod,
            temp_adiabat,
            temp_prev,
            params["k_a"],
            params["removal_fraction"],
        )
        rho_prime = -rho * alphav * temp_init
        rho_inc_temp = rho + rho_prime
        isoref = rho_inc_temp.sum() * dx
        h_ref = isoref / params["rho_a"]
        elev_init = max_depth - h_ref
        lab_depths.append(lab_depth)

        # Find starting depth if using only a one-pass erosion type
        if num_pass == 1:
            while curtime < t_total:
                curtime += dt
                vx_hist[idx] = vx
                idx += 1
                vx, _ = calculate_erosion_rate(
                    t_total,
                    curtime,
                    params["ero_type"],
                    params["ero_option1"],
                    params["ero_option2"],
                    params["ero_option3"],
                    params["ero_option4"],
                    params["ero_option5"],
                )
            for i in range(len(surface_times_ma)):
                time_inc_now = myr2sec(params["t_total"] - surface_times_ma[i])
                nt_now = int(np.floor(time_inc_now / dt))
                depths[i] = (vx_hist[:nt_now] * dt).sum()
                # Adjust depths for footwall if using ero type 4
                if params["ero_type"] == 4 and params["ero_option3"] > 0.0:
                    if params["ero_option2"] > 0.0:
                        depths[i] = (vx_hist[:nt_now] * dt).sum() - kilo2base(
                            params["ero_option1"]
                        )
                if params["debug"]:
                    print(f"Calculated starting depth {i}: {depths[i]} m")

            # Reset loop variables
            curtime = 0.0
            idx = 0
            vx, _ = calculate_erosion_rate(
                t_total,
                curtime,
                params["ero_type"],
                params["ero_option1"],
                params["ero_option2"],
                params["ero_option3"],
                params["ero_option4"],
                params["ero_option5"],
            )

        if not params["batch_mode"]:
            print("")
            print(
                f"--- Calculating transient thermal model (Pass {j + 1}/{num_pass}) ---"
            )
            print("")
        while curtime < t_total:
            # if (idx+1) % 100 == 0:
            if not params["batch_mode"]:
                print(
                    f"- Step {idx + 1:5d} of {nt} (Time: {curtime / myr2sec(1):5.1f} Myr, Erosion rate: {vx / mmyr2ms(1):5.2f} mm/yr)\r",
                    end="",
                )
            else:
                # Print progress dot if using batch model. 1 dot = 10%
                if (idx + 1) % round(nt / 10, 0) == 0:
                    print(".", end="", flush=True)
            curtime += dt

            # Modify temperatures and material properties for ero_types 4 and 5
            if (
                (params["ero_type"] == 4)
                or (params["ero_type"] == 5)
                and (not fault_activated)
            ):
                in_fault_interval = (
                    (curtime - (dt / 2)) / myr2sec(1)
                    <= params["ero_option3"]
                    < (curtime + (dt / 2)) / myr2sec(1)
                )
                if in_fault_interval:
                    (
                        temp_prev,
                        moho_depth,
                        rho,
                        cp,
                        k,
                        heat_prod,
                        alphav,
                    ) = init_ero_types(params, x, xstag, temp_prev, moho_depth)
                    # TODO: Check does this work for ero_type 5?
                    if params["ero_option2"] > 0.0:
                        depths[i] += kilo2base(params["ero_option1"])
                    fault_activated = True

            if (params["removal_fraction"] > 0.0) and (not delaminated):
                in_removal_interval = (
                    (curtime - (dt / 2)) / myr2sec(1)
                    <= params["removal_time"]
                    < (curtime + (dt / 2)) / myr2sec(1)
                )
                if in_removal_interval:
                    for ix in range(params["nx"]):
                        if x[ix] > (max_depth - removal_thickness):
                            temp_prev[ix] = (
                                params["temp_base"] + (x[ix] - max_depth) * adiabat_m
                            )
                    delaminated = True

            rho, cp, k, heat_prod, lab_depth = update_materials(
                x,
                xstag,
                moho_depth,
                params["rho_crust"],
                params["rho_mantle"],
                rho,
                params["cp_crust"],
                params["cp_mantle"],
                cp,
                params["k_crust"],
                params["k_mantle"],
                k,
                micro2base(params["heat_prod_crust"]),
                micro2base(params["heat_prod_mantle"]),
                heat_prod,
                temp_adiabat,
                temp_prev,
                params["k_a"],
                params["removal_fraction"],
            )

            # Fill velocity array to be able to have crust-only uplift
            if params["crustal_uplift"]:
                vx_array[x <= moho_depth] = vx
                vx_array[x > moho_depth] = 0.0
            else:
                vx_array[:] = vx

            if params["implicit"]:
                temp_new[:] = temp_transient_implicit(
                    params["nx"],
                    dx,
                    dt,
                    temp_prev,
                    params["temp_surf"],
                    params["temp_base"],
                    vx_array,
                    rho,
                    cp,
                    k,
                    heat_prod,
                )
            else:
                temp_new[:] = temp_transient_explicit(
                    temp_prev,
                    temp_new,
                    params["temp_surf"],
                    params["temp_base"],
                    params["nx"],
                    dx,
                    vx_array,
                    dt,
                    rho,
                    cp,
                    k,
                    heat_prod,
                )

            temp_prev[:] = temp_new[:]

            rho_prime = -rho * alphav * temp_new
            rho_temp_new = rho + rho_prime
            lab_depths.append(lab_depth - moho_depth)

            # Blend materials when the Moho lies between two nodes
            isonew = 0.0
            for i in range(len(rho_temp_new) - 1):
                rho_inc = rho_temp_new[i]
                if (moho_depth < x[i + 1]) and (moho_depth >= x[i]):
                    crust_frac = (moho_depth - x[i]) / dx
                    mantle_frac = 1.0 - crust_frac
                    rho_inc = (
                        crust_frac * rho_temp_new[i] + mantle_frac * rho_temp_new[i + 1]
                    )
                isonew += rho_inc * dx

            h_asthenosphere = isonew / params["rho_a"]
            elev = max_depth - h_asthenosphere

            # Update Moho depth
            if not params["fixed_moho"]:
                moho_depth -= vx * dt

            # Store tracked surface elevations and advection velocities
            if j == 0:
                elev_list.append(elev - elev_init)
                time_list.append(curtime / myr2sec(1.0))

            # Save Temperature-depth history
            if j == num_pass - 1:
                # Store temperature, time, depth, pressure
                interp_temp_new = interp1d(x, temp_new)
                # Calculate lithostatic pressure
                pressure = calculate_pressure(rho_temp_new, dx)
                interp_pressure = interp1d(x, pressure)
                for i in range(len(surface_times_ma)):
                    if curtime <= myr2sec(params["t_total"] - surface_times_ma[i]):
                        depths[i] -= vx * dt
                        depth_hists[i][idx] = depths[i]
                        time_hists[i][idx] = curtime
                        # Store temperature history
                        # Check whether point is very close to the surface
                        if abs(depths[i]) <= 1e-6:
                            temp_hists[i][idx] = 0.0
                        # Check whether point is below the Moho for fixed-moho models
                        # If so, set temperature to Moho temperature
                        elif depths[i] > moho_depth and params["fixed_moho"]:
                            temp_hists[i][idx] = interp_temp_new(moho_depth)
                        # Otherwise, record temperature at current depth
                        else:
                            temp_hists[i][idx] = interp_temp_new(depths[i])
                        # Store pressure history
                        # Check whether point is very close to the surface
                        if abs(depths[i]) <= 1e-6:
                            pressure_hists[i][idx] = 0.0
                        elif depths[i] > moho_depth and params["fixed_moho"]:
                            pressure_hists[i][idx] = interp_pressure(moho_depth)
                        else:
                            pressure_hists[i][idx] = interp_pressure(depths[i])
                        if params["debug"]:
                            print("")
                            print(
                                f"Current time: {curtime} s ({curtime / myr2sec(1):.2f} Myr)"
                            )
                            print(
                                f"Time span for surface time {i}: {myr2sec(params['t_total'] - surface_times_ma[i]):.2f} s ({params['t_total'] - surface_times_ma[i]} Myr)"
                            )
                            print(
                                f"Depth for surface time {i}: {depth_hists[i][idx] / kilo2base(1):.2f} km"
                            )
                            print(
                                f"Pressure for surface time {i}: {pressure_hists[i][idx] * micro2base(1):.2f} MPa"
                            )
                            print(
                                f"Time for surface time {i}: {time_hists[i][idx] / myr2sec(1):.2f} Myr"
                            )
                            print(
                                f"Temp for surface time {i}: {temp_hists[i][idx]:.1f} °C"
                            )

            # Update index
            idx += 1

            # Update erosion rate
            vx, _ = calculate_erosion_rate(
                t_total,
                curtime,
                params["ero_type"],
                params["ero_option1"],
                params["ero_option2"],
                params["ero_option3"],
                params["ero_option4"],
                params["ero_option5"],
            )

            if j == num_pass - 1:
                if params["plot_results"] and more_plots:
                    if curtime > t_plots[plotidx]:
                        ax1.plot(
                            temp_new,
                            -x / 1000,
                            "-",
                            label=f"{t_plots[plotidx] / myr2sec(1):.1f} Myr",
                            color=colors[plotidx],
                        )
                        ax2.plot(
                            rho_temp_new,
                            -x / 1000,
                            label=f"{t_plots[plotidx] / myr2sec(1):.1f} Myr",
                            color=colors[plotidx],
                        )
                        if plotidx == len(t_plots) - 1:
                            more_plots = False
                        plotidx += 1
                        # tplot = t_plots[plotidx]

        if not params["batch_mode"]:
            print("")

    rho_prime = -rho * alphav * temp_new
    rho_temp_new = rho + rho_prime

    interp_temp_new = interp1d(x, temp_new)
    final_moho_temp = interp_temp_new(moho_depth)
    final_heat_flow = kilo2base((k[0] + k[1]) / 2 * (temp_new[1] - temp_new[0]) / dx)

    if not params["batch_mode"]:
        print("")

    if params["echo_thermal_info"]:
        print("")
        print("--- Final thermal model values ---")
        print("")
        print(f"- Final surface heat flow: {final_heat_flow:.1f} mW/m^2")
        print(f"- Final Moho temperature: {final_moho_temp:.1f}°C")
        print(f"- Final Moho depth: {moho_depth / kilo2base(1):.1f} km")
        print(f"- Final LAB depth: {lab_depth / kilo2base(1):.1f} km")

    if params["calc_ages"]:
        # Convert time since model start to time before end of simulation
        time_ma = t_total - time_hists[-1]
        time_ma = time_ma / myr2sec(1)

        corr_ahe_ages = np.zeros(len(surface_times_ma))
        ahe_temps = np.zeros(len(surface_times_ma))
        aft_ages = np.zeros(len(surface_times_ma))
        aft_temps = np.zeros(len(surface_times_ma))
        corr_zhe_ages = np.zeros(len(surface_times_ma))
        zhe_temps = np.zeros(len(surface_times_ma))
        zft_ages = np.zeros(len(surface_times_ma))
        zft_temps = np.zeros(len(surface_times_ma))
        for i in range(len(surface_times_ma)):
            (
                corr_ahe_ages[i],
                ahe_age,
                ahe_temps[i],
                aft_ages[i],
                aft_mean_ftl,
                aft_temps[i],
                corr_zhe_ages[i],
                zhe_age,
                zhe_temps[i],
                zft_ages[i],
                zft_temps[i],
            ) = calculate_ages_and_tcs(
                params,
                time_hists[i],
                temp_hists[i],
                depth_hists[i],
                pressure_hists[i],
            )
            if params["debug"]:
                print(f"")
                print(f"--- Predicted ages for cooling history {i} ---")
                print(
                    f"- AHe age: {corr_ahe_ages[i]:.2f} Ma (Tc: {ahe_temps[i]:.2f} °C)"
                )
                print(f"- AFT age: {aft_ages[i]:.2f} Ma (Tc: {aft_temps[i]:.2f} °C)")
                print(
                    f"- ZHe age: {corr_zhe_ages[i]:.2f} Ma (Tc: {zhe_temps[i]:.2f} °C)"
                )
                print(f"- ZFT age: {zft_ages[i]:.2f} Ma (Tc: {zft_temps[i]:.2f} °C)")

        # Only do this for the final ages/histories!
        if params["batch_mode"]:
            tt_filename = params["model_id"] + "-time_temp_hist.csv"
            ttd_filename = params["model_id"] + "-time_temp_depth_pressure_hist.csv"
            ftl_filename = params["model_id"] + "-ft_length.csv"
            os.rename("time_temp_hist.csv", "batch_output/" + tt_filename)
            os.rename(
                "time_temp_depth_pressure_hist.csv", "batch_output/" + ttd_filename
            )
            os.rename("ft_length.csv", "batch_output/" + ftl_filename)

        if params["echo_ages"]:
            print("")
            print("--- Predicted thermochronometer ages ---")
            print("")
            print(
                f"- AHe age: {float(corr_ahe_ages[-1]):.2f} Ma (uncorrected age: {float(ahe_age):.2f} Ma)"
            )
            if params["madtrax_aft"]:
                print(f"- AFT age: {aft_ages[-1] / 1e6:.2f} Ma (MadTrax)")
            if params["ketch_aft"]:
                print(f"- AFT age: {float(aft_ages[-1]):.2f} Ma (Ketcham)")
            print(
                f"- ZHe age: {float(corr_zhe_ages[-1]):.2f} Ma (uncorrected age: {float(zhe_age):.2f} Ma)"
            )
            print(f"- ZFT age: {zft_ages[-1]:.2f} Ma (MadTrax)")

        # If measured ages have been provided, calculate misfit
        if (
            len(params["obs_ahe"])
            + len(params["obs_aft"])
            + len(params["obs_zhe"])
            + len(params["obs_zft"])
            > 0
        ):
            # Create single arrays of ages for misfit calculation
            pred_ages = []
            obs_ages = []
            obs_stdev = []
            for i in range(len(params["obs_ahe"])):
                pred_ages.append(float(corr_ahe_ages[-1]))
                obs_ages.append(params["obs_ahe"][i])
                obs_stdev.append(params["obs_ahe_stdev"][i])
            for i in range(len(params["obs_aft"])):
                pred_ages.append(float(aft_ages[-1]))
                obs_ages.append(params["obs_aft"][i])
                obs_stdev.append(params["obs_aft_stdev"][i])
            for i in range(len(params["obs_zhe"])):
                pred_ages.append(float(corr_zhe_ages[-1]))
                obs_ages.append(params["obs_zhe"][i])
                obs_stdev.append(params["obs_zhe_stdev"][i])
            for i in range(len(params["obs_zft"])):
                pred_ages.append(float(zft_ages[-1]))
                obs_ages.append(params["obs_zft"][i])
                obs_stdev.append(params["obs_zft_stdev"][i])

            # Convert lists to NumPy arrays
            pred_ages = np.array(pred_ages)
            obs_ages = np.array(obs_ages)
            obs_stdev = np.array(obs_stdev)

            # Calculate misfit
            misfit = calculate_misfit(
                pred_ages,
                obs_ages,
                obs_stdev,
                params["misfit_type"],
                params["misfit_num_params"],
            )

            # Print misfit to the screen
            if params["echo_ages"]:
                print("")
                print("--- Predicted and observed age misfit ---")
                print("")
                print(
                    f"- Misfit: {misfit:.4f} (misfit type {params['misfit_type']}, {len(pred_ages)} age(s))"
                )

    if (
        (params["plot_results"] and params["save_plots"])
        or params["write_temps"]
        or params["read_temps"]
        or (params["past_age_increment"] > 0.0 and params["write_past_ages"])
    ):
        # fp = "/Users/whipp/Work/Modeling/Source/Python/tc1d-git/"
        fp = ""
        print("")
        print("--- Writing output file(s) ---")
        print("")

    # Write past ages to file if requested
    if (params["past_age_increment"] > 0.0) and (params["write_past_ages"]):
        past_ages_out = np.zeros([len(surface_times_ma), 5])
        past_ages_out[:, 0] = surface_times_ma
        past_ages_out[:, 1] = corr_ahe_ages + surface_times_ma
        past_ages_out[:, 2] = aft_ages + surface_times_ma
        past_ages_out[:, 3] = corr_zhe_ages + surface_times_ma
        past_ages_out[:, 4] = zft_ages + surface_times_ma
        savefile = os.path.join("csv", "past_ages.csv")
        np.savetxt(
            fp + savefile,
            past_ages_out,
            delimiter=",",
            fmt="%.8f",
            header="Time (Ma),Predicted Apatite (U-Th)/He age (Ma),Predicted Apatite fission-track age (Ma),Predicted "
            "Zircon (U-Th)/He age (Ma),Predicted Zircon fission-track age (Ma)",
        )
        print(f"- Past ages written to {savefile}")

    if params["plot_results"]:
        # Plot the final temperature field
        xmin = 0.0
        # xmax = params['temp_base'] + 100
        xmax = 1600.0
        ax1.plot(
            temp_new,
            -x / 1000,
            "-",
            label=f"{curtime / myr2sec(1):.1f} Myr",
            color=colors[-1],
        )
        ax1.plot(
            [xmin, xmax],
            [-moho_depth / kilo2base(1), -moho_depth / kilo2base(1)],
            linestyle="--",
            color="black",
            lw=0.5,
        )
        ax1.plot(
            [xmin, xmax],
            [-params["init_moho_depth"], -params["init_moho_depth"]],
            linestyle="--",
            color="gray",
            lw=0.5,
        )

        if params["crust_solidus"]:
            crust_solidus_comp_text = {
                "wet_felsic": "Wet felsic",
                "wet_intermediate": "Wet intermediate",
                "wet_basalt": "Wet basalt",
                "dry_felsic": "Dry felsic",
                "dry_basalt": "Dry basalt",
            }
            crust_slice = x / 1000.0 <= moho_depth / kilo2base(1)
            pressure = calculate_pressure(rho_temp_new, dx)
            crust_pressure = pressure[crust_slice]
            crust_solidus = calculate_crust_solidus(
                params["crust_solidus_comp"], crust_pressure
            )
            crust_solidus_plot_text = crust_solidus_comp_text[
                params["crust_solidus_comp"]
            ]
            ax1.plot(
                crust_solidus,
                -x[crust_slice] / 1000.0,
                color="gray",
                linestyle=":",
                lw=1.5,
                label=f"Crust solidus ({crust_solidus_plot_text})",
            )

        if params["mantle_solidus"]:
            mantle_slice = x / 1000 >= moho_depth / kilo2base(1)
            pressure = calculate_pressure(rho_temp_new, dx)
            mantle_solidus = calculate_mantle_solidus(
                pressure / 1.0e9, xoh=params["mantle_solidus_xoh"]
            )
            ax1.plot(
                mantle_solidus[mantle_slice],
                -x[mantle_slice] / 1000,
                color="gray",
                linestyle="--",
                lw=1.5,
                label=f"Mantle solidus ({params['mantle_solidus_xoh']:.1f} μg/g H$_{2}$O)",
            )

        if params["solidus_ranges"]:
            # Crust solidii
            crust_solidus_comp_text = {
                "wet_felsic": "Wet felsic",
                "wet_intermediate": "Wet intermediate",
                "wet_basalt": "Wet basalt",
                "dry_felsic": "Dry felsic",
                "dry_basalt": "Dry basalt",
            }
            crust_thickness = max(params["init_moho_depth"], moho_depth / kilo2base(1))
            crust_slice = x / kilo2base(1) <= crust_thickness
            pressure = calculate_pressure(rho_temp_new, dx)
            crust_pressure = pressure[crust_slice]
            wet_felsic_solidus = calculate_crust_solidus("wet_felsic", crust_pressure)
            dry_basalt_solidus = calculate_crust_solidus("dry_basalt", crust_pressure)
            wet_felsic_solidus_plot_text = crust_solidus_comp_text["wet_felsic"]
            dry_basalt_solidus_plot_text = crust_solidus_comp_text["dry_basalt"]

            # Mantle solidii
            min_moho_depth = min(params["init_moho_depth"], moho_depth / kilo2base(1))
            mantle_slice = x / 1000 >= min_moho_depth
            mantle_pressure = pressure[mantle_slice]
            # TODO: Find a suitable value for xoh
            wet_mantle_solidus = calculate_mantle_solidus(
                mantle_pressure / 1.0e9, xoh=100000.0
            )
            dry_mantle_solidus = calculate_mantle_solidus(
                mantle_pressure / 1.0e9, xoh=0.0
            )

            # Plots
            ax1.plot(
                wet_felsic_solidus,
                -x[crust_slice] / 1000.0,
                color="gray",
                linestyle="--",
                lw=1.5,
                label=f"{wet_felsic_solidus_plot_text} solidus",
            )
            ax1.plot(
                dry_basalt_solidus,
                -x[crust_slice] / 1000.0,
                color="gray",
                linestyle="-.",
                lw=1.5,
                label=f"{dry_basalt_solidus_plot_text} solidus",
            )
            ax1.fill_betweenx(
                -x[crust_slice] / 1000.0,
                wet_felsic_solidus,
                dry_basalt_solidus,
                color="tab:olive",
                alpha=0.5,
                lw=1.5,
                # label=f"Crust solidus: {wet_felsic_solidus_plot_text}, {dry_basalt_solidus_plot_text}",
            )
            ax1.fill_betweenx(
                -x[mantle_slice] / 1000.0,
                wet_mantle_solidus,
                dry_mantle_solidus,
                color="tab:gray",
                alpha=0.5,
                lw=1.5,
                label=f"Mantle solidus",
            )

        ax1.text(20.0, -moho_depth / kilo2base(1) + 1.0, "Final Moho")
        ax1.text(20.0, -params["init_moho_depth"] - 3.0, "Initial Moho", color="gray")
        ax1.legend()
        ax1.axis([xmin, xmax, -max_depth / 1000, 0])
        ax1.set_xlabel("Temperature (°C)")
        ax1.set_ylabel("Depth (km)")
        # ax1.grid()

        xmin = 2700
        xmax = 3300
        ax2.plot(
            rho_temp_new,
            -x / 1000,
            label=f"{t_total / myr2sec(1):.1f} Myr",
            color=colors[-1],
        )
        ax2.plot(
            [xmin, xmax],
            [-moho_depth / kilo2base(1), -moho_depth / kilo2base(1)],
            linestyle="--",
            color="black",
            lw=0.5,
        )
        ax2.plot(
            [xmin, xmax],
            [-params["init_moho_depth"], -params["init_moho_depth"]],
            linestyle="--",
            color="gray",
            lw=0.5,
        )
        ax2.axis([xmin, xmax, -max_depth / 1000, 0])
        ax2.set_xlabel("Density (kg m$^{-3}$)")
        ax2.set_ylabel("Depth (km)")
        ax2.legend()
        # ax2.grid()

        plt.tight_layout()
        if params["save_plots"]:
            savefile = os.path.join("png", "T_rho_hist.png")
            plt.savefig(fp + savefile, dpi=300)
            print(f"- Temperature/density history plot written to {savefile}")
        if params["display_plots"]:
            plt.show()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        # ax1.plot(time_list, elev_list, 'k-')
        ax1.plot(time_list, elev_list)
        ax1.set_xlabel("Time (Myr)")
        ax1.set_ylabel("Elevation (m)")
        ax1.set_xlim(0.0, t_total / myr2sec(1))
        ax1.set_title("Elevation history")
        # plt.axis([0.0, t_total/myr2sec(1), 0, 750])
        # ax1.grid()

        ax2.plot(time_hists[-1] / myr2sec(1), vx_hist / mmyr2ms(1))
        ax2.fill_between(
            time_hists[-1] / myr2sec(1),
            vx_hist / mmyr2ms(1),
            0.0,
            alpha=0.33,
            color="tab:blue",
            label=f"Total erosional exhumation: {exhumation_magnitude:.1f} km",
        )
        ax2.set_xlabel("Time (Myr)")
        ax2.set_ylabel("Erosion rate (mm/yr)")
        ax2.set_xlim(0.0, t_total / myr2sec(1))
        # if params["ero_option1"] >= 0.0:
        #    ax2.set_ylim(ymin=0.0)
        # plt.axis([0.0, t_total/myr2sec(1), 0, 750])
        # ax2.grid()
        ax2.legend()

        plt.tight_layout()
        if params["save_plots"]:
            savefile = os.path.join("png", "elev_hist.png")
            plt.savefig(fp + savefile, dpi=300)
            print(f"- Surface elevation history plot written to {savefile}")
        if params["display_plots"]:
            plt.show()

        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Plot cooling history and ages only if ages were calculated
        if params["calc_ages"]:
            # create objects
            fig = plt.figure(figsize=(12, 8))
            gs = GridSpec(3, 3, figure=fig)

            # create sub plots as grid
            ax1 = fig.add_subplot(gs[0:2, :])
            ax2 = fig.add_subplot(gs[2, :-1])
            ax3 = fig.add_subplot(gs[2, -1])
            # ax1.plot(time_ma, temp_hist, 'r-', lw=2)

            # Calculate synthetic uncertainties
            ahe_uncert = 0.1
            aft_uncert = 0.2
            zhe_uncert = 0.1
            zft_uncert = 0.2
            ahe_min, ahe_max = (1.0 - ahe_uncert) * float(corr_ahe_ages[-1]), (
                1.0 + ahe_uncert
            ) * float(corr_ahe_ages[-1])
            aft_min, aft_max = (1.0 - aft_uncert) * float(aft_ages[-1]), (
                1.0 + aft_uncert
            ) * float(aft_ages[-1])
            zhe_min, zhe_max = (1.0 - zhe_uncert) * float(corr_zhe_ages[-1]), (
                1.0 + zhe_uncert
            ) * float(corr_zhe_ages[-1])
            zft_min, zft_max = (1.0 - zft_uncert) * float(zft_ages[-1]), (
                1.0 + zft_uncert
            ) * float(zft_ages[-1])
            ax1.plot(time_ma, temp_hists[-1])

            # Plot delamination time, if enabled
            if params["removal_fraction"] > 0.0:
                removal_time_ma = params["t_total"] - params["removal_time"]
                ax1.plot(
                    [removal_time_ma, removal_time_ma],
                    [params["temp_surf"], params["temp_base"]],
                    "--",
                    color="gray",
                    label="Time of mantle delamination",
                )
                ax1.text(
                    removal_time_ma - 1.0,
                    (temp_hists[-1].max() + temp_hists[-1].min()) / 2.0,
                    "Mantle lithosphere delaminates",
                    rotation=90,
                    ha="center",
                    va="center",
                    color="gray",
                )

            # Plot shaded uncertainty area and AHe age if no measured ages exist
            if len(params["obs_ahe"]) == 0:
                ax1.axvspan(
                    ahe_min,
                    ahe_max,
                    alpha=0.33,
                    color="tab:blue",
                    label=f"Predicted AHe age ({float(corr_ahe_ages[-1]):.2f} Ma ± {ahe_uncert * 100.0:.0f}% uncertainty; T$_c$ = {ahe_temps[-1]:.1f}°C)",
                )
                ax1.plot(
                    float(corr_ahe_ages[-1]),
                    ahe_temps[-1],
                    marker="o",
                    color="tab:blue",
                )
            # Plot predicted age + observed AHe age(s)
            else:
                ax1.scatter(
                    float(corr_ahe_ages[-1]),
                    ahe_temps[-1],
                    marker="o",
                    color="tab:blue",
                    label=f"Predicted AHe age ({float(corr_ahe_ages[-1]):.2f} Ma; T$_c$ = {ahe_temps[-1]:.1f}°C)",
                )
                ahe_temps_obs = []
                for i in range(len(params["obs_ahe"])):
                    ahe_temps_obs.append(ahe_temps[-1])
                ax1.errorbar(
                    params["obs_ahe"],
                    ahe_temps_obs,
                    xerr=params["obs_ahe_stdev"],
                    marker="s",
                    color="tab:blue",
                    label="Measured AHe age(s)",
                )

            # Plot shaded uncertainty area and AFT age if no measured ages exist
            if len(params["obs_aft"]) == 0:
                ax1.axvspan(
                    aft_min,
                    aft_max,
                    alpha=0.33,
                    color="tab:orange",
                    label=f"Predicted AFT age ({float(aft_ages[-1]):.2f} Ma ± {aft_uncert * 100.0:.0f}% uncertainty; T$_c$ = {aft_temps[-1]:.1f}°C)",
                )
                ax1.plot(
                    float(aft_ages[-1]), aft_temps[-1], marker="o", color="tab:orange"
                )
            # Plot predicted age + observed AFT age(s)
            else:
                ax1.scatter(
                    float(aft_ages[-1]),
                    aft_temps[-1],
                    marker="o",
                    color="tab:orange",
                    label=f"Predicted AFT age ({float(aft_ages[-1]):.2f} Ma; T$_c$ = {aft_temps[-1]:.1f}°C)",
                )
                aft_temps_obs = []
                for i in range(len(params["obs_aft"])):
                    aft_temps_obs.append(aft_temps[-1])
                ax1.errorbar(
                    params["obs_aft"],
                    aft_temps_obs,
                    xerr=params["obs_aft_stdev"],
                    marker="s",
                    color="tab:orange",
                    label="Measured AFT age(s)",
                )

            # Plot shaded uncertainty area and ZHe age if no measured ages exist
            if len(params["obs_zhe"]) == 0:
                ax1.axvspan(
                    zhe_min,
                    zhe_max,
                    alpha=0.33,
                    color="tab:green",
                    label=f"Predicted ZHe age ({float(corr_zhe_ages[-1]):.2f} Ma ± {zhe_uncert * 100.0:.0f}% uncertainty; T$_c$ = {zhe_temps[-1]:.1f}°C)",
                )
                ax1.plot(
                    float(corr_zhe_ages[-1]),
                    zhe_temps[-1],
                    marker="o",
                    color="tab:green",
                )
            # Plot predicted age + observed ZHe age(s)
            else:
                ax1.scatter(
                    float(corr_zhe_ages[-1]),
                    zhe_temps[-1],
                    marker="o",
                    color="tab:green",
                    label=f"Predicted ZHe age ({float(corr_zhe_ages[-1]):.2f} Ma; T$_c$ = {zhe_temps[-1]:.1f}°C)",
                )
                zhe_temps_obs = []
                for i in range(len(params["obs_zhe"])):
                    zhe_temps_obs.append(zhe_temps[-1])
                ax1.errorbar(
                    params["obs_zhe"],
                    zhe_temps_obs,
                    xerr=params["obs_zhe_stdev"],
                    marker="s",
                    color="tab:green",
                    label="Measured ZHe age(s)",
                )

            # Plot shaded uncertainty area and ZFT age if no measured ages exist
            if len(params["obs_zft"]) == 0:
                ax1.axvspan(
                    zft_min,
                    zft_max,
                    alpha=0.33,
                    color="tab:red",
                    label=f"Predicted ZFT age ({float(zft_ages[-1]):.2f} Ma ± {zft_uncert * 100.0:.0f}% uncertainty; T$_c$ = {zft_temps[-1]:.1f}°C)",
                )
                ax1.plot(
                    float(zft_ages[-1]), zft_temps[-1], marker="o", color="tab:red"
                )
            # Plot predicted age + observed ZFT age(s)
            else:
                ax1.scatter(
                    float(zft_ages[-1]),
                    zft_temps[-1],
                    marker="o",
                    color="tab:red",
                    label=f"Predicted ZFT age ({float(zft_ages[-1]):.2f} Ma; T$_c$ = {zft_temps[-1]:.1f}°C)",
                )
                zft_temps_obs = []
                for i in range(len(params["obs_zft"])):
                    zft_temps_obs.append(zft_temps[-1])
                ax1.errorbar(
                    params["obs_zft"],
                    zft_temps_obs,
                    xerr=params["obs_zft_stdev"],
                    marker="s",
                    color="tab:red",
                    label="Measured ZFT age(s)",
                )

            ax1.set_xlim(t_total / myr2sec(1), 0.0)
            ax1.set_ylim(params["temp_surf"], 1.05 * temp_hists[-1].max())
            ax1.set_xlabel("Time (Ma)")
            ax1.set_ylabel("Temperature (°C)")
            # Include misfit in title if there are measured ages
            if (
                len(params["obs_ahe"])
                + len(params["obs_aft"])
                + len(params["obs_zhe"])
                + len(params["obs_zft"])
                == 0
            ):
                ax1.set_title("Thermal history for surface sample")
            else:
                ax1.set_title(
                    f"Thermal history for surface sample (misfit = {misfit:.4f}; {len(obs_ages)} age(s))"
                )
            if params["pad_thist"] and params["pad_time"] > 0.0:
                ax1.annotate(
                    f"Initial holding time: +{params['pad_time']:.1f} Myr",
                    xy=(time_ma.max(), temp_hists[-1][0]),
                    xycoords="data",
                    xytext=(0.95 * time_ma.max(), 0.65 * temp_hists[-1].max()),
                    textcoords="data",
                    arrowprops=dict(
                        arrowstyle="->", connectionstyle="arc3", fc="black"
                    ),
                    bbox=dict(boxstyle="round4,pad=0.3", fc="white", lw=0),
                )
            # ax1.grid()
            ax1.legend()

            ax2.plot(time_ma, vx_hist / mmyr2ms(1))
            ax2.fill_between(
                time_ma,
                vx_hist / mmyr2ms(1),
                0.0,
                alpha=0.33,
                color="tab:blue",
                label=f"Total erosional exhumation: {exhumation_magnitude:.1f} km",
            )
            ax2.set_xlabel("Time (Ma)")
            ax2.set_ylabel("Erosion rate (mm/yr)")
            ax2.set_xlim(t_total / myr2sec(1), 0.0)
            # if params["ero_option1"] >= 0.0:
            #    ax2.set_ylim(ymin=0.0)
            # plt.axis([0.0, t_total/myr2sec(1), 0, 750])
            # ax2.grid()
            ax2.legend()
            ax2.set_title("Erosion history for surface sample")

            ft_lengths = np.genfromtxt("ft_length.csv", delimiter=",", skip_header=1)
            length = ft_lengths[:, 0]
            prob = ft_lengths[:, 1]
            ax3.plot(length, prob)
            ax3.plot(
                [float(aft_mean_ftl), float(aft_mean_ftl)],
                [0.0, 1.05 * prob.max()],
                label=f"Mean: {float(aft_mean_ftl):.1f} µm",
            )
            ax3.set_xlabel("Track length (um)")
            ax3.set_ylabel("Probability")
            ax3.set_xlim([0.0, 20.0])
            ax3.set_ylim([0.0, 1.05 * prob.max()])
            ax3.legend()
            ax3.set_title("Apatite fission-track length distribution")

            plt.tight_layout()
            if params["save_plots"]:
                savefile = os.path.join("png", "cooling_hist.png")
                plt.savefig(fp + savefile, dpi=300)
                print(f"- Thermal history and ages plot written to {savefile}")
            if params["display_plots"]:
                plt.show()

            # Display plot of past ages if more than one surface age is calculated
            if len(surface_times_ma) > 1:
                # Make figure and plot axes
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

                # Plot ages and reference line for ages at time of exposure
                ax1.plot(
                    surface_times_ma,
                    corr_ahe_ages,
                    marker="o",
                    label="Predicted AHe age",
                )
                ax1.plot(
                    surface_times_ma, aft_ages, marker="o", label="Predicted AFT age"
                )
                ax1.plot(
                    surface_times_ma,
                    corr_zhe_ages,
                    marker="o",
                    label="Predicted ZHe age",
                )
                ax1.plot(
                    surface_times_ma, zft_ages, marker="o", label="Predicted ZFT age"
                )
                ax1.plot(
                    [params["t_total"], 0.0],
                    [0.0 + params["pad_time"], params["t_total"] + params["pad_time"]],
                    "--",
                    color="gray",
                    label="Unreset ages",
                )

                # Add axis labels
                ax1.set_xlabel("Surface exposure time (Ma)")
                ax1.set_ylabel("Age (Ma)")

                # Set axis ranges
                ax1.set_xlim(params["t_total"], 0.0)
                ax1.set_ylim(
                    0.0,
                    1.05
                    * max(
                        corr_ahe_ages.max(),
                        aft_ages.max(),
                        corr_zhe_ages.max(),
                        zft_ages.max(),
                    ),
                )

                # Enable legend and title
                ax1.legend()
                ax1.set_title("Predicted ages at the time of exposure")

                # Plot ages and reference line for ages including time since exposure
                ax2.plot(
                    surface_times_ma,
                    corr_ahe_ages + surface_times_ma,
                    marker="o",
                    label="Predicted AHe age",
                )
                ax2.plot(
                    surface_times_ma,
                    aft_ages + surface_times_ma,
                    marker="o",
                    label="Predicted AFT age",
                )
                ax2.plot(
                    surface_times_ma,
                    corr_zhe_ages + surface_times_ma,
                    marker="o",
                    label="Predicted ZHe age",
                )
                ax2.plot(
                    surface_times_ma,
                    zft_ages + surface_times_ma,
                    marker="o",
                    label="Predicted ZFT age",
                )
                ax2.plot(
                    [params["t_total"], 0.0],
                    [
                        params["t_total"] + params["pad_time"],
                        params["t_total"] + params["pad_time"],
                    ],
                    "--",
                    color="gray",
                    label="Unreset ages",
                )

                # Add axis labels
                ax2.set_xlabel("Surface exposure time (Ma)")
                ax2.set_ylabel("Age (Ma)")

                # Set axis ranges
                ax2.set_xlim(params["t_total"], 0.0)
                ax2.set_ylim(0.0, 1.05 * (params["t_total"] + params["pad_time"]))

                # Enable legend and title
                ax2.legend()
                ax2.set_title("Predicted ages including time since exposure")

                # Use tight layout and save/display plot if requested
                plt.tight_layout()
                if params["save_plots"]:
                    savefile = os.path.join("png", "past_ages.png")
                    plt.savefig(fp + savefile, dpi=300)
                    print(f"- Past ages plot written to {savefile}")
                if params["display_plots"]:
                    plt.show()

        # Plot LAB depths
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        ax.plot(time_hists[-1] / myr2sec(1), lab_depths[1:], "o-")

        plt.tight_layout()
        plt.show()
        """

    if params["read_temps"]:
        load_file = "py/output_temps.csv"
        data = np.genfromtxt(fp + load_file, delimiter=",", skip_header=1)
        temps = data[:, 1]
        temp_diff = temps[1:] - temp_new[1:]
        pct_diff = temp_diff / temps[1:] * 100.0
        plt.figure(figsize=(12, 6))
        plt.plot(pct_diff, -x[1:] / 1000, "k-")
        plt.xlabel("Percent temperature difference")
        plt.ylabel("Depth (km)")
        plt.grid()
        plt.title("Percent difference from explicit FD solution")
        if params["display_plots"]:
            plt.show()

    if params["write_temps"]:
        print("")
        print("--- Writing temperature output to file ---")
        print("")
        temp_x_out = np.zeros([len(x), 3])
        temp_x_out[:, 0] = x
        temp_x_out[:, 1] = temp_new
        temp_x_out[:, 2] = temp_init
        savefile = os.path.join("csv", "output_temps.csv")
        np.savetxt(
            fp + savefile,
            temp_x_out,
            delimiter=",",
            header="Depth (m),Temperature (deg. C),Initial temperature (deg. C)",
            comments="",
        )
        print("- Temperature output saved to file\n  " + fp + savefile)

    # Write header in log file if needed
    if params["log_output"]:
        log_output(params, batch_mode=False)

    if params["batch_mode"] or params["log_output"]:
        # Write output to a file
        outfile = params["log_file"]

        # Define measured ages for batch output
        if len(params["obs_ahe"]) == 0:
            obs_ahe = -9999.0
            obs_ahe_stdev = -9999.0
        else:
            obs_ahe = params["obs_ahe"][0]
            obs_ahe_stdev = params["obs_ahe_stdev"][0]
        if len(params["obs_aft"]) == 0:
            obs_aft = -9999.0
            obs_aft_stdev = -9999.0
        else:
            obs_aft = params["obs_aft"][0]
            obs_aft_stdev = params["obs_aft_stdev"][0]
        if len(params["obs_zhe"]) == 0:
            obs_zhe = -9999.0
            obs_zhe_stdev = -9999.0
        else:
            obs_zhe = params["obs_zhe"][0]
            obs_zhe_stdev = params["obs_zhe_stdev"][0]
        if len(params["obs_zft"]) == 0:
            obs_zft = -9999.0
            obs_zft_stdev = -9999.0
        else:
            obs_zft = params["obs_zft"][0]
            obs_zft_stdev = params["obs_zft_stdev"][0]

        # Define misfit details for output
        if (
            len(params["obs_ahe"])
            + len(params["obs_aft"])
            + len(params["obs_zhe"])
            + len(params["obs_zft"])
            == 0
        ):
            misfit = -9999.0
            misfit_type = -9999.0
            misfit_ages = 0
        else:
            misfit_type = params["misfit_type"]
            misfit_ages = len(obs_ages)

        # Open file for writing
        with open(outfile, "a+") as f:
            # TODO: Add output of ero_options 4 and 5 here
            f.write(
                f'{t_total / myr2sec(1):.4f},{dt / yr2sec(1):.4f},{max_depth / kilo2base(1):.4f},{params["nx"]},'
                f'{params["temp_surf"]:.4f},{params["temp_base"]:.4},{params["mantle_adiabat"]},'
                f'{params["rho_crust"]:.4f},{params["removal_fraction"]:.4f},{params["removal_time"]:.4f},'
                f'{params["ero_type"]},{params["ero_option1"]:.4f},'
                f'{params["ero_option2"]:.4f},{params["ero_option3"]:.4f},{params["init_moho_depth"]:.4f},{init_moho_temp:.4f},'
                f"{init_heat_flow:.4f},{elev_list[1] / kilo2base(1):.4f},"
                f"{moho_depth / kilo2base(1):.4f},{final_moho_temp:.4f},{final_heat_flow:.4f},"
                f'{elev_list[-1] / kilo2base(1):.4f},{exhumation_magnitude:.4f},{params["ap_rad"]:.4f},{params["ap_uranium"]:.4f},'
                f'{params["ap_thorium"]:.4f},{params["zr_rad"]:.4f},{params["zr_uranium"]:.4f},'
                f'{params["zr_thorium"]:.4f},{float(corr_ahe_ages[-1]):.4f},'
                f"{ahe_temps[-1]:.4f},{obs_ahe:.4f},"
                f"{obs_ahe_stdev:.4f},{float(aft_ages[-1]):.4f},"
                f"{aft_temps[-1]:.4f},{obs_aft:.4f},"
                f"{obs_aft_stdev:.4f},{float(corr_zhe_ages[-1]):.4f},"
                f"{zhe_temps[-1]:.4f},{obs_zhe:.4f},"
                f"{obs_zhe_stdev:.4f},{float(zft_ages[-1]):.4f},"
                f"{zft_temps[-1]:.4f},{obs_zft:.4f},"
                f"{obs_zft_stdev:.4f},{misfit:.6f},{misfit_type},{misfit_ages}\n"
            )
    
    if not params["batch_mode"]:
        print("")
        print(30 * "-" + " Execution complete " + 30 * "-")
    
        #Returns misfit for inverse_mode
    if 'misfit' in locals():
            #print("- Returning misfit")
            return misfit
