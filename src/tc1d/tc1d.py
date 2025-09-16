#!/usr/bin/env python3

# Required core libraries
import csv
from .madtrax import madtrax_apatite, madtrax_zircon
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.linalg import solve
import shutil
import subprocess
import time
import warnings

# Batch mode libraries
from sklearn.model_selection import ParameterGrid

# Inverse mode libraries
import copy  # TODO: Could this be removed?
import corner  # BG: Corner plots for MCMC
import emcee  # BG: For MCMC sampling
from itertools import combinations
from mpi4py import MPI
from neighpy import NASearcher, NAAppraiser
import os
from scipy.spatial import Voronoi, voronoi_plot_2d
from schwimmbad import MPIPool
import sys  # TODO: Could this be removed?

# Versioning
import importlib.metadata

__version__ = importlib.metadata.version("tc1d")


# Exceptions
class UnstableSolutionError(Exception):
    pass


# Other classes
class Intrusion:
    def __init__(self, temperature, start_time, duration, base_depth, thickness):
        self.temperature = temperature
        self.start_time = myr2sec(start_time)
        self.duration = myr2sec(duration)
        self.base_depth = kilo2base(base_depth)
        self.thickness = kilo2base(thickness)
        self.end_time = self.start_time + self.duration


# Unit conversions
def yr2sec(time: float) -> float:
    """Converts time from years to seconds."""
    return time * 60.0 * 60.0 * 24.0 * 365.25


def myr2sec(time: float) -> float:
    """Converts time from million years to seconds."""
    return yr2sec(time) * 1.0e6


def kilo2base(value: float) -> float:
    """Converts value from kilo-units to the base unit."""
    return value * 1000.0


def milli2base(value: float) -> float:
    """Converts value from milli-units to the base unit."""
    return value / 1.0e3


def micro2base(value: float) -> float:
    """Converts value from micro-units to the base unit."""
    return value / 1.0e6


def mmyr2ms(rate: float) -> float:
    """Converts rate from mm/yr to m/s."""
    return milli2base(rate) / yr2sec(1)


def deg2rad(value: float) -> float:
    """Converts value degrees to radians."""
    return value * np.pi / 180.0


def round_to_base(x: float, base: int = 50) -> float:
    return base * round(x / base)


# Define function for calculating effective uranium concentration
def calculate_eu(uranium: float, thorium: float) -> float:
    """Calculates effective uranium concentration from U, Th inputs (Cooperdock et al., 2019)"""
    return uranium + 0.238 * thorium


def calculate_heat_flow(temperature, conductivity, dx, nstart=0, nx=1):
    """Calculates heat flow in W/m2."""
    return kilo2base(
        (conductivity[nstart] + conductivity[nstart + nx])
        / 2
        * (temperature[nstart + nx] - temperature[nstart])
        / (nx * dx)
    )


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
        7: "Convergent/extensional exhumation",
    }
    print(f"- Erosion model: {ero_models[ero_type]}")
    print(f"- Total erosional exhumation: {exhumation_magnitude:.2f} km")


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
        raise UnstableSolutionError(
            f"Heat conduction solution unstable: {cond_stab:.3f} > {cond_crit:.4f}. Decrease nx or dt."
        )
    adv_stab = vx * dt / dx
    if adv_stab >= adv_crit:
        raise UnstableSolutionError(
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
    heat_prod_decay_depth,
    heat_prod_mantle,
    heat_prod,
    temp_adiabat,
    temp_prev,
    k_a,
    delaminated,
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
    if removal_fraction > 0.0 and delaminated:
        lab_depth = xstag[temp_stag >= temp_adiabat].min()
    else:
        lab_depth = x.max()

    heat_prod[:] = heat_prod_crust
    # Make exponential decay in heat production in crust, if enabled
    if heat_prod_decay_depth > 0.0:
        heat_prod *= np.exp(-x / heat_prod_decay_depth)
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
    # Use exponential decay in heat production, if enabled
    if params["heat_prod_decay_depth"] > 0.0:
        heat_prod *= np.exp(-x / kilo2base(params["heat_prod_decay_depth"]))
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
        a_matrix[ix, ix] = (rho[ix] * cp[ix]) / dt + k[ix] / dx**2 + k[ix - 1] / dx**2
        a_matrix[ix, ix + 1] = (rho[ix] * cp[ix] * -vx[ix]) / (2 * dx) - k[ix] / dx**2
        b[ix] = heat_prod[ix] + ((rho[ix] * cp[ix]) / dt) * temp_prev[ix]

    temp = solve(a_matrix, b)
    return temp


def create_intrusion(temperature, start_time, base_depth, thickness, duration=0.0):
    """Creates an instance of an intrusion."""
    intrusion = Intrusion(temperature, start_time, duration, base_depth, thickness)
    return intrusion


def apply_intrusion(intrusion, x, model_temperatures):
    """Applies intrusion temperatures to the intrusion depths."""
    intrusion_slice = (x >= intrusion.base_depth - intrusion.thickness) & (
        x <= intrusion.base_depth
    )
    model_temperatures[intrusion_slice] = intrusion.temperature
    return model_temperatures


def check_execs():
    """Checks whether all required executables exist."""

    # Check that executables are in $PATH
    for executable in ("RDAAM_He", "ketch_aft"):
        exec_path = shutil.which(executable)
        if exec_path is None:
            raise FileNotFoundError(
                f"{executable} executable not found. See instructions at https://github.com/HUGG/Tc_core to fix this."
            )

    return None


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

    # Run executable to calculate age
    exec_path = shutil.which("RDAAM_He")
    command = (
        exec_path
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
    if retval != 0:
        print(f"RDAAM_He execution failed with return code: {retval}!")

    return ahe_age, corr_ahe_age, zhe_age, corr_zhe_age


def ft_ages(file, write_track_lengths):
    """Calculates AFT ages."""

    # Run executable to calculate age
    exec_path = shutil.which("ketch_aft")
    command = exec_path + " " + file
    if write_track_lengths:
        command += " 1"
    p = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )

    stdout = p.stdout.readlines()
    if stdout[0] == b"usage: ketch_aft tT_file\n":
        raise RuntimeError(
            "Incompatible version of Tc_core. Must be >=0.2. See https://github.com/HUGG/Tc_core."
        )
    aft_age = stdout[0].split()[4][:-1].decode("UTF-8")
    mean_ft_length = stdout[0].split()[9][:-1].decode("UTF-8")

    retval = p.wait()
    if retval != 0:
        print(f"ketch_aft execution failed with return code: {retval}!")

    return aft_age, mean_ft_length


def calculate_closure_temp(age, time_history, temp_history):
    """Calculates closure temperature from predicted age and cooling history."""
    closure_temp = np.interp(age, time_history, temp_history)
    return closure_temp


def tt_hist_to_ma(time_history):
    """Converts a time-temperature history in Myr since model start to Ma."""
    current_max_time = time_history.max()
    time_ma = current_max_time - time_history
    time_ma = time_ma / myr2sec(1)
    return time_ma


def get_write_increment(params, time_ma):
    """Determines frequency for output in tt and ttd histories."""
    if len(time_ma) > 1000.0:
        write_increment = int(round(len(time_ma) / 100, 0))
    elif len(time_ma) > 100.0:
        write_increment = int(round(len(time_ma) / 10, 0))
    else:
        write_increment = 2
    # Use highest possible density of points in thermal history for ero_types 4 and 5
    # Gradients are very high following thrust emplacement or tectonic exhumation
    if (params["ero_type"] == 4) or (params["ero_type"] == 5):
        write_increment = len(time_ma) // 1000 + 1

    return write_increment


def write_tt_history(params, tt_filename, time_history, temp_history):
    """Writes a time-temperature history to a file."""
    # Get time history in Ma
    time_ma = tt_hist_to_ma(time_history)

    # Get time-temp history output increment
    write_increment = get_write_increment(params, time_ma)

    # Write time-temperature history file
    with open(tt_filename, "w") as csvfile:
        writer = csv.writer(csvfile, delimiter=",", lineterminator="\n")

        # Write time-temperature history in reverse order!
        for i in range(-1, -(len(time_ma) + 1), -write_increment):
            writer.writerow([time_ma[i], temp_history[i]])

        # Write fake times if time history padding is enabled
        if params["pad_time"] > 0.0:
            # Make array of pad times with 1.0 Myr time increments
            pad_times = np.arange(
                time_ma.max(),
                time_ma.max() + params["pad_time"] + 0.1,
                1.0,
            )
            for pad_time in pad_times:
                writer.writerow([pad_time, temp_history[i]])


def write_ttdp_history(
    params, ttdp_filename, time_history, temp_history, depth_history, pressure_history
):
    """Writes a time-temperature-depth-pressure history to a file."""
    # Get time history in Ma
    time_ma = tt_hist_to_ma(time_history)

    # Get time-temp history output increment
    write_increment = get_write_increment(params, time_ma)

    with open(ttdp_filename, "w") as csvfile:
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


def calculate_ages_and_tcs(
    params,
    time_history,
    temp_history,
    depth_history,
    pressure_history,
    tt_filename,
    ttdp_filename,
    write_track_lengths,
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

    # Get time history in Ma
    time_ma = tt_hist_to_ma(time_history)

    # Calculate AFT age using MadTrax
    if params["madtrax_aft"]:
        aft_age, _, _, _ = madtrax_apatite(
            time_ma, temp_history, len(time_ma), 1, params["madtrax_aft_kinetic_model"]
        )

    # Calculate ZFT age using MadTrax
    # FIXME: out_flag temporarily set to zero!
    zft_age, _, _, _ = madtrax_zircon(
        time_ma, temp_history, params["madtrax_zft_kinetic_model"], 0
    )

    # Write time-temperature history to file for (U-Th)/He, Ketcham AFT age calculation
    write_tt_history(params, tt_filename, time_history, temp_history)

    # Write pressure-time-temperature-depth history to file for reference
    write_ttdp_history(
        params,
        ttdp_filename,
        time_history,
        temp_history,
        depth_history,
        pressure_history,
    )

    ahe_age, corr_ahe_age, zhe_age, corr_zhe_age = he_ages(
        file=tt_filename,
        ap_rad=params["ap_rad"],
        ap_uranium=params["ap_uranium"],
        ap_thorium=params["ap_thorium"],
        zr_rad=params["zr_rad"],
        zr_uranium=params["zr_uranium"],
        zr_thorium=params["zr_thorium"],
    )
    if params["ketch_aft"]:
        aft_age, aft_mean_ftl = ft_ages(tt_filename, write_track_lengths)

    # Find effective closure temperatures
    ahe_temp = calculate_closure_temp(
        float(corr_ahe_age), np.flip(time_ma), np.flip(temp_history)
    )
    aft_temp = calculate_closure_temp(
        float(aft_age), np.flip(time_ma), np.flip(temp_history)
    )
    zhe_temp = calculate_closure_temp(
        float(corr_zhe_age), np.flip(time_ma), np.flip(temp_history)
    )
    zft_temp = calculate_closure_temp(
        float(zft_age), np.flip(time_ma), np.flip(temp_history)
    )

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
    params,
    dt,
    t_total,
    current_time,
    x,
    vx_array,
    fault_depth,
    moho_depth,
    fw_reference_frame,
):
    """Defines the way in which erosion should be applied.

    Erosion model types:

    1. Constant erosion rate
    2. Constant rate with a step-function change at a specified time
    3. Exponential decay
    4. Emplacement and erosional removal of a thrust sheet
    5. Tectonic exhumation and erosion
    6. Linear increase in erosion rate from a specified starting time
    7. Convergent/extensional tectonics

    Parameters
    ----------
    params : dict
        Dictionary of model parameters.
    dt : numeric
        Model time step in years.
    t_total : numeric
        Total model run time in Myr.
    current_time : numeric
        Current time in the model.
    x : numpy array
        Model spatial coordinates (depths)
    vx_array : numpy array
        Array of velocities across the model depth range.
    fault_depth : numeric
        Fault depth used for erosion type 7.
    moho_depth : numeric
        Moho depth.
    fw_reference_frame : Boolean
        Reference frame for erosion_type 7.

    Returns
    -------
    vx_array : numpy array
        Array of velocities across the model depth range.
    vx_surf : numeric
        Velocity at the model surface.
    vx_max : numeric
        Magnitude of the maximum velocity in the model.
    fault_depth : numeric
        Fault depth used for erosion type 7.
    """

    # Split the code below into separate functions?
    # Could have tests integrated more easily that way.

    # Constant erosion rate
    # Convert to inputting rate directly?
    if params["ero_type"] == 1:
        vx_array[:] = kilo2base(params["ero_option1"]) / t_total
        vx_surf = vx_array[0]
        vx_max = vx_surf

    # BG: Extended step-function erosion model to support multiple intervals
    # Collect erosion thicknesses (odd options) and transition times (even options)
    # TODO: Can this be simplified?
    elif params["ero_type"] == 2:
        thicknesses = []
        transition_times_sec = []
        total_time_Myr = (
            params["t_total"]
            if isinstance(params["t_total"], float)
            else params["t_total"][0]
        )
        # Ensure total_time_Myr is in Myr (params["t_total"] is likely already in Myr here)
        for opt_idx in range(1, 11, 2):  # Check ero_option1,3,5,7,9
            thick = params.get(f"ero_option{opt_idx}", 0.0)
            trans_idx = opt_idx + 1
            # Stop if no corresponding transition or transition is 0 (end of intervals)
            if trans_idx > 10:
                thicknesses.append(thick)
                break
            trans_time = params.get(f"ero_option{trans_idx}", 0.0)
            # If transition time is zero or beyond total time, this is the final interval
            if (not trans_time) or (trans_time >= total_time_Myr - 1e-8):
                thicknesses.append(thick)
                break
            # Otherwise, record this interval and continue
            thicknesses.append(thick)
            transition_times_sec.append(myr2sec(trans_time))
        # Compute erosion rates (m/s) for each interval
        rates = []
        prev_time = 0.0
        for j, thick_km in enumerate(thicknesses):
            if j < len(transition_times_sec):
                interval_duration = transition_times_sec[j] - prev_time
                prev_time = transition_times_sec[j]
            else:
                interval_duration = t_total - prev_time  # t_total is in seconds
            # Convert thickness (km) to m, then rate = thickness / duration
            rates.append(
                kilo2base(thick_km) / interval_duration
                if interval_duration > 0
                else 0.0
            )
        # Assign velocity according to current time
        # current_time is the simulation time (seconds) passed into this function
        applied_rate = rates[-1]  # default to last rate
        for k, t_sec in enumerate(transition_times_sec):
            if current_time < t_sec:
                applied_rate = rates[k]
                break
        vx_array[:] = applied_rate
        vx_surf = vx_array[0]
        vx_max = max(abs(r) for r in rates)

    # Exponential erosion rate decay with a set characteristic time
    # Convert to inputting rate directly?
    elif params["ero_type"] == 3:
        erosion_magnitude = kilo2base(params["ero_option1"])
        decay_time = myr2sec(params["ero_option2"])
        ero_start = myr2sec(params["ero_option3"])
        max_rate = erosion_magnitude / (
            decay_time
            * (np.exp(0.0 / decay_time) - np.exp(-(t_total - ero_start) / decay_time))
        )
        if current_time < ero_start:
            vx_array[:] = 0.0
        else:
            vx_array[:] = max_rate * np.exp(-(current_time - ero_start) / decay_time)
        vx_surf = vx_array[0]
        vx_max = max_rate

    # Emplacement and erosional removal of a thrust sheet
    elif params["ero_type"] == 4:
        # Calculate erosion magnitude
        erosion_magnitude = kilo2base(params["ero_option1"] + params["ero_option2"])
        ero_start = myr2sec(params["ero_option4"])
        if current_time < ero_start:
            vx_array[:] = 0.0
        else:
            vx_array[:] = erosion_magnitude / (t_total - ero_start)
        vx_surf = vx_array[0]
        vx_max = erosion_magnitude / (t_total - ero_start)

    # Emplacement and erosional removal of a thrust sheet
    elif params["ero_type"] == 5:
        # Calculate erosion magnitude
        erosion_magnitude = kilo2base(params["ero_option2"])
        ero_start = myr2sec(params["ero_option4"])
        if current_time < ero_start:
            vx_array[:] = 0.0
        else:
            vx_array[:] = erosion_magnitude / (t_total - ero_start)
        vx_surf = vx_array[0]
        vx_max = erosion_magnitude / (t_total - ero_start)

    # Linear change in erosion rate from a starting rate/time until an ending time
    elif params["ero_type"] == 6:
        init_rate = mmyr2ms(params["ero_option1"])
        rate_change_start = myr2sec(params["ero_option2"])
        final_rate = mmyr2ms(params["ero_option3"])
        if abs(params["ero_option4"]) <= 1.0e-8:
            rate_change_end = t_total
        else:
            rate_change_end = myr2sec(params["ero_option4"])
        if current_time < rate_change_start:
            vx_array[:] = init_rate
        elif current_time < rate_change_end:
            vx_array[:] = init_rate + (current_time - rate_change_start) / (
                rate_change_end - rate_change_start
            ) * (final_rate - init_rate)
        else:
            vx_array[:] = final_rate
        vx_surf = vx_array[0]
        vx_max = max(init_rate, final_rate)

    # Convergent / extensional tectonic model
    elif params["ero_type"] == 7:
        init_rate = mmyr2ms(params["ero_option5"])
        rate_change_time1 = myr2sec(params["ero_option6"])
        final_rate = mmyr2ms(params["ero_option7"])
        if abs(params["ero_option8"]) <= 1.0e-8:
            rate_change_time2 = t_total
        else:
            rate_change_time2 = myr2sec(params["ero_option8"])
        if current_time <= rate_change_time1:
            vx_array[:] = init_rate
            vx_surf = vx_array[0]
            fault_depth -= vx_array[0] * dt
        elif current_time <= rate_change_time2:
            slip_velocity = mmyr2ms(params["ero_option1"])
            part_factor = abs(params["ero_option2"])
            dip_angle = deg2rad(params["ero_option3"])
            # Test if fault depth is above free surface
            hw_velo = -(1 - part_factor) * slip_velocity * np.sin(dip_angle)
            fw_velo = part_factor * slip_velocity * np.sin(dip_angle)
            if fault_depth <= 0.0:
                vx_array[:] = fw_velo
            # Test if fault depth is below model base
            elif fault_depth > kilo2base(params["max_depth"]):
                vx_array[:] = hw_velo
            # Catch case that fault is within model thickness
            else:
                vx_array[x <= fault_depth] = hw_velo
                vx_array[x > fault_depth] = fw_velo
            vx_surf = vx_array[0]
            if fw_reference_frame:
                fault_depth -= fw_velo * dt
            else:
                fault_depth -= hw_velo * dt
        else:
            vx_array[:] = final_rate
            vx_surf = vx_array[0]
            fault_depth -= vx_array[0] * dt
        vx_max = max(abs(vx_array))

    # Catch bad cases
    else:
        raise ValueError(
            f"Bad erosion type: {params['ero_type']}. Must be between 1 and 7."
        )

    # Set velocities below Moho to 0.0 if using crustal uplift only
    if params["crustal_uplift"]:
        vx_array[x > moho_depth] = 0.0

    return vx_array, vx_surf, vx_max, fault_depth


def calculate_exhumation_magnitude(
    ero_type,
    ero_option1,
    ero_option2,
    ero_option3,
    ero_option4,
    ero_option5,
    ero_option6,
    ero_option7,
    ero_option8,
    ero_option9,
    ero_option10,
    t_total,
):
    """Calculates erosion magnitude in kilometers."""

    # Initialize fw_ref_frame Boolean for ero types != 7
    fw_ref_frame = False

    # Constant erosion rate
    if ero_type == 1:
        magnitude = ero_option1

    elif ero_type == 2:
        magnitude = ero_option1 + ero_option3 + ero_option5 + ero_option7 + ero_option9

    elif ero_type == 3:
        magnitude = ero_option1

    elif ero_type == 4:
        magnitude = ero_option1 + ero_option2

    elif ero_type == 5:
        magnitude = ero_option2

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
        magnitude /= kilo2base(1.0)

    elif ero_type == 7:
        # Initial exhumation phase, if applicable
        magnitude1 = myr2sec(ero_option6) * mmyr2ms(ero_option5)
        # Handle case that ero_option8 is not specified (i.e., second phase of constant exhumation)
        if abs(ero_option8) <= 1.0e-8:
            rate_change_time2 = t_total
        else:
            rate_change_time2 = myr2sec(ero_option8)
        # Convergent / extensional tectonics phase
        magnitude_hw = (rate_change_time2 - myr2sec(ero_option6)) * (
            -(1 - abs(ero_option2))
            * mmyr2ms(ero_option1)
            * np.sin(deg2rad(ero_option3))
        )
        magnitude_fw = (rate_change_time2 - myr2sec(ero_option6)) * (
            abs(ero_option2) * mmyr2ms(ero_option1) * np.sin(deg2rad(ero_option3))
        )
        # Final exhumation phase, if applicable
        magnitude3 = (t_total - rate_change_time2) * mmyr2ms(ero_option7)

        # Determine amount of hanging wall and footwall exhumation in total
        fw_total_magnitude = magnitude1 + magnitude_fw + magnitude3
        hw_total_magnitude = magnitude1 + magnitude_hw + magnitude3

        # Raise exception if using compression and a forbidden initial fault depth
        if ero_option1 < 0.0:
            if fw_total_magnitude <= kilo2base(ero_option4) <= hw_total_magnitude:
                raise ValueError(
                    f"Impossible value for initial fault depth (ero_option4). Must be less than {fw_total_magnitude / kilo2base(1.0):.1f} or greater than {hw_total_magnitude / kilo2base(1.0):.1f}."
                )

        # Check the amount of hanging wall exhumation for convergent models
        if ero_option1 < 0.0:
            magnitude = magnitude1 + magnitude_hw + magnitude3
        # Otherwise do the same for the footwall for extension
        else:
            magnitude = magnitude1 + magnitude_fw + magnitude3

        # If exhumation magnitude exceeds initial fault depth, define exhumation magnitude using footwall
        # Note: if a negative value is given for the partitioning factor in extension, initial fault locations
        #       that would normally be having footwall exhumation will use hanging wall kinematics!
        # TODO: Check whether this logic should be tweaked to prevent incorrect behavior if users define initial
        #       fault depths outside the zone of multiple kinematics being possible!
        if magnitude > kilo2base(ero_option4) and not (
            (ero_option1 > 0.0) and (ero_option2 < 0.0)
        ):
            magnitude = magnitude1 + magnitude_fw + magnitude3
            fw_ref_frame = True
        # Otherwise, use the hanging wall
        else:
            magnitude = magnitude1 + magnitude_hw + magnitude3
            fw_ref_frame = False
        magnitude /= kilo2base(1.0)

    else:
        raise ValueError(f"Bad erosion type: {ero_type}. Must be between 1 and 7.")

    # Return values in km
    return magnitude, fw_ref_frame


def calculate_pressure(density, dx, g=9.81):
    """Calculates lithostatic pressure"""
    pressure = np.zeros(len(density))
    for i in range(1, len(density)):
        pressure[i] = pressure[i - 1] + density[i] * g * dx

    return pressure


def update_density(rho, alphav, temperature):
    """Calculates density including thermal expansion"""
    rho_prime = -rho * alphav * temperature
    updated_rho = rho + rho_prime

    return updated_rho


def calculate_isostatic_elevation(
    density, x, dx, moho_depth, astheno_density, max_depth
):
    """Calculates elevation of surface due to isostasy"""
    new_mass = 0.0
    for i in range(len(density) - 1):
        incremental_density = density[i]
        # Blend materials when the Moho lies between two nodes
        if x[i] <= moho_depth < x[i + 1]:
            crust_frac = (moho_depth - x[i]) / dx
            mantle_frac = 1.0 - crust_frac
            incremental_density = crust_frac * density[i] + mantle_frac * density[i + 1]
        new_mass += incremental_density * dx

    height = new_mass / astheno_density
    elevation = max_depth - height

    return elevation


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
    fp = Path(__file__).parent / "melts_data" / compositions[composition]
    comp_data = np.genfromtxt(fp, delimiter=",")

    # Create interpolation function for composition
    # crust_interp = RectBivariateSpline(
    #     comp_data[2:, 0], comp_data[0, 1:], comp_data[2:, 1:], kx=1, ky=1
    # )

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


def plot_predictions_no_data(
    x, y, xerr=0.0, ax=None, marker="o", color="tab:blue", label=""
):
    """Creates an errorbar plot of predicted ages with no measurements to plot."""
    ax.errorbar(
        x, y, xerr=xerr, marker=marker, color=color, linestyle="None", label=label
    )
    return ax


def plot_predictions_with_data(x, y, ax=None, marker="o", color="tab:blue", label=""):
    """Creates a plot of predicted ages when measurements are also plotted."""
    ax.plot(x, y, marker=marker, color=color, linestyle="None", label=label)
    return ax


def plot_measurements(x, y, xerr=0.0, ax=None, marker="o", color="tab:blue", label=""):
    """"""
    ax.errorbar(
        x,
        y,
        xerr=xerr,
        marker=marker,
        color="white",
        markeredgecolor=color,
        markeredgewidth=1.5,
        ecolor=color,
        linestyle="None",
        label=label,
    )
    return ax


# Function for reading age data file
def read_age_data_file(file, params):
    """
    Read in age data from a csv file and store sample data.

    Parameters
    ----------
    file : Path object or string
        A character string with the relative path of the age data file.
    params : dict
        Dictionary of model parameters.

    Returns
    -------
    age_type : numpy.ndarray
        A list containing the age types for all samples/grains.
    age : numpy.ndarray
        A list containing the age values for all samples/grains.
    uncertainty : numpy.ndarray
        A list containing the uncertainty values for all samples/grains.
    u : numpy.ndarray
        A list containing the uranium values for all samples/grains.
    th : numpy.ndarray
        A list containing the thorium values for all samples/grains.
    radius : numpy.ndarray
        A list containing the equivalent spherical grain radius for all samples/grains.
    sample_id : numpy.ndarray
        A list containing the sample ID data.
    depo_age : numpy.ndarray
        A list containing the depositional age data.
    """
    # Read in data file and create nested lists of values
    with open(file, "r") as file:
        data = file.read().splitlines()

        # Create NumPy arrays for file data
        data_len = len(data) - 1
        age_type = np.empty(data_len, dtype="<U6")
        age = np.zeros(data_len, dtype=float)
        uncertainty = np.zeros(data_len, dtype=float)
        u = np.zeros(data_len, dtype=float)
        th = np.zeros(data_len, dtype=float)
        radius = np.zeros(data_len, dtype=float)
        sample_id = np.empty(data_len, dtype="<U50")
        depo_age = np.zeros(data_len, dtype=float)

        for i in range(1, len(data)):
            np_index = i - 1
            # Split lines by commas
            data[i] = data[i].split(",")
            # Ensure the correct number of items are on each line
            line_items = 7
            if len(data[i]) != line_items:
                raise ValueError(f"Each line should contain {line_items} values.")
            # Strip whitespace
            data[i] = [line.strip() for line in data[i]]
            # Append measured age data to lists, starting with age type
            if data[i][0].lower() == "ahe":
                age_type[np_index] = "AHe"
            elif data[i][0].lower() == "aft":
                age_type[np_index] = "AFT"
            elif data[i][0].lower() == "zhe":
                age_type[np_index] = "ZHe"
            elif data[i][0].lower() == "zft":
                age_type[np_index] = "ZFT"
            else:
                warnings.warn(
                    f"Unsupported age type ({data[i][0].lower()}) on age data file line {i + 1}.",
                    stacklevel=2,
                )
            age[np_index] = float(data[i][1])
            uncertainty[np_index] = float(data[i][2])
            u[np_index] = -1.0
            th[np_index] = -1.0
            radius[np_index] = -1.0

            if (data[i][0].lower() == "ahe") or (data[i][0].lower() == "zhe"):
                # Append eU value as U if it exists, use defaults if missing (keeps list lengths consistent)
                if len(data[i][3]) > 0:
                    u[np_index] = float(data[i][3])
                    th[np_index] = 0.0
                else:
                    warnings.warn(
                        f"No eU value provided for observed {data[i][0]} age on data file line {i + 1}.",
                        stacklevel=2,
                    )
                    if data[i][0].lower() == "ahe":
                        u[np_index] = float(params["ap_uranium"])
                        th[np_index] = float(params["ap_thorium"])
                    elif data[i][0].lower() == "zhe":
                        u[np_index] = float(params["zr_uranium"])
                        th[np_index] = float(params["zr_thorium"])
                    print(
                        f"         Using default U ({u[np_index]:.1f} ppm) and Th ({th[np_index]:.1f} ppm) values."
                    )
                # Append radius value if it exists, use default if missing (keeps list lengths consistent)
                if len(data[i][4]) > 0:
                    radius[np_index] = float(data[i][4])
                else:
                    warnings.warn(
                        f"No grain radius value provided for observed {data[i][0]} age on data file line {i + 1}.",
                        stacklevel=2,
                    )
                    if data[i][0].lower() == "ahe":
                        radius[np_index] = float(params["ap_rad"])
                    elif data[i][0].lower() == "zhe":
                        radius[np_index] = float(params["zr_rad"])
                    print(
                        f"         Using default radius ({radius[np_index]:.1f} um) value."
                    )
            # Append sample ID to list
            if len(data[i][5]) > 0:
                sample_id[np_index] = data[i][5]
            else:
                sample_id[np_index] = ""
            # Append depositional age to list
            if len(data[i][6]) > 0:
                depo_age[np_index] = float(data[i][6])
            else:
                depo_age[np_index] = 0.0

        # Sort arrays in reverse order using depositional ages (if there are nonzero depo ages)
        if depo_age.any() > 0.0:
            rev_depo_indices = np.argsort(depo_age)[::-1]
            age_type = age_type[rev_depo_indices]
            age = age[rev_depo_indices]
            uncertainty = uncertainty[rev_depo_indices]
            u = u[rev_depo_indices]
            th = th[rev_depo_indices]
            radius = radius[rev_depo_indices]
            sample_id = sample_id[rev_depo_indices]
            depo_age = depo_age[rev_depo_indices]

    return age_type, age, uncertainty, u, th, radius, sample_id, depo_age


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
    run_type="forward",
    length=125.0,
    nx=251,
    time=50.0,
    dt=5000.0,
    init_moho_depth=50.0,
    crustal_uplift=False,
    fixed_moho=False,
    removal_fraction=0.0,
    removal_start_time=0.0,
    removal_end_time=-1.0,
    rho_crust=2850.0,
    cp_crust=800.0,
    k_crust=2.75,
    heat_prod_crust=0.5,
    heat_prod_decay_depth=-1.0,
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
    intrusion_temperature=750.0,
    intrusion_start_time=-1.0,
    intrusion_duration=-1.0,
    intrusion_thickness=-1.0,
    intrusion_base_depth=-1.0,
    vx_init=0.0,
    ero_type=1,
    ero_option1=0.0,
    ero_option2=0.0,
    ero_option3=0.0,
    ero_option4=0.0,
    ero_option5=0.0,
    ero_option6=0.0,
    ero_option7=0.0,
    ero_option8=0.0,
    ero_option9=0.0,
    ero_option10=0.0,
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
    obs_age_file="",
    misfit_num_params=0,
    misfit_type=1,
    plot_results=True,
    display_plots=True,
    plot_ma=True,
    plot_depth_history=False,
    plot_fault_depth_history=False,
    invert_tt_plot=False,
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
    write_age_output=False,
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
    run_type : str, default="forward"
        Define type of run: forward, batch, na, or mcmc.
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
    removal_start_time : float or int, default=0.0
        Timing of start of removal of lithospheric mantle in Myr from start of simulation.
    removal_end_time : float or int, default=-1.0
        Timing of end of removal of lithospheric mantle in Myr from start of simulation.
    rho_crust : float or int, default=2850.0
        Crustal density in kg/m^3.
    cp_crust : float or int, default=800.0
        Crustal heat capacity in J/kg/K.
    k_crust : float or int, default=2.75
        Crustal thermal conductivity in W/m/K.
    heat_prod_crust : float or int, default=0.5
        Crustal heat production in uW/m^3.
    heat_prod_decay_depth : float or int, default=-1.0
        Crustal heat production decay depth in km.
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
    intrusion_temperature : float or int, default=750.0
        Intrusion temperature in deg. C.
    intrusion_start_time : float or int, default=-1.0
        Time for when magmatic intrusion becomes active in Myr.
    intrusion_duration : float or int, default=-1.0
        Duration for which a magmatic intrusion is active in Myr.
    intrusion_thickness : float or int, default=-1.0
        Thickness of magmatic intrusion in km.
    intrusion_base_depth : float or int, default=-1.0
        Depth of base of intrusion in km.
    vx_init : float or int, default=0.0
        Initial steady-state advection velocity in mm/yr.
    ero_type : int, default=1
        Type of erosion model (1, 2, 3, 4, 5, 6, 7 - see https://tc1d.readthedocs.io/en/latest/erosion-models.html).
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
    ero_option6 : float or int, default=0.0
        Erosion model option 6 (see https://tc1d.readthedocs.io/en/latest/erosion-models.html).
    ero_option7 : float or int, default=0.0
        Erosion model option 7 (see https://tc1d.readthedocs.io/en/latest/erosion-models.html).
    ero_option8 : float or int, default=0.0
        Erosion model option 8 (see https://tc1d.readthedocs.io/en/latest/erosion-models.html).
    ero_option9 : float or int, default=0.0
        Erosion model option 8 (see https://tc1d.readthedocs.io/en/latest/erosion-models.html).
    ero_option10 : float or int, default=0.0
        Erosion model option 8 (see https://tc1d.readthedocs.io/en/latest/erosion-models.html).
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
    pad_time : float or int, default=0.0
        Additional time added at starting temperature in t-T history in Myr.
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
    obs_age_file : str, default=""
        CSV file containing measured ages.
    misfit_num_params : int, default=0
        Number of model parameters to use in misfit calculation. Only applies to misfit type 2.
    misfit_type : int, default=1
        Misfit type for misfit calculation.
    plot_results : bool, default=True
        Plot calculated results.
    display_plots : bool, default=True
        Display plots on screen.
    plot_ma : bool, default=True
        Plot time in Ma rather than Myr from start of model.
    plot_depth_history : bool, default=False
        Plot depth history on thermal history plot.
    plot_fault_depth_history : bool, default=False
        Plot fault depth history on thermal history plot.
    invert_tt_plot : bool, default=False
        Invert depth/temperature axis on thermal history plot.
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
    write_age_output : bool, default=False
        Save predicted and observed ages to a file.
    write_past_ages : bool, default=False
        Write out incremental past ages to csv file.
    save_plots : bool, default=False
        Save plots to a file.
    read_temps : bool, default=False
        Read temperatures from a file.
    compare_temps : bool, default=False
        Compare model temperatures to those from a file.

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
        "plot_ma": plot_ma,
        "plot_depth_history": plot_depth_history,
        "plot_fault_depth_history": plot_fault_depth_history,
        "invert_tt_plot": invert_tt_plot,
        "run_type": run_type,
        # Batch mode not supported when called as a function
        "batch_mode": False,
        # Inverse mode not supported when called as a function
        "inverse_mode": False,
        "mantle_adiabat": mantle_adiabat,
        "implicit": implicit,
        "read_temps": read_temps,
        "compare_temps": compare_temps,
        "write_temps": write_temps,
        "write_age_output": write_age_output,
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
        "removal_start_time": removal_start_time,
        "removal_end_time": removal_end_time,
        "crustal_uplift": crustal_uplift,
        "fixed_moho": fixed_moho,
        "ero_type": ero_type,
        "ero_option1": ero_option1,
        "ero_option2": ero_option2,
        "ero_option3": ero_option3,
        "ero_option4": ero_option4,
        "ero_option5": ero_option5,
        "ero_option6": ero_option6,
        "ero_option7": ero_option7,
        "ero_option8": ero_option8,
        "ero_option9": ero_option9,
        "ero_option10": ero_option10,
        "temp_surf": temp_surf,
        "temp_base": temp_base,
        "t_total": time,
        "dt": dt,
        "vx_init": vx_init,
        "rho_crust": rho_crust,
        "cp_crust": cp_crust,
        "k_crust": k_crust,
        "heat_prod_crust": heat_prod_crust,
        "heat_prod_decay_depth": heat_prod_decay_depth,
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
        "obs_age_file": obs_age_file,
        "misfit_num_params": misfit_num_params,
        "misfit_type": misfit_type,
        "log_output": log_output,
        "log_file": log_file,
        "model_id": model_id,
        "intrusion_temperature": intrusion_temperature,
        "intrusion_start_time": intrusion_start_time,
        "intrusion_duration": intrusion_duration,
        "intrusion_thickness": intrusion_thickness,
        "intrusion_base_depth": intrusion_base_depth,
    }

    return params


def create_output_directory(wd, dir=""):
    """Creates a new directory in working directory."""
    newdir = wd / dir
    newdir.mkdir(parents=True, exist_ok=True)
    return newdir


def prep_model(params):
    """Prepares models to be run as single models or in batch mode.

    Parameters
    ----------
    params : dict
        Dictionary of model parameter values.

    Returns
    -------
    None
    """
    # Define working directory path
    wd = Path.cwd()

    # Create needed output directories
    if (
        params["log_output"]
        or params["write_past_ages"]
        or params["write_temps"]
        or params["calc_ages"]
        or params["write_age_output"]
    ):
        create_output_directory(wd, dir="csv")
    if params["save_plots"]:
        create_output_directory(wd, dir="png")

    # Check the needed executable files exist
    check_execs()

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
        "removal_start_time",
        "removal_end_time",
        "ero_type",
        "ero_option1",
        "ero_option2",
        "ero_option3",
        "ero_option4",
        "ero_option5",
        "ero_option6",
        "ero_option7",
        "ero_option8",
        "ero_option9",
        "ero_option10",
        "mantle_adiabat",
        "rho_crust",
        "cp_crust",
        "k_crust",
        "heat_prod_crust",
        "heat_prod_decay_depth",
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
        "pad_time",
        "intrusion_temperature",
        "intrusion_start_time",
        "intrusion_duration",
        "intrusion_thickness",
        "intrusion_base_depth",
    ]

    # Create empty dictionary for batch model parameters, if any
    batch_params = {}

    # Get MPI rank for controlling screen output
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Announce version before proceeding
    if rank == 0:
        print("")
        print(f"{26 * '='} This is Tc1D version {__version__} {26 * '='}\n")

    # Check that prep_model was called from the command line
    if params["cmd_line_call"]:
        # We know all batch model values are lists, check their lengths
        # If all are 1 then run in forward mode
        params["batch_mode"] = False
        params["inverse_mode"] = False

        for key in batch_keys:
            if len(params[key]) != 1:
                params["batch_mode"] = True
            batch_params[key] = params[key]

        # Check to ensure a suitable run mode has been defined
        run_types = ["forward", "batch", "na", "mcmc"]
        if params["run_type"].lower() not in run_types:
            raise ValueError(
                f"Invalid run mode: {params['run_type']}. Must be one of {run_types}."
            )

        # Check whether inverse mode should be enabled, if not make sure batch mode is set
        if params["batch_mode"]:
            if (params["run_type"].lower() == "na") or (
                params["run_type"].lower() == "mcmc"
            ):
                params["inverse_mode"] = True
            else:
                params["run_type"] = "batch"

        # Create batch_mode output directories if using batch mode
        if params["batch_mode"]:
            create_output_directory(wd, dir="csv")

        # Now we see what to do for running the model(s)
        if params["run_type"].lower() == "forward":
            # Convert list values, run single forward model
            for key in batch_keys:
                params[key] = params[key][0]
            run_model(params)
        if params["run_type"].lower() == "batch":
            # Run batch model
            batch_run(params, batch_params)
        elif params["run_type"].lower() == "na":
            # Run NA inversion
            batch_run_na(params, batch_params)
        elif params["run_type"].lower() == "mcmc":
            # Run MCMC inversion
            batch_run_mcmc(params, batch_params)

    else:
        # If called as a function, check for lists and their lengths
        params["batch_mode"] = False
        params["inverse_mode"] = False

        for key in batch_keys:
            if isinstance(params[key], list):
                if len(params[key]) != 1:
                    params["batch_mode"] = True
                batch_params[key] = params[key]
            # TODO: Raise exception here? Batch and inverse mode not supported, right?

        # TODO: Remove stuff below except run_model()?
        # Create batch_mode output directories if using batch mode
        if params["batch_mode"]:
            create_output_directory(wd, dir="csv")

        # Now we see what to do for running the model(s)
        if not params["batch_mode"]:
            # Convert list values, run single model
            run_model(params)
        else:
            # Run in batch mode
            batch_run(params, batch_params)


def log_output(params, batch_mode=False):
    """Writes model summary output to a csv file"""
    # Define working directory path
    wd = Path.cwd()

    # Define log file name if undefined
    if params["log_file"] == "":
        if batch_mode:
            params["log_file"] = "TC1D_batch_log.csv"
        else:
            params["log_file"] = "TC1D_run_log.csv"

    # Create output file path
    outfile = wd / "csv" / params["log_file"]

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
    except FileNotFoundError:
        outfile.touch()
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
                "Crustal density (kg m^-3),Mantle removal fraction,Mantle removal start time (Ma),"
                "Mantle removal end time (Ma),Erosion model type,Erosion model option 1,"
                "Erosion model option 2,Erosion model option 3,Erosion model option 4,Erosion model option 5,"
                "Erosion model option 6,Erosion model option 7,Erosion model option 8,"
                "Erosion model option 9,Erosion model option 10,"
                "Initial Moho depth (km),Initial Moho temperature (C),"
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

    return outfile


def batch_run(params, batch_params):
    """Runs TC1D in batch mode"""
    # Define working directory path
    # wd = Path.cwd()

    # Make parameter permutation list
    param_list = list(ParameterGrid(batch_params))

    print(
        f"{19 * '-'} Starting batch processor for {len(param_list):4} models {19 * '-'}\n"
    )
    exec_start = time.time()

    # Initialize counters
    success = 0
    failed = 0

    # Loop over list of parameters and run all permutations
    for i in range(len(param_list)):
        outfile = log_output(params, batch_mode=True)
        model = param_list[i]
        print(f"Iteration {i + 1}", end="", flush=True)
        # Update model parameters
        for key in batch_params:
            params[key] = model[key]

        try:
            run_model(params)
            print("Complete")
            success += 1
        # TODO: Should make a complete list of exceptions to catch
        except ValueError:
            print("FAILED!")
            # FIXME: outfile is not defined when models crash...
            # Should this also be handled in log_output()?
            with open(outfile, "a+") as f:
                f.write(
                    f"{params['t_total']:.4f},{params['dt']:.4f},{params['max_depth']:.4f},{params['nx']},"
                    f"{params['temp_surf']:.4f},{params['temp_base']:.4f},{params['mantle_adiabat']},"
                    f"{params['rho_crust']:.4f},{params['removal_fraction']:.4f},{params['removal_start_time']:.4f},"
                    f"{params['removal_end_time']:.4f},"
                    f"{params['ero_type']},{params['ero_option1']:.4f},"
                    f"{params['ero_option2']:.4f},{params['ero_option3']:.4f},{params['ero_option4']:.4f},{params['ero_option5']:.4f},{params['ero_option6']:.4f},{params['ero_option7']:.4f},{params['ero_option8']:.4f},{params['ero_option9']:.4f},{params['ero_option10']:.4f},{params['init_moho_depth']:.4f},,,,,,,,,{params['ap_rad']:.4f},{params['ap_uranium']:.4f},"
                    f"{params['ap_thorium']:.4f},{params['zr_rad']:.4f},{params['zr_uranium']:.4f},{params['zr_thorium']:.4f},,,,,,,,,,,,,,,\n"
                )
            failed += 1

    exec_end = time.time()
    print(
        f"\n{3 * '-'} Execution completed in {exec_end - exec_start:10.4f} seconds ({success:4} succeeded, {failed:4} failed) {4 * '-'}"
    )


def batch_run_na(params, batch_params):
    """Runs TC1D in inverse mode with the Neighbourhood Algorithm"""
    # Define working directory path
    wd = Path.cwd()

    print(f"{27 * '-'} Starting NA inverse mode {27 * '-'}\n")
    exec_start = time.time()

    # Objective function to be minimised, run for misfit
    def objective(x):
        # Map sampled values x to the corresponding parameter names
        for key, value in zip(filtered_params, x):
            filtered_params[key] = value

        # BG: Get erosion parameters with default fallback from global params
        ero1 = filtered_params.get("ero_option1", params.get("ero_option1", 0.0))
        ero3_final = filtered_params.get("ero_option3", params.get("ero_option3", 0.0))
        ero5_final = filtered_params.get("ero_option5", params.get("ero_option5", 0.0))
        ero7_final = filtered_params.get("ero_option7", params.get("ero_option7", 0.0))
        ero9_final = filtered_params.get("ero_option9", params.get("ero_option9", 0.0))

        # BG: Apply constraint to ero_option3 if it is part of the inverted parameters
        # Ero option 3
        if "ero_option3" in filtered_params:
            upper_bound = max_exhumation - ero1  # cannot exhume >35 km
            lower_bound = -max_burial - ero1  # cannot bury >15 km
            ero3_final = max(lower_bound, min(ero3_final, upper_bound))
            filtered_params["ero_option3"] = ero3_final

        # Ero option 5
        if "ero_option5" in filtered_params:
            upper_bound = max_exhumation - (ero1 + ero3_final)
            lower_bound = -max_burial - (ero1 + ero3_final)
            ero5_final = max(lower_bound, min(ero5_final, upper_bound))
            filtered_params["ero_option5"] = ero5_final

        # Ero option 7
        if "ero_option7" in filtered_params:
            upper_bound = max_exhumation - (ero1 + ero3_final + ero5_final)
            lower_bound = -max_burial - (ero1 + ero3_final + ero5_final)
            ero7_final = max(lower_bound, min(ero7_final, upper_bound))
            filtered_params["ero_option7"] = ero7_final

        # Ero option 9
        if "ero_option9" in filtered_params:
            upper_bound = max_exhumation - (ero1 + ero3_final + ero5_final + ero7_final)
            lower_bound = -max_burial - (ero1 + ero3_final + ero5_final + ero7_final)
            ero9_final = max(lower_bound, min(ero9_final, upper_bound))
            filtered_params["ero_option9"] = ero9_final

        # Add bounds to parameters
        params.update(filtered_params)
        cleaned_filtered = {k: float(v) for k, v in filtered_params.items()}
        print(f"The current values are: {cleaned_filtered}")

        # BG: Cumulative constraint to prevent elevation above the surface
        cumulative_erosion = ero1 + ero3_final + ero5_final + ero7_final + ero9_final
        if cumulative_erosion < 0:
            print("Rejected: model would place the sample above surface.")
            return 1e10  # Or some large misfit to reject the model

        misfit = run_model(params)
        print(f"The current misfit is: {misfit}\n")
        return misfit

    param_list = list(ParameterGrid(batch_params))
    log_output(params, batch_mode=True)

    # FIXME: Are these needed???
    # success = 0
    # failed = 0

    # Define model limitations
    # FIXME: Could the line below be set to the initial Moho depth???
    max_exhumation = 35.0
    max_burial = 15.0

    # Starting model
    model = param_list[0]
    for key in batch_params:
        params[key] = model[key]

    # Filter params for multiple supplied values, use these as bounds for the NA
    filtered_params = {}
    for key, value in batch_params.items():
        if len(value) > 1:
            filtered_params[key] = value

    # Bounds of the parameter space
    bounds = list(filtered_params.values())

    # Initialize NA searcher
    searcher = NASearcher(
        objective,
        ns=8,  # 16 #100, # number of samples per iteration #10
        nr=4,  # 8 #10, # number of cells to resample #1
        ni=20,  # 100, # size of initial random search #1
        n=5,  # 20, # number of iterations #1
        bounds=bounds,
    )

    # Run the direct search phase
    searcher.run()  # results stored in searcher.samples and searcher.objectives

    # BG: Apply constraints after search using parameter names to avoid index errors
    for i in searcher.samples:
        param_dict = dict(zip(filtered_params.keys(), i))

        ero1 = param_dict.get("ero_option1", 0.0)
        ero3 = param_dict.get("ero_option3", 0.0)
        ero5 = param_dict.get("ero_option5", 0.0)
        ero7 = param_dict.get("ero_option7", 0.0)
        ero9 = param_dict.get("ero_option9", 0.0)

        if "ero_option3" in param_dict:
            upper = max_exhumation - ero1
            lower = -max_burial - ero1
            param_dict["ero_option3"] = max(lower, min(ero3, upper))

        if "ero_option5" in param_dict:
            upper = max_exhumation - (ero1 + param_dict.get("ero_option3", 0.0))
            lower = -max_burial - (ero1 + param_dict.get("ero_option3", 0.0))
            param_dict["ero_option5"] = max(lower, min(ero5, upper))

        if "ero_option7" in param_dict:
            upper = max_exhumation - (
                ero1
                + param_dict.get("ero_option3", 0.0)
                + param_dict.get("ero_option5", 0.0)
            )
            lower = -max_burial - (
                ero1
                + param_dict.get("ero_option3", 0.0)
                + param_dict.get("ero_option5", 0.0)
            )
            param_dict["ero_option7"] = max(lower, min(ero7, upper))

        if "ero_option9" in param_dict:
            upper = max_exhumation - (
                ero1
                + param_dict.get("ero_option3", 0.0)
                + param_dict.get("ero_option5", 0.0)
                + param_dict.get("ero_option7", 0.0)
            )
            lower = -max_burial - (
                ero1
                + param_dict.get("ero_option3", 0.0)
                + param_dict.get("ero_option5", 0.0)
                + param_dict.get("ero_option7", 0.0)
            )
            param_dict["ero_option9"] = max(lower, min(ero9, upper))

        i[:] = [param_dict[k] for k in filtered_params.keys()]

    appraiser = NAAppraiser(
        initial_ensemble=searcher.samples,  # points of parameter space already sampled
        log_ppd=-searcher.objectives,  # objective function values
        bounds=bounds,
        n_resample=2000,  # number of desired new samples #100
        n_walkers=5,  # number of parallel walkers #1
    )

    appraiser.run()  # Results stored in appraiser.samples
    print(f"Appraiser mean: {appraiser.mean}")
    print(f"Appraiser mean error: {appraiser.sample_mean_error}")
    print(f"Appraiser covariance: {appraiser.covariance}")
    print(f"Appraiser covariance error: {appraiser.sample_covariance_error}")

    # BG: Safely extract best parameter set using param names
    best = searcher.samples[np.argmin(searcher.objectives)]
    best_dict = dict(zip(filtered_params.keys(), best))
    ero1 = best_dict.get("ero_option1", 0.0)
    # FIXME: Delete line below???
    ero3 = best_dict.get("ero_option3", 0.0)

    # BG: Apply constraints only to parameters being inverted
    ero3 = best_dict.get("ero_option3", 0.0)
    ero5 = best_dict.get("ero_option5", 0.0)
    ero7 = best_dict.get("ero_option7", 0.0)
    ero9 = best_dict.get("ero_option9", 0.0)

    if "ero_option3" in best_dict:
        upper = max_exhumation - ero1
        lower = -max_burial - ero1
        best_dict["ero_option3"] = max(lower, min(ero3, upper))

    if "ero_option5" in best_dict:
        upper = max_exhumation - (ero1 + best_dict.get("ero_option3", 0.0))
        lower = -max_burial - (ero1 + best_dict.get("ero_option3", 0.0))
        best_dict["ero_option5"] = max(lower, min(ero5, upper))

    if "ero_option7" in best_dict:
        upper = max_exhumation - (
            ero1 + best_dict.get("ero_option3", 0.0) + best_dict.get("ero_option5", 0.0)
        )
        lower = -max_burial - (
            ero1 + best_dict.get("ero_option3", 0.0) + best_dict.get("ero_option5", 0.0)
        )
        best_dict["ero_option7"] = max(lower, min(ero7, upper))

    if "ero_option9" in best_dict:
        upper = max_exhumation - (
            ero1
            + best_dict.get("ero_option3", 0.0)
            + best_dict.get("ero_option5", 0.0)
            + best_dict.get("ero_option7", 0.0)
        )
        lower = -max_burial - (
            ero1
            + best_dict.get("ero_option3", 0.0)
            + best_dict.get("ero_option5", 0.0)
            + best_dict.get("ero_option7", 0.0)
        )
        best_dict["ero_option9"] = max(lower, min(ero9, upper))

    # BG: Rebuild best list in correct parameter order
    best[:] = [best_dict[k] for k in filtered_params.keys()]
    print(f" The best parameters are: {best}")

    # Write searcher results to file
    outfile = wd / "csv" / "na_searcher_results.csv"
    paramkeys = list(filtered_params.keys())
    header = paramkeys + ["misfit"]
    combined_array = np.column_stack((searcher.samples, searcher.objectives))
    with open(outfile, "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(combined_array)

    # Stop timer before displaying plots if plots are to be displayed
    if params["display_plots"]:
        exec_end = time.time()

    # Plot for misfit
    best_i = np.argmin(searcher.objectives)
    plt.plot(searcher.objectives, marker=".", linestyle="", markersize=2)
    plt.scatter(best_i, searcher.objectives[best_i], c="g", s=10, zorder=10)
    plt.axvline(searcher.ni, c="k", ls="--")
    plt.yscale("log")
    plt.text(0.05, 0.95, "Initial Search", transform=plt.gca().transAxes, ha="left")
    plt.text(
        0.95,
        0.95,
        "Neighbourhood Search",
        transform=plt.gca().transAxes,
        ha="right",
    )
    if params["save_plots"]:
        savefile = wd / "png" / "na_misfit.png"
        plt.savefig(savefile, dpi=300)
        print(f"- NA misfit plot written to {savefile}")
    if params["display_plots"]:
        plt.show()
    else:
        plt.close()

    # Plot for 2 params
    if len(bounds) == 2:
        # Other params
        x_searcher = searcher.samples[:, 0]
        y_searcher = searcher.samples[:, 1]
        # Appraiser params
        x_appraiser = appraiser.samples[:, 0]
        y_appraiser = appraiser.samples[:, 1]

        # Plot
        fig = plt.figure(constrained_layout=True)
        # Gridspec axes
        gs = fig.add_gridspec(4, 4)
        ax = fig.add_subplot(gs[1:, :-1])
        ax_histx = fig.add_subplot(gs[0, :-1], sharex=ax)
        ax_histy = fig.add_subplot(gs[1:, -1], sharey=ax)
        bestlabel = f"Best: {'{:.2f}'.format(best[0])}, {'{:.2f}'.format(best[1])}"
        # Scatterplots
        scatter2 = ax.scatter(x_appraiser, y_appraiser, color="grey", marker="x")
        scatter1 = ax.scatter(
            x_searcher,
            y_searcher,
            c=searcher.objectives,
            cmap="viridis",
            marker="x",
        )
        scatter3 = ax.scatter(best[0], best[1], color="red", marker="x")
        ax.legend(
            handles=[scatter1, scatter2, scatter3],
            labels=["Searcher samples", "Appraiser samples", bestlabel],
            loc="upper right",
        )
        ax.set_title("Neighbourhood Algorithm samples")
        ax.set_xlabel(list(filtered_params.keys())[0])
        ax.set_ylabel(list(filtered_params.keys())[1])
        fig.colorbar(scatter1, location="bottom", label="Misfit")

        # Histograms
        ax_histx.hist(x_appraiser, bins=15, color="grey")
        ax_histy.hist(y_appraiser, bins=15, color="grey", orientation="horizontal")
        if params["save_plots"]:
            savefile = wd / "png" / "na_samples.png"
            plt.savefig(savefile, dpi=300)
            print(f"- NA samples scatter plot written to {savefile}")
        if params["display_plots"]:
            plt.show()
        else:
            plt.close()

    # NA covariance matrix plot
    paramkeys = list(filtered_params.keys())
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(appraiser.covariance, interpolation="nearest")
    fig.colorbar(cax)
    x_axis = np.arange(len(paramkeys))
    ax.set_xticks(x_axis)
    ax.set_yticks(x_axis)
    ax.set_xticklabels(paramkeys)
    ax.set_yticklabels(paramkeys)
    plt.title("Covariance Matrix")
    for i in range(len(paramkeys)):
        for j in range(len(paramkeys)):
            ax.text(
                j,
                i,
                round(appraiser.covariance[i, j], 4),
                color="white",
                ha="center",
                va="center",
            )
    if params["save_plots"]:
        savefile = wd / "png" / "na_covariance_matrix.png"
        plt.savefig(savefile, dpi=300)
        print(f"- NA covariance matrix plot written to {savefile}")
    if params["display_plots"]:
        plt.show()
    else:
        plt.close()

    samples = searcher.samples
    nparams = samples.shape[1]

    if nparams == 2:
        # BG: Classic 2D Voronoi plot
        vor = Voronoi(samples)
        fig, ax = plt.subplots(figsize=(6, 6))
        voronoi_plot_2d(
            vor, ax=ax, show_vertices=False, show_points=False, line_width=0.5
        )
        ax.scatter(
            best[0], best[1], c="g", marker="x", s=100, label="Best model", zorder=10
        )
        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])
        ax.set_xlabel(list(filtered_params.keys())[0])
        ax.set_ylabel(list(filtered_params.keys())[1])
        ax.legend(loc="lower right")
        plt.tight_layout()
        if params["save_plots"]:
            savefile = wd / "png" / "na_voronoi.png"
            plt.savefig(savefile, dpi=300)
            print(f"- NA Voronoi diagram written to {savefile}")
        if params["display_plots"]:
            plt.show()
        else:
            plt.close()

    elif nparams > 2:
        # BG: Pairwise Voronoi plots for all parameter pairs (lower triangle)
        fig, axs = plt.subplots(
            nparams, nparams, figsize=(2.5 * nparams, 2.5 * nparams), tight_layout=True
        )
        plt.suptitle("Voronoi pairwise plots")
        for i in range(nparams):
            for j in range(nparams):
                if j < i:
                    vor = Voronoi(samples[:, [j, i]])
                    voronoi_plot_2d(
                        vor,
                        ax=axs[i, j],
                        show_vertices=False,
                        show_points=False,
                        line_width=0.5,
                    )
                    axs[i, j].scatter(
                        best[j],
                        best[i],
                        c="g",
                        marker="x",
                        s=100,
                        label="Best model",
                        zorder=10,
                    )
                    axs[i, j].set_xlim(searcher.bounds[j])
                    axs[i, j].set_ylim(searcher.bounds[i])
                    axs[i, j].set_xticks([])
                    axs[i, j].set_yticks([])
                    if i == nparams - 1:
                        axs[i, j].set_xlabel(paramkeys[j])
                    if j == 0:
                        axs[i, j].set_ylabel(paramkeys[i])
                else:
                    axs[i, j].set_visible(False)

        handles, labels = axs[1, 0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(
            by_label.values(),
            by_label.keys(),
            loc="lower left",
            bbox_to_anchor=(0.6, 0.25),
        )
        if params["save_plots"]:
            savefile = wd / "png" / "na_voronoi.png"
            plt.savefig(savefile, dpi=300)
            print(f"- NA Voronoi diagram written to {savefile}")
        if params["display_plots"]:
            plt.show()
        else:
            plt.close()

    # BG: Plot univariate histograms for each inverted parameter
    fig, axs = plt.subplots(nparams, 1, figsize=(5, 2.5 * nparams), tight_layout=True)
    for i in range(nparams):
        axs[i].hist(searcher.samples[:, i], bins=15, color="grey", edgecolor="black")
        axs[i].set_xlabel(paramkeys[i])
        axs[i].set_ylabel("Count")
        axs[i].set_title(f"Histogram of {paramkeys[i]}")

    plt.suptitle("Parameter distributions from NA search", y=1.02)
    if params["save_plots"]:
        savefile = wd / "png" / "na_histograms.png"
        plt.savefig(savefile, dpi=300)
        print(f"- NA histograms written to {savefile}")
    if params["display_plots"]:
        plt.show()
    else:
        plt.close()

    # Stop timer here if not displaying plots
    if not params["display_plots"]:
        exec_end = time.time()
    print(
        f"\n{17 * '-'} NA execution completed in {exec_end - exec_start:10.4f} seconds {17 * '-'}"
    )

    # FIXME: Does the line below work???
    # success += 1


# FIXME: These should be defined only when using MCMC
# BG: Global variables used by log_probability
global_bounds = None
global_param_names = None
global_params = None
global_max_exhumation = 35.0
global_max_burial = 15.0


def log_prior(x):
    for val, (low, high) in zip(x, global_bounds):
        if not (low <= val <= high):
            return -np.inf

    param_dict = dict(zip(global_param_names, x))
    ero1 = param_dict.get("ero_option1", global_params.get("ero_option1", 0.0))
    ero3 = param_dict.get("ero_option3", global_params.get("ero_option3", 0.0))
    ero5 = param_dict.get("ero_option5", global_params.get("ero_option5", 0.0))
    ero7 = param_dict.get("ero_option7", global_params.get("ero_option7", 0.0))
    ero9 = param_dict.get("ero_option9", global_params.get("ero_option9", 0.0))

    # Interval 3
    if "ero_option3" in param_dict:
        upper = global_max_exhumation - ero1
        lower = -global_max_burial - ero1
        if not (lower <= ero3 <= upper):
            return -np.inf

    # Interval 5
    if "ero_option5" in param_dict:
        upper = global_max_exhumation - (ero1 + ero3)
        lower = -global_max_burial - (ero1 + ero3)
        if not (lower <= ero5 <= upper):
            return -np.inf

    # Interval 7
    if "ero_option7" in param_dict:
        upper = global_max_exhumation - (ero1 + ero3 + ero5)
        lower = -global_max_burial - (ero1 + ero3 + ero5)
        if not (lower <= ero7 <= upper):
            return -np.inf

    # Interval 9
    if "ero_option9" in param_dict:
        upper = global_max_exhumation - (ero1 + ero3 + ero5 + ero7)
        lower = -global_max_burial - (ero1 + ero3 + ero5 + ero7)
        if not (lower <= ero9 <= upper):
            return -np.inf

    # Total thickness must remain ≥ 0 km (cannot end above surface)
    cumulative = ero1 + ero3 + ero5 + ero7 + ero9
    if cumulative < 0.0:
        return -np.inf

    return 0.0


def log_likelihood(x):
    param_dict = dict(zip(global_param_names, x))
    new_dict = {}
    for k, v in param_dict.items():
        try:
            new_dict[k] = float(v[0]) if isinstance(v, list) else float(v)
        except (ValueError, TypeError):
            print(f"[WARNING] Could not convert {k}={v} to float.")
            return -np.inf
    params_local = copy.deepcopy(global_params)
    params_local.update(new_dict)
    cleaned_dict = {k: float(v) for k, v in new_dict.items()}
    print(f"[MCMC] Testing params: {cleaned_dict}")
    try:
        misfit = run_model(params_local)
        print(f"Misfit: {misfit}")
        return -misfit
    except Exception as e:
        print(f"[ERROR] run_model failed: {e}")
        return -np.inf


def log_probability(x):
    lp = log_prior(x)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(x)
    return lp + ll


def batch_run_mcmc(params, batch_params):
    """Runs TC1D in inverse mode using MCMC"""
    # Define working directory path
    wd = Path.cwd()

    # print("--- Starting MCMC inverse mode ---\n")
    log_output(params, batch_mode=True)

    # FIXME: Are the lines below needed???
    param_list = list(ParameterGrid(batch_params))
    # print(f"--- Starting batch processor for {len(param_list)} models ---\n")
    # success = 0
    # failed = 0

    # BG: Extract parameters with more than one value (i.e., varied ones) to define search space
    filtered_params = {k: v for k, v in batch_params.items() if len(v) > 1}
    bounds = list(filtered_params.values())
    param_names = list(filtered_params.keys())
    ndim = len(param_names)  # Number of parameters to invert
    max_exhumation = 35.0  # BG: Maximum total exhumation constraint
    max_burial = 15.0

    # BG: Use the first parameter set to update the base parameters
    model = param_list[0]  # BG: Start from the first parameter combination
    for key in batch_params:
        params[key] = model[key]

    # BG: Set global variables for MPI pickling compatibility
    global \
        global_bounds, \
        global_param_names, \
        global_params, \
        global_max_exhumation, \
        global_max_burial
    global_bounds = bounds
    global_param_names = param_names
    global_params = params
    global_max_exhumation = max_exhumation
    global_max_burial = max_burial

    # BG: MCMC setup - number of walkers and initial positions sampled from uniform priors
    nwalkers = 16
    nsteps = 20
    discard = 3
    thin = 3

    p0 = [
        [np.random.uniform(low, high) for (low, high) in bounds]
        for _ in range(nwalkers)
    ]

    # Handle serial and parallel use nicely
    try:
        pool = MPIPool()
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        if pool.rank == 0:
            print(
                f"{8 * '-'} Starting MCMC inverse mode (parallel mode; {pool.size:4} MPI processes) {8 * '-'}\n"
            )
            exec_start = time.time()
    except ValueError:
        pool = None
        print(f"{16 * '-'} Starting MCMC inverse mode (single processor) {17 * '-'}\n")
        exec_start = time.time()

    # BG: Create the sampler and run the MCMC
    sampler = emcee.EnsembleSampler(
        nwalkers=nwalkers,
        ndim=ndim,
        log_prob_fn=log_probability,
        pool=pool,
    )
    sampler.run_mcmc(initial_state=p0, nsteps=nsteps, progress=True)
    if pool is not None:
        pool.close()

    # BG: Post-processing with legacy-compatible attributes
    chain = sampler.chain
    log_probs = sampler.lnprobability

    # BG: Flatten chains manually (legacy emcee version does not support .get_chain())
    flat_samples = chain[:, discard::thin, :].reshape(-1, ndim)
    flat_log_probs = log_probs[:, discard::thin].reshape(-1)

    # Write sample results to file
    outfile = wd / "csv" / "mcmc_searcher_results.csv"
    header = param_names + ["log probability"]
    combined_array = np.column_stack((flat_samples, flat_log_probs))
    with open(outfile, "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(combined_array)

    # BG: Check if any valid samples remain after burn-in
    if len(log_probs) == 0:
        raise RuntimeError("No valid samples after burn-in. Aborting analysis.")

    # BG: Identify and print the best parameter set (lowest misfit)
    best_idx = np.argmax(flat_log_probs)
    best = flat_samples[best_idx]
    best_dict = dict(zip(param_names, best))  # Manquait ici
    print(f"The best parameters are: { {k: float(v) for k, v in best_dict.items()} }")

    # Stop timer before displaying plots if plots are to be displayed
    if params["display_plots"]:
        exec_end = time.time()

    # BG: Plot evolution of misfit values
    plt.figure()
    neg_log_probs = -flat_log_probs
    plt.plot(neg_log_probs, ".", markersize=2)
    plt.scatter(best_idx, neg_log_probs[best_idx], c="g", s=10)
    plt.xlabel("Sample Index")
    plt.ylabel("Misfit")
    plt.title("MCMC Misfit Values")
    plt.yscale("log")
    if params["save_plots"]:
        savefile = wd / "png" / "mcmc_misfit.png"
        plt.savefig(savefile, dpi=300)
        print(f"- MCMC misfit plot written to {savefile}")
    if params["display_plots"]:
        plt.show()
    else:
        plt.close()

    # BG: Plot MCMC chains to assess parameter convergence
    plt.figure(figsize=(10, ndim * 2))
    for i in range(ndim):
        plt.subplot(ndim, 1, i + 1)
        for walker in chain[:, :, i]:
            plt.plot(walker, alpha=0.4)
        plt.ylabel(param_names[i])
        if i == 0:
            plt.title("MCMC Chains for Each Parameter")
    plt.xlabel("Step")
    plt.tight_layout()
    if params["save_plots"]:
        savefile = wd / "png" / "mcmc_chains.png"
        plt.savefig(savefile, dpi=300)
        print(f"- MCMC chain plot written to {savefile}")
    if params["display_plots"]:
        plt.show()
    else:
        plt.close()

    # BG: Use corner plot to visualize marginal distributions and parameter correlations
    # TODO: Define variable for plot if modifying plot format
    corner.corner(
        flat_samples,
        labels=param_names,
        truths=best,
        show_titles=True,
        title_fmt=".2f",
        title_kwargs={"fontsize": 10},
    )
    if params["save_plots"]:
        savefile = wd / "png" / "mcmc_corner.png"
        plt.savefig(savefile, dpi=300)
        print(f"- MCMC corner plot written to {savefile}")
    if params["display_plots"]:
        plt.show()
    else:
        plt.close()

    for i, j in combinations(range(ndim), 2):
        x, y = flat_samples[:, i], flat_samples[:, j]
        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(4, 4)
        ax = fig.add_subplot(gs[1:, :-1])
        ax_histx = fig.add_subplot(gs[0, :-1], sharex=ax)
        ax_histy = fig.add_subplot(gs[1:, -1], sharey=ax)

        sc = ax.scatter(x, y, c=neg_log_probs, cmap="viridis", marker="x")
        ax.scatter(best[i], best[j], color="red", marker="x", label="Best")
        ax.set_xlabel(param_names[i])
        ax.set_ylabel(param_names[j])
        fig.colorbar(sc, ax=ax, orientation="horizontal", label="Misfit")

        ax_histx.hist(x, bins=15, color="grey")
        ax_histy.hist(y, bins=15, orientation="horizontal", color="grey")
        fig.suptitle(f"Scatter: {param_names[i]} vs {param_names[j]}")
        if params["save_plots"]:
            savefile = (
                wd / "png" / f"mcmc_scatter_{param_names[i]}_vs_{param_names[j]}.png"
            )
            plt.savefig(savefile, dpi=300)
            print(
                f"- MCMC scatter plot for {param_names[i]} versus {param_names[j]} written to {savefile}"
            )
        if params["display_plots"]:
            plt.show()
        else:
            plt.close()

    # FIXME: Is the line below needed???
    # success += 1

    # Stop timer here if not displaying plots
    if not params["display_plots"]:
        exec_end = time.time()
    print(
        f"\n{16 * '-'} MCMC execution completed in {exec_end - exec_start:10.4f} seconds {16 * '-'}"
    )


def run_model(params):
    # Say hello
    if not params["batch_mode"]:
        print(f"{23 * '-'} Forward model execution started {24 * '-'}")
        exec_start = time.time()

    # Define working directory
    wd = Path.cwd()

    # Set flags if using batch mode (or not)
    write_track_lengths = False
    if params["batch_mode"]:
        params["echo_info"] = False
        params["echo_thermal_info"] = False
        params["echo_ages"] = False
        params["plot_results"] = False
    else:
        # Only write AFT track lengths to file in forward models
        write_track_lengths = True

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
    if params["plot_fault_depth_history"]:
        fault_depth_history = np.zeros(nt)

    # Calculate exhumation magnitude
    exhumation_magnitude, fw_reference_frame = calculate_exhumation_magnitude(
        params["ero_type"],
        params["ero_option1"],
        params["ero_option2"],
        params["ero_option3"],
        params["ero_option4"],
        params["ero_option5"],
        params["ero_option6"],
        params["ero_option7"],
        params["ero_option8"],
        params["ero_option9"],
        params["ero_option10"],
        t_total,
    )

    # Create velocity arrays for heat transfer
    vx_init = np.zeros(len(x))
    vx_array = np.zeros(len(x))

    # Set initial exhumation velocity
    vx_init[:] = mmyr2ms(params["vx_init"])

    # Set number of passes needed based on erosion model type
    # Types 1-7 need only 1 pass
    if params["ero_type"] < 8:
        num_pass = 1

    # Create array of plot times
    t_plots = myr2sec(np.array(params["t_plots"]))
    t_plots.sort()

    # Populate t_plots array if only one value given (treated like a plot increment)
    if len(t_plots) == 1:
        t_plots = np.arange(t_plots[0], t_total, t_plots[0])

    # Set flag if more than one plot to produce
    if len(t_plots) > 0:
        more_plots = True
    else:
        more_plots = False

    # Determine thickness of mantle to remove
    mantle_lith_thickness = max_depth - moho_depth
    # FIXME: This perhaps should be handled better
    if params["removal_start_time"] < 1.0e-6:
        removal_thickness = params["removal_fraction"] * mantle_lith_thickness
    else:
        # NOTE: This assumes the removal time is considerably longer than dt
        removal_thickness = 0.0

    # Create intrusion instance(s)
    intrusion = create_intrusion(
        temperature=params["intrusion_temperature"],
        start_time=params["intrusion_start_time"],
        duration=params["intrusion_duration"],
        thickness=params["intrusion_thickness"],
        base_depth=params["intrusion_base_depth"],
    )

    # Calculate vx_moho to get fault depth for ero type 7
    # TODO: Check that all uses of vx now work correctly when defining vx_array
    vx_moho = np.array([0.0])
    vx_moho, vx, vx_max, _ = calculate_erosion_rate(
        params,
        dt,
        t_total,
        0.0,
        moho_depth,
        vx_moho,
        kilo2base(params["ero_option4"]),
        moho_depth,
        fw_reference_frame,
    )

    # Define final fault depth for erosion model 7
    if params["ero_type"] == 7:
        # Set fault depth for extension
        fault_depth = kilo2base(params["ero_option4"]) - kilo2base(exhumation_magnitude)
    else:
        fault_depth = 0.0

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

    if params["calc_ages"]:
        num_file_ages = 0
        ages_from_data_file = False
        obs_unique_depo_ages = []

        # Read age data from file if one has been specified
        if len(params["obs_age_file"]) > 0:
            ages_from_data_file = True

            # Read age data from file
            obs_age_file = Path(params["obs_age_file"])
            (
                obs_age_type_file,
                obs_ages_file,
                obs_uncertainty_file,
                obs_u_file,
                obs_th_file,
                obs_radius_file,
                obs_sample_id_file,
                obs_depo_age_file,
            ) = read_age_data_file(obs_age_file, params)
            num_file_ages = len(obs_ages_file)

            # Find indices of each different age type
            obs_ahe_indices = np.where(obs_age_type_file == "AHe")[0]
            obs_aft_indices = np.where(obs_age_type_file == "AFT")[0]
            obs_zhe_indices = np.where(obs_age_type_file == "ZHe")[0]
            obs_zft_indices = np.where(obs_age_type_file == "ZFT")[0]

            # Find indices of samples with zero/nonzero depositional ages
            eps = 1e-8
            obs_zero_depo_indices = np.where(obs_depo_age_file <= eps)[0]
            obs_nz_depo_indices = np.where(obs_depo_age_file > eps)[0]
            obs_unique_depo_ages = np.unique(obs_depo_age_file[obs_nz_depo_indices])

            # Create list of indices for samples with nonzero depositional ages
            obs_ahe_depo_indices = np.intersect1d(obs_ahe_indices, obs_nz_depo_indices)
            obs_aft_depo_indices = np.intersect1d(obs_aft_indices, obs_nz_depo_indices)
            obs_zhe_depo_indices = np.intersect1d(obs_zhe_indices, obs_nz_depo_indices)
            obs_zft_depo_indices = np.intersect1d(obs_zft_indices, obs_nz_depo_indices)

            # Update indices of different age types to remove samples with nonzero depositional ages
            obs_ahe_indices = np.intersect1d(obs_ahe_indices, obs_zero_depo_indices)
            obs_aft_indices = np.intersect1d(obs_aft_indices, obs_zero_depo_indices)
            obs_zhe_indices = np.intersect1d(obs_zhe_indices, obs_zero_depo_indices)
            obs_zft_indices = np.intersect1d(obs_zft_indices, obs_zero_depo_indices)

            if params["debug"]:
                print(f"\n{num_file_ages} ages read from data file.")

        # Create array of past ages at which ages should be calculated, if not zero
        if len(obs_unique_depo_ages) > 0:
            # Issue warning if trying to use depositional ages and a past age increment
            if params["past_age_increment"] > 0.0:
                warnings.warn(
                    "Depositional ages in data file and past age increment specified. Only file ages will be used!",
                    stacklevel=2,
                )
            # Create surface times array
            surface_times_ma = np.zeros(len(obs_unique_depo_ages) + 1, dtype=float)
            # Find unique nonzero depositional ages and fill
            surface_times_ma[:-1] = np.sort(obs_unique_depo_ages)[::-1]
        else:
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
    vx_pts = np.zeros(len(surface_times_ma))

    # Create empty numpy arrays for depth, temperature, and time histories
    nt_surf_times = np.zeros(len(surface_times_ma))
    for i in range(len(surface_times_ma)):
        time_inc_now = myr2sec(params["t_total"] - surface_times_ma[i])
        nt_now = int(np.floor(time_inc_now / dt))
        depth_hists.append(np.zeros(nt_now))
        pressure_hists.append(np.zeros(nt_now))
        temp_hists.append(np.zeros(nt_now))
        time_hists.append(np.zeros(nt_now))
        nt_surf_times[i] = nt_now

    # Calculate mantle adiabat (or fill with dummy values)
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
    # Decrease crustal heat production exponentially, if enabled
    if params["heat_prod_decay_depth"] > 0.0:
        heat_prod *= np.exp(-x / kilo2base(params["heat_prod_decay_depth"]))
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
        vx_init,
        rho,
        cp,
        k,
        heat_prod,
    )
    # Apply intrusion to initial thermal field, if applicable
    if (intrusion.start_time <= 0.0) and (intrusion.duration > 0.0):
        apply_intrusion(intrusion=intrusion, x=x, model_temperatures=temp_init)
    interp_temp_init = interp1d(x, temp_init)
    init_moho_temp = interp_temp_init(moho_depth)

    # Calculate initial heat flow
    init_heat_flow = calculate_heat_flow(temp_init, k, dx)

    # Echo thermal model values
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
    density_init = update_density(rho, alphav, temp_init)

    # Set temperatures at 0 Ma
    temp_prev = temp_init.copy()

    # Modify temperatures and material properties for ero_types 4 and 5
    # TODO: Remove this?
    fault_activated = False
    """
    if (params["ero_type"] == 4 or params["ero_type"] == 5) and (
        params["ero_option3"] < 1.0e-6
    ):
        temp_prev, moho_depth, rho, cp, k, heat_prod, alphav = init_ero_types(
            params, x, xstag, temp_prev, moho_depth
        )
        fault_activated = True
    """

    # TODO: Remove this?
    # FIXME: This should handle the gradual removal case too!
    delaminated = False
    if (params["removal_fraction"] > 0.0) and (params["removal_start_time"] < 1e-6):
        for ix in range(params["nx"]):
            if x[ix] > (max_depth - removal_thickness):
                temp_prev[ix] = params["temp_base"] + (x[ix] - max_depth) * adiabat_m
        delaminated = True

    # Set plot parameters if plotting requested
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
        if params["plot_ma"]:
            time_label = f"{params['t_total']:.1f} Ma"
        else:
            time_label = "0.0 Myr"
        ax1.plot(temp_prev, -x / 1000, "k-", label=time_label)
        ax2.plot(density_init, -x / 1000, "k-", label=time_label)

    # Calculate model times when particles reach surface
    surface_times = myr2sec(params["t_total"] - surface_times_ma)

    # Loop over number of required passes
    for j in range(num_pass):
        # Start the loop over time steps
        curtime = t_total
        idx = 0
        if j == num_pass - 1:
            plotidx = 0

        # Restore values if using more than 1 pass
        if num_pass > 1:
            # Reset moho depth
            moho_depth = moho_depth_init
            # FIXME: The code below is not currently used for any scenario, but
            #        should be updated for gradual mantle delamination
            # Reset initial temperatures
            for ix in range(params["nx"]):
                if x[ix] > (max_depth - removal_thickness):
                    temp_prev[ix] = (
                        params["temp_base"] + (x[ix] - max_depth) * adiabat_m
                    )
                else:
                    temp_prev[ix] = temp_init[ix]

        # Reset erosion rate
        vx_array, vx, vx_max, _ = calculate_erosion_rate(
            params,
            dt,
            t_total,
            curtime,
            x,
            vx_array,
            fault_depth,
            moho_depth,
            fw_reference_frame,
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
            kilo2base(params["heat_prod_decay_depth"]),
            micro2base(params["heat_prod_mantle"]),
            heat_prod,
            temp_adiabat,
            temp_prev,
            params["k_a"],
            delaminated,
            params["removal_fraction"],
        )
        elev_init = calculate_isostatic_elevation(
            density_init, x, dx, moho_depth, params["rho_a"], max_depth
        )

        lab_depths.append(lab_depth)

        # Find starting depth if using only a one-pass erosion type
        if num_pass == 1:
            # Loop over all times and use -dt to run erosion model backwards
            # TODO: Make this a function???
            while curtime > 0.0:
                # Find particle velocities and move incrementally to starting depths
                vx_pts, vx, vx_max, fault_depth = calculate_erosion_rate(
                    params,
                    -dt,
                    t_total,
                    curtime,
                    depths,
                    vx_pts,
                    fault_depth,
                    moho_depth,
                    fw_reference_frame,
                )
                move_particles = surface_times >= curtime
                depths[move_particles] -= vx_pts[move_particles] * -dt
                # Store exhumation velocity history for particle reaching surface at 0 Ma
                vx_hist[idx] = vx_pts[-1]
                if params["plot_fault_depth_history"]:
                    fault_depth_history[idx] = fault_depth

                # Increment current time and idx
                curtime -= dt
                idx += 1

            # Reverse exhumation history for plotting
            vx_hist = np.flip(vx_hist)
            if params["plot_fault_depth_history"]:
                fault_depth_history = np.flip(fault_depth_history)

            # TODO: Can this be made into a function???
            # Adjust depths for footwall if using ero type 4 or all cases for ero type 5
            if (params["ero_type"] == 4) or (params["ero_type"] == 5):
                for i in range(len(surface_times)):
                    nt_now = int(nt_surf_times[i])
                    surf_exhumation_magnitude = (vx_hist[:nt_now] * dt).sum()
                    # Subtract thrust sheet if the thickness eroded exceeds the sheet thickness
                    if params["ero_type"] == 4:
                        if (params["ero_option2"] > 0.0) and (
                            surf_exhumation_magnitude
                            >= kilo2base(params["ero_option1"])
                        ):
                            depths[i] = surf_exhumation_magnitude - kilo2base(
                                params["ero_option1"]
                            )
                            # Print starting depths if debugging is on
                            if params["debug"]:
                                print(f"Calculated starting depth {i}: {depths[i]} m")
                    # Add tectonic erosional thickness for tectonic exhumation (ero_type 5)
                    if params["ero_type"] == 5:
                        depths[i] = surf_exhumation_magnitude + kilo2base(
                            params["ero_option1"]
                        )
                        # Print starting depths if debugging is on
                        if params["debug"]:
                            print(f"Calculated starting depth {i}: {depths[i]} m")

            # Reset loop variables
            curtime = 0.0
            idx = 0
            vx_array, vx, vx_max, _ = calculate_erosion_rate(
                params,
                dt,
                t_total,
                curtime,
                x,
                vx_array,
                fault_depth,
                moho_depth,
                fw_reference_frame,
            )

        if not params["batch_mode"]:
            print("")
            print(
                f"--- Calculating transient thermal model (Pass {j + 1}/{num_pass}) ---"
            )
            print("")

        # Start main time loop
        while curtime < t_total:
            if not params["batch_mode"]:
                print(
                    f"- Step {idx + 1:5d} of {nt} (Time: {curtime / myr2sec(1):5.1f} Myr, Erosion rate: {vx / mmyr2ms(1):5.2f} mm/yr)\r",
                    end="",
                )
            elif not params["inverse_mode"]:
                # Print progress dot if using batch model. 1 dot = 10%
                if (idx + 1) % round(nt / 10, 0) == 0:
                    print(".", end="", flush=True)

            # Modify temperatures and material properties for ero_types 4 and 5
            if ((params["ero_type"] == 4) or (params["ero_type"] == 5)) and (
                not fault_activated
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
                    # Adjust tracked particle depths
                    for i in range(len(surface_times)):
                        nt_now = int(nt_surf_times[i])
                        surf_exhumation_magnitude = (vx_hist[:nt_now] * dt).sum()
                        # Add thrust sheet if the thickness eroded exceeds the sheet thickness (ero_type = 4)
                        if params["ero_type"] == 4:
                            if (params["ero_option2"] > 0.0) and (
                                surf_exhumation_magnitude
                                >= kilo2base(params["ero_option1"])
                            ):
                                depths[i] += kilo2base(params["ero_option1"])
                                if params["debug"]:
                                    print(
                                        f"Adjusted depth of particle {i}, reaching surface at {surface_times_ma[i]} Ma: {depths[i]} m"
                                    )
                        # Adjust tracked particle depth following tectonic exhumation
                        if params["ero_type"] == 5:
                            depths[i] -= kilo2base(params["ero_option1"])
                            # Print adjusted depth if debugging is on
                            if params["debug"]:
                                print(
                                    f"Adjusted depth of particle {i}, reaching surface at {surface_times_ma[i]} Ma: {depths[i]} m"
                                )
                    fault_activated = True

            # Set mantle temperatures to adiabat if in removal interval
            if (params["removal_fraction"] > 0.0) and (not delaminated):
                # Episodic removal
                if params["removal_end_time"] - params[
                    "removal_start_time"
                ] < dt / myr2sec(1):
                    in_removal_interval = (
                        (curtime - (dt / 2)) / myr2sec(1)
                        <= params["removal_start_time"]
                        < (curtime + (dt / 2)) / myr2sec(1)
                    )
                    removal_thickness = params["removal_fraction"] * (
                        max_depth - moho_depth
                    )
                # Gradual removal
                else:
                    in_removal_interval = (
                        params["removal_start_time"]
                        <= curtime / myr2sec(1)
                        <= params["removal_end_time"]
                    )
                    removal_thickness = (
                        (curtime / myr2sec(1) - params["removal_start_time"])
                        / (params["removal_end_time"] - params["removal_start_time"])
                        * params["removal_fraction"]
                        * (max_depth - moho_depth)
                    )
                if in_removal_interval:
                    for ix in range(params["nx"]):
                        if x[ix] > (max_depth - removal_thickness):
                            temp_prev[ix] = (
                                params["temp_base"] + (x[ix] - max_depth) * adiabat_m
                            )
                    if curtime / myr2sec(1) > params["removal_end_time"]:
                        delaminated = True

            # Apply intrusion(s), if applicable
            in_intrusion_interval = (
                intrusion.start_time <= curtime <= intrusion.end_time
            )
            if in_intrusion_interval:
                temp_prev = apply_intrusion(
                    intrusion=intrusion, x=x, model_temperatures=temp_prev
                )

            # Update material properties
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
                kilo2base(params["heat_prod_decay_depth"]),
                micro2base(params["heat_prod_mantle"]),
                heat_prod,
                temp_adiabat,
                temp_prev,
                params["k_a"],
                delaminated,
                params["removal_fraction"],
            )

            # Calculate updated temperatures
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

            # Calculate maximum temperature difference if using debug output
            if params["debug"]:
                max_temp_diff = abs(temp_prev - temp_new).max()

            # Store new temperatures for next temperature calculation
            temp_prev[:] = temp_new[:]

            # Store updated densities
            density_new = update_density(rho, alphav, temp_new)

            # Store current LAB depth
            lab_depths.append(lab_depth - moho_depth)

            # Calculate model mass for isostasy
            elev = calculate_isostatic_elevation(
                density_new, x, dx, moho_depth, params["rho_a"], max_depth
            )

            # Update Moho depth
            if not params["fixed_moho"]:
                # Update Moho depth to use hanging wall Moho for hw in erosion type 7
                if (params["ero_type"] == 7) and (not fw_reference_frame):
                    vx_moho = np.interp(float(fault_depth - dx), x, vx_array)
                else:
                    vx_moho = np.interp(float(moho_depth), x, vx_array)
                moho_depth -= vx_moho * dt

            # Update base depth for any active intrusions
            if in_intrusion_interval:
                if (params["ero_type"] == 7) and (not fw_reference_frame):
                    vx_intrusion = np.interp(float(fault_depth - dx), x, vx_array)
                else:
                    vx_intrusion = np.interp(intrusion.base_depth, x, vx_array)
                intrusion.base_depth -= vx_intrusion * dt

            # Store tracked surface elevations and current time
            if j == 0:
                elev_list.append(elev - elev_init)
                time_list.append(curtime / myr2sec(1.0))

            # Save Temperature, depth, and pressure histories
            if j == num_pass - 1:
                # Create temperature interpolation function
                interp_temp_new = interp1d(x, temp_new)
                # Calculate lithostatic pressure
                pressure = calculate_pressure(density_new, dx)
                interp_pressure = interp1d(x, pressure)

                # Find particle velocities and move incrementally toward surface
                # Note: This does NOT update the fault depth!
                vx_pts, vx, vx_max, _ = calculate_erosion_rate(
                    params,
                    dt,
                    t_total,
                    curtime,
                    depths,
                    vx_pts,
                    fault_depth,
                    moho_depth,
                    fw_reference_frame,
                )
                move_particles = surface_times >= curtime
                depths[move_particles] -= vx_pts[move_particles] * dt

                # Loop over all times when particles reach the surface
                for i in range(len(surface_times_ma)):
                    if curtime < myr2sec(params["t_total"] - surface_times_ma[i]):
                        # Store depth and time histories
                        depth_hists[i][idx] = depths[i]
                        time_hists[i][idx] = curtime

                        # Store temperature histories
                        # Check whether point is within 10 cm of the surface
                        if abs(depths[i]) <= 5.0e-2:
                            temp_hists[i][idx] = 0.0
                        # Check whether point is below the Moho for fixed-moho models
                        # If so, set temperature to Moho temperature
                        elif depths[i] > moho_depth and params["fixed_moho"]:
                            temp_hists[i][idx] = interp_temp_new(moho_depth)
                        # Otherwise, record temperature at current depth
                        else:
                            temp_hists[i][idx] = interp_temp_new(depths[i])

                        # Store pressure history
                        # Check whether point is within 10 cm of the surface
                        if abs(depths[i]) <= 5.0e-2:
                            pressure_hists[i][idx] = 0.0
                        elif depths[i] > moho_depth and params["fixed_moho"]:
                            pressure_hists[i][idx] = interp_pressure(moho_depth)
                        else:
                            pressure_hists[i][idx] = interp_pressure(depths[i])

                        # Print array values if debugging is on
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

            if params["debug"]:
                print(
                    f"Maximum temp difference at time {curtime / myr2sec(1):.4f} Myr: {max_temp_diff:.4f} °C"
                )

            # Update current time and index
            curtime += dt
            idx += 1

            # Update erosion rate (and fault depth when it applies)
            vx_array, vx, vx_max, fault_depth = calculate_erosion_rate(
                params,
                dt,
                t_total,
                curtime,
                x,
                vx_array,
                fault_depth,
                moho_depth,
                fw_reference_frame,
            )

            # Plot temperature and density profiles
            if j == num_pass - 1:
                if params["plot_results"] and more_plots:
                    if curtime > t_plots[plotidx]:
                        if params["plot_ma"]:
                            time_label = f"{(params['t_total'] - t_plots[plotidx] / myr2sec(1)):.1f} Ma"
                        else:
                            time_label = f"{t_plots[plotidx] / myr2sec(1):.1f} Myr"
                        ax1.plot(
                            temp_new,
                            -x / 1000,
                            "-",
                            label=time_label,
                            color=colors[plotidx],
                        )
                        ax2.plot(
                            density_new,
                            -x / 1000,
                            label=time_label,
                            color=colors[plotidx],
                        )
                        if plotidx == len(t_plots) - 1:
                            more_plots = False
                        plotidx += 1
                        # tplot = t_plots[plotidx]

        if not params["batch_mode"]:
            print("")

    # Calculate final densities
    density_new = update_density(rho, alphav, temp_new)

    # Calculate final Moho temperature and heat flow
    interp_temp_new = interp1d(x, temp_new)
    final_moho_temp = interp_temp_new(moho_depth)
    final_heat_flow = calculate_heat_flow(temp_new, k, dx)

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

    # Calculate ages
    if params["calc_ages"]:
        # Define time-temperature-depth filenames
        if params["inverse_mode"]:
            # Get process ID for file naming
            pid = os.getpid()
            tt_filename = f"time_temp_hist_{pid}.csv"
            ttdp_filename = f"time_temp_depth_pressure_hist_{pid}.csv"
        else:
            tt_filename = "time_temp_hist.csv"
            ttdp_filename = "time_temp_depth_pressure_hist.csv"

        # Convert time since model start to time before end of simulation
        time_ma = tt_hist_to_ma(time_hists[-1])

        # Calculate generic present (and possibly past) ages using default params
        surf_age_array_size = len(surface_times_ma)
        corr_ahe_ages = np.zeros(surf_age_array_size)
        ahe_temps = np.zeros(surf_age_array_size)
        aft_ages = np.zeros(surf_age_array_size)
        aft_temps = np.zeros(surf_age_array_size)
        corr_zhe_ages = np.zeros(surf_age_array_size)
        zhe_temps = np.zeros(surf_age_array_size)
        zft_ages = np.zeros(surf_age_array_size)
        zft_temps = np.zeros(surf_age_array_size)
        for i in range(surf_age_array_size):
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
                tt_filename,
                ttdp_filename,
                write_track_lengths,
            )

            if params["debug"]:
                print("")
                print(f"--- Predicted ages for cooling history {i} ---")
                print(
                    f"- AHe age: {corr_ahe_ages[i]:.2f} Ma (Tc: {ahe_temps[i]:.2f} °C)"
                )
                print(f"- AFT age: {aft_ages[i]:.2f} Ma (Tc: {aft_temps[i]:.2f} °C)")
                print(
                    f"- ZHe age: {corr_zhe_ages[i]:.2f} Ma (Tc: {zhe_temps[i]:.2f} °C)"
                )
                print(f"- ZFT age: {zft_ages[i]:.2f} Ma (Tc: {zft_temps[i]:.2f} °C)")

        # Move/rename/remove time-temp histories
        # Only do this for the final ages/histories!
        tt_orig = Path(tt_filename)
        ttdp_orig = Path(ttdp_filename)
        if params["batch_mode"] and not params["inverse_mode"]:
            # Rename and move files to batch output directory
            tt_newfile = params["model_id"] + "-time_temp_hist.csv"
            tt_new = tt_orig.rename(wd / "csv" / tt_newfile)
            ttdp_newfile = params["model_id"] + "-time_temp_depth_pressure_hist.csv"
            ttdp_new = ttdp_orig.rename(wd / "csv" / ttdp_newfile)
        else:
            tt_new = tt_orig.rename(wd / "csv" / tt_orig)
            ttdp_new = ttdp_orig.rename(wd / "csv" / ttdp_orig)

        # Move track length file to csv directory
        if write_track_lengths:
            ftl_orig = Path("ft_length.csv")
            ftl_new = ftl_orig.rename(wd / "csv" / ftl_orig)

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

        # TODO?: Make a separate function to handle observed age data
        # If measured ages have been provided, calculate ages/misfit
        num_passed_ages = (
            len(params["obs_ahe"])
            + len(params["obs_aft"])
            + len(params["obs_zhe"])
            + len(params["obs_zft"])
        )

        if ages_from_data_file:
            # Issue warning if measured ages provided in file and passed as params
            if num_passed_ages > 0:
                warnings.warn(
                    "Measured ages provided in data file and passed as arguments. Only file ages will be used!",
                    stacklevel=2,
                )

            # Create array to store predicted ages
            pred_data_ages = np.zeros(num_file_ages)
            pred_data_temps = np.zeros(num_file_ages)

            # Loop over all ages in age data file can calculate predicted age equivalents
            tt_hist_index = -1
            depo_age_old = 5000.0
            for i in range(len(obs_ages_file)):
                # Increment time-temperature history index whenever it changes, write new tt-history file
                depo_age_now = obs_depo_age_file[i]

                if depo_age_now < depo_age_old:
                    tt_hist_index += 1

                    # Write time-temperature history to file for age prediction
                    write_tt_history(
                        params,
                        tt_orig,
                        time_hists[tt_hist_index],
                        temp_hists[tt_hist_index],
                    )

                    # Update time_ma array
                    time_ma = tt_hist_to_ma(time_hists[tt_hist_index])

                    # Calculate time to be added to predicted age
                    extra_depo_time = depo_age_now

                    # Update depositional age reference
                    depo_age_old = depo_age_now

                if obs_age_type_file[i] == "AHe":
                    # Calculate AHe age
                    _, corr_ahe_age, _, _ = he_ages(
                        file=tt_orig.as_posix(),
                        ap_rad=obs_radius_file[i],
                        ap_uranium=obs_u_file[i],
                        ap_thorium=obs_th_file[i],
                        zr_rad=params["zr_rad"],
                        zr_uranium=params["zr_uranium"],
                        zr_thorium=params["zr_thorium"],
                    )
                    pred_data_ages[i] = float(corr_ahe_age) + depo_age_now
                    pred_data_temps[i] = calculate_closure_temp(
                        float(corr_ahe_age),
                        np.flip(time_ma),
                        np.flip(temp_hists[tt_hist_index]),
                    )
                    if params["debug"]:
                        print(
                            f"AHe age calculated from file data including depositional age of {depo_age_now:.2f} Ma: {pred_data_ages[i]:.2f} Ma"
                        )
                        print(
                            f"eU: {calculate_eu(obs_u_file[i], obs_th_file[i])} ppm, grain radius: {obs_radius_file[i]} um"
                        )

                elif obs_age_type_file[i] == "AFT":
                    # Calculate predicted AFT ages
                    pred_data_ages[i] = float(aft_ages[tt_hist_index]) + depo_age_now
                    pred_data_temps[i] = aft_temps[tt_hist_index]

                elif obs_age_type_file[i] == "ZHe":
                    # Calculate ZHe age
                    _, _, _, corr_zhe_age = he_ages(
                        file=tt_orig.as_posix(),
                        ap_rad=params["ap_rad"],
                        ap_uranium=params["ap_uranium"],
                        ap_thorium=params["ap_thorium"],
                        zr_rad=obs_radius_file[i],
                        zr_uranium=obs_u_file[i],
                        zr_thorium=obs_th_file[i],
                    )
                    pred_data_ages[i] = float(corr_zhe_age) + depo_age_now
                    pred_data_temps[i] = calculate_closure_temp(
                        float(corr_zhe_age),
                        np.flip(time_ma),
                        np.flip(temp_hists[tt_hist_index]),
                    )
                    if params["debug"]:
                        print(
                            f"ZHe age calculated from file data including depositional age of {depo_age_now:.2f} Ma: {pred_data_ages[i]:.2f} Ma"
                        )
                        print(
                            f"eU: {calculate_eu(obs_u_file[i], obs_th_file[i])} ppm, grain radius: {obs_radius_file[i]} um"
                        )

                elif obs_age_type_file[i] == "ZFT":
                    # Calculate predicted ZFT ages
                    pred_data_ages[i] = float(zft_ages[tt_hist_index]) + depo_age_now
                    pred_data_temps[i] = zft_temps[tt_hist_index]

            # Delete unneeded tt file
            tt_orig.unlink()

            # Log number of different age types
            n_obs_ahe = len(obs_ahe_indices)
            n_obs_aft = len(obs_aft_indices)
            n_obs_zhe = len(obs_zhe_indices)
            n_obs_zft = len(obs_zft_indices)
        else:
            # Log number of different age types
            n_obs_ahe = len(params["obs_ahe"])
            n_obs_aft = len(params["obs_aft"])
            n_obs_zhe = len(params["obs_zhe"])
            n_obs_zft = len(params["obs_zft"])

        # Delete the tt files if using inverse mode
        if params["inverse_mode"]:
            tt_new.unlink()
            ttdp_new.unlink()

        # END FIXME?

        depo_ages_in_file = False
        if (num_passed_ages > 0) or (num_file_ages > 0):
            # Create single arrays of ages for misfit calculation
            if ages_from_data_file:
                pred_ages = pred_data_ages
                obs_ages = obs_ages_file
                obs_stdev = obs_uncertainty_file
                obs_eu = [calculate_eu(u, th) for u, th in zip(obs_u_file, obs_th_file)]
                obs_radius = obs_radius_file
            else:
                pred_ages = []
                obs_ages = []
                obs_stdev = []
                obs_eu = []
                obs_radius = []
                for i in range(n_obs_ahe):
                    pred_ages.append(float(corr_ahe_ages[-1]))
                    obs_ages.append(params["obs_ahe"][i])
                    obs_stdev.append(params["obs_ahe_stdev"][i])
                    obs_eu.append("")
                    obs_radius.append("")
                for i in range(n_obs_aft):
                    pred_ages.append(float(aft_ages[-1]))
                    obs_ages.append(params["obs_aft"][i])
                    obs_stdev.append(params["obs_aft_stdev"][i])
                    obs_eu.append("")
                    obs_radius.append("")
                for i in range(n_obs_zhe):
                    pred_ages.append(float(corr_zhe_ages[-1]))
                    obs_ages.append(params["obs_zhe"][i])
                    obs_stdev.append(params["obs_zhe_stdev"][i])
                    obs_eu.append("")
                    obs_radius.append("")
                for i in range(n_obs_zft):
                    pred_ages.append(float(zft_ages[-1]))
                    obs_ages.append(params["obs_zft"][i])
                    obs_stdev.append(params["obs_zft_stdev"][i])
                    obs_eu.append("")
                    obs_radius.append("")

            # Convert lists to NumPy arrays
            pred_ages = np.array(pred_ages)
            obs_ages = np.array(obs_ages)
            obs_stdev = np.array(obs_stdev)
            obs_eu = np.array(obs_eu)
            obs_radius = np.array(obs_radius)

            # Calculate misfit
            misfit = calculate_misfit(
                pred_ages,
                obs_ages,
                obs_stdev,
                params["misfit_type"],
                params["misfit_num_params"],
            )

            # Calculate misfit components if using age data file
            # Do we use an age data file with depositional ages?
            depo_ages_in_file = (ages_from_data_file) and (
                len(pred_ages[obs_depo_age_file > eps]) > 0
            )
            if depo_ages_in_file:
                nondepo_misfit = calculate_misfit(
                    pred_ages[obs_depo_age_file <= eps],
                    obs_ages[obs_depo_age_file <= eps],
                    obs_stdev[obs_depo_age_file <= eps],
                    params["misfit_type"],
                    params["misfit_num_params"],
                )
                depo_misfit = calculate_misfit(
                    pred_ages[obs_depo_age_file > eps],
                    obs_ages[obs_depo_age_file > eps],
                    obs_stdev[obs_depo_age_file > eps],
                    params["misfit_type"],
                    params["misfit_num_params"],
                )

            # Print misfit to the screen
            if params["echo_ages"]:
                print("")
                print("--- Predicted and observed age misfit ---")
                print("")
                print(
                    f"- Total misfit: {misfit:.4f} (misfit type {params['misfit_type']}, {len(pred_ages)} age(s))"
                )
                if depo_ages_in_file:
                    print(
                        f"  - Non-depositional age misfit: {nondepo_misfit:.4f} ({len(pred_ages[obs_depo_age_file <= eps])} age(s))"
                    )
                    print(
                        f"  - Depositional age misfit: {depo_misfit:.4f} ({len(pred_ages[obs_depo_age_file > eps])} age(s))"
                    )

    # Write output files
    if (
        (params["plot_results"] and params["save_plots"])
        or params["write_temps"]
        or params["read_temps"]
        or (params["past_age_increment"] > 0.0 and params["write_past_ages"])
        or params["write_age_output"]
    ):
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
        savefile = wd / "csv" / "past_ages.csv"
        np.savetxt(
            savefile,
            past_ages_out,
            delimiter=",",
            fmt="%.8f",
            header="Time (Ma),Predicted Apatite (U-Th)/He age (Ma),Predicted Apatite fission-track age (Ma),Predicted "
            "Zircon (U-Th)/He age (Ma),Predicted Zircon fission-track age (Ma)",
        )
        print(f"- Past ages written to {savefile}")

    # Make final set of plots
    if params["plot_results"]:
        # Stop the timer to avoid plots viewing adding to execution time
        exec_end = time.time()

        # Calculate model time in Ma for final plots
        time_ma = tt_hist_to_ma(time_hists[-1])

        # Plot the final temperature field
        xmin = params["temp_surf"]
        # Add 10% to max T and round to nearest 100
        xmax = round(1.1 * temp_new.max(), -2)
        if params["plot_ma"]:
            time_label = "0.0 Ma"
        else:
            time_label = f"{curtime / myr2sec(1):.1f} Myr"
        ax1.plot(
            temp_new,
            -x / 1000,
            "-",
            label=time_label,
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
            pressure = calculate_pressure(density_new, dx)
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
            pressure = calculate_pressure(density_new, dx)
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
            pressure = calculate_pressure(density_new, dx)
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
                label="Mantle solidus",
            )

        ax1.text(20.0, (-moho_depth + 0.01 * x.max()) / kilo2base(1), "Final Moho")
        if moho_depth < x.max():
            ax1.text(
                20.0,
                -params["init_moho_depth"] - (0.025 * x.max()) / kilo2base(1),
                "Initial Moho",
                color="gray",
            )
        ax1.legend()
        ax1.axis([xmin, xmax, -max_depth / 1000, 0])
        ax1.set_xlabel("Temperature (°C)")
        ax1.set_ylabel("Depth (km)")
        # Round density ranges to nearest 50
        density_base = 50.0
        xmin = round_to_base(density_new.min(), density_base) - density_base
        xmax = round_to_base(density_new.max(), density_base) + density_base
        ax2.plot(
            density_new,
            -x / 1000,
            label=time_label,
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

        plt.tight_layout()
        if params["save_plots"]:
            savefile = wd / "png" / "T_rho_hist.png"
            plt.savefig(savefile, dpi=300)
            print(f"- Temperature/density history plot written to {savefile}")
        if params["display_plots"]:
            plt.show()
        else:
            plt.close()

        # Plot elevation history
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        # ax1.plot(time_list, elev_list, 'k-')
        if params["plot_ma"]:
            time_list = [params["t_total"] - time_now for time_now in time_list]
            time_xlabel = "Time (Ma)"
            time_xlim = [params["t_total"], 0.0]
        else:
            time_xlabel = "Time (Myr)"
            time_xlim = [0.0, params["t_total"]]
        ax1.plot(time_list, elev_list)
        ax1.set_xlabel(time_xlabel)
        ax1.set_ylabel("Elevation (m)")
        ax1.set_xlim(time_xlim)
        ax1.set_title("Elevation history")
        # plt.axis([0.0, t_total/myr2sec(1), 0, 750])
        # ax1.grid()

        if params["plot_ma"]:
            plot_time = params["t_total"] - time_hists[-1] / myr2sec(1)
        else:
            plot_time = time_hists[-1] / myr2sec(1)
        ax2.plot(plot_time, vx_hist / mmyr2ms(1))
        ax2.fill_between(
            plot_time,
            vx_hist / mmyr2ms(1),
            0.0,
            alpha=0.33,
            color="tab:blue",
            label=f"Total erosional exhumation: {exhumation_magnitude:.2f} km",
        )
        ax2.set_xlabel(time_xlabel)
        ax2.set_ylabel("Erosion rate (mm/yr)")
        ax2.set_xlim(time_xlim)
        # if params["ero_option1"] >= 0.0:
        #    ax2.set_ylim(ymin=0.0)
        # plt.axis([0.0, t_total/myr2sec(1), 0, 750])
        # ax2.grid()
        ax2.legend()

        plt.tight_layout()
        if params["save_plots"]:
            savefile = wd / "png" / "elev_hist.png"
            plt.savefig(savefile, dpi=300)
            print(f"- Surface elevation history plot written to {savefile}")
        if params["display_plots"]:
            plt.show()
        else:
            plt.close()

        # Plot cooling history and ages only if ages were calculated
        if params["calc_ages"]:
            # create objects
            fig = plt.figure(figsize=(12, 8))
            gs = GridSpec(3, 3, figure=fig)

            # create sub plots as grid
            ax1 = fig.add_subplot(gs[0:2, :])
            if params["plot_depth_history"] or params["plot_fault_depth_history"]:
                ax1b = ax1.twinx()
            ax2 = fig.add_subplot(gs[2, :-1])
            ax3 = fig.add_subplot(gs[2, -1])

            # Calculate synthetic uncertainties
            ahe_uncert = 0.1
            aft_uncert = 0.2
            zhe_uncert = 0.1
            zft_uncert = 0.2
            ahe_min, ahe_max = (
                (1.0 - ahe_uncert) * float(corr_ahe_ages[-1]),
                (1.0 + ahe_uncert) * float(corr_ahe_ages[-1]),
            )
            aft_min, aft_max = (
                (1.0 - aft_uncert) * float(aft_ages[-1]),
                (1.0 + aft_uncert) * float(aft_ages[-1]),
            )
            zhe_min, zhe_max = (
                (1.0 - zhe_uncert) * float(corr_zhe_ages[-1]),
                (1.0 + zhe_uncert) * float(corr_zhe_ages[-1]),
            )
            zft_min, zft_max = (
                (1.0 - zft_uncert) * float(zft_ages[-1]),
                (1.0 + zft_uncert) * float(zft_ages[-1]),
            )
            ax1.plot(time_ma, temp_hists[-1], color="dimgray", label="Thermal history")
            if params["plot_depth_history"]:
                ax1b.plot(
                    time_ma,
                    depth_hists[-1] / kilo2base(1),
                    "--",
                    color="darkgray",
                    label="Sample depth history",
                )
            if params["plot_fault_depth_history"]:
                ax1b.plot(
                    time_ma,
                    fault_depth_history / kilo2base(1),
                    "-.",
                    color="darkgray",
                    label="Fault depth history",
                )

            # Plot delamination time, if enabled
            if params["removal_fraction"] > 0.0:
                removal_start_time_ma = params["t_total"] - params["removal_start_time"]
                removal_end_time_ma = params["t_total"] - params["removal_end_time"]
                # TODO: Make this work for instantaneous and gradual removal
                ax1.plot(
                    [removal_start_time_ma, removal_start_time_ma],
                    [params["temp_surf"], params["temp_base"]],
                    "--",
                    color="gray",
                    label="Time of mantle delamination",
                )
                ax1.text(
                    removal_start_time_ma - 0.02 * t_total / myr2sec(1.0),
                    (temp_hists[-1].max() + temp_hists[-1].min()) / 2.0,
                    "Mantle lithosphere delaminates",
                    rotation=90,
                    ha="center",
                    va="center",
                    color="gray",
                )

            # Plot uncertainty error bars and AHe age if no measured ages exist
            if n_obs_ahe == 0:
                ax1 = plot_predictions_no_data(
                    float(corr_ahe_ages[-1]),
                    ahe_temps[-1],
                    xerr=ahe_uncert * float(corr_ahe_ages[-1]),
                    ax=ax1,
                    marker="o",
                    color="tab:blue",
                    label=f"Predicted AHe age ({float(corr_ahe_ages[-1]):.2f} ± {ahe_uncert * float(corr_ahe_ages[-1]):.2f} Ma ({ahe_uncert * 100.0:.0f}% error); T$_c$ = {ahe_temps[-1]:.1f}°C)",
                )
            # Plot predicted age + observed AHe age(s)
            else:
                if ages_from_data_file:
                    # Plot predicted and observed AHe age(s) from file
                    if n_obs_ahe == 1:
                        ahe_label = f"Predicted AHe age ({float(pred_ages[obs_ahe_indices[0]]):.2f} Ma; T$_c$ = {pred_data_temps[obs_ahe_indices[0]]:.1f}°C)"
                    else:
                        min_file_ahe_age = min(pred_ages[obs_ahe_indices])
                        max_file_ahe_age = max(pred_ages[obs_ahe_indices])
                        min_file_ahe_temp = min(pred_ages[obs_ahe_indices])
                        max_file_ahe_temp = max(pred_ages[obs_ahe_indices])
                        ahe_label = f"Predicted AHe ages ({min_file_ahe_age:.2f}–{max_file_ahe_age:.2f} Ma; T$_c$ = {min_file_ahe_temp:.1f}–{max_file_ahe_temp:.1f}°C)"
                    ax1 = plot_predictions_with_data(
                        pred_ages[obs_ahe_indices],
                        pred_data_temps[obs_ahe_indices],
                        ax=ax1,
                        marker="o",
                        color="tab:blue",
                        label=ahe_label,
                    )
                    ax1 = plot_measurements(
                        obs_ages[obs_ahe_indices],
                        pred_data_temps[obs_ahe_indices],
                        ax=ax1,
                        xerr=obs_stdev[obs_ahe_indices],
                        marker="o",
                        color="tab:blue",
                        label="Measured AHe age(s)",
                    )
                else:
                    # Plot predicted and passed-in observed AHe age(s)
                    ax1 = plot_predictions_with_data(
                        float(corr_ahe_ages[-1]),
                        ahe_temps[-1],
                        ax=ax1,
                        marker="o",
                        color="tab:blue",
                        label=f"Predicted AHe age ({float(corr_ahe_ages[-1]):.2f} Ma; T$_c$ = {ahe_temps[-1]:.1f}°C)",
                    )
                    ahe_temps_obs = []
                    for i in range(len(params["obs_ahe"])):
                        ahe_temps_obs.append(ahe_temps[-1])
                    ax1 = plot_measurements(
                        params["obs_ahe"],
                        ahe_temps_obs,
                        ax=ax1,
                        xerr=params["obs_ahe_stdev"],
                        marker="o",
                        color="tab:blue",
                        label="Measured AHe age(s)",
                    )

            # Plot uncertainty error bars and AFT age if no measured ages exist
            if n_obs_aft == 0:
                ax1 = plot_predictions_no_data(
                    float(aft_ages[-1]),
                    aft_temps[-1],
                    xerr=aft_uncert * float(aft_ages[-1]),
                    ax=ax1,
                    marker="s",
                    color="tab:orange",
                    label=f"Predicted AFT age ({float(aft_ages[-1]):.2f} ± {aft_uncert * float(aft_ages[-1]):.2f} Ma ({aft_uncert * 100.0:.0f}% error); T$_c$ = {aft_temps[-1]:.1f}°C)",
                )
            # Plot predicted age + observed AFT age(s)
            else:
                if ages_from_data_file:
                    ax1 = plot_predictions_with_data(
                        pred_ages[obs_aft_indices[0]],
                        pred_data_temps[obs_aft_indices[0]],
                        ax=ax1,
                        marker="s",
                        color="tab:orange",
                        label=f"Predicted AFT age ({pred_ages[obs_aft_indices[0]]:.2f} Ma; T$_c$ = {pred_data_temps[obs_aft_indices[0]]:.1f}°C)",
                    )
                    ax1 = plot_measurements(
                        obs_ages[obs_aft_indices],
                        pred_data_temps[obs_aft_indices],
                        xerr=obs_stdev[obs_aft_indices],
                        ax=ax1,
                        marker="s",
                        color="tab:orange",
                        label="Measured AFT age(s)",
                    )
                else:
                    ax1 = plot_predictions_with_data(
                        float(aft_ages[-1]),
                        aft_temps[-1],
                        ax=ax1,
                        marker="s",
                        color="tab:orange",
                        label=f"Predicted AFT age ({float(aft_ages[-1]):.2f} Ma; T$_c$ = {aft_temps[-1]:.1f}°C)",
                    )
                    aft_temps_obs = []
                    for i in range(len(params["obs_aft"])):
                        aft_temps_obs.append(aft_temps[-1])
                    ax1 = plot_measurements(
                        params["obs_aft"],
                        aft_temps_obs,
                        xerr=params["obs_aft_stdev"],
                        ax=ax1,
                        marker="s",
                        color="tab:orange",
                        label="Measured AFT age(s)",
                    )

            # Plot shaded uncertainty area and ZHe age if no measured ages exist
            if n_obs_zhe == 0:
                ax1 = plot_predictions_no_data(
                    float(corr_zhe_ages[-1]),
                    zhe_temps[-1],
                    xerr=zhe_uncert * float(corr_zhe_ages[-1]),
                    ax=ax1,
                    marker="d",
                    color="tab:green",
                    label=f"Predicted ZHe age ({float(corr_zhe_ages[-1]):.2f} ± {zhe_uncert * float(corr_zhe_ages[-1]):.2f} Ma ({zhe_uncert * 100.0:.0f}% error); T$_c$ = {zhe_temps[-1]:.1f}°C)",
                )
            # Plot predicted age + observed ZHe age(s)
            else:
                if ages_from_data_file:
                    if n_obs_zhe == 1:
                        zhe_label = f"Predicted ZHe age ({float(pred_ages[obs_zhe_indices[0]]):.2f} Ma; T$_c$ = {pred_data_temps[obs_zhe_indices[0]]:.1f}°C)"
                    else:
                        min_file_zhe_age = min(pred_ages[obs_zhe_indices])
                        max_file_zhe_age = max(pred_ages[obs_zhe_indices])
                        min_file_zhe_temp = min(pred_ages[obs_zhe_indices])
                        max_file_zhe_temp = max(pred_ages[obs_zhe_indices])
                        zhe_label = f"Predicted ZHe ages ({min_file_zhe_age:.2f}–{max_file_zhe_age:.2f} Ma; T$_c$ = {min_file_zhe_temp:.1f}–{max_file_zhe_temp:.1f}°C)"
                    # Plot predicted and observed ZHe age(s) from file
                    ax1 = plot_predictions_with_data(
                        pred_ages[obs_zhe_indices],
                        pred_data_temps[obs_zhe_indices],
                        ax=ax1,
                        marker="d",
                        color="tab:green",
                        label=zhe_label,
                    )
                    ax1 = plot_measurements(
                        obs_ages[obs_zhe_indices],
                        pred_data_temps[obs_zhe_indices],
                        xerr=obs_stdev[obs_zhe_indices],
                        ax=ax1,
                        marker="d",
                        color="tab:green",
                        label="Measured ZHe age(s)",
                    )
                else:
                    # Plot predicted and passed-in observed AHe age(s)
                    ax1 = plot_predictions_with_data(
                        float(corr_zhe_ages[-1]),
                        zhe_temps[-1],
                        ax=ax1,
                        marker="d",
                        color="tab:green",
                        label=f"Predicted ZHe age ({float(corr_zhe_ages[-1]):.2f} Ma; T$_c$ = {zhe_temps[-1]:.1f}°C)",
                    )
                    zhe_temps_obs = []
                    for i in range(len(params["obs_zhe"])):
                        zhe_temps_obs.append(zhe_temps[-1])
                    ax1 = plot_measurements(
                        params["obs_zhe"],
                        zhe_temps_obs,
                        xerr=params["obs_zhe_stdev"],
                        ax=ax1,
                        marker="d",
                        color="tab:green",
                        label="Measured ZHe age(s)",
                    )

            # Plot shaded uncertainty area and ZFT age if no measured ages exist
            if n_obs_zft == 0:
                ax1 = plot_predictions_no_data(
                    float(zft_ages[-1]),
                    zft_temps[-1],
                    xerr=zft_uncert * float(zft_ages[-1]),
                    ax=ax1,
                    marker="^",
                    color="tab:red",
                    label=f"Predicted ZFT age ({float(zft_ages[-1]):.2f} ± {zft_uncert * float(zft_ages[-1]):.2f} Ma ({zft_uncert * 100.0:.0f}% error); T$_c$ = {zft_temps[-1]:.1f}°C)",
                )
            # Plot predicted age + observed ZFT age(s)
            else:
                if ages_from_data_file:
                    ax1 = plot_predictions_with_data(
                        pred_ages[obs_zft_indices[0]],
                        pred_data_temps[obs_zft_indices[0]],
                        ax=ax1,
                        marker="^",
                        color="tab:red",
                        label=f"Predicted ZFT age ({pred_ages[obs_zft_indices[0]]:.2f} Ma; T$_c$ = {pred_data_temps[obs_zft_indices[0]]:.1f}°C)",
                    )
                    ax1 = plot_measurements(
                        obs_ages[obs_zft_indices],
                        pred_data_temps[obs_zft_indices],
                        xerr=obs_stdev[obs_zft_indices],
                        ax=ax1,
                        marker="^",
                        color="tab:red",
                        label="Measured ZFT age(s)",
                    )
                else:
                    ax1 = plot_predictions_with_data(
                        float(zft_ages[-1]),
                        zft_temps[-1],
                        ax=ax1,
                        marker="^",
                        color="tab:red",
                        label=f"Predicted ZFT age ({float(zft_ages[-1]):.2f} Ma; T$_c$ = {zft_temps[-1]:.1f}°C)",
                    )
                    zft_temps_obs = []
                    for i in range(len(params["obs_zft"])):
                        zft_temps_obs.append(zft_temps[-1])
                    ax1 = plot_measurements(
                        params["obs_zft"],
                        zft_temps_obs,
                        xerr=params["obs_zft_stdev"],
                        ax=ax1,
                        marker="^",
                        color="tab:red",
                        label="Measured ZFT age(s)",
                    )

            ax1.set_xlim(t_total / myr2sec(1), 0.0)
            ax1.set_ylim(params["temp_surf"], 1.05 * temp_hists[-1].max())
            if params["invert_tt_plot"]:
                ax1.set_ylim(1.05 * temp_hists[-1].max(), params["temp_surf"])
            ax1.set_xlabel("Time (Ma)")
            ax1.set_ylabel("Temperature (°C)")
            if params["plot_depth_history"] or params["plot_fault_depth_history"]:
                # Make left y-axis dimgray
                ax1.set_ylabel("Temperature (°C)", color="dimgray")
                ax1.tick_params(axis="y", colors="dimgray")

                ax1b.set_xlim(t_total / myr2sec(1), 0.0)
                depth_hist_min = 0.0
                if params["plot_depth_history"]:
                    depth_hist_max_particle = depth_hists[-1].max()
                else:
                    depth_hist_max_particle = -1.0e6
                if params["plot_fault_depth_history"]:
                    depth_hist_max_fault = fault_depth_history.max()
                else:
                    depth_hist_max_fault = -1.0e6
                depth_hist_max = max(depth_hist_max_particle, depth_hist_max_fault)
                depth_hist_max = (depth_hist_max / kilo2base(1.0)) * 1.05
                ax1b.set_ylim(depth_hist_min, depth_hist_max)
                if params["invert_tt_plot"]:
                    ax1b.set_ylim(depth_hist_max, depth_hist_min)
                ax1b.set_ylabel("Depth (km)", color="darkgray")
                ax1b.tick_params(axis="y", colors="darkgray")
            # Set plot title
            title_string = "Surface sample thermal history"
            # Include misfit in title if there are measured ages
            if num_passed_ages > 0 or ((num_file_ages > 0) and (not depo_ages_in_file)):
                title_string += f" (misfit = {misfit:.4f}; {len(obs_ages)} age(s))"
            elif depo_ages_in_file:
                title_string += f" (Misfit = {misfit:.4f}; {len(obs_ages)} age(s) including {len(pred_ages[obs_depo_age_file > eps])} depositional ages)"
            ax1.set_title(title_string)

            if params["pad_time"] > 0.0:
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
            if params["plot_depth_history"] or params["plot_fault_depth_history"]:
                ax1.grid(None)
                ax1b.grid(None)
                lines, labels = ax1.get_legend_handles_labels()
                lines2, labels2 = ax1b.get_legend_handles_labels()
                ax1.legend(lines + lines2, labels + labels2)
            else:
                ax1.legend()

            ax2.plot(time_ma, vx_hist / mmyr2ms(1))
            ax2.fill_between(
                time_ma,
                vx_hist / mmyr2ms(1),
                0.0,
                alpha=0.33,
                color="tab:blue",
                label=f"Total erosional exhumation: {exhumation_magnitude:.2f} km",
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

            ft_lengths = np.genfromtxt(ftl_new, delimiter=",", skip_header=1)
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
                savefile = wd / "png" / "cooling_hist.png"
                plt.savefig(savefile, dpi=300)
                print(f"- Thermal history and ages plot written to {savefile}")
            if params["display_plots"]:
                plt.show()
            else:
                plt.close()

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
                    surface_times_ma, aft_ages, marker="s", label="Predicted AFT age"
                )
                ax1.plot(
                    surface_times_ma,
                    corr_zhe_ages,
                    marker="d",
                    label="Predicted ZHe age",
                )
                ax1.plot(
                    surface_times_ma, zft_ages, marker="^", label="Predicted ZFT age"
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
                    marker="s",
                    label="Predicted AFT age",
                )
                ax2.plot(
                    surface_times_ma,
                    corr_zhe_ages + surface_times_ma,
                    marker="d",
                    label="Predicted ZHe age",
                )
                ax2.plot(
                    surface_times_ma,
                    zft_ages + surface_times_ma,
                    marker="^",
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
                    savefile = wd / "png" / "past_ages.png"
                    plt.savefig(savefile, dpi=300)
                    print(f"- Past ages plot written to {savefile}")
                if params["display_plots"]:
                    plt.show()
                else:
                    plt.close()

        # Plot LAB depths
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        ax.plot(time_hists[-1] / myr2sec(1), lab_depths[1:], "o-")

        plt.tight_layout()
        plt.show()
        """

        # Plot vectors, if enabled
        quiver_plot = False
        if quiver_plot:
            # Define crustal box info
            crust_rect_center = 0.0
            crust_rect_base = 0.0
            crust_rwidth = kilo2base(20.0)
            crust_rheight = moho_depth

            # Define mantle box info
            mantle_rect_center = 0.0
            mantle_rect_base = moho_depth
            mantle_rwidth = kilo2base(20.0)
            mantle_rheight = kilo2base(params["max_depth"])

            # Define vector info
            v_ymin = 0.0
            v_ymax = kilo2base(params["max_depth"])
            moho = kilo2base(params["max_depth"] - moho_depth)
            n_vect = len(x)
            vect_mag = 1.0

            # Calculate crustal rectangle values
            crust_anchor_x = crust_rect_center - crust_rwidth / 2.0
            crust_anchor_y = crust_rect_base

            # Calculate mantle rectangle values
            mantle_anchor_x = mantle_rect_center - mantle_rwidth / 2.0
            mantle_anchor_y = mantle_rect_base

            # Create figure and axes
            fig, ax = plt.subplots(1, 1)

            # Create a Rectangle patch
            crust_rect = patches.Rectangle(
                (crust_anchor_x, crust_anchor_y),
                crust_rwidth,
                crust_rheight,
                linewidth=1,
                edgecolor="k",
                facecolor="lightpink",
            )

            # Create a Rectangle patch
            mantle_rect = patches.Rectangle(
                (mantle_anchor_x, mantle_anchor_y),
                mantle_rwidth,
                mantle_rheight,
                linewidth=1,
                edgecolor="k",
                facecolor="lightgray",
            )

            # Add the patches to the Axes
            ax.add_patch(crust_rect)
            ax.add_patch(mantle_rect)

            # Define vectors
            x_vect = np.zeros(len(x))
            y_vect = x
            u_vect = np.zeros(n_vect)
            v_vect = vx_array

            # Plot vectors
            ax.quiver(x_vect, y_vect, u_vect, v_vect)

            # Set plot axis ranges
            # xpadding = 2.0
            # ypadding = 15.0
            # ax.set_aspect('equal', adjustable='box')
            # ax.set_xlim([anchor_x - xpadding, anchor_x + rwidth + xpadding])
            # ax.set_ylim([anchor_y - ypadding, anchor_y + rheight + ypadding])

            ax.axis("equal")
            plt.gca().invert_yaxis()
            # plt.tight_layout()

            # Show plot
            if params["display_plots"]:
                plt.show()
            else:
                plt.close()

    # Read temperature data from file
    if params["read_temps"]:
        load_file = wd / "py" / "output_temps.csv"
        data = np.genfromtxt(load_file, delimiter=",", skip_header=1)
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
        else:
            plt.close()

    # Write temperature data to file
    if params["write_temps"]:
        temp_x_out = np.zeros([len(x), 3])
        temp_x_out[:, 0] = x
        temp_x_out[:, 1] = temp_new
        temp_x_out[:, 2] = temp_init
        savefile = wd / "csv" / "output_temps.csv"
        np.savetxt(
            savefile,
            temp_x_out,
            delimiter=",",
            header="Depth (m),Temperature (deg. C),Initial temperature (deg. C)",
            comments="",
        )
        print(f"- Temperature output writen to {savefile}")

    # Write header in log file if needed
    if params["log_output"]:
        log_output(params, batch_mode=False)

    # Write output to log file
    if params["batch_mode"] or params["log_output"]:
        # Write output to a file
        outfile = wd / "csv" / params["log_file"]

        # Define measured ages for batch output
        if n_obs_ahe == 0:
            obs_ahe = -9999.0
            obs_ahe_stdev = -9999.0
        else:
            if ages_from_data_file:
                obs_ahe = obs_ages[obs_ahe_indices[0]]
                obs_ahe_stdev = obs_stdev[obs_ahe_indices[0]]
            else:
                obs_ahe = params["obs_ahe"][0]
                obs_ahe_stdev = params["obs_ahe_stdev"][0]
        if n_obs_aft == 0:
            obs_aft = -9999.0
            obs_aft_stdev = -9999.0
        else:
            if ages_from_data_file:
                obs_aft = obs_ages[obs_aft_indices[0]]
                obs_aft_stdev = obs_stdev[obs_aft_indices[0]]
            else:
                obs_aft = params["obs_aft"][0]
                obs_aft_stdev = params["obs_aft_stdev"][0]
        if n_obs_zhe == 0:
            obs_zhe = -9999.0
            obs_zhe_stdev = -9999.0
        else:
            if ages_from_data_file:
                obs_zhe = obs_ages[obs_ahe_indices[0]]
                obs_zhe_stdev = obs_stdev[obs_ahe_indices[0]]
            else:
                obs_zhe = params["obs_zhe"][0]
                obs_zhe_stdev = params["obs_zhe_stdev"][0]
        if n_obs_zft == 0:
            obs_zft = -9999.0
            obs_zft_stdev = -9999.0
        else:
            if ages_from_data_file:
                obs_zft = obs_ages[obs_zft_indices[0]]
                obs_zft_stdev = obs_stdev[obs_zft_indices[0]]
            else:
                obs_zft = params["obs_zft"][0]
                obs_zft_stdev = params["obs_zft_stdev"][0]

        # Define misfit details for output
        if (num_passed_ages == 0) and (num_file_ages == 0):
            misfit = -9999.0
            misfit_type = -9999.0
            misfit_ages = 0
        else:
            misfit_type = params["misfit_type"]
            misfit_ages = len(obs_ages)

        # Print warnings if there are multiple observed ages to write to the log file
        age_types = ["AHe", "AFT", "ZHe", "ZFT"]
        obs_age_nums = [n_obs_ahe, n_obs_aft, n_obs_zhe, n_obs_zft]
        if (not params["batch_mode"]) or (not params["inverse_mode"]):
            if (n_obs_ahe > 1) or (n_obs_aft > 1) or (n_obs_zhe > 1) or (n_obs_zft > 1):
                print("")
                for i in range(len(age_types)):
                    if obs_age_nums[i] > 1:
                        warnings.warn(
                            f"More than one measured {age_types[i]} age supplied, only the first will be written to the output file!",
                            stacklevel=2,
                        )

        # Open log file for writing
        with open(outfile, "a+") as f:
            f.write(
                f"{t_total / myr2sec(1):.4f},{dt / yr2sec(1):.4f},{max_depth / kilo2base(1):.4f},{params['nx']},"
                f"{params['temp_surf']:.4f},{params['temp_base']:.4},{params['mantle_adiabat']},"
                f"{params['rho_crust']:.4f},{params['removal_fraction']:.4f},{params['removal_start_time']:.4f},"
                f"{params['removal_end_time']:.4f},{params['ero_type']},{params['ero_option1']:.4f},"
                f"{params['ero_option2']:.4f},{params['ero_option3']:.4f},{params['ero_option4']:.4f},"
                f"{params['ero_option5']:.4f},{params['ero_option6']:.4f},{params['ero_option7']:.4f},{params['ero_option8']:.4f},{params['ero_option9']:.4f},{params['ero_option10']:.4f},{params['init_moho_depth']:.4f},{init_moho_temp:.4f},"
                f"{init_heat_flow:.4f},{elev_list[1] / kilo2base(1):.4f},{moho_depth / kilo2base(1):.4f},"
                f"{final_moho_temp:.4f},{final_heat_flow:.4f},{elev_list[-1] / kilo2base(1):.4f},"
                f"{exhumation_magnitude:.4f},{params['ap_rad']:.4f},{params['ap_uranium']:.4f},"
                f"{params['ap_thorium']:.4f},{params['zr_rad']:.4f},{params['zr_uranium']:.4f},"
                f"{params['zr_thorium']:.4f},{float(corr_ahe_ages[-1]):.4f},"
                f"{ahe_temps[-1]:.4f},{obs_ahe:.4f},"
                f"{obs_ahe_stdev:.4f},{float(aft_ages[-1]):.4f},"
                f"{aft_temps[-1]:.4f},{obs_aft:.4f},"
                f"{obs_aft_stdev:.4f},{float(corr_zhe_ages[-1]):.4f},"
                f"{zhe_temps[-1]:.4f},{obs_zhe:.4f},"
                f"{obs_zhe_stdev:.4f},{float(zft_ages[-1]):.4f},"
                f"{zft_temps[-1]:.4f},{obs_zft:.4f},"
                f"{obs_zft_stdev:.4f},{misfit:.6f},{misfit_type},{misfit_ages}\n"
            )

    # Write summary age output to file
    if params["write_age_output"]:
        savefile = wd / "csv" / "age_summary.csv"
        # Use sample IDs from data file, or None otherwise
        if ages_from_data_file:
            # Fill in age types
            obs_age_types = (
                ["AHe"] * n_obs_ahe
                + ["AFT"] * n_obs_aft
                + ["ZHe"] * n_obs_zhe
                + ["ZFT"] * n_obs_zft
            )
            # Use sample IDs from data file
            sample_id_out = obs_sample_id_file
            # Store predicted age eU, grain radius
            pred_eu = obs_eu
            pred_radius = obs_radius
        else:
            # Fill in age types
            obs_age_types = (
                ["AHe"] * len(params["obs_ahe"])
                + ["AFT"] * len(params["obs_aft"])
                + ["ZHe"] * len(params["obs_zhe"])
                + ["ZFT"] * len(params["obs_zft"])
            )
            obs_age_types = np.array(obs_age_types)
            # Use empty sample IDs
            sample_id_out = np.array([""] * len(obs_ages))
            # Create array of predicted age eU values
            pred_eu = np.empty(len(obs_ages))
            if len(params["obs_ahe"]) > 0:
                pred_eu[obs_age_types == "AHe"] = calculate_eu(
                    params["ap_uranium"], params["ap_thorium"]
                )
            if len(params["obs_aft"]) > 0:
                pred_eu[obs_age_types == "AFT"] = None
            if len(params["obs_zhe"]) > 0:
                pred_eu[obs_age_types == "ZHe"] = calculate_eu(
                    params["zr_uranium"], params["zr_thorium"]
                )
            if len(params["obs_zft"]) > 0:
                pred_eu[obs_age_types == "ZFT"] = None
            pred_eu = pred_eu.astype("str")
            pred_eu[pred_eu == "nan"] = ""
            # Create array of predicted age radius values
            pred_radius = np.empty(len(obs_ages))
            if len(params["obs_ahe"]) > 0:
                pred_radius[np.strings.lower(obs_age_types) == "ahe"] = params["ap_rad"]
            if len(params["obs_aft"]) > 0:
                pred_radius[np.strings.lower(obs_age_types) == "aft"] = None
            if len(params["obs_zhe"]) > 0:
                pred_radius[np.strings.lower(obs_age_types) == "zhe"] = params["zr_rad"]
            if len(params["obs_zft"]) > 0:
                pred_radius[np.strings.lower(obs_age_types) == "zft"] = None
            pred_radius = pred_radius.astype("str")
            pred_radius[pred_radius == "nan"] = ""
        # Create output list, rounding predicted ages to 2 decimals
        summary_ages = [
            list(x)
            for x in zip(
                obs_age_types,
                obs_ages,
                obs_stdev,
                obs_eu,
                obs_radius,
                sample_id_out,
                pred_ages.round(2),
                pred_eu,
                pred_radius,
            )
        ]
        np.savetxt(
            savefile,
            summary_ages,
            delimiter=",",
            header="Age type, Observed age (Ma), Observed age stdev (Ma), Observed age eU (ppm), Observed age grain radius (um), Sample ID, Predicted age (Ma), Predicted age eU (ppm), Predicted age grain radius (um)",
            comments="",
            fmt="%s",
        )
        print(f"- Summary age output written to {savefile}")

    if not params["batch_mode"]:
        print("")
        if not params["plot_results"]:
            exec_end = time.time()
        print(
            f"{18 * '-'} Execution completed in {exec_end - exec_start:10.4f} seconds {19 * '-'}"
        )

        # Returns misfit for inverse_mode
    if "misfit" in locals():
        # print(f"- Returning misfit: {misfit}")
        return misfit
