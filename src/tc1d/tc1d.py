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
import time
import math
from typing import Tuple
import warnings
from ctypes import CDLL, POINTER, c_double, c_float, c_int, c_void_p, pointer
import platform

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
try:
    from importlib.resources import files
    import importlib.metadata

    __version__ = importlib.metadata.version("tc1d")
except ImportError:
    __version__ = "dev"


# Shared object loading
def load_lib(base_name="RDAAM"):  # or "ketch"
    root = files("tc1d")
    system = platform.system().lower()
    if system == "linux":
        libname = f"lib{base_name}.so"
    elif system == "darwin":
        libname = f"lib{base_name}.so"
    else:
        libname = f"{base_name}.dll"
    return CDLL(os.fspath(root.joinpath(libname)))


# RDAAM library for AHe and ZHe
def load_rdaam():
    rdaam = load_lib("RDAAM")
    rdaam.make_path.restype = c_void_p
    rdaam.path_push.argtypes = [c_void_p, c_double, c_double]
    rdaam.path_push.restype = None
    rdaam.del_path.argtypes = [c_void_p]
    rdaam.del_path.restype = None
    rdaam.run_RDAAM_He.argtypes = [
        c_void_p,  # TTPath*
        c_double,  # ap_rad
        c_double,  # ap_U
        c_double,  # ap_Th
        POINTER(c_double),  # ap_age*
        POINTER(c_double),  # ap_corrAge*
        c_double,  # total_He
        c_double,  # zr_rad
        c_double,  # zr_U
        c_double,  # zr_Th
        POINTER(c_double),  # zr_age*
        POINTER(c_double),  # zr_corrAge*
        POINTER(c_int),  # ap_success*
        POINTER(c_int),  # zr_success*
    ]
    rdaam.run_RDAAM_He.restype = None
    return rdaam


# ketch_aft for AFT
def load_ketch():
    ketch = load_lib("ketch")
    return ketch


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
def yr2sec(time_yr: float) -> float:
    """Converts time from years to seconds."""
    return time_yr * 60.0 * 60.0 * 24.0 * 365.25


def myr2sec(time_myr: float) -> float:
    """Converts time from million years to seconds."""
    return yr2sec(time_myr) * 1.0e6


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


def calculate_heat_flow(
    temperature: np.ndarray,
    conductivity: np.ndarray,
    dx: float,
    nstart: int = 0,
    nx: int = 1,
) -> float:
    """Calculates heat flow in W/m2."""
    return kilo2base(
        (conductivity[nstart] + conductivity[nstart + nx])
        / 2
        * (temperature[nstart + nx] - temperature[nstart])
        / (nx * dx)
    )


def echo_model_info(
    dx: float,
    nt: int,
    dt: float,
    t_total: float,
    solution_type: int,
    bc_type: int,
    ero_type: int,
    exhumation_magnitude: float,
    cond_stab: float,
    adv_stab: float,
    cond_crit: float,
    adv_crit: float,
) -> None:
    print("")
    print("--- General model information ---")
    print("")
    print(f"- Node spacing: {dx} m")
    print(f"- Total simulation time: {t_total / myr2sec(1):.1f} million years")
    print(f"- Time steps: {nt} @ {dt / yr2sec(1):.1f} years each")

    # Output thermal calculation type
    solution_types = {
        1: "Explicit",
        2: "Implicit",
        3: "Crank-Nicolson",
    }
    print(f"- Solution type: {solution_types[solution_type]}")

    boundary_condition_types = {
        1: "Constant temperature (Dirichlet)",
        2: "Constant heat flux (Neumann)",
    }
    # Check stability conditions
    if solution_type == 1:
        print(
            f"- Conductive stability: {(cond_stab < cond_crit)} ({cond_stab:.3f} < {cond_crit:.4f})"
        )
        print(
            f"- Advective stability: {(adv_stab < adv_crit)} ({adv_stab:.3f} < {adv_crit:.4f})"
        )
    print(f"- Basal boundary condition: {boundary_condition_types[bc_type]}")

    # Output erosion model
    ero_models = {
        0: "Erosion model: Multi-stage erosion from YAML (ero_stages)",
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
    vx: float,
    k_crust: float,
    rho_crust: float,
    cp_crust: float,
    k_mantle: float,
    rho_mantle: float,
    cp_mantle: float,
    k_a: float,
    dt: float,
    dx: float,
    cond_crit: float,
    adv_crit: float,
) -> Tuple[float, float]:
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


# Peclet number calculation
def calculate_peclet_number(
    ero_type: int,
    vx: float,
    x: np.ndarray,
    xstag: np.ndarray,
    k: np.ndarray,
    rho: np.ndarray,
    cp: np.ndarray,
    moho_depth: float,
    max_depth: float,
    crustal_uplift: bool,
):
    """Calculates the Peclet number."""
    # FIXME: This does not currently support ero type 7
    if ero_type == 7:
        peclet = -1.0
    else:
        if crustal_uplift:
            kappa_eff = k[xstag <= moho_depth].mean() / (
                rho[x <= moho_depth].mean() * cp[x <= moho_depth].mean()
            )
            peclet = abs(vx) * moho_depth / kappa_eff
        else:
            kappa_eff = k.mean() / (rho.mean() * cp.mean())
            peclet = abs(vx) * max_depth / kappa_eff

    return peclet


# Dimensionless heat production calculation
def calculate_nd_heat_prod(
    heat_prod: np.ndarray,
    k: np.ndarray,
    reference_temp: float,
    max_depth: float,
):
    """Calculates a dimensionless heat production factor."""
    # TODO: Handle the fixed temperature at depth better?
    nd_heat_prod = (heat_prod.mean() * max_depth**2) / (2 * k.mean() * reference_temp)

    return nd_heat_prod


# Mantle adiabat from Turcotte and Schubert (eqn 4.254)
def adiabat(alphav: float, temp: float, cp: float) -> float:
    """Calculates a mantle adiabat in degress / m."""
    grav = 9.81
    return alphav * grav * temp / cp


# Conductive steady-state heat transfer
def temp_ss_implicit(
    nx: int,
    dx: float,
    a_matrix: np.ndarray,
    b_vector: np.ndarray,
    temp_surf: float,
    temp_base: float,
    vx: np.ndarray,
    rho: np.ndarray,
    cp: np.ndarray,
    k: np.ndarray,
    heat_prod: np.ndarray,
) -> np.ndarray:
    """Calculates a steady-state thermal solution."""
    # Set B.C. values in the coefficient array and in the r.h.s. array
    a_matrix[0, 0] = 1
    b_vector[0] = temp_surf
    a_matrix[nx - 1, nx - 1] = 1
    b_vector[nx - 1] = temp_base

    # Matrix loop
    for ix in range(1, nx - 1):
        a_matrix[ix, ix - 1] = (-(rho[ix] * cp[ix] * -vx[ix]) / (2 * dx)) - k[
            ix - 1
        ] / dx**2
        a_matrix[ix, ix] = k[ix] / dx**2 + k[ix - 1] / dx**2
        a_matrix[ix, ix + 1] = (rho[ix] * cp[ix] * -vx[ix]) / (2 * dx) - k[ix] / dx**2
        b_vector[ix] = heat_prod[ix]

    temp = solve(a_matrix, b_vector)
    return temp


def update_materials(
    x: np.ndarray,
    xstag: np.ndarray,
    moho_depth: float,
    rho_crust: float,
    rho_mantle: float,
    rho: np.ndarray,
    cp_crust: float,
    cp_mantle: float,
    cp: np.ndarray,
    k_crust: float,
    k_mantle: float,
    k: np.ndarray,
    heat_prod_crust: float,
    heat_prod_decay_depth: float,
    heat_prod_mantle: float,
    heat_prod: np.ndarray,
    temp_adiabat: np.ndarray,
    temp_prev: np.ndarray,
    k_a: float,
    delaminated: bool,
    removal_fraction: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
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


def init_ero_types(
    params: dict,
    x: np.ndarray,
    xstag: np.ndarray,
    temp_prev: np.ndarray,
    moho_depth: float,
) -> Tuple[
    np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
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


# TODO: Add support for basal flux boundary condition
def temp_transient_explicit(
    nx: int,
    dx: float,
    dt: float,
    temp_prev: np.ndarray,
    temp_new: np.ndarray,
    bc_type: int,
    temp_surf: float,
    temp_base: float,
    flux_base: float,
    vx: np.ndarray,
    rho: np.ndarray,
    cp: np.ndarray,
    k: np.ndarray,
    heat_prod: np.ndarray,
) -> np.ndarray:
    """Updates a transient thermal solution."""
    # Set boundary conditions
    temp_new[0] = temp_surf
    if bc_type == 1:
        # Assign constant basal temperature
        temp_new[nx - 1] = temp_base
    elif bc_type == 2:
        temp_new[nx - 1] = temp_prev[-1] + (2 * k[-1] * dt) / (
            rho[-1] * cp[-1] * dx**2
        ) * (temp_prev[-2] - temp_prev[-1] + (flux_base * dx) / k[-1])
    else:
        raise ValueError(f"Bad boundary condition type: {bc_type}. Must be 1 or 2.")

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
    nx: int,
    dx: float,
    dt: float,
    a_matrix: np.ndarray,
    b_vector: np.ndarray,
    temp_prev: np.ndarray,
    vx: np.ndarray,
    rho: np.ndarray,
    cp: np.ndarray,
    k: np.ndarray,
    heat_prod: np.ndarray,
) -> np.ndarray:
    """Calculates a transient implicit thermal solution."""
    # Matrix loop
    for ix in range(1, nx - 1):
        a_matrix[ix, ix - 1] = (
            -(rho[ix] * cp[ix] * -vx[ix]) / (2 * dx) - k[ix - 1] / dx**2
        )
        a_matrix[ix, ix] = (rho[ix] * cp[ix]) / dt + k[ix] / dx**2 + k[ix - 1] / dx**2
        a_matrix[ix, ix + 1] = (rho[ix] * cp[ix] * -vx[ix]) / (2 * dx) - k[ix] / dx**2
        b_vector[ix] = heat_prod[ix] + ((rho[ix] * cp[ix]) / dt) * temp_prev[ix]

    temp = solve(a_matrix, b_vector)
    return temp


# Transient heat transfer equation solved using Crank-Nicolson method
def temp_transient_crank_nicolson(
    nx: int,
    dx: float,
    dt: float,
    a_matrix: np.ndarray,
    b_vector: np.ndarray,
    temp_prev: np.ndarray,
    vx: np.ndarray,
    rho: np.ndarray,
    cp: np.ndarray,
    k: np.ndarray,
    heat_prod: np.ndarray,
) -> np.ndarray:
    """Calculates a steady-state thermal solution."""
    # Matrix loop
    for ix in range(1, nx - 1):
        a_matrix[ix, ix - 1] = -k[ix - 1] / dx**2 - (rho[ix] * cp[ix] * -vx[ix]) / (
            2 * dx
        )
        a_matrix[ix, ix] = (
            k[ix] / dx**2 + k[ix - 1] / dx**2 + (2 * rho[ix] * cp[ix]) / dt
        )
        a_matrix[ix, ix + 1] = -k[ix] / dx**2 + (rho[ix] * cp[ix] * -vx[ix]) / (2 * dx)
        b_vector[ix] = (
            (k[ix] / dx**2 - (rho[ix] * cp[ix] * -vx[ix]) / (2 * dx))
            * temp_prev[ix + 1]
            + (-k[ix] / dx**2 - k[ix - 1] / dx**2 + (2 * rho[ix] * cp[ix]) / dt)
            * temp_prev[ix]
            + (k[ix - 1] / dx**2 + (rho[ix] * cp[ix] * -vx[ix]) / (2 * dx))
            * temp_prev[ix - 1]
            + 2 * heat_prod[ix]
        )

    temp = solve(a_matrix, b_vector)
    return temp


def create_intrusion(
    temperature: float,
    start_time: float,
    base_depth: float,
    thickness: float,
    duration: float = 0.0,
) -> Intrusion:
    """Creates an instance of an intrusion."""
    intrusion = Intrusion(temperature, start_time, duration, base_depth, thickness)
    return intrusion


def apply_intrusion(
    intrusion: Intrusion, x: np.ndarray, model_temperatures: np.ndarray
) -> np.ndarray:
    """Applies intrusion temperatures to the intrusion depths."""
    intrusion_slice = (x >= intrusion.base_depth - intrusion.thickness) & (
        x <= intrusion.base_depth
    )
    model_temperatures[intrusion_slice] = intrusion.temperature
    return model_temperatures


# TODO: Sort out why type hinting is problematic for this function
def he_ages(
    rdaam,
    pa,
    ap_rad,
    ap_uranium,
    ap_thorium,
    zr_rad,
    zr_uranium,
    zr_thorium,
):
    """Calculates (U-Th)/He ages."""

    # Initialize library parameters
    ap_age = c_double(0.0)
    ap_corrAge = c_double(0.0)
    zr_age = c_double(0.0)
    zr_corrAge = c_double(0.0)
    ap_success = c_int(0)
    zr_success = c_int(0)

    # Calculate age
    rdaam.run_RDAAM_He(
        pa,  # path
        c_double(ap_rad),  # ap_rad
        c_double(ap_uranium),  # ap_U
        c_double(ap_thorium),  # ap_Th
        pointer(ap_age),  # ap_age
        pointer(ap_corrAge),  # ap_corrAge
        c_double(0.0),  # total_He
        c_double(zr_rad),  # zr_rad
        c_double(zr_uranium),  # zr_U
        c_double(zr_thorium),  # zr_Th
        pointer(zr_age),  # zr_age
        pointer(zr_corrAge),  # zr_corrAge
        pointer(ap_success),  # ap_success
        pointer(zr_success),  # zr_success
    )

    # Exit if age calculation fails
    if ap_success.value != 1:
        print(f"RDAAM_calculate for ap failed! {ap_success}")
        exit(1)
    if zr_success.value != 1:
        print(f"RDAAM_calculate for zr failed! {zr_success}")
        exit(1)

    return ap_age.value, ap_corrAge.value, zr_age.value, zr_corrAge.value


# TODO: Sort out why type hinting is problematic for this function
def ft_ages(ti_arr, te_arr, n, write_track_lengths):
    """Calculates AFT ages."""

    # Load library to calculate AFT age
    ketch = load_ketch()

    # Initialize library parameters
    ntime = c_int(n)
    alo = c_double(16.3)
    final_age = c_double(0.0)
    oldest_age = c_double(0.0)
    fmean = c_double(0.0)
    fdist = (c_double * 201)()

    # Calculate age
    ketch.ketch_main(
        pointer(ntime),  # int *ntime
        ti_arr,  # float ketchtime[]
        te_arr,  # float ketchtemp[]
        pointer(alo),  # double *alo
        pointer(final_age),  # double *final_age
        pointer(oldest_age),  # double *oldest_age
        pointer(fmean),  # double *fmean
        fdist,  # double fdist[]
    )

    if write_track_lengths:
        with open("ft_length.csv", "w") as out_csv:
            for i, d in enumerate(fdist):
                out_csv.write(f"{(i*1.0+0.5)*20.0/200},{d}\n")

    return float(final_age.value), float(fmean.value)


def calculate_closure_temp(
    age: float, time_history: np.ndarray, temp_history: np.ndarray
) -> float:
    """Calculates closure temperature from predicted age and cooling history."""
    closure_temp = np.interp(age, time_history, temp_history)
    return closure_temp


def tt_hist_to_ma(time_history: np.ndarray) -> np.ndarray:
    """Converts a time-temperature history in Myr since model start to Ma."""
    current_max_time = time_history.max()
    time_ma = current_max_time - time_history
    time_ma = time_ma / myr2sec(1)
    return time_ma


def get_write_increment(params: dict, time_ma: np.ndarray) -> int:
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


def save_tt_history(
    params: dict, tt_filename: str, time_history: np.ndarray, temp_history: np.ndarray
) -> None:
    "Writes a time-temperature history to a file."
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


def save_ttdp_history(
    params: dict,
    ttdp_filename: str,
    time_history: np.ndarray,
    temp_history: np.ndarray,
    depth_history: np.ndarray,
    pressure_history: np.ndarray,
) -> None:
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
    write_tt_history,
    tt_filename,
    write_ttdp_history,
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

    # Write time-temperature history to file, if requested
    if write_tt_history:
        save_tt_history(params, tt_filename, time_history, temp_history)

    # Write pressure-time-temperature-depth history to file, if requested
    if write_ttdp_history:
        save_ttdp_history(
            params,
            ttdp_filename,
            time_history,
            temp_history,
            depth_history,
            pressure_history,
        )

    rdaam = load_rdaam()
    pa = rdaam.make_path()
    time_ma = tt_hist_to_ma(time_history)
    write_increment = get_write_increment(params, time_ma)
    ti_hist = []
    te_hist = []

    for i in range(len(time_ma) - 1, -1, -write_increment):
        ti_hist.append(time_ma[i])
        te_hist.append(temp_history[i])
        rdaam.path_push(pa, c_double(time_ma[i]), c_double(temp_history[i]))
    if params["pad_time"] > 0.0:
        pad_times = np.arange(
            time_ma.max(), time_ma.max() + params["pad_time"] + 0.1, 1.0
        )
        for pad_time in pad_times:
            ti_hist.append(pad_time)
            te_hist.append(temp_history[i])
            rdaam.path_push(pa, c_double(pad_time), c_double(temp_history[i]))

    ahe_age, corr_ahe_age, zhe_age, corr_zhe_age = he_ages(
        rdaam=rdaam,
        pa=pa,
        ap_rad=params["ap_rad"],
        ap_uranium=params["ap_uranium"],
        ap_thorium=params["ap_thorium"],
        zr_rad=params["zr_rad"],
        zr_uranium=params["zr_uranium"],
        zr_thorium=params["zr_thorium"],
    )
    rdaam.del_path(pa)

    ti_arr = (c_float * len(ti_hist))()
    te_arr = (c_float * len(te_hist))()
    for i in range(len(ti_hist)):
        ti_arr[i] = c_float(ti_hist[i])
        te_arr[i] = c_float(te_hist[i])
    if params["ketch_aft"]:
        aft_age, aft_mean_ftl = ft_ages(
            ti_arr, te_arr, len(ti_hist), write_track_lengths
        )

    # Find effective closure temperatures
    ahe_temp = calculate_closure_temp(
        corr_ahe_age, np.flip(time_ma), np.flip(temp_history)
    )
    aft_temp = calculate_closure_temp(aft_age, np.flip(time_ma), np.flip(temp_history))
    zhe_temp = calculate_closure_temp(
        corr_zhe_age, np.flip(time_ma), np.flip(temp_history)
    )
    zft_temp = calculate_closure_temp(zft_age, np.flip(time_ma), np.flip(temp_history))

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


def parse_ero_stage_row(row: dict, stage_idx: int = 1) -> dict:
    """
    BG: Parse one erosion stage from YAML and build the internal rate-based
    parameterization used by the forward model (ero_type=0).

    YAML stage schema
    -----------------
    Each stage is a dict with:
      type: "constant" / "linear" / "exponential"
      unit: "erosion_rate" / "thickness"
      duration_myr: float > 0
      p1/p2/p3 (or parameter1/2/3): floats (some optional depending on type/unit)

    Parameter conventions
    ---------------------
    - constant + erosion_rate : p1 = r_const
    - constant + thickness    : p1 = M_total   (converted internally to mean rate)

    - linear + erosion_rate   : p1 = r_start, p2 = r_end
    - linear + thickness      : p1 = M_total, p2 = s (dimensionless in [-1, 1])

    - exponential + erosion_rate : p1 = r_start, p2 = tau, p3 = r_target
    - exponential + thickness    : p1 = M_total, p2 = tau, p3 = M_base
                                   (baseline thickness; internal rates are reconstructed)

    Returns
    -------
    stage : dict
        {
          "type": str,
          "unit": str,
          "duration_myr": float,
          "dt_sec": float,
          "params": np.ndarray,      # internal: constant [r], linear [r0,r1], exponential [r0,rt,tau]
          "input_params": list,      # raw YAML p1/p2/p3 for echo/debug
        }
    """

    if not isinstance(row, dict):
        raise ValueError(f"BG: erosion YAML stage {stage_idx}: stage must be a dict.")

    def _as_float(v, name):
        # BG: YAML may give float/int directly; also accept strings.
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip()
        if s == "":
            return None
        try:
            return float(s)
        except ValueError:
            raise ValueError(
                f"BG: erosion YAML stage {stage_idx}: {name} must be a float (got {v!r})."
            )

    stage_type = str(row.get("type", "")).strip().lower()
    unit = str(row.get("unit", "")).strip().lower()

    if stage_type not in ("constant", "linear", "exponential"):
        raise ValueError(
            f"BG: erosion YAML stage {stage_idx}: invalid type={stage_type!r}. "
            "Valid: constant, linear, exponential."
        )

    if unit not in ("erosion_rate", "thickness"):
        raise ValueError(
            f"BG: erosion YAML stage {stage_idx}: invalid unit={unit!r}. "
            "Valid: erosion_rate, thickness."
        )

    duration_myr = _as_float(row.get("duration_myr", None), "duration_myr")
    if duration_myr is None or duration_myr <= 0.0:
        raise ValueError(
            f"BG: erosion YAML stage {stage_idx}: duration_myr must be > 0 (got {duration_myr})."
        )
    dt_stage_myr = duration_myr

    p1 = _as_float(row.get("p1", row.get("parameter1", None)), "p1/parameter1")
    p2 = _as_float(row.get("p2", row.get("parameter2", None)), "p2/parameter2")
    p3 = _as_float(row.get("p3", row.get("parameter3", None)), "p3/parameter3")
    input_params = [p1, p2, p3]

    # ============================
    # BG: Build internal parameterization (rate-based)
    # ============================

    if stage_type == "constant":
        if unit == "erosion_rate":
            if p1 is None:
                raise ValueError(
                    f"BG: erosion YAML stage {stage_idx}: constant + erosion_rate requires parameter1=r_const."
                )
            r_const = p1
        else:  # thickness
            if p1 is None:
                raise ValueError(
                    f"BG: erosion YAML stage {stage_idx}: constant + thickness requires parameter1=M_total."
                )
            r_const = p1 / dt_stage_myr  # km/Myr
        params = np.array([r_const], dtype=float)

    elif stage_type == "linear":
        if unit == "erosion_rate":
            if p1 is None or p2 is None:
                raise ValueError(
                    f"BG: erosion YAML stage {stage_idx}: linear + erosion_rate requires p1=r_start, p2=r_end."
                )
            r_start, r_end = p1, p2
        else:  # thickness
            if p1 is None or p2 is None:
                raise ValueError(
                    f"BG: erosion YAML stage {stage_idx}: linear + thickness requires p1=M_total, p2=s in [-1,1]."
                )
            M_total, s = p1, p2
            if s < -1.0 or s > 1.0:
                raise ValueError(
                    f"BG: erosion YAML stage {stage_idx}: linear + thickness requires s in [-1,1] (got {s})."
                )
            r_mean = M_total / dt_stage_myr
            r_start = r_mean * (1.0 - s)
            r_end = r_mean * (1.0 + s)
        params = np.array([r_start, r_end], dtype=float)

    else:  # exponential
        if p2 is None or p2 <= 0.0:
            raise ValueError(
                f"BG: erosion YAML stage {stage_idx}: exponential requires parameter2=tau > 0."
            )
        tau = p2

        if unit == "erosion_rate":
            if p1 is None or p3 is None:
                raise ValueError(
                    f"BG: erosion YAML stage {stage_idx}: exponential + erosion_rate requires p1=r_start, p2=tau, p3=r_target."
                )
            r_start, r_target = p1, p3
        else:  # thickness
            if p1 is None or p3 is None:
                raise ValueError(
                    f"BG: erosion YAML stage {stage_idx}: exponential + thickness requires p1=M_total, p2=tau, p3=M_base."
                )
            M_total, M_base = p1, p3
            r_target = M_base / dt_stage_myr

            # BG: enforce strict thickness conservation over the stage duration
            A = tau * (1.0 - math.exp(-dt_stage_myr / tau))
            if abs(A) < 1e-12:
                raise ValueError(
                    f"BG: erosion YAML stage {stage_idx}: cannot compute r_start (tau too small or duration too small). "
                    f"tau={tau}, duration={dt_stage_myr}."
                )
            r_start = r_target + (M_total - r_target * dt_stage_myr) / A

        params = np.array([r_start, r_target, tau], dtype=float)

    return {
        "type": stage_type,
        "unit": unit,
        "duration_myr": dt_stage_myr,
        "dt_sec": myr2sec(dt_stage_myr),
        "params": params,
        "input_params": input_params,
    }


def read_ero_stages_from_yaml(ero_stages_input) -> list:
    """
    BG: Read and validate YAML erosion stages for ero_type=0.

    Parameters
    ----------
    ero_stages_input : list of dict
        Raw YAML list under:
          erosion_model:
            ero_stages: [...]

    Returns
    -------
    stages : list of dict
        Parsed stages with internal 'params' representation (rate-based).
    """
    if ero_stages_input is None:
        raise ValueError(
            "BG: ero_type=0 requires erosion_model.ero_stages in the YAML input file."
        )

    if not isinstance(ero_stages_input, list) or len(ero_stages_input) == 0:
        raise ValueError("BG: erosion_model.ero_stages must be a non-empty list.")

    stages = []
    for i, row in enumerate(ero_stages_input, start=1):
        stages.append(parse_ero_stage_row(row, stage_idx=i))

    return stages


def _stage_idx_from_na_param_name(pname):
    """
    BG: Extract 1-based stage index from NA param name like:
        'ero_stage01_duration_myr' or 'ero_stage01_p2'
    Returns int stage index (1-based).
    """
    try:
        token = pname.split("_")[1]  # "stage01"
        idx = int(token.replace("stage", ""))
    except Exception as exc:
        raise ValueError(
            f"BG: cannot parse erosion stage index from {pname!r}."
        ) from exc
    return idx


def _get_stage_param_value(st, key):
    """
    BG: Read stage parameter with backward-compatible YAML keys.
    Accept both p1/p2/p3 and parameter1/2/3.
    """
    if key == "p1":
        return st.get("p1", st.get("parameter1", None))
    if key == "p2":
        return st.get("p2", st.get("parameter2", None))
    if key == "p3":
        return st.get("p3", st.get("parameter3", None))
    raise ValueError(f"BG: unknown stage parameter key {key!r}.")


def _collect_ero_stage_duration_bounds(ero_stages_template):
    """
    BG: Collect duration bounds from YAML ero_stages (raw template).

    Expected syntax for inversion:
        duration_myr: [min, max]

    Returns
    -------
    dict mapping NA param name -> [min, max]
        e.g. "ero_stage01_duration_myr": [2.0, 6.0]
    """
    bounds = {}
    if not ero_stages_template:
        return bounds

    for i, st in enumerate(ero_stages_template, start=1):
        dur = st.get("duration_myr", None)

        # BG: inversion syntax uses a 2-item range [min, max]
        if isinstance(dur, (list, tuple)) and len(dur) == 2:
            dmin = float(dur[0])
            dmax = float(dur[1])

            if dmin <= 0.0 or dmax <= 0.0 or dmax <= dmin:
                raise ValueError(
                    f"BG: erosion YAML stage {i}: duration_myr bounds must satisfy "
                    f"0 < min < max (got {dur})."
                )

            key = f"ero_stage{i:02d}_duration_myr"
            bounds[key] = [dmin, dmax]

    return bounds


def _collect_ero_stage_param_bounds(ero_stages_template):
    """
    BG: Collect p1/p2/p3 bounds from YAML ero_stages (raw template).

    Expected syntax for inversion (either key style supported):
        p1: [min, max]   or  parameter1: [min, max]
        p2: [min, max]   or  parameter2: [min, max]
        p3: [min, max]   or  parameter3: [min, max]

    Returns
    -------
    dict mapping NA param name -> [min, max]
        e.g. "ero_stage01_p1": [0.0, 2.0]
    """
    bounds = {}
    if not ero_stages_template:
        return bounds

    for i, st in enumerate(ero_stages_template, start=1):
        for key in ("p1", "p2", "p3"):
            v = _get_stage_param_value(st, key)

            # BG: inversion syntax uses a 2-item range [min, max]
            if isinstance(v, (list, tuple)) and len(v) == 2:
                vmin = float(v[0])
                vmax = float(v[1])

                if vmax <= vmin:
                    raise ValueError(
                        f"BG: erosion YAML stage {i}: {key} bounds must satisfy "
                        f"min < max (got {v})."
                    )

                pname = f"ero_stage{i:02d}_{key}"
                bounds[pname] = [vmin, vmax]

    return bounds


def _apply_na_stage_durations(
    ero_stages_template,
    duration_param_names,
    x,
    t_total_myr,
    balance=True,
):
    """
    BG: Apply NA-sampled durations into a copy of erosion stages.

    Parameters
    ----------
    ero_stages_template : list[dict]
        Raw YAML stages (no dt_sec cache).
    duration_param_names : list[str]
        Names like 'ero_stage01_duration_myr' in order matching x.
    x : array-like
        NA sampled values.
    t_total_myr : float
        Total model duration in Myr.
    balance : bool
        If True, adjust one non-inverted stage to keep sum(durations)=t_total_myr.

    Returns
    -------
    new_stages : list[dict]
        Updated stages (still raw YAML dicts).
    """
    stages = copy.deepcopy(ero_stages_template)
    nst = len(stages)

    # BG: Map x values to stage index
    inverted_stage_idxs = set()
    for pname, val in zip(duration_param_names, x):
        idx = _stage_idx_from_na_param_name(pname)

        if idx < 1 or idx > nst:
            raise ValueError(
                f"BG: duration param {pname!r} refers to stage {idx}, "
                f"but only {nst} stages exist."
            )

        inverted_stage_idxs.add(idx)
        stages[idx - 1]["duration_myr"] = float(val)

    # BG: Balancing to ensure sum durations == t_total
    if balance:
        # pick last stage not inverted (preferably)
        balance_idx = None
        for j in range(nst, 0, -1):
            if j not in inverted_stage_idxs:
                balance_idx = j
                break

        if balance_idx is not None:
            current_sum = 0.0
            for k, st in enumerate(stages, start=1):
                if k == balance_idx:
                    continue
                current_sum += float(st.get("duration_myr", 0.0))

            remainder = float(t_total_myr) - current_sum

            # BG: invalid draw -> let objective penalize it (no crash)
            if remainder <= 0.0:
                return None

            stages[balance_idx - 1]["duration_myr"] = remainder

    return stages


def _apply_na_stage_params(
    ero_stages_template,
    param_names,
    x,
):
    """
    BG: Apply NA-sampled p1/p2/p3 into a copy of erosion stages.

    Parameters
    ----------
    ero_stages_template : list[dict]
        Raw YAML stages (no dt_sec cache).
    param_names : list[str]
        Names like 'ero_stage01_p1' in order matching x.
    x : array-like
        NA sampled values.

    Returns
    -------
    new_stages : list[dict]
        Updated stages (still raw YAML dicts).
    """
    stages = copy.deepcopy(ero_stages_template)
    nst = len(stages)

    for pname, val in zip(param_names, x):
        idx = _stage_idx_from_na_param_name(pname)

        if idx < 1 or idx > nst:
            raise ValueError(
                f"BG: stage param {pname!r} refers to stage {idx}, "
                f"but only {nst} stages exist."
            )

        # pname: ero_stageXX_pY -> last token is 'p1'/'p2'/'p3'
        key = pname.split("_")[-1]
        if key not in ("p1", "p2", "p3"):
            raise ValueError(
                f"BG: stage param name {pname!r} does not end with p1/p2/p3."
            )

        # BG: write canonical keys (p1/p2/p3); parse_ero_stage_row supports fallback
        stages[idx - 1][key] = float(val)

    return stages


def _rebuild_ero_stages_from_sample(params_local: dict, sample_dict: dict) -> bool:
    """
    BG: Rebuild parsed erosion stages for ero_type=0 using YAML template and sampled
    stage parameters (duration_myr and/or p1/p2/p3).

    Conservative behavior:
    - No duration balancing (balance=False).
    - If sum(durations) > t_total: erosion is naturally truncated by forward model.
    - If sum(durations) < t_total: erosion naturally drops to 0 after last stage.

    Parameters
    ----------
    params_local : dict
        Local copy of params used for this MCMC evaluation.
        Must contain 'ero_stages_template' for ero_type=0 stage inversion.
    sample_dict : dict
        Sampled parameter values for this evaluation (floats).

    Returns
    -------
    bool
        True if stages were rebuilt (or not needed), False if invalid (reject sample).
    """
    # BG: robust ero_type scalar
    ero_type_val = params_local.get("ero_type", None)
    ero_type0 = (
        ero_type_val[0]
        if isinstance(ero_type_val, (list, tuple, np.ndarray))
        else ero_type_val
    ) == 0
    if not ero_type0:
        return True

    # BG: Only rebuild if sample includes any stage parameters
    duration_param_names = [
        k
        for k in sample_dict.keys()
        if k.startswith("ero_stage") and k.endswith("_duration_myr")
    ]
    stage_p_param_names = [
        k
        for k in sample_dict.keys()
        if k.startswith("ero_stage") and k.endswith(("_p1", "_p2", "_p3"))
    ]

    if len(duration_param_names) == 0 and len(stage_p_param_names) == 0:
        return True  # nothing to do

    template = params_local.get("ero_stages_template", None)
    if template is None:
        raise ValueError(
            "BG: ero_type=0 stage inversion requires params['ero_stages_template'] "
            "(raw YAML erosion_model.ero_stages)."
        )

    new_template = copy.deepcopy(template)

    # ---- durations (NO balance) ----
    if len(duration_param_names) > 0:
        x_dur = [float(sample_dict[k]) for k in duration_param_names]
        new_template = _apply_na_stage_durations(
            new_template,
            duration_param_names,
            x_dur,
            t_total_myr=float(params_local["t_total"]),
            balance=False,  # BG: key choice for truncate/drop-to-zero behavior
        )
        if new_template is None:
            return False

    # ---- p1/p2/p3 ----
    if len(stage_p_param_names) > 0:
        x_p = [float(sample_dict[k]) for k in stage_p_param_names]
        new_template = _apply_na_stage_params(
            new_template,
            stage_p_param_names,
            x_p,
        )

    # BG: Parse and cache for forward model usage
    params_local["ero_stages"] = read_ero_stages_from_yaml(new_template)
    params_local["ero_total_stage_sec"] = sum(
        st["dt_sec"] for st in params_local["ero_stages"]
    )

    return True


def erosion_constant(t_local_myr: float, r_const: float) -> float:
    """
    BG: Constant erosion rate over a stage.

    Parameters
    ----------
    t_local_myr : float
        Time since the start of the stage, in Myr.
    r_const : float
        Constant erosion rate [km/Myr].

    Returns
    -------
    float
        Erosion rate [km/Myr] at t_local_myr (always r_const).
    """
    return r_const


def erosion_linear(
    t_local_myr: float, r_start: float, r_end: float, dt_stage_myr: float
) -> float:
    """
    BG: Linear change in erosion rate over a stage.

    Rate evolves linearly from r_start (at t_local=0) to r_end
    (at t_local = dt_stage_myr).

    Parameters
    ----------
    t_local_myr : float
        Time since the start of the stage [Myr].
    r_start : float
        Erosion rate at the start of the stage [km/Myr].
    r_end : float
        Erosion rate at the end of the stage [km/Myr].
    dt_stage_myr : float
        Duration of the stage [Myr].

    Returns
    -------
    float
        Erosion rate [km/Myr] at t_local_myr.
    """
    if dt_stage_myr <= 0.0:
        return r_start
    t = min(max(t_local_myr, 0.0), dt_stage_myr)
    return r_start + (r_end - r_start) * (t / dt_stage_myr)


def erosion_exponential(
    t_local_myr: float, r_start: float, r_target: float, tau_myr: float
) -> float:
    """
    BG: Exponential change in erosion rate over a stage.

    Rate follows an exponential relaxation from r_start towards r_target
    with timescale tau_myr.

    Parameters
    ----------
    t_local_myr : float
        Time since the start of the stage [Myr].
    r_start : float
        Initial erosion rate at stage start [km/Myr].
    r_target : float
        Target erosion rate [km/Myr].
    tau_myr : float
        Characteristic timescale [Myr].

    Returns
    -------
    float
        Erosion rate [km/Myr] at t_local_myr.
    """
    if tau_myr <= 0.0:
        return r_start
    t = max(t_local_myr, 0.0)
    return r_target + (r_start - r_target) * math.exp(-t / tau_myr)


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
    mantle_velocity,
):
    """Defines the way in which erosion should be applied.

    Erosion model types:

    0. Multi-stage erosion defined from the YAML input file (BG)
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
    mantle_velocity : numeric
        Velocity for mantle movement in fixed-Moho models (mm/yr)

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

    # BG: Normalize ero_type to a scalar int (CLI stores lists; YAML may also store lists).
    ero_type = (
        params["ero_type"][0]
        if isinstance(params.get("ero_type", None), (list, tuple, np.ndarray))
        else int(params["ero_type"])
    )

    # BG: Multi-stage erosion model (ero_type = 0)
    if ero_type == 0:
        # BG: Multi-stage erosion from YAML stages (ero_type=0).
        stages = params.get("ero_stages", None)

        # BG: YAML-only: ero_type=0 must provide erosion stages in params.
        if not isinstance(stages, list) or len(stages) == 0:
            raise ValueError(
                "BG: ero_type=0 requires 'ero_stages' in params (from YAML input file). "
            )

        if (
            stages is not None
            and (len(stages) > 0)
            and isinstance(stages[0], dict)
            and ("dt_sec" not in stages[0])
        ):
            # BG: YAML pathway: parse + validate stages once (raw YAML list -> internal stage dicts).
            stages = read_ero_stages_from_yaml(stages)
            params["ero_stages"] = (
                stages  # BG: cache parsed stages to avoid re-parsing every timestep
            )
            params["ero_total_stage_sec"] = sum(st["dt_sec"] for st in stages)

        else:
            # BG: stages already parsed/cached in params (dt_sec exists)
            stages = params["ero_stages"]
            if "ero_total_stage_sec" not in params:
                params["ero_total_stage_sec"] = sum(st["dt_sec"] for st in stages)

        # 2) Find which stage we are currently in, using current_time [s]
        t = current_time  # current model time [s] since start
        t0 = 0.0
        found = False

        for i, st in enumerate(stages):
            t1 = t0 + st["dt_sec"]
            is_last = i == len(stages) - 1

            if (t >= t0) and (t < t1 or (is_last and t <= t1)):
                # Local time since stage start in Myr
                t_local_myr = (t - t0) / myr2sec(1.0)

                stage_type = st["type"]
                dt_stage_myr = st["duration_myr"]
                p = st["params"]

                # 3) Compute instantaneous erosion rate [km/Myr] for this stage
                if stage_type == "constant":
                    rate_km_myr = erosion_constant(t_local_myr, p[0])

                elif stage_type == "linear":
                    rate_km_myr = erosion_linear(t_local_myr, p[0], p[1], dt_stage_myr)

                elif stage_type == "exponential":
                    rate_km_myr = erosion_exponential(t_local_myr, p[0], p[1], p[2])

                else:
                    raise ValueError(f"Unknown erosion stage type: {stage_type}")

                found = True
                break

            t0 = t1

        if not found:
            # BG: If stages end before t_total, set erosion rate to zero after the last stage.
            total_stage_sec = params.get(
                "ero_total_stage_sec", sum(st["dt_sec"] for st in stages)
            )
            eps_sec = max(1.0e-6, 2.0 * abs(dt))

            if t >= total_stage_sec - eps_sec:
                rate_km_myr = 0.0
            else:
                raise ValueError(
                    "current_time is outside erosion stages (unexpected). "
                    "Check erosion stages durations and time handling."
                )

        # 4) Convert rate [km/Myr] to velocity [m/s]
        vx_array[:] = mmyr2ms(rate_km_myr)
        vx_surf = vx_array[0]
        vx_max = abs(vx_surf)

    # Constant erosion rate
    # Convert to inputting rate directly?
    elif params["ero_type"] == 1:
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
            f"Bad erosion type: {params['ero_type']}. Must be 0 (multi-stage) or between 1 and 7."
        )

    # Set velocities below Moho to mantle velocity if using crustal uplift only
    if params["crustal_uplift"]:
        vx_array[x > moho_depth] = mantle_velocity

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
    ero_stages=None,
):
    """
    Calculates total erosion / exhumation magnitude over the full model run.

    For ero_type = 0 (YAML-defined multi-stage erosion), the magnitude is computed
    by analytically integrating the erosion rate over each stage, using the
    internal rate-based parameterization reconstructed by read_ero_stages_from_yaml().

    If the total duration of erosion stages exceeds t_total, only the portion
    within t_total is integrated. If stages are shorter than t_total, no erosion
    is applied beyond the last stage.

    Returns erosion magnitude in kilometers.
    """

    # Initialize fw_ref_frame Boolean for ero types != 7
    fw_ref_frame = False

    # BG: Normalize ero_type to scalar (CLI/YAML can pass list-like)
    ero_type = (
        ero_type[0]
        if isinstance(ero_type, (list, tuple, np.ndarray))
        else int(ero_type)
    )

    # BG: Multi-stage erosion from YAML stages (ero_type = 0)
    if ero_type == 0:
        stages = ero_stages
        if not isinstance(stages, list) or len(stages) == 0:
            raise ValueError(
                "BG: ero_type=0 requires 'ero_stages' (from YAML input file)."
            )

        if isinstance(stages[0], dict) and ("dt_sec" not in stages[0]):
            stages = read_ero_stages_from_yaml(stages)

        thickness_km = 0.0
        # Convert total model time from seconds to Myr
        t_total_myr = t_total / myr2sec(1.0)
        remaining_myr = t_total_myr
        total_stage_myr = sum(st["duration_myr"] for st in stages)

        # BG: warn if stages extend beyond model time (magnitude will be truncated)
        if total_stage_myr > t_total_myr + 1.0e-12:
            warnings.warn(
                "Sum of erosion stage durations exceeds t_total. "
                "Exhumation magnitude will be computed only up to the model time (t_total).",
                RuntimeWarning,
                stacklevel=2,
            )

        for st in stages:
            dt_stage = st["duration_myr"]  # Myr
            dt = min(dt_stage, remaining_myr)
            if dt <= 0.0:
                break

            p = st["params"]
            stage_type = st["type"]

            if stage_type == "constant":
                r_const = p[0]
                thickness_km += r_const * dt

            elif stage_type == "linear":
                r_start, r_end = p[0], p[1]
                # Integral of a linear erosion rate from r_start to r_end over dt
                thickness_km += (
                    r_start * dt + 0.5 * (r_end - r_start) * (dt**2) / dt_stage
                )

            elif stage_type == "exponential":
                r_start, r_target, tau = p[0], p[1], p[2]
                # Integral of r(t) = r_target + (r_start - r_target)*exp(-t/tau)
                thickness_km += r_target * dt + (r_start - r_target) * tau * (
                    1.0 - math.exp(-dt / tau)
                )

            else:
                raise ValueError(f"Unknown stage erosion type: {stage_type}")

            remaining_myr -= dt

        if total_stage_myr < t_total_myr - 1.0e-12:
            # BG: If stages are shorter than t_total, erosion is set to zero afterwards
            # (no additional exhumation beyond what is prescribed in the file).
            pass

        magnitude = thickness_km

    # Constant erosion rate
    elif ero_type == 1:
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
        raise ValueError(
            f"Bad erosion type: {ero_type}. Must be 0 (multi-stage) or between 1 and 7."
        )

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
        # TODO: Make this a function?
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
    solution_type=3,
    bc_type=1,
    temp_surf=0.0,
    temp_base=1300.0,
    flux_base=20.0,
    temp_adiabat_ref=1300.0,
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
    ero_stages=None,
    mantle_velocity=0.0,
    calc_ages=True,
    ketch_aft=True,
    madtrax_aft=False,
    madtrax_aft_kinetic_model=1,
    madtrax_zft_kinetic_model=1,
    ap_rad=60.0,
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
    # BG: Neighbourhood Algorithm (NA) parameters
    na_ns=24,  # BG: samples per iteration (default: 24)
    na_nr=12,  # BG: number of Voronoi cells to resample (default: 12)
    na_ni=50,  # BG: size of initial random search (default: 50)
    na_n=6,  # BG: number of NA iterations (default: 6)
    na_n_resample=2000,  # BG: NA appraiser - total new samples
    na_n_walkers=5,  # BG: NA appraiser - parallel walkers
    # BG: MCMC (emcee) parameters
    mcmc_nwalkers=30,  # BG: number of walkers in the ensemble (default: 30)
    mcmc_nsteps=150,  # BG: number of steps per walker (default: 150)
    mcmc_discard=50,  # BG: burn-in steps to discard (default: 50)
    mcmc_thin=3,  # BG: thinning factor (default: 3)
    plot_results=True,
    display_plots=True,
    plot_ma=True,
    plot_depth_history=False,
    plot_fault_depth_history=False,
    plot_density=False,
    plot_elevation_history=False,
    plot_peclet_number=False,
    plot_ft_length_dist=False,
    invert_tt_plot=False,
    t_plots=[0.1, 1, 5, 10, 20, 30, 50],
    watch_it_exhume=False,
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
    write_tt_history=False,
    write_ttdp_history=False,
    write_ft_lengths=False,
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
    solution_type : int, default=3
        Finite-difference solution type (1 = explicit, 2 = implicit, 3 = Crank-Nicolson).
    bc_type : int, default=1
        Basal boundary condition type (1 = constant temperature, 2 = constant heat flux).
    temp_surf : float or int, default=0.0
        Surface boundary condition temperature in °C.
    temp_base : float or int, default=1300.0
        Basal boundary condition temperature in °C.
    flux_base : float or int, default=20.0
        Basal heat flux in mW/m^2.
    temp_adiabat_ref : float or int, default=1300.0
        Reference temperature for mantle adiabat calculation
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
        Type of erosion model (0, 1, 2, 3, 4, 5, 6, 7 - see https://tc1d.readthedocs.io/en/latest/erosion-models.html).
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
    ero_stages : list of dict, default=None
        Erosion stages from YAML input file for ero_type 0
    mantle_velocity : float or int, default=0.0
        Velocity for mantle movement in fixed-Moho models (mm/yr)
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
    ap_rad : float or int, default=60.0
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
    plot_density : bool, default=False
        Plot density beside geotherms plot.
    plot_elevation_history : bool, default=False
        Plot surface elevation history.
    plot_peclet_number : bool, default=False
        Plot Peclet number on erosion history plot.
    plot_ft_length_dist : bool, default=False
        Plot apatite fission-track length distribution.
    invert_tt_plot : bool, default=False
        Invert depth/temperature axis on thermal history plot.
    t_plots : list of float or int, default=[0.1, 1, 5, 10, 20, 30, 50]
        Output times for temperature plotting in Myr. Treated as increment if only one value given.
    watch_it_exhume : bool, default=False
        Use animation plot for thermal history rather than static plotting
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
    write_tt_history : bool, default=False
        Write time-temperature history to a csv file.
    write_ttdp_history : bool, default=False
        Write time-temperature-depth-pressure history to a csv file.
    write_ft_lengths : bool, default=False
        Write apatite fission track-length distribution to a file.
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
        "plot_density": plot_density,
        "plot_elevation_history": plot_elevation_history,
        "plot_peclet_number": plot_peclet_number,
        "plot_ft_length_dist": plot_ft_length_dist,
        "invert_tt_plot": invert_tt_plot,
        "watch_it_exhume": watch_it_exhume,
        "run_type": run_type,
        # Batch mode defaults to false and will set itself true if batch parameters exist
        "batch_mode": False,
        # Inverse mode not supported when called as a function
        "inverse_mode": False,
        "mantle_adiabat": mantle_adiabat,
        "solution_type": solution_type,
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
        "ero_stages": ero_stages,
        "mantle_velocity": mantle_velocity,
        "bc_type": bc_type,
        "temp_surf": temp_surf,
        "temp_base": temp_base,
        "flux_base": flux_base,
        "temp_adiabat_ref": temp_adiabat_ref,
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
        "write_tt_history": write_tt_history,
        "write_ttdp_history": write_ttdp_history,
        "write_ft_lengths": write_ft_lengths,
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
        # BG: Neighbourhood Algorithm (NA) hyper-parameters
        "na_ns": na_ns,  # BG: NA - samples per iteration
        "na_nr": na_nr,  # BG: NA - cells to resample
        "na_ni": na_ni,  # BG: NA - initial random search size
        "na_n": na_n,  # BG: NA - number of iterations
        "na_n_resample": na_n_resample,  # BG: NA appraiser - total new samples
        "na_n_walkers": na_n_walkers,  # BG: NA appraiser - parallel walkers
        # BG: MCMC (emcee) hyper-parameters
        "mcmc_nwalkers": mcmc_nwalkers,  # BG: MCMC - number of walkers
        "mcmc_nsteps": mcmc_nsteps,  # BG: MCMC - steps per walker
        "mcmc_discard": mcmc_discard,  # BG: MCMC - burn-in steps to discard
        "mcmc_thin": mcmc_thin,  # BG: MCMC - thinning factor
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
        or params["write_tt_history"]
        or params["write_ttdp_history"]
        or params["write_ft_lengths"]
        or params["write_age_output"]
    ):
        create_output_directory(wd, dir="csv")
    if params["save_plots"]:
        create_output_directory(wd, dir="png")

    batch_keys = [
        "max_depth",
        "nx",
        "solution_type",
        "bc_type",
        "temp_surf",
        "temp_base",
        "flux_base",
        "temp_adiabat_ref",
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
        "mantle_velocity",
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
        version_len = len(__version__)
        second_star_len = 31 - version_len
        print("")
        print(
            f"{26 * '='} This is Tc1D version {__version__} {second_star_len * '='}\n"
        )

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

        # Set inverse mode to true for NA and MCMC runs
        if (params["run_type"].lower() == "na") or (
            params["run_type"].lower() == "mcmc"
        ):
            params["inverse_mode"] = True

        # Set run_type to batch if batch mode is detected
        if params["batch_mode"]:
            params["run_type"] = "batch"

        # Create batch_mode output directories if using batch, NA, or MCMC mode
        if params["batch_mode"] or params["inverse_mode"]:
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
            params["log_file"] = "tc1d_batch_log.csv"
        else:
            params["log_file"] = "tc1d_run_log.csv"

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
                "Erosion model option 9,Erosion model option 10,Mantle velocity (mm/yr),"
                "Initial Moho depth (km),Initial Moho temperature (C),"
                "Initial surface heat flow (mW m^-2),Initial surface elevation (km),Initial dimensionless heat production,"
                "Final Moho depth (km),Final Moho temperature (C),Final surface heat flow (mW m^-2),"
                "Final surface elevation (km),Final dimensionless heat production,Total exhumation (km),Average Peclet number,Minimum Peclet number, Maximum Peclet number,Apatite grain radius (um),Apatite U "
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
                    f"{params['t_total']:.4f},{params['dt']:.4f},{params['max_depth']:.4f},{params['nx']},{params['bc_type']},"
                    f"{params['temp_surf']:.4f},{params['temp_base']:.4f},{params['flux_base']:.4f},{params['temp_adiabat_ref']:.4f},{params['mantle_adiabat']},"
                    f"{params['rho_crust']:.4f},{params['removal_fraction']:.4f},{params['removal_start_time']:.4f},"
                    f"{params['removal_end_time']:.4f},"
                    f"{params['ero_type']},{params['ero_option1']:.4f},"
                    f"{params['ero_option2']:.4f},{params['ero_option3']:.4f},{params['ero_option4']:.4f},{params['ero_option5']:.4f},{params['ero_option6']:.4f},{params['ero_option7']:.4f},{params['ero_option8']:.4f},{params['ero_option9']:.4f},{params['ero_option10']:.4f},{params['mantle_velocity']:.4f},{params['init_moho_depth']:.4f},,,,,,,,,,,,,,{params['ap_rad']:.4f},{params['ap_uranium']:.4f},"
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

    # BG: robust ero_type scalar for NA (CLI often stores lists)
    ero_type_val = params.get("ero_type", None)
    ero_type0 = (
        ero_type_val[0]
        if isinstance(ero_type_val, (list, tuple, np.ndarray))
        else ero_type_val
    ) == 0

    # Objective function to be minimised, run for misfit
    def objective(x):
        # Map sampled values x to the corresponding parameter names
        for key, value in zip(param_names, x):
            filtered_params[key] = value

        # ------------------------------------------------------------
        # BG: If NA is inverting YAML ero_type=0 stage durations and/or p1/p2/p3,
        # rebuild params['ero_stages'] (parsed)
        # ------------------------------------------------------------
        duration_param_names = [
            k
            for k in param_names
            if k.startswith("ero_stage") and k.endswith("_duration_myr")
        ]

        stage_p_param_names = [
            k
            for k in param_names
            if k.startswith("ero_stage") and k.endswith(("_p1", "_p2", "_p3"))
        ]

        if ero_type0 and (
            len(duration_param_names) > 0 or len(stage_p_param_names) > 0
        ):
            # BG: start from raw YAML template
            new_template = copy.deepcopy(params["ero_stages_template"])

            # ---- durations (with balance) ----
            if len(duration_param_names) > 0:
                x_dur = [filtered_params[k] for k in duration_param_names]

                new_template = _apply_na_stage_durations(
                    new_template,  # raw YAML template
                    duration_param_names,
                    x_dur,
                    t_total_myr=float(params["t_total"]),
                    balance=True,
                )

                # BG: reject invalid duration combinations (no crash)
                if new_template is None:
                    return 1e9

            # ---- p1/p2/p3 ----
            if len(stage_p_param_names) > 0:
                x_p = [filtered_params[k] for k in stage_p_param_names]
                new_template = _apply_na_stage_params(
                    new_template,
                    stage_p_param_names,
                    x_p,
                )

            # Parsed stages used by the forward model
            params["ero_stages"] = read_ero_stages_from_yaml(new_template)
            params["ero_total_stage_sec"] = sum(
                st["dt_sec"] for st in params["ero_stages"]
            )

            # Optional: keep for debug
            # params["ero_stages_current_template"] = new_template

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

        # ------------------------------------------------------------
        # BG: Silence forward-model verbosity during NA evaluations
        # by forcing batch_mode=True just for this run_model() call.
        # This prevents printing:
        #   --- Calculating initial thermal model ---
        #   --- Calculating transient thermal model ---
        #   - Step ...
        # ------------------------------------------------------------
        _prev_batch_mode = params.get("batch_mode", False)
        params["batch_mode"] = True
        try:
            misfit = run_model(params)
        finally:
            params["batch_mode"] = _prev_batch_mode

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

    # ------------------------------------------------------------
    # BG: Add NA bounds for YAML ero_type=0 stage durations and/or p1/p2/p3
    # ------------------------------------------------------------
    if ero_type0:
        ero_stages_template = params.get("ero_stages_template", None)
        if ero_stages_template is None:
            raise ValueError(
                "BG: NA duration inversion requires params['ero_stages_template'] "
                "(raw YAML erosion_model.ero_stages)."
            )

        duration_bounds = _collect_ero_stage_duration_bounds(ero_stages_template)

        # Merge duration bounds into filtered_params (do not overwrite)
        for k, v in duration_bounds.items():
            if k in filtered_params:
                raise ValueError(f"BG: NA param name conflict for {k}.")
            filtered_params[k] = v

        # BG: Add NA bounds for YAML ero_type=0 stage p1/p2/p3
        param_bounds = _collect_ero_stage_param_bounds(ero_stages_template)

        # Merge param bounds into filtered_params
        for k, v in param_bounds.items():
            if k in filtered_params:
                raise ValueError(f"BG: NA param name conflict for {k}.")
            filtered_params[k] = v

    # BG: freeze parameter order once for NA mapping (critical)
    param_names = list(filtered_params.keys())
    bounds = [filtered_params[k] for k in param_names]

    # BG: Read Neighbourhood Algorithm (NA) parameters from params
    na_ns = int(params["na_ns"])  # BG: NA - samples per iteration
    na_nr = int(params["na_nr"])  # BG: NA - number of Voronoi cells to resample
    na_ni = int(params["na_ni"])  # BG: NA - size of initial random search
    na_n = int(params["na_n"])  # BG: NA - number of NA iterations

    # Initialize NA searcher (BG: controlled by CLI flags)
    searcher = NASearcher(
        objective,
        ns=na_ns,
        nr=na_nr,
        ni=na_ni,
        n=na_n,
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

    # BG: Read NA appraiser settings from params (BG)
    na_n_resample = int(params["na_n_resample"])  # BG: total new samples
    na_n_walkers = int(params["na_n_walkers"])  # BG: parallel walkers

    # Initialize NA appraiser (BG: controlled by CLI flags)
    appraiser = NAAppraiser(
        initial_ensemble=searcher.samples,  # points of parameter space already sampled
        log_ppd=-searcher.objectives,  # objective function values
        bounds=bounds,
        n_resample=na_n_resample,  # BG: number of desired new samples
        n_walkers=na_n_walkers,  # BG: number of parallel walkers
    )

    appraiser.run()  # Results stored in appraiser.samples
    print(f"Appraiser mean: {appraiser.mean}")
    print(f"Appraiser mean error: {appraiser.sample_mean_error}")
    print(f"Appraiser covariance: {appraiser.covariance}")
    print(f"Appraiser covariance error: {appraiser.sample_covariance_error}")

    # BG: Safely extract best parameter set using param names
    best = searcher.samples[np.argmin(searcher.objectives)]
    best_dict = dict(zip(filtered_params.keys(), best))

    # ------------------------------------------------------------
    # BG: Rebuild ero_stages for best model if durations, p1, p2, p3 are inverted
    # ------------------------------------------------------------
    duration_param_names = [
        k
        for k in param_names
        if k.startswith("ero_stage") and k.endswith("_duration_myr")
    ]
    stage_p_param_names = [
        k
        for k in param_names
        if k.startswith("ero_stage") and k.endswith(("_p1", "_p2", "_p3"))
    ]

    if ero_type0 and (len(duration_param_names) > 0 or len(stage_p_param_names) > 0):
        new_template_best = copy.deepcopy(params["ero_stages_template"])

        if len(duration_param_names) > 0:
            x_dur_best = [best_dict[k] for k in duration_param_names]
            new_template_best = _apply_na_stage_durations(
                new_template_best,
                duration_param_names,
                x_dur_best,
                t_total_myr=float(params["t_total"]),
                balance=True,
            )

        if len(stage_p_param_names) > 0:
            x_p_best = [best_dict[k] for k in stage_p_param_names]
            new_template_best = _apply_na_stage_params(
                new_template_best,
                stage_p_param_names,
                x_p_best,
            )

        params["ero_stages"] = read_ero_stages_from_yaml(new_template_best)
        params["ero_total_stage_sec"] = sum(st["dt_sec"] for st in params["ero_stages"])

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

    # BG: robust ero_type scalar
    ero_type_val = global_params.get("ero_type", None)
    ero_type0 = (
        ero_type_val[0]
        if isinstance(ero_type_val, (list, tuple, np.ndarray))
        else ero_type_val
    ) == 0

    # BG: For ero_type=0, skip ero_option* cumulative constraints
    # BG: (they apply to ero_type>0 only).
    # BG: Apply only stage-based constraints (e.g., sum of sampled stage durations <= t_total).
    if ero_type0:
        # Reject samples where sum(duration) > t_total
        t_total = global_params.get("t_total", 0.0)
        t_total = float(
            t_total[0] if isinstance(t_total, (list, tuple, np.ndarray)) else t_total
        )

        param_dict = dict(zip(global_param_names, x))

        # BG: collect all sampled stage durations automatically
        durs = [
            float(v)
            for k, v in param_dict.items()
            if isinstance(k, str) and k.endswith("_duration_myr")
        ]

        if durs and sum(durs) > t_total:
            return -np.inf

        return 0.0

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

    # Total thickness must remain >= 0 km (cannot end above surface)
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

    # BG: Force MCMC/inverse context (safety belt)
    params_local["inverse_mode"] = True
    params_local["run_type"] = "mcmc"

    cleaned_dict = {k: float(v) for k, v in new_dict.items()}
    print(f"[MCMC] Testing params: {cleaned_dict}")

    # BG: Rebuild YAML ero_type=0 stages if stage params are sampled in this MCMC step
    ok = _rebuild_ero_stages_from_sample(params_local, cleaned_dict)
    if not ok:
        return -np.inf

    try:
        # BG: Silence forward-model verbosity during MCMC evaluations
        _prev_batch_mode = params_local.get("batch_mode", False)
        params_local["batch_mode"] = True
        try:
            misfit = run_model(params_local)
        finally:
            params_local["batch_mode"] = _prev_batch_mode

        if not np.isfinite(misfit):
            return -np.inf

        print(f"Misfit: {misfit}")

        # BG: Standard Gaussian log-likelihood assuming independent normal
        # BG: errors and using chi-square misfit.
        return -0.5 * misfit

    except Exception as e:
        print(f"[ERROR] run_model failed: {e}")
        if params_local.get("debug", False):
            import traceback

            traceback.print_exc()
        return -np.inf


def log_probability(x):
    lp = log_prior(x)
    if not np.isfinite(lp):
        return -np.inf

    ll = log_likelihood(x)
    if not np.isfinite(ll):
        return -np.inf

    return lp + ll


def batch_run_mcmc(params, batch_params):
    """Runs TC1D in inverse mode using MCMC"""
    # Define working directory path
    wd = Path.cwd()

    # print("--- Starting MCMC inverse mode ---\n")
    log_output(params, batch_mode=True)

    # ------------------------------------------------------------
    # BG: MCMC does NOT need a ParameterGrid (that is for batch/grids).
    #     The previous code only used ParameterGrid(batch_params)[0] to
    #     inject fixed values into `params`. We preserve the same intent
    #     without building an unnecessary cartesian product.
    # ------------------------------------------------------------

    # BG: Extract parameters with more than one value (i.e., varied ones)
    #     These define the MCMC search space (bounds).
    filtered_params = {
        k: v
        for k, v in batch_params.items()
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) > 1
    }

    # BG: Inject fixed batch params (scalars or singletons) into base `params`.
    #     This preserves the previous behavior where the "first grid point"
    #     was used as the base model for non-inverted params.
    for k, v in batch_params.items():
        if isinstance(v, (list, tuple, np.ndarray)):
            if len(v) == 1:
                params[k] = v[0]
        else:
            params[k] = v

    # BG: ero_type scalar for MCMC
    ero_type_val = params.get("ero_type", None)
    ero_type0 = (
        ero_type_val[0]
        if isinstance(ero_type_val, (list, tuple, np.ndarray))
        else ero_type_val
    ) == 0

    # ------------------------------------------------------------
    # BG: Add MCMC bounds for YAML ero_type=0 stage durations and/or p1/p2/p3
    #     (same conservative logic as NA)
    # ------------------------------------------------------------
    if ero_type0:
        ero_stages_template = params.get("ero_stages_template", None)
        if ero_stages_template is None:
            raise ValueError(
                "BG: MCMC inversion for ero_type=0 requires params['ero_stages_template'] "
                "(raw YAML erosion_model.ero_stages)."
            )

        # BG: duration_myr bounds -> ero_stageXX_duration_myr
        duration_bounds = _collect_ero_stage_duration_bounds(ero_stages_template)
        for k, v in duration_bounds.items():
            if k in filtered_params:
                raise ValueError(f"BG: MCMC param name conflict for {k}.")
            filtered_params[k] = v

        # BG: p1/p2/p3 bounds -> ero_stageXX_p1/p2/p3
        param_bounds = _collect_ero_stage_param_bounds(ero_stages_template)
        for k, v in param_bounds.items():
            if k in filtered_params:
                raise ValueError(f"BG: MCMC param name conflict for {k}.")
            filtered_params[k] = v

    # BG: Keep deterministic ordering for reproducibility and safer x<->name mapping
    param_names = sorted(filtered_params.keys())
    bounds = [filtered_params[k] for k in param_names]

    ndim = len(param_names)  # Number of parameters to invert
    if ndim == 0:
        raise ValueError(
            "BG: MCMC requires at least one parameter range to invert (ndim=0)."
        )

    # BG: Conservative physical constraints (prior guards)
    max_exhumation = 35.0  # km (or equivalent model unit, consistent with your prior)
    max_burial = 15.0  # km (or equivalent model unit)

    # BG: Set global variables for MPI pickling compatibility
    global \
        global_bounds, \
        global_param_names, \
        global_params, \
        global_max_exhumation, \
        global_max_burial
    global_bounds = bounds
    global_param_names = param_names
    global_params = copy.deepcopy(params)
    global_max_exhumation = max_exhumation
    global_max_burial = max_burial

    # BG: MCMC setup - number of walkers and initial positions sampled from uniform priors
    mcmc_nwalkers = int(params["mcmc_nwalkers"])  # BG: number of walkers
    mcmc_nsteps = int(params["mcmc_nsteps"])  # BG: steps per walker
    mcmc_discard = int(params["mcmc_discard"])  # BG: burn-in steps
    mcmc_thin = int(params["mcmc_thin"])  # BG: thinning factor

    nwalkers = mcmc_nwalkers
    nsteps = mcmc_nsteps
    discard = mcmc_discard
    thin = mcmc_thin

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

    # BG: Build sampler (optionally parallelized with an emcee pool)
    sampler = emcee.EnsembleSampler(
        nwalkers=nwalkers,
        ndim=ndim,
        log_prob_fn=log_probability,
        pool=pool,
    )

    # BG: Always close the pool, even if the MCMC run errors out
    try:
        sampler.run_mcmc(initial_state=p0, nsteps=nsteps, progress=True)
    finally:
        if pool is not None:
            pool.close()

    # BG: Post-processing compatible with emcee v2 and v3
    if hasattr(sampler, "get_chain"):
        # emcee v3 returns (nsteps, nwalkers, ndim) -> convert to (nwalkers, nsteps, ndim)
        chain = sampler.get_chain()
        chain = np.transpose(chain, (1, 0, 2))

        log_probs = sampler.get_log_prob()
        log_probs = np.transpose(log_probs, (1, 0))
    else:
        # emcee v2 returns (nwalkers, nsteps, ndim) directly
        chain = sampler.chain
        log_probs = sampler.lnprobability

    # BG: Flatten chains after burn-in and thinning
    flat_samples = chain[:, discard::thin, :].reshape(-1, ndim)
    flat_log_probs = log_probs[:, discard::thin].reshape(-1)

    if flat_log_probs.size == 0:
        raise RuntimeError(
            "No valid samples after burn-in/thinning. Aborting analysis."
        )

    # BG: With the current uniform hard-bound prior, valid samples have log_prior = 0,
    # BG: so log_probability = log_likelihood = -0.5 * misfit.
    # BG: This allows exact reconstruction of the chi-square misfit from flat_log_probs.
    flat_misfits = -2.0 * flat_log_probs

    # Write sample results to file
    outfile = wd / "csv" / "mcmc_searcher_results.csv"
    header = param_names + ["log_probability", "misfit"]
    combined_array = np.column_stack((flat_samples, flat_log_probs, flat_misfits))
    with open(outfile, "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(combined_array)

    # BG: Identify and print the best parameter set (minimum misfit)
    best_idx = np.argmin(flat_misfits)
    best = flat_samples[best_idx]
    best_dict = dict(zip(param_names, best))
    print(f"The best parameters are: { {k: float(v) for k, v in best_dict.items()} }")
    print(f"Best misfit: {float(flat_misfits[best_idx])}")
    print(f"Best log probability: {float(flat_log_probs[best_idx])}")

    # Stop timer before displaying plots if plots are to be displayed
    if params["display_plots"]:
        exec_end = time.time()

    # BG: Plot evolution of negative log-posterior values
    plt.figure()
    neg_log_probs = -flat_log_probs
    plt.plot(neg_log_probs, ".", markersize=2)
    plt.scatter(best_idx, neg_log_probs[best_idx], c="g", s=10)
    plt.xlabel("Sample Index")
    plt.ylabel(r"$-\log P$")
    plt.title("MCMC Negative Log-Posterior Values")
    plt.yscale("log")
    if params["save_plots"]:
        savefile = wd / "png" / "mcmc_neg_log_posterior.png"
        plt.savefig(savefile, dpi=300)
        print(f"- MCMC negative log-posterior plot written to {savefile}")
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

    # BG: Plot evolution of misfit values
    plt.figure()
    plt.plot(flat_misfits, ".", markersize=2)
    plt.scatter(best_idx, flat_misfits[best_idx], c="g", s=10)
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

        sc = ax.scatter(x, y, c=flat_misfits, cmap="viridis", marker="x")
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
    if params["batch_mode"]:
        params["echo_info"] = False
        params["echo_thermal_info"] = False
        params["echo_ages"] = False
        params["plot_results"] = False

    write_tt_history = params["write_tt_history"]
    write_ttdp_history = params["write_ttdp_history"]
    write_track_lengths = params["write_ft_lengths"]
    if params["inverse_mode"]:
        # Only allow tT histories and track lengths to be saved to file in forward and batch modes
        write_tt_history = False
        write_ttdp_history = False
        write_track_lengths = False

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

    # Create arrays for velocity history, peclet numbers, fault depth history
    vx_hist = np.zeros(nt)
    peclet = np.zeros(nt)
    if params["plot_fault_depth_history"]:
        fault_depth_history = np.zeros(nt)

    # BG: Preload multi-stage erosion stages from YAML (ero_type = 0)
    ero_type = (
        params["ero_type"][0]
        if isinstance(params.get("ero_type", None), (list, tuple))
        else int(params["ero_type"])
    )

    if ero_type == 0:
        stages = params.get("ero_stages", None)
        if stages is None or (isinstance(stages, list) and len(stages) == 0):
            raise ValueError(
                "BG: ero_type=0 requires 'ero_stages' from YAML input file."
            )

        # BG: Parse once if raw YAML (no dt_sec), otherwise assume already parsed
        if (
            isinstance(stages, list)
            and len(stages) > 0
            and isinstance(stages[0], dict)
            and ("dt_sec" not in stages[0])
        ):
            stages = read_ero_stages_from_yaml(stages)
            params["ero_stages"] = stages  # cache parsed

        # BG: Validate total duration vs t_total (seconds)
        total_stage_sec = sum(st["dt_sec"] for st in params["ero_stages"])
        eps_sec = max(
            1.0e-6, 2.0 * dt
        )  # dt is model timestep [s] (already converted in run_model)

        if total_stage_sec > t_total + eps_sec:
            total_stage_myr = total_stage_sec / myr2sec(1.0)
            t_total_myr = t_total / myr2sec(1.0)

            warnings.warn(
                f"The cumulative erosion duration ({total_stage_myr:.2f} Myr) "
                f"exceeds the model time ({t_total_myr:.2f} Myr).\n"
                "Tc1D will apply erosion stages only until the model end time. "
                "Any erosion defined after this time will be ignored.",
                RuntimeWarning,
            )

        # Total duration of all erosion stages (seconds), used for tail handling
        params["ero_total_stage_sec"] = total_stage_sec

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
        params["ero_stages"],
    )

    # Ensure that initial moho depth is greater than exhumation magnitude if using crustal uplift
    if params["crustal_uplift"]:
        if exhumation_magnitude > params["init_moho_depth"]:
            raise ValueError(
                f"Exhumation magnitude ({exhumation_magnitude:.1f} km) exceeds initial Moho depth ({params['init_moho_depth']} km)."
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
        mmyr2ms(params["mantle_velocity"]),
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
    if params["solution_type"] == 1:
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
            params["solution_type"],
            params["bc_type"],
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
            # FIXME: Should this be skipped to not exclude depo ages from final plots?
            # obs_ahe_indices = np.intersect1d(obs_ahe_indices, obs_zero_depo_indices)
            # obs_aft_indices = np.intersect1d(obs_aft_indices, obs_zero_depo_indices)
            # obs_zhe_indices = np.intersect1d(obs_zhe_indices, obs_zero_depo_indices)
            # obs_zft_indices = np.intersect1d(obs_zft_indices, obs_zero_depo_indices)

            if params["debug"]:
                print(f"\n{num_file_ages} ages read from data file.")

        # Create array of past ages at which ages should be calculated, if not zero
        if len(obs_unique_depo_ages) > 0:
            # Issue warning about depositional ages being experimental
            warnings.warn(
                "Support for depositional ages is experimental and may produce unexpected results.",
                stacklevel=2,
            )

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
                n_steps = int(
                    np.floor(params["t_total"] / params["past_age_increment"])
                )
                surface_times_ma = np.linspace(
                    0.0,
                    params["t_total"] - params["past_age_increment"],
                    num=n_steps,
                    endpoint=True,
                    dtype=float,
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
            temp=params["temp_adiabat_ref"] + 273.15,
            cp=params["cp_mantle"],
        )
        temp_adiabat = params["temp_adiabat_ref"] + (xstag - max_depth) * adiabat_m
    else:
        adiabat_m = 0.0
        temp_adiabat = params["temp_adiabat_ref"]

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

    # Create the empty (zero) coefficient and right hand side arrays for temperature calculations
    a_matrix = np.zeros(
        (params["nx"], params["nx"])
    )  # 2-dimensional array, ny rows, ny columns
    b_vector = np.zeros(params["nx"])

    temp_init = temp_ss_implicit(
        params["nx"],
        dx,
        a_matrix,
        b_vector,
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

    # Calculate initial Peclet number
    if params["ero_type"] == 7:
        warnings.warn(
            f"Peclet number calculation is not yet supported for ero_type {params['ero_type']}. A value of -1.0 will be reported.",
            stacklevel=2,
        )
    init_peclet = calculate_peclet_number(
        params["ero_type"],
        vx,
        x,
        xstag,
        k,
        rho,
        cp,
        moho_depth,
        max_depth,
        params["crustal_uplift"],
    )

    # Calculate initial dimensionless heat production
    init_nd_hp = calculate_nd_heat_prod(
        heat_prod,
        k,
        params["temp_base"],
        max_depth,
    )

    # Echo thermal model values
    if params["echo_thermal_info"]:
        print(f"- Initial surface heat flow: {init_heat_flow:.1f} mW/m^2")
        print(f"- Initial Moho temperature: {init_moho_temp:.1f}°C")
        print(f"- Initial Moho depth: {params['init_moho_depth']:.1f} km")
        print(
            f"- Initial LAB depth: {(max_depth - removal_thickness) / kilo2base(1):.1f} km"
        )
        print(f"- Initial Peclet number: {init_peclet:.2f}")
        print(f"- Initial dimensionless heat production: {init_nd_hp:.2f}")

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

    # Calculate model times when particles reach surface
    surface_times = myr2sec(params["t_total"] - surface_times_ma)

    # Assign basal boundary condition for implicit and Crank-Nicolson solutions
    # NOTE: This assumes the boundary conditions do NOT change with time
    # FIXME: Should handle cases where b/cs change somehow...
    # Constant surface temperature
    a_matrix[0, 0] = 1
    b_vector[0] = params["temp_surf"]

    if params["bc_type"] == 1:
        # Assign constant basal temperature
        a_matrix[-1, -1] = 1
        b_vector[-1] = params["temp_base"]
    elif params["bc_type"] == 2:
        # Assign constant basal heat flux
        a_matrix[-1, -3] = 1.0
        a_matrix[-1, -2] = -4.0
        a_matrix[-1, -1] = 3.0
        b_vector[-1] = 2 * milli2base(params["flux_base"]) * dx / k[-1]
    else:
        raise ValueError(
            f"Bad boundary condition type: {params['bc_type']}. Must be 1 or 2."
        )

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
            mmyr2ms(params["mantle_velocity"]),
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
                # Increment current time
                curtime -= dt

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
                    mmyr2ms(params["mantle_velocity"]),
                )
                move_particles = surface_times >= curtime
                depths[move_particles] -= vx_pts[move_particles] * -dt
                # Store exhumation velocity history for particle reaching surface at 0 Ma
                vx_hist[idx] = vx_pts[-1]
                if params["plot_fault_depth_history"]:
                    fault_depth_history[idx] = fault_depth

                # Increment idx
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
                mmyr2ms(params["mantle_velocity"]),
            )

        # Create initial plots and set plot parameters if plotting requested
        if params["plot_results"]:
            # Set plot style
            plt.style.use("seaborn-v0_8-darkgrid")

            # Plot animation of temperatures and particle depth, if requested
            if params["watch_it_exhume"]:
                n_anim_steps = 200
                anim_plot_increment = int(round(nt / n_anim_steps, 0))
                plt.ion()
                fig_anim, (ax1_anim, ax2_anim) = plt.subplots(1, 2, figsize=(12, 8))
                # Create initial temperature plot
                (temp_line,) = ax1_anim.plot(
                    temp_init, -x / 1000.0, label="Current temp."
                )
                particle_temp = np.interp(depths[0], x, temp_init)
                (particle,) = ax1_anim.plot(
                    particle_temp, -depths[0] / 1000.0, "o", label="Tracking particle"
                )
                time_text = ax1_anim.text(
                    0.05 * params["temp_base"],
                    -0.95 * max_depth / 1000.0,
                    f"Time: {t_total / myr2sec(1):.1f} Ma",
                )
                ax1_anim.legend()

                # Create initial thermal history plot
                (thist_line,) = ax2_anim.plot(
                    t_total, particle_temp, "o", label="Thermal history"
                )
                ax2_anim.xaxis.set_inverted(True)
                ax2_anim.yaxis.set_inverted(True)

                # Use tight layout
                plt.tight_layout()

            # Create static plots otherwise
            else:
                # Plot initial temperature field
                if params["plot_density"]:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
                else:
                    fig, ax1 = plt.subplots(1, 1, figsize=(6, 8))
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
                if params["plot_density"]:
                    ax2.plot(density_init, -x / 1000, "k-", label=time_label)

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
            if params["solution_type"] == 1:
                temp_new[:] = temp_transient_explicit(
                    params["nx"],
                    dx,
                    dt,
                    temp_prev,
                    temp_new,
                    params["bc_type"],
                    params["temp_surf"],
                    params["temp_base"],
                    milli2base(params["flux_base"]),
                    vx_array,
                    rho,
                    cp,
                    k,
                    heat_prod,
                )
            elif params["solution_type"] == 2:
                temp_new[:] = temp_transient_implicit(
                    params["nx"],
                    dx,
                    dt,
                    a_matrix,
                    b_vector,
                    temp_prev,
                    vx_array,
                    rho,
                    cp,
                    k,
                    heat_prod,
                )
            elif params["solution_type"] == 3:
                temp_new[:] = temp_transient_crank_nicolson(
                    params["nx"],
                    dx,
                    dt,
                    a_matrix,
                    b_vector,
                    temp_prev,
                    vx_array,
                    rho,
                    cp,
                    k,
                    heat_prod,
                )
            else:
                raise ValueError(
                    f"Bad solution type: {params['solution_type']}. Must be 1, 2 or 3."
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
                    mmyr2ms(params["mantle_velocity"]),
                )
                # Determine which particle positions to update
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

                if params["plot_results"] and params["watch_it_exhume"]:
                    if idx % anim_plot_increment == 0:
                        # Update geotherm plot
                        temp_line.set_data(temp_new, -x / 1000.0)
                        particle_temp = np.interp(depths[0], x, temp_new)
                        particle.set_data([particle_temp], [-depths[0] / 1000.0])
                        time_text.set_text(
                            f"Time: {(t_total - curtime) / myr2sec(1):.1f} Ma"
                        )

                        # Update thermal history plot
                        thist_line.set_data(
                            (t_total - time_hists[-1][:idx]) / myr2sec(1.0),
                            temp_hists[-1][:idx],
                        )

                        # Rescale axes
                        ax1_anim.relim()
                        ax1_anim.autoscale_view()
                        ax2_anim.relim()
                        ax2_anim.autoscale_view()

                        # Add axis labels
                        ax1_anim.set_xlabel("Temperature (°C)")
                        ax1_anim.set_ylabel("Depth (km)")
                        ax2_anim.set_xlabel("Time (Ma)")
                        ax2_anim.set_ylabel("Temperature (°C)")

                        # Redraw
                        fig_anim.canvas.draw()
                        fig_anim.canvas.flush_events()

                    if idx % (40 * anim_plot_increment) == 0:
                        ax1_anim.plot(
                            temp_new,
                            -x / 1000.0,
                            label=f"{(t_total - curtime) / myr2sec(1):.1f} Ma",
                        )
                        ax1_anim.legend()

            if params["debug"]:
                print(
                    f"Maximum temp difference at time {curtime / myr2sec(1):.4f} Myr: {max_temp_diff:.4f} °C"
                )

            # Calculate current Peclet number using surface velocity
            peclet[idx] = calculate_peclet_number(
                params["ero_type"],
                vx_array[0],
                x,
                xstag,
                k,
                rho,
                cp,
                moho_depth,
                max_depth,
                params["crustal_uplift"],
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
                mmyr2ms(params["mantle_velocity"]),
            )

            # Plot temperature and density profiles
            if j == num_pass - 1:
                if params["plot_results"] and more_plots:
                    if not params["watch_it_exhume"]:
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
                            if params["plot_density"]:
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

    if params["watch_it_exhume"]:
        plt.ioff()
        # plt.show()

    # Calculate final densities
    density_new = update_density(rho, alphav, temp_new)

    # Calculate final Moho temperature and heat flow
    interp_temp_new = interp1d(x, temp_new)
    final_moho_temp = interp_temp_new(moho_depth)
    final_heat_flow = calculate_heat_flow(temp_new, k, dx)

    # Calculate final dimensionless heat production
    final_nd_hp = calculate_nd_heat_prod(
        heat_prod,
        k,
        params["temp_base"],
        max_depth,
    )

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
        print(f"- Final Peclet number: {peclet[-1]:.2f}")
        print(f"  - Average for all steps: {peclet.mean():.2f}")
        print(f"  - Range for all steps: {peclet.min():.2f}-{peclet.max():.2f}")
        print(f"- Final dimensionless heat production: {final_nd_hp:.2f}")

    # Calculate ages
    if params["calc_ages"]:
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

        # Define string for time-temperature and time-temperature-depth ages
        for i in range(surf_age_array_size):
            # Define tT and tTd filenames
            if len(surface_times_ma) == 1:
                file_age_text = ""
            else:
                file_age_text = f"_{surface_times_ma[i]}Ma"
            tt_filename = "time_temp_hist" + file_age_text + ".csv"
            ttdp_filename = "time_temp_depth_pressure_hist" + file_age_text + ".csv"

            # Calculate ages
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
                write_tt_history,
                tt_filename,
                write_ttdp_history,
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

            # Rename and/or move time-temperature history file, if requested
            # TODO: Make this a function!
            if write_tt_history:
                tt_orig = Path(tt_filename)
                if params["batch_mode"]:
                    # Rename and move files to batch output directory
                    tt_newfile = params["model_id"] + "-" + tt_filename
                    tt_new = tt_orig.rename(wd / "csv" / tt_newfile)
                else:
                    tt_new = tt_orig.rename(wd / "csv" / tt_orig)

            # Rename and/or move time-temperature-depth-pressure history file, if requested
            if write_ttdp_history:
                ttdp_orig = Path(ttdp_filename)
                if params["batch_mode"]:
                    # Rename and move files to batch output directory
                    ttdp_newfile = params["model_id"] + "-" + ttdp_filename
                    ttdp_new = ttdp_orig.rename(wd / "csv" / ttdp_newfile)
                else:
                    ttdp_new = ttdp_orig.rename(wd / "csv" / ttdp_orig)

            # Move track length file to csv directory
            if write_track_lengths:
                ftl_orig = Path("ft_length.csv")
                ftl_filename = "ft_length" + file_age_text + ".csv"
                if params["batch_mode"]:
                    # Rename and move files to batch output directory
                    ftl_newfile = params["model_id"] + "-" + ftl_filename
                    ftl_new = ftl_orig.rename(wd / "csv" / ftl_newfile)
                else:
                    ftl_new = ftl_orig.rename(wd / "csv" / ftl_filename)

        if params["echo_ages"]:
            print("")
            print("--- Predicted thermochronometer ages ---")
            print("")
            print(
                f"- AHe age: {corr_ahe_ages[-1]:.2f} Ma (uncorrected age: {ahe_age:.2f} Ma)"
            )
            if params["madtrax_aft"]:
                print(f"- AFT age: {aft_ages[-1] / 1e6:.2f} Ma (MadTrax)")
            if params["ketch_aft"]:
                print(f"- AFT age: {aft_ages[-1]:.2f} Ma (Ketcham)")
            print(
                f"- ZHe age: {corr_zhe_ages[-1]:.2f} Ma (uncorrected age: {zhe_age:.2f} Ma)"
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
            rdaam = load_rdaam()
            # pa = pointer()
            # Does this work?
            pa = rdaam.make_path()
            for i in range(len(obs_ages_file)):
                # Increment time-temperature history index whenever it changes, write new tt-history file
                depo_age_now = obs_depo_age_file[i]

                if depo_age_now < depo_age_old:
                    tt_hist_index += 1

                    rdaam.del_path(pa)
                    pa = rdaam.make_path()
                    time_ma = tt_hist_to_ma(time_hists[tt_hist_index])
                    write_increment = get_write_increment(params, time_ma)
                    for j in range(len(time_ma) - 1, -1, -write_increment):
                        rdaam.path_push(
                            pa,
                            c_double(time_ma[j]),
                            c_double(temp_hists[tt_hist_index][j]),
                        )
                    if params["pad_time"] > 0.0:
                        pad_times = np.arange(
                            time_ma.max(), time_ma.max() + params["pad_time"] + 0.1, 1.0
                        )
                        for pad_time in pad_times:
                            rdaam.path_push(
                                pa,
                                c_double(pad_time),
                                c_double(temp_hists[tt_hist_index][i]),
                            )

                    # Calculate time to be added to predicted age
                    extra_depo_time = depo_age_now

                    # Update depositional age reference
                    depo_age_old = depo_age_now

                if obs_age_type_file[i] == "AHe":
                    # Calculate AHe age
                    _, corr_ahe_age, _, _ = he_ages(
                        rdaam=rdaam,
                        pa=pa,
                        ap_rad=obs_radius_file[i],
                        ap_uranium=obs_u_file[i],
                        ap_thorium=obs_th_file[i],
                        zr_rad=params["zr_rad"],
                        zr_uranium=params["zr_uranium"],
                        zr_thorium=params["zr_thorium"],
                    )
                    # FIXME: Should this same calculation be done for a separate array of depo ages and plotted???
                    pred_data_ages[i] = corr_ahe_age + depo_age_now
                    pred_data_temps[i] = calculate_closure_temp(
                        corr_ahe_age,
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
                    pred_data_ages[i] = aft_ages[tt_hist_index] + depo_age_now
                    pred_data_temps[i] = aft_temps[tt_hist_index]

                elif obs_age_type_file[i] == "ZHe":
                    # Calculate ZHe age
                    _, _, _, corr_zhe_age = he_ages(
                        rdaam=rdaam,
                        pa=pa,
                        ap_rad=params["ap_rad"],
                        ap_uranium=params["ap_uranium"],
                        ap_thorium=params["ap_thorium"],
                        zr_rad=obs_radius_file[i],
                        zr_uranium=obs_u_file[i],
                        zr_thorium=obs_th_file[i],
                    )
                    pred_data_ages[i] = corr_zhe_age + depo_age_now
                    pred_data_temps[i] = calculate_closure_temp(
                        corr_zhe_age,
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
                    pred_data_ages[i] = zft_ages[tt_hist_index] + depo_age_now
                    pred_data_temps[i] = zft_temps[tt_hist_index]

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
                    pred_ages.append(corr_ahe_ages[-1])
                    obs_ages.append(params["obs_ahe"][i])
                    obs_stdev.append(params["obs_ahe_stdev"][i])
                    obs_eu.append("")
                    obs_radius.append("")
                for i in range(n_obs_aft):
                    pred_ages.append(aft_ages[-1])
                    obs_ages.append(params["obs_aft"][i])
                    obs_stdev.append(params["obs_aft_stdev"][i])
                    obs_eu.append("")
                    obs_radius.append("")
                for i in range(n_obs_zhe):
                    pred_ages.append(corr_zhe_ages[-1])
                    obs_ages.append(params["obs_zhe"][i])
                    obs_stdev.append(params["obs_zhe_stdev"][i])
                    obs_eu.append("")
                    obs_radius.append("")
                for i in range(n_obs_zft):
                    pred_ages.append(zft_ages[-1])
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

            # Set misfit type to type 3 for inverse models
            if params["inverse_mode"]:
                params["misfit_type"] = 3

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
        # Update plot and plot final geotherm if using watch it exhume mode
        if params["watch_it_exhume"]:
            # Update geotherm plot
            temp_line.set_data(temp_new, -x / 1000.0)
            particle_temp = np.interp(depths[0], x, temp_new)
            particle.set_data([particle_temp], [-depths[0] / 1000.0])
            time_text.set_text(f"Time: {(t_total - curtime) / myr2sec(1):.1f} Ma")

            # Update thermal history plot
            thist_line.set_data(
                (t_total - time_hists[-1][:idx]) / myr2sec(1.0), temp_hists[-1][:idx]
            )

            # Rescale axes
            ax1_anim.relim()
            ax1_anim.autoscale_view()
            ax2_anim.relim()
            ax2_anim.autoscale_view()

            # Redraw
            fig_anim.canvas.draw()
            fig_anim.canvas.flush_events()

            # Plot final geotherm
            ax1_anim.plot(temp_new, -x / 1000.0, label=time_label)
            ax1_anim.legend()

            # Use tight layout
            plt.tight_layout()

        # Plot final geotherm on static plot
        else:
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
                crust_thickness = max(
                    params["init_moho_depth"], moho_depth / kilo2base(1)
                )
                crust_slice = x / kilo2base(1) <= crust_thickness
                pressure = calculate_pressure(density_new, dx)
                crust_pressure = pressure[crust_slice]
                wet_felsic_solidus = calculate_crust_solidus(
                    "wet_felsic", crust_pressure
                )
                dry_basalt_solidus = calculate_crust_solidus(
                    "dry_basalt", crust_pressure
                )
                wet_felsic_solidus_plot_text = crust_solidus_comp_text["wet_felsic"]
                dry_basalt_solidus_plot_text = crust_solidus_comp_text["dry_basalt"]

                # Mantle solidii
                min_moho_depth = min(
                    params["init_moho_depth"], moho_depth / kilo2base(1)
                )
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

            # Plot density, if requested
            if params["plot_density"]:
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
                plot_filename = "temperature_history.png"
                if params["plot_density"]:
                    plot_filename = "temperature_density_history.png"
                savefile = wd / "png" / plot_filename
                plt.savefig(savefile, dpi=300)
                if params["plot_density"]:
                    print(f"- Temperature/density history plot written to {savefile}")
                else:
                    print(f"- Temperature history plot written to {savefile}")
            if params["display_plots"]:
                plt.show()
            else:
                plt.close()

        # Plot elevation history
        if params["plot_elevation_history"]:
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
            ax2 = fig.add_subplot(gs[2, :])
            # Add second y axis if plotting Peclet number
            if params["plot_peclet_number"]:
                ax2b = ax2.twinx()

            # Calculate synthetic uncertainties
            ahe_uncert = 0.1
            aft_uncert = 0.2
            zhe_uncert = 0.1
            zft_uncert = 0.2
            ahe_min, ahe_max = (
                (1.0 - ahe_uncert) * corr_ahe_ages[-1],
                (1.0 + ahe_uncert) * corr_ahe_ages[-1],
            )
            aft_min, aft_max = (
                (1.0 - aft_uncert) * aft_ages[-1],
                (1.0 + aft_uncert) * aft_ages[-1],
            )
            zhe_min, zhe_max = (
                (1.0 - zhe_uncert) * corr_zhe_ages[-1],
                (1.0 + zhe_uncert) * corr_zhe_ages[-1],
            )
            zft_min, zft_max = (
                (1.0 - zft_uncert) * zft_ages[-1],
                (1.0 + zft_uncert) * zft_ages[-1],
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

            # Plot intrusions and their time, if enabled
            """
            if "intrusion" in locals():
                intrusion_start_time_ma = params["t_total"] - intrusion.start_time / myr2sec(1.0)
                intrusion_end_time_ma = params["t_total"] - intrusion.end_time / myr2sec(1.0)
                ax1.plot(
                    [intrusion_start_time_ma, intrusion_start_time_ma],
                    [params["temp_surf"], params["temp_base"]],
                    "--",
                    color="gray",
                    label="Time of intrusion activity",
                )
                if intrusion.duration / myr2sec(1.0) > 2.0:
                    ax1.plot(
                        [intrusion_end_time_ma, intrusion_end_time_ma],
                        [params["temp_surf"], params["temp_base"]],
                        "-.",
                        color="gray",
                        #label="Time of intrusion activity",
                    )
                ax1.text(
                    intrusion_start_time_ma - 0.02 * t_total / myr2sec(1.0),
                    (temp_hists[-1].max() + temp_hists[-1].min()) / 4.0,
                    "Intrusion emplaced",
                    rotation=90,
                    ha="center",
                    va="center",
                    color="gray",
                )
            """

            # Plot uncertainty error bars and AHe age if no measured ages exist
            if n_obs_ahe == 0:
                ax1 = plot_predictions_no_data(
                    corr_ahe_ages[-1],
                    ahe_temps[-1],
                    xerr=ahe_uncert * corr_ahe_ages[-1],
                    ax=ax1,
                    marker="o",
                    color="tab:blue",
                    label=f"Predicted AHe age ({corr_ahe_ages[-1]:.2f} ± {ahe_uncert * corr_ahe_ages[-1]:.2f} Ma ({ahe_uncert * 100.0:.0f}% error); T$_c$ = {ahe_temps[-1]:.1f}°C)",
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
                        min_file_ahe_temp = min(pred_data_temps[obs_ahe_indices])
                        max_file_ahe_temp = max(pred_data_temps[obs_ahe_indices])
                        ahe_label = f"Predicted AHe ages ({min_file_ahe_age:.2f}–{max_file_ahe_age:.2f} Ma; T$_c$ = {min_file_ahe_temp:.1f}–{max_file_ahe_temp:.1f}° C)"
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
                        corr_ahe_ages[-1],
                        ahe_temps[-1],
                        ax=ax1,
                        marker="o",
                        color="tab:blue",
                        label=f"Predicted AHe age ({corr_ahe_ages[-1]:.2f} Ma; T$_c$ = {ahe_temps[-1]:.1f}° C)",
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
                    aft_ages[-1],
                    aft_temps[-1],
                    xerr=aft_uncert * aft_ages[-1],
                    ax=ax1,
                    marker="s",
                    color="tab:orange",
                    label=f"Predicted AFT age ({aft_ages[-1]:.2f} ± {aft_uncert * aft_ages[-1]:.2f} Ma ({aft_uncert * 100.0:.0f}% error); T$_c$ = {aft_temps[-1]:.1f}° C)",
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
                        label=f"Predicted AFT age ({pred_ages[obs_aft_indices[0]]:.2f} Ma; T$_c$ = {pred_data_temps[obs_aft_indices[0]]:.1f}° C)",
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
                        aft_ages[-1],
                        aft_temps[-1],
                        ax=ax1,
                        marker="s",
                        color="tab:orange",
                        label=f"Predicted AFT age ({aft_ages[-1]:.2f} Ma; T$_c$ = {aft_temps[-1]:.1f}° C)",
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
                    corr_zhe_ages[-1],
                    zhe_temps[-1],
                    xerr=zhe_uncert * corr_zhe_ages[-1],
                    ax=ax1,
                    marker="d",
                    color="tab:green",
                    label=f"Predicted ZHe age ({corr_zhe_ages[-1]:.2f} ± {zhe_uncert * corr_zhe_ages[-1]:.2f} Ma ({zhe_uncert * 100.0:.0f}% error); T$_c$ = {zhe_temps[-1]:.1f}° C)",
                )
            # Plot predicted age + observed ZHe age(s)
            else:
                if ages_from_data_file:
                    if n_obs_zhe == 1:
                        zhe_label = f"Predicted ZHe age ({float(pred_ages[obs_zhe_indices[0]]):.2f} Ma; T$_c$ = {pred_data_temps[obs_zhe_indices[0]]:.1f}° C)"
                    else:
                        min_file_zhe_age = min(pred_ages[obs_zhe_indices])
                        max_file_zhe_age = max(pred_ages[obs_zhe_indices])
                        min_file_zhe_temp = min(pred_data_temps[obs_zhe_indices])
                        max_file_zhe_temp = max(pred_data_temps[obs_zhe_indices])
                        zhe_label = f"Predicted ZHe ages ({min_file_zhe_age:.2f}–{max_file_zhe_age:.2f} Ma; T$_c$ = {min_file_zhe_temp:.1f}–{max_file_zhe_temp:.1f}° C)"
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
                        corr_zhe_ages[-1],
                        zhe_temps[-1],
                        ax=ax1,
                        marker="d",
                        color="tab:green",
                        label=f"Predicted ZHe age ({corr_zhe_ages[-1]:.2f} Ma; T$_c$ = {zhe_temps[-1]:.1f}° C)",
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
                    zft_ages[-1],
                    zft_temps[-1],
                    xerr=zft_uncert * zft_ages[-1],
                    ax=ax1,
                    marker="^",
                    color="tab:red",
                    label=f"Predicted ZFT age ({zft_ages[-1]:.2f} ± {zft_uncert * zft_ages[-1]:.2f} Ma ({zft_uncert * 100.0:.0f}% error); T$_c$ = {zft_temps[-1]:.1f}° C)",
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
                        label=f"Predicted ZFT age ({pred_ages[obs_zft_indices[0]]:.2f} Ma; T$_c$ = {pred_data_temps[obs_zft_indices[0]]:.1f}° C)",
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
                        zft_ages[-1],
                        zft_temps[-1],
                        ax=ax1,
                        marker="^",
                        color="tab:red",
                        label=f"Predicted ZFT age ({zft_ages[-1]:.2f} Ma; T$_c$ = {zft_temps[-1]:.1f}° C)",
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
            if not params["plot_peclet_number"]:
                ax1.set_xlabel("Time (Ma)")
            ax1.set_ylabel("Temperature (° C)")
            if params["plot_depth_history"] or params["plot_fault_depth_history"]:
                # Make left y-axis dimgray
                ax1.set_ylabel("Temperature (° C)", color="dimgray")
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

            if not params["plot_peclet_number"]:
                ax2.plot(time_ma, vx_hist / mmyr2ms(1))
                ax2.fill_between(
                    time_ma,
                    vx_hist / mmyr2ms(1),
                    0.0,
                    alpha=0.33,
                    color="tab:blue",
                    label=f"Total erosional exhumation: {exhumation_magnitude:.2f} km",
                )
            else:
                ax2.plot(
                    time_ma,
                    vx_hist / mmyr2ms(1),
                    color="dimgray",
                    label=f"Erosion rate (Total exhumation: {exhumation_magnitude:.2f} km)",
                )
                ax2b.plot(
                    time_ma, peclet, "--", color="darkgray", label="Péclet number"
                )
            ax2.set_xlabel("Time (Ma)")
            if not params["plot_peclet_number"]:
                ax2.set_ylabel("Erosion rate (mm/yr)")
            else:
                ax2.set_ylabel("Erosion rate (mm/yr)", color="dimgray")
                ax2.tick_params(axis="y", colors="dimgray")
                ax2b.set_ylabel("Péclet number", color="darkgray")
                ax2b.tick_params(axis="y", colors="darkgray")
            ax2.set_xlim(t_total / myr2sec(1), 0.0)
            if params["plot_peclet_number"]:
                ax2.grid(None)
                ax2b.grid(None)
                lines, labels = ax2.get_legend_handles_labels()
                lines2, labels2 = ax2b.get_legend_handles_labels()
                ax2.legend(lines + lines2, labels + labels2)
            else:
                ax2.legend()
            ax2.set_title("Erosion history for surface sample")

            plt.tight_layout()
            if params["save_plots"]:
                savefile = wd / "png" / "thermal_history.png"
                plt.savefig(savefile, dpi=300)
                print(f"- Thermal history and ages plot written to {savefile}")
            if params["display_plots"]:
                plt.show()
            else:
                plt.close()

            if params["plot_ft_length_dist"]:
                ft_lengths = np.genfromtxt(ftl_new, delimiter=",", skip_header=1)
                length = ft_lengths[:, 0]
                prob = ft_lengths[:, 1]
                fig, ax = plt.subplots(1, 1)
                ax.plot(length, prob)
                ax.plot(
                    [float(aft_mean_ftl), float(aft_mean_ftl)],
                    [0.0, 1.05 * prob.max()],
                    label=f"Mean: {float(aft_mean_ftl):.1f} µm",
                )
                ax.set_xlabel("Track length (um)")
                ax.set_ylabel("Probability")
                ax.set_xlim([0.0, 20.0])
                ax.set_ylim([0.0, 1.05 * prob.max()])
                ax.legend()
                ax.set_title("Apatite fission-track length distribution")

                plt.tight_layout()
                if params["save_plots"]:
                    savefile = wd / "png" / "apatite_ft_length_dist.png"
                    plt.savefig(savefile, dpi=300)
                    print(
                        f"- Apatite fission-track length distribution plot written to {savefile}"
                    )
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
                f"{t_total / myr2sec(1):.4f},{dt / yr2sec(1):.4f},{max_depth / kilo2base(1):.4f},{params['nx']},{params['bc_type']},"
                f"{params['temp_surf']:.4f},{params['temp_base']:.4f},{params['flux_base']:.4f},{params['temp_adiabat_ref']:.4f},{params['mantle_adiabat']},"
                f"{params['rho_crust']:.4f},{params['removal_fraction']:.4f},{params['removal_start_time']:.4f},"
                f"{params['removal_end_time']:.4f},{params['ero_type']},{params['ero_option1']:.4f},"
                f"{params['ero_option2']:.4f},{params['ero_option3']:.4f},{params['ero_option4']:.4f},"
                f"{params['ero_option5']:.4f},{params['ero_option6']:.4f},{params['ero_option7']:.4f},{params['ero_option8']:.4f},{params['ero_option9']:.4f},{params['ero_option10']:.4f},{params['mantle_velocity']:.4f},{params['init_moho_depth']:.4f},{init_moho_temp:.4f},"
                f"{init_heat_flow:.4f},{elev_list[1] / kilo2base(1):.4f},{init_nd_hp:.4f},{moho_depth / kilo2base(1):.4f},"
                f"{final_moho_temp:.4f},{final_heat_flow:.4f},{elev_list[-1] / kilo2base(1):.4f},{final_nd_hp:.4f},"
                f"{exhumation_magnitude:.4f},{peclet.mean():.4f},{peclet.min():.4f},{peclet.max():.4f},{params['ap_rad']:.4f},{params['ap_uranium']:.4f},"
                f"{params['ap_thorium']:.4f},{params['zr_rad']:.4f},{params['zr_uranium']:.4f},"
                f"{params['zr_thorium']:.4f},{corr_ahe_ages[-1]:.4f},"
                f"{ahe_temps[-1]:.4f},{obs_ahe:.4f},"
                f"{obs_ahe_stdev:.4f},{aft_ages[-1]:.4f},"
                f"{aft_temps[-1]:.4f},{obs_aft:.4f},"
                f"{obs_aft_stdev:.4f},{corr_zhe_ages[-1]:.4f},"
                f"{zhe_temps[-1]:.4f},{obs_zhe:.4f},"
                f"{obs_zhe_stdev:.4f},{zft_ages[-1]:.4f},"
                f"{zft_temps[-1]:.4f},{obs_zft:.4f},"
                f"{obs_zft_stdev:.4f},{misfit:.6f},{misfit_type},{misfit_ages}\n"
            )

    # Write summary age output to file
    if params["write_age_output"]:
        savefile = wd / "csv" / "age_summary.csv"
        # Use sample IDs from data file, or None otherwise
        if ages_from_data_file:
            # Fill in age types
            obs_age_types = obs_age_type_file
            # Set eU and grain radii to None for FT ages
            obs_eu[obs_age_types == "AFT"] = None
            obs_eu[obs_age_types == "ZFT"] = None
            obs_eu = obs_eu.astype("str")
            obs_eu[obs_eu == "nan"] = ""
            obs_radius[obs_age_types == "AFT"] = None
            obs_radius[obs_age_types == "ZFT"] = None
            obs_radius = obs_radius.astype("str")
            obs_radius[obs_radius == "nan"] = ""
            # Use sample IDs from data file
            sample_id_out = obs_sample_id_file
            # Use depo ages from file
            depo_age_out = obs_depo_age_file
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
            # Use empty depositional ages
            depo_age_out = np.array([""] * len(obs_ages))
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
                depo_age_out,
                pred_ages.round(2),
                pred_eu,
                pred_radius,
            )
        ]
        np.savetxt(
            savefile,
            summary_ages,
            delimiter=",",
            header="Age type,Measured age (Ma),Measured age stdev (Ma),Measured age eU (ppm),Measured age grain radius (um),Sample ID,Depositional age (Ma),Predicted age (Ma),Predicted age eU (ppm),Predicted age grain radius (um)",
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
        return misfit
