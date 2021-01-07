#!/usr/bin/env python3

# Import libaries we need
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from scipy.interpolate import interp1d, make_interp_spline, BSpline
import argparse
import subprocess
import csv
import os
from sklearn.model_selection import ParameterGrid

# Import user functions
from mad_trax import *

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


def echo_model_info(dx, nt, dt, t_total, implicit, vx, k_crust,
                    rho_crust, Cp_crust, k_mantle, rho_mantle, Cp_mantle, k_a,
                    erotype, cond_crit=0.5, adv_crit=0.5):
    print('')
    print('--- General model information ---')
    print('')
    print('- Node spacing: {0} m'.format(dx))
    print('- Total simulation time: {0:.1f} million years'.format(t_total / myr2sec(1)))
    print('- Time steps: {0} @ {1:.1f} years each'.format(nt, dt / yr2sec(1)))

    if implicit:
        print('- Solution type: Implicit')
    else:
        print('- Solution type: Explicit')

    # Check stability conditions
    if not implicit:
        kappa_crust = k_crust / (rho_crust * Cp_crust)
        kappa_mantle = k_mantle / (rho_mantle * Cp_mantle)
        kappa_a = k_a / (rho_mantle * Cp_mantle)
        kappa = max(kappa_crust, kappa_mantle, kappa_a)
        cond_stab = kappa * dt / dx ** 2
        print("- Conductive stability: {0} ({1:.3f} < {2:.4f})".format((cond_stab < cond_crit), cond_stab, cond_crit))
        if cond_stab >= cond_crit:
            raise UnstableSolutionException('Heat conduction solution unstable. Decrease nx or dt.')

        adv_stab = vx * dt / dx
        print("- Advective stability: {0} ({1:.3f} < {2:.4f})".format((adv_stab < adv_crit), adv_stab, adv_crit))
        if adv_stab >= adv_crit:
            raise UnstableSolutionException(
                'Heat advection solution unstable. Decrease nx, dt, or vx (change in Moho over model time).')

    # Output erosion model
    ero_models = {1: 'Constant', 2: 'Step-function', 3: 'Exponential decay'}
    print('- Erosion model: {0}'.format(ero_models[erotype]))


# Mantle adiabat from Turcotte and Schubert (eqn 4.254)
def adiabat(alphav, T, Cp):
    """Calculates a mantle adiabat in degress / m."""
    g = 9.81
    return alphav * g * T / Cp


# Conductive steady-state heat transfer
def temp_ss_implicit(nx, dx, Tsurf, Tbase, vx, rho, Cp, k, H):
    """Calculates a steady-state thermal solution."""
    # Create the empty (zero) coefficient and right hand side arrays
    A = np.zeros((nx, nx))  # 2-dimensional array, ny rows, ny columns
    b = np.zeros(nx)

    # Set B.C. values in the coefficient array and in the r.h.s. array
    A[0, 0] = 1
    b[0] = Tsurf
    A[nx - 1, nx - 1] = 1
    b[nx - 1] = Tbase

    # Matrix loop
    for ix in range(1, nx - 1):
        A[ix, ix - 1] = (-(rho[ix - 1] * Cp[ix - 1] * -vx) / (2 * dx)) - k[ix - 1] / dx ** 2
        A[ix, ix] = k[ix] / dx ** 2 + k[ix - 1] / dx ** 2
        A[ix, ix + 1] = (rho[ix + 1] * Cp[ix + 1] * -vx) / (2 * dx) - k[ix] / dx ** 2
        b[ix] = H[ix]

    T = solve(A, b)
    return T


def update_materials(x, xstag, moho_depth, rho_crust, rho_mantle, rho, Cp_crust,
                     Cp_mantle, Cp, k_crust, k_mantle, k, H_crust, H_mantle, H,
                     Tadiabat, Tprev, k_a):
    """Updates arrays of material properties."""
    rho[:] = rho_crust
    rho[x > moho_depth] = rho_mantle
    Cp[:] = Cp_crust
    Cp[x > moho_depth] = Cp_mantle
    k[:] = k_crust
    k[xstag > moho_depth] = k_mantle

    interpTprev = interp1d(x, Tprev)
    Tstag = interpTprev(xstag)
    k[Tstag >= Tadiabat] = k_a
    lab_depth = xstag[Tstag >= Tadiabat].min()

    H[:] = H_crust
    H[x > moho_depth] = H_mantle
    return rho, Cp, k, H, lab_depth


def temp_transient_explicit(Tprev, Tnew, Tsurf, Tbase, nx, dx, vx, dt,
                            rho, Cp, k, H):
    """Updates a transient thermal solution."""
    # Set boundary conditions
    Tnew[0] = Tsurf
    Tnew[nx - 1] = Tbase

    # Calculate internal grid point temperatures
    for ix in range(1, nx - 1):
        Tnew[ix] = ((1 / (rho[ix] * Cp[ix])) * (
                k[ix] * (Tprev[ix + 1] - Tprev[ix]) - k[ix - 1] * (Tprev[ix] - Tprev[ix - 1])) / dx ** 2 + H[ix] / (
                            rho[ix] * Cp[ix]) + vx * (Tprev[ix + 1] - Tprev[ix - 1]) / (2 * dx)) * dt + Tprev[ix]

    return Tnew


# Conductive steady-state heat transfer
def temp_transient_implicit(nx, dx, dt, Tprev, Tsurf, Tbase, vx, rho, Cp, k, H):
    """Calculates a steady-state thermal solution."""
    # Create the empty (zero) coefficient and right hand side arrays
    A = np.zeros((nx, nx))  # 2-dimensional array, ny rows, ny columns
    b = np.zeros(nx)

    # Set B.C. values in the coefficient array and in the r.h.s. array
    A[0, 0] = 1
    b[0] = Tsurf
    A[nx - 1, nx - 1] = 1
    b[nx - 1] = Tbase

    # Matrix loop
    for ix in range(1, nx - 1):
        A[ix, ix - 1] = -(rho[ix - 1] * Cp[ix - 1] * -vx) / (2 * dx) - k[ix - 1] / dx ** 2
        A[ix, ix] = (rho[ix] * Cp[ix]) / dt + k[ix] / dx ** 2 + k[ix - 1] / dx ** 2
        A[ix, ix + 1] = (rho[ix + 1] * Cp[ix + 1] * -vx) / (2 * dx) - k[ix] / dx ** 2
        b[ix] = H[ix] + ((rho[ix] * Cp[ix]) / dt) * Tprev[ix]

    T = solve(A, b)
    return T


def He_ages(file, ap_rad=60.0, ap_U=10.0, ap_Th=40.0, zr_rad=60.0, zr_U=10.0, zr_Th=40.0):
    """Calculates (U-Th)/He ages."""

    command = '../bin/RDAAM_He ' + file + ' ' + str(ap_rad) + ' ' + str(ap_U) + ' ' + str(ap_Th) + ' ' + str(
        zr_rad) + ' ' + str(zr_U) + ' ' + str(zr_U)
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    stdout = p.stdout.readlines()

    ahe_age = stdout[0].split()[3][:-1].decode('UTF-8')
    corr_ahe_age = stdout[0].split()[7].decode('UTF-8')
    zhe_age = stdout[1].split()[3][:-1].decode('UTF-8')
    corr_zhe_age = stdout[1].split()[7].decode('UTF-8')

    retval = p.wait()
    return ahe_age, corr_ahe_age, zhe_age, corr_zhe_age


def FT_ages(file):
    """Calculates AFT ages."""

    command = '../bin/ketch_aft ' + file
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    stdout = p.stdout.readlines()
    aft_age = stdout[0].split()[4][:-1].decode('UTF-8')

    retval = p.wait()
    return aft_age


def calculate_erosion_rate(t_total, current_time, magnitude, erotype, erotype_opt1, erotype_opt2):
    """Defines the way in which erosion should be applied."""

    # Constant erosion rate
    if erotype == 1:
        vx = magnitude / t_total

    # Constant erosion rate with a step-function change at a specified time
    elif erotype == 2:
        init_rate = mmyr2ms(erotype_opt1)
        rate_change_time = myr2sec(erotype_opt2)
        remaining_magnitude = magnitude - (init_rate * rate_change_time)
        # First stage of erosion
        if current_time < rate_change_time:
            vx = init_rate
        # Second stage of erosion
        else:
            vx = remaining_magnitude / (t_total - rate_change_time)

    # Exponential erosion rate decay with a set characteristic time
    elif erotype == 3:
        decay_time = myr2sec(erotype_opt1)
        # Calculate max erosion rate for exponential
        max_rate = magnitude / (decay_time * (np.exp(0.0 / decay_time) - np.exp(-t_total / decay_time)))
        vx = max_rate * np.exp(-current_time / decay_time)

    # Catch bad cases
    else:
        raise MissingOption('Bad erosion type. Type should be 1 or 2.')

    return vx


def crust_solidus():
    """Reads a file with a crustal solidus"""
    fp = '../csv/granite_solidus.csv'
    data = np.genfromtxt(fp, delimiter=',', skip_header=3)
    depth = data[:, 0]
    wet = data[:, 1]
    dry = data[:, 2]
    return depth, wet, dry


def mantle_solidus():
    """Reads a file with a crustal solidus"""
    fp = '../csv/olivine_solidus.csv'
    data = np.genfromtxt(fp, delimiter=',', skip_header=4)
    depth = data[:, 1]
    wet = data[:, 2]
    dry = data[:, 3]
    return depth, wet, dry

def prep_model(params):
    """Prepares models to be run as single models or in batch mode"""

    batch_keys = ['L', 'nx', 'Tsurf', 'Tbase', 't_total', 'dt', 'vx_init',
                  'init_moho_depth', 'final_moho_depth', 'removal_fraction',
                  'crustal_flux',
                  'erotype', 'erotype_opt1', 'erotype_opt2', 'mantle_adiabat',
                  'rho_crust', 'Cp_crust', 'k_crust', 'H_crust', 'alphav_crust',
                  'rho_mantle', 'Cp_mantle', 'k_mantle', 'H_mantle',
                  'alphav_mantle', 'rho_a', 'k_a', 'ap_rad', 'ap_U', 'ap_Th',
                  'zr_rad', 'zr_U', 'zr_Th']

    # Create empty dictionary for batch model parameters, if any
    batch_params = {}

    # Check that prep_model was called from the command line
    if params['cmd_line_call']:
        # We know all batch model values are lists, check their lengths
        # If all are 1 then run in single-model mode
        params['batch_mode'] = False

        for key in batch_keys:
            if len(params[key]) != 1:
                params['batch_mode'] = True
            batch_params[key] = params[key]

        # Now we see what to do for running the model(s)
        if not params['batch_mode']:
            # Convert list values, run single model
            for key in batch_keys:
                params[key] = params[key][0]
            run_model(params)
        else:
            # Run in batch mode
            batch_run(params, batch_params)

    else:
        # If called as a function, check for lists and their lengths
        params['batch_mode'] = False

        for key in batch_keys:
            if isinstance(params[key], list):
                if len(params[key]) != 1:
                    params['batch_mode'] = True
                batch_params[key] = params[key]

        # Now we see what to do for running the model(s)
        if not params['batch_mode']:
            # Convert list values, run single model
            run_model(params)
        else:
            # Run in batch mode
            batch_run(params, batch_params)

def batch_run(params, batch_params):
    """Runs delam1d in batch mode"""
    param_list = list(ParameterGrid(batch_params))

    print('--- Starting batch processor for {0} models ---\n'.format(len(param_list)))

    # Check number of past models and write header as needed
    # Define output file
    outfile = 'delam1D_batch_log.csv'

    # Open file for writing
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

    success = 0
    failed = 0

    for i in range(len(param_list)):
        model_count += 1
        model_id = 'M{0}'.format(str(model_count).zfill(4))
        model = param_list[i]
        print('Iteration {0}...'.format(i + 1), end='', flush=True)
        # Update model parameters
        for key in batch_params:
            params[key] = model[key]

        # Open file for writing
        with open(outfile, 'a+') as f:
            if write_header == True:
                f.write('Model ID,Simulation time (Myr),Time step (yr),Model thickness (km),Node points,'
                        'Surface temperature (C),Basal temperature (C),Mantle adiabat,'
                        'Crustal density (kg m^-3),Mantle removal fraction,'
                        'Erosion model type,Erosion model option 1,'
                        'Erosion model option 2,Initial Moho depth (km),Initial Moho temperature (C),'
                        'Initial surface heat flow (mW m^-2),Initial surface elevation (km),'
                        'Final Moho depth (km),Final Moho temperature (C),Final surface heat flow (mW m^-2),'
                        'Final surface elevation (km),Apatite grain radius (um),Apatite U concentration (ppm),'
                        'Apatite Th concentration (ppm),Zircon grain radius (um),Zircon U concentration (ppm),'
                        'Zircon Th concentration (ppm),Apatite (U-Th)/He age (Ma),'
                        'Apatite fission-track age (Ma),Zircon (U-Th)/He age (Ma)\n')
                write_header = False
            f.write('{0},'.format(model_id))
        params['model_id'] = model_id

        try:
            run_model(params)
            print('{0}'.format('Complete'))
            success += 1
        except:
            print('{0}'.format('FAILED!'))
            with open(outfile, 'a+') as f:
                f.write('{0:.4f},{1:.4f},{2:.4f},{3},{4:.4f},{5:.4},{6},{7:.4f},'
                        '{8:.4f},{9},{10:.4f},{11:.4f},{12:.4f},,,,{13:.4f},'
                        ',,,{14:.4f},{15:.4f},{16:.4f},{17:.4f},{18:.4f},'
                        '{19:.4f},,,'
                        '\n'.format(params['t_total'], params['dt'], params['L'],
                                    params['nx'], params['Tsurf'], params['Tbase'], \
                                    params['mantle_adiabat'], params['rho_crust'], \
                                    params['removal_fraction'], params['erotype'], \
                                    params['erotype_opt1'], params['erotype_opt2'], \
                                    params['init_moho_depth'], params['final_moho_depth'], \
                                    params['ap_rad'], params['ap_U'], params['ap_Th'], \
                                    params['zr_rad'], params['zr_U'], params['zr_Th']))
                #f.write(',,,,,,,,,,,,,,,,,,,,,,,,,,,,\n')
            failed += 1

    print('\n--- Execution complete ({0} succeeded, {1} failed) ---'.format(success, failed))

def run_model(params):
    # Say hello
    if not params['batch_mode']:
        print('')
        print(30 * '-' + ' Execution started ' + 31 * '-')

    # Set flags if using batch mode
    if params['batch_mode']:
        params['echo_info'] = False
        params['echo_thermal_info'] = False
        params['echo_tc_ages'] = False
        params['plot_results'] = False

    # Conversion factors and unit conversions
    L = kilo2base(params['L'])
    moho_depth_init = kilo2base(params['init_moho_depth'])
    moho_depth = moho_depth_init
    delta_moho = kilo2base(params['init_moho_depth'] - params['final_moho_depth'])

    t_total = myr2sec(params['t_total'])
    dt = yr2sec(params['dt'])

    vx_init = mmyr2ms(params['vx_init'])
    crustal_flux = mmyr2ms(params['crustal_flux'])
    vx = calculate_erosion_rate(t_total, 0.0, delta_moho, params['erotype'], params['erotype_opt1'], params['erotype_opt2'])

    H_crust = micro2base(params['H_crust'])
    H_mantle = micro2base(params['H_mantle'])

    t_plots = myr2sec(np.array(params['t_plots']))
    t_plots.sort()
    if len(t_plots) > 0:
        more_plots = True
    else:
        more_plots = False

    # Determine thickness of mantle to remove
    mantle_lith_thickness = L - moho_depth
    removal_thickness = params['removal_fraction'] * mantle_lith_thickness

    # Calculate node spacing
    dx = L / (params['nx'] - 1)  # m

    # Calculate time step
    nt = int(np.floor(t_total / dt))  # -

    # Echo model info if requested
    if params['echo_info']:
        echo_model_info(dx, nt, dt, t_total, params['implicit'], vx, params['k_crust'],
                        params['rho_crust'], params['Cp_crust'], params['k_mantle'],
                        params['rho_mantle'], params['Cp_mantle'], params['k_a'],
                        params['erotype'], cond_crit=0.5, adv_crit=0.5)

    # Create arrays to hold temperature fields
    Tnew = np.zeros(params['nx'])
    Tprev = np.zeros(params['nx'])

    # Create coordinates of the grid points
    x = np.linspace(0, L, params['nx'])
    xstag = x[:-1] + dx / 2
    vx_hist = np.zeros(nt)
    depth_hist = np.zeros(nt)
    T_hist = np.zeros(nt)
    time_hist = np.zeros(nt)
    if params['mantle_adiabat']:
        adiabat_m = adiabat(alphav=params['alphav_mantle'], T=params['Tbase'] + 273.15, Cp=params['Cp_mantle'])
        Tadiabat = params['Tbase'] + (xstag - L) * adiabat_m
    else:
        adiabat_m = 0.0
        Tadiabat = params['Tbase']

    # Create material property arrays
    rho = np.ones(len(x)) * params['rho_crust']
    rho[x > moho_depth] = params['rho_mantle']
    Cp = np.ones(len(x)) * params['Cp_crust']
    Cp[x > moho_depth] = params['Cp_mantle']
    k = np.ones(len(xstag)) * params['k_crust']
    k[xstag > moho_depth] = params['k_mantle']
    H = np.ones(len(x)) * H_crust
    H[x > moho_depth] = H_mantle
    alphav = np.ones(len(x)) * params['alphav_crust']
    alphav[x > moho_depth] = params['alphav_mantle']

    # Generate initial temperature field
    if not params['batch_mode']:
        print('')
        print('--- Calculating initial thermal model ---')
        print('')
    Tinit = temp_ss_implicit(params['nx'], dx, params['Tsurf'], params['Tbase'], vx_init, rho, Cp, k, H)
    interpTinit = interp1d(x, Tinit)
    init_moho_temp = interpTinit(moho_depth)
    init_heat_flow = kilo2base((k[0] + k[1]) / 2 * (Tinit[1] - Tinit[0]) / dx)
    if params['echo_thermal_info']:
        print('- Initial surface heat flow: {0:.1f} mW/m^2'.format(init_heat_flow))
        print('- Initial Moho temperature: {0:.1f}°C'.format(init_moho_temp))
        print('- Initial Moho depth: {0:.1f} km'.format(params['init_moho_depth']))
        print('- Initial LAB depth: {0:.1f} km'.format((L - removal_thickness) / kilo2base(1)))
        print('- Crustal flux: {0:.1f} mm/yr'.format(crustal_flux / mmyr2ms(1)))

    # Calculate initial densities
    rho_prime = -rho * alphav * Tinit
    rhoT = rho + rho_prime
    isoref = rhoT.sum() * dx
    h_ref = isoref / params['rho_a']
    elev_init = L - h_ref

    elev_list = []
    time_list = []
    elev_list.append(0.0)
    time_list.append(0.0)

    for ix in range(params['nx']):
        if x[ix] > (L - removal_thickness):
            Tprev[ix] = params['Tbase'] + (x[ix] - L) * adiabat_m
        else:
            Tprev[ix] = Tinit[ix]

    if params['plot_results']:
        # Set plot style
        plt.style.use('seaborn-darkgrid')

        # Plot initial temperature field
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        if t_plots.max() < t_total - 1.0:
            # Add an extra color for the final temperature if it is not in the
            # list of times for plotting
            colors = plt.cm.viridis_r(np.linspace(0, 1, len(t_plots) + 1))
        else:
            colors = plt.cm.viridis_r(np.linspace(0, 1, len(t_plots)))
        ax1.plot(Tinit, -x / 1000, 'k:', label='Initial')
        ax1.plot(Tprev, -x / 1000, 'k-', label='0 Myr')
        ax2.plot(rhoT, -x / 1000, 'k-', label='0 Myr')

    # Start the loop over time steps
    curtime = 0.0
    idx = 0
    if not params['batch_mode']:
        print('')
        print('--- Calculating transient thermal model (Pass 1/2) ---')
        print('')
    while curtime < t_total:
        # if (idx+1) % 100 == 0:
        if not params['batch_mode']:
            # print('- Step {0:5d} of {1} ({2:3d}%)\r'.format(idx+1, nt, int(round(100*(idx+1)/nt, 0))), end="")
            print('- Step {0:5d} of {1} (Time: {2:5.1f} Myr, Erosion rate: {3:5.2f} mm/yr)\r'.format(idx + 1, nt,
                                                                                                     curtime / myr2sec(
                                                                                                         1),
                                                                                                     vx / mmyr2ms(1)),
                  end="")
        curtime += dt

        rho, Cp, k, H, lab_depth = update_materials(x, xstag, moho_depth,
                                                    params['rho_crust'], params['rho_mantle'], rho,
                                                    params['Cp_crust'], params['Cp_mantle'], Cp,
                                                    params['k_crust'], params['k_mantle'], k,
                                                    H_crust, H_mantle, H,
                                                    Tadiabat, Tprev, params['k_a'])
        if params['implicit']:
            Tnew[:] = temp_transient_implicit(params['nx'], dx, dt, Tprev, params['Tsurf'], params['Tbase'], vx, rho, Cp, k, H)
        else:
            Tnew[:] = temp_transient_explicit(Tprev, Tnew, params['Tsurf'], params['Tbase'], params['nx'], dx, vx, dt, rho, Cp, k, H)

        Tprev[:] = Tnew[:]

        rho_prime = -rho * alphav * Tnew
        rhoTnew = rho + rho_prime

        # Blend materials when the Moho lies between two nodes
        isonew = 0.0
        for i in range(len(rhoTnew) - 1):
            rho_inc = rhoTnew[i]
            if (moho_depth < x[i + 1]) and (moho_depth >= x[i]):
                crust_frac = (moho_depth - x[i]) / dx
                mantle_frac = 1.0 - crust_frac
                rho_inc = crust_frac * rhoTnew[i] + mantle_frac * rhoTnew[i + 1]
            isonew += rho_inc * dx

        h_asthenosphere = isonew / params['rho_a']
        elev = L - h_asthenosphere
        elev_list.append(elev - elev_init)
        time_list.append(curtime / myr2sec(1.0))

        # Update Moho depth
        moho_depth -= (vx - crustal_flux) * dt

        # Save vx
        vx_hist[idx] = vx
        idx += 1

        # Update erosion rate
        vx = calculate_erosion_rate(t_total, curtime, delta_moho, params['erotype'], params['erotype_opt1'], params['erotype_opt2'])

    if not params['batch_mode']:
        print('')

    depth = (vx_hist * dt).sum()

    moho_depth = moho_depth_init

    for ix in range(params['nx']):
        if x[ix] > (L - removal_thickness):
            Tprev[ix] = params['Tbase'] + (x[ix] - L) * adiabat_m
        else:
            Tprev[ix] = Tinit[ix]

    curtime = 0.0
    # tplot = t_plots[0]
    plotidx = 0
    idx = 0

    # Reset erosion rate
    vx = calculate_erosion_rate(t_total, curtime, delta_moho, params['erotype'], params['erotype_opt1'], params['erotype_opt2'])

    # Calculate initial densities
    rho, Cp, k, H, lab_depth = update_materials(x, xstag, moho_depth, params['rho_crust'],
                                                params['rho_mantle'], rho, params['Cp_crust'],
                                                params['Cp_mantle'], Cp, params['k_crust'], params['k_mantle'],
                                                k, H_crust, H_mantle, H,
                                                Tadiabat, Tprev, params['k_a'])
    rho_prime = -rho * alphav * Tinit
    rhoT = rho + rho_prime
    isoref = rhoT.sum() * dx
    h_ref = isoref / params['rho_a']
    # elev_init = L - h_ref

    if not params['batch_mode']:
        print('')
        print('--- Calculating transient thermal model (Pass 2/2) ---')
        print('')
    while curtime < t_total:
        # if (idx+1) % 100 == 0:
        if not params['batch_mode']:
            # print('- Step {0:5d} of {1} ({2:3d}%)\r'.format(idx+1, nt, int(round(100*(idx+1)/nt, 0))), end="")
            print('- Step {0:5d} of {1} (Time: {2:5.1f} Myr, Erosion rate: {3:5.2f} mm/yr)\r'.format(idx + 1, nt,
                                                                                                     curtime / myr2sec(
                                                                                                         1),
                                                                                                     vx / mmyr2ms(1)),
                  end="")
        curtime += dt

        rho, Cp, k, H, lab_depth = update_materials(x, xstag, moho_depth,
                                                    params['rho_crust'], params['rho_mantle'], rho,
                                                    params['Cp_crust'], params['Cp_mantle'], Cp,
                                                    params['k_crust'], params['k_mantle'], k, H_crust,
                                                    H_mantle, H, Tadiabat, Tprev,
                                                    params['k_a'])
        if params['implicit']:
            Tnew[:] = temp_transient_implicit(params['nx'], dx, dt, Tprev, params['Tsurf'], params['Tbase'], vx, rho, Cp, k, H)
        else:
            Tnew[:] = temp_transient_explicit(Tprev, Tnew, params['Tsurf'], params['Tbase'], params['nx'], dx, vx, dt, rho, Cp, k, H)
        Tprev[:] = Tnew[:]

        rho_prime = -rho * alphav * Tnew
        rhoTnew = rho + rho_prime

        # Blend materials when the Moho lies between two nodes
        isonew = 0.0
        for i in range(len(rhoTnew) - 1):
            rho_inc = rhoTnew[i]
            if (moho_depth < x[i + 1]) and (moho_depth >= x[i]):
                crust_frac = (moho_depth - x[i]) / dx
                mantle_frac = 1.0 - crust_frac
                rho_inc = crust_frac * rhoTnew[i] + mantle_frac * rhoTnew[i + 1]
            isonew += rho_inc * dx

        h_asthenosphere = isonew / params['rho_a']
        elev = L - h_asthenosphere

        # Update Moho depth
        moho_depth -= (vx - crustal_flux) * dt

        # Store temperature, time, depth
        interpTnew = interp1d(x, Tnew)
        depth -= vx * dt
        depth_hist[idx] = depth
        time_hist[idx] = curtime
        if abs(depth) <= 1e-6:
            T_hist[idx] = 0.0
        else:
            T_hist[idx] = interpTnew(depth)
        idx += 1

        # Update erosion rate
        vx = calculate_erosion_rate(t_total, curtime, delta_moho, params['erotype'], params['erotype_opt1'], params['erotype_opt2'])

        if params['plot_results'] and more_plots:
            if curtime > t_plots[plotidx]:
                ax1.plot(Tnew, -x / 1000, '-', label='{0:.1f} Myr'.format(t_plots[plotidx] / myr2sec(1)),
                         color=colors[plotidx])
                ax2.plot(rhoTnew, -x / 1000, label='{0:.1f} Myr'.format(t_plots[plotidx] / myr2sec(1)),
                         color=colors[plotidx])
                if plotidx == len(t_plots) - 1:
                    more_plots = False
                plotidx += 1
                # tplot = t_plots[plotidx]

    rho_prime = -rho * alphav * Tnew
    rhoTnew = rho + rho_prime
    isonew = rhoTnew.sum() * dx

    interpTnew = interp1d(x, Tnew)
    final_moho_temp = interpTnew(moho_depth)
    final_heat_flow = kilo2base((k[0] + k[1]) / 2 * (Tnew[1] - Tnew[0]) / dx)

    if not params['batch_mode']:
        print('')

    if params['echo_thermal_info']:
        print('')
        print('--- Final thermal model values ---')
        print('')
        print('- Final surface heat flow: {0:.1f} mW/m^2'.format(final_heat_flow))
        print('- Final Moho temperature: {0:.1f}°C'.format(final_moho_temp))
        print('- Final Moho depth: {0:.1f} km ({1:+.1f} km from crustal flux)'.format(moho_depth / kilo2base(1), (
                crustal_flux / mmyr2ms(1)) * t_total / myr2sec(1)))
        print('- Final LAB depth: {0:.1f} km'.format(lab_depth / kilo2base(1)))

    if params['calc_tc_ages']:
        # INPUT
        # time_i:the time values (in Myr) in descending order at which the thermal history 
        # is given (ex: 100,50,20,10,0); the last value should always be 0; the first value
        # should be smaller than 1000.
        # temp_i: the thermal history in degree Celsius
        # n: the number of time-temperature pairs used  to describe the temperature history
        # out_flag:  =0 only calculate fission track age
        #            =1 also calculate track length distribution and statistics
        # param_flag : =1 uses Laslett et al, 1987 parameters
        #              =2 uses Crowley et al., 1991 Durango parameters
        #              =3 uses Crowley et al., 1991 F-apatite parameters
        time_ma = t_total - time_hist
        time_ma = time_ma / myr2sec(1)
        if params['madtrax']:
            age, _, _, _ = Mad_Trax(time_ma, T_hist, len(time_ma), 1, 2)

        # Write time-temperature history to file for (U-Th)/He age prediction
        with open('tT_hist.csv', 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', lineterminator="\n")
            # Write time-temperature history in reverse order!
            for i in range(-1, -(len(time_ma) + 1), -100):
                writer.writerow([time_ma[i], T_hist[i]])

        ahe_age, corr_ahe_age, zhe_age, corr_zhe_age = He_ages(file='tT_hist.csv', ap_rad=params['ap_rad'], ap_U=params['ap_U'],
                                                               ap_Th=params['ap_Th'], zr_rad=params['zr_rad'], zr_U=params['zr_U'], zr_Th=params['zr_Th'])
        if params['ketch_aft']:
            aft_age = FT_ages('tT_hist.csv')

        if params['batch_mode']:
            tt_filename = params['model_id']+'-tT_hist.csv'
            os.rename('tT_hist.csv', tt_filename)

        if params['echo_tc_ages']:
            print('')
            print('--- Predicted thermochronometer ages ---')
            print('')
            print('- AHe age: {0:.2f} Ma (uncorrected age: {1:.2f} Ma)'.format(float(corr_ahe_age), float(ahe_age)))
            if params['madtrax']:
                print('- AFT age: {0:.2f} Ma (MadTrax)'.format(age / 1e6))
            if params['ketch_aft']:
                print('- AFT age: {0:.2f} Ma (Ketcham)'.format(float(aft_age)))
            print('- ZHe age: {0:.2f} Ma (uncorrected age: {1:.2f} Ma)'.format(float(corr_zhe_age), float(zhe_age)))

    if (params['plot_results'] and params['save_plots']) or params['write_temps'] or params['read_temps']:
        fp = '/Users/whipp/Work/Documents/projects/Kellett-Coutand-Canadian-Cordillera/delamination-1D/'

    if params['plot_results']:
        # Plot the final temperature field
        xmin = 0.0
        xmax = params['Tbase'] + 100
        ax1.plot(Tnew, -x / 1000, '-', label='{0:.1f} Myr'.format(curtime / myr2sec(1)), color=colors[-1])
        ax1.plot([xmin, xmax], [-moho_depth / kilo2base(1), -moho_depth / kilo2base(1)], linestyle='--', color='black',
                 lw=0.5)
        ax1.plot([xmin, xmax], [-params['init_moho_depth'], -params['init_moho_depth']], linestyle='--', color='gray', lw=0.5)

        plot_crust_solidus = True
        if plot_crust_solidus:
            crust_depth, crust_wet, crust_dry = crust_solidus()
            crust_slice = crust_depth <= params['init_moho_depth']
            # crust_depth_interp = np.linspace(crust_depth.min(), crust_depth.max(), nx)
            # spl = make_interp_spline(crust_depth, crust_wet, k=3)  # type: BSpline
            # power_smooth = spl(crust_depth_interp)
            # ax1.plot(power_smooth, -crust_depth_interp, label='Granite, wet')
            # ax1.plot(crust_wet[crust_slice], -crust_depth[crust_slice], color='gray', label='Granite, wet')
            # ax1.plot(crust_dry[crust_slice], -crust_depth[crust_slice], color='gray', linestyle='--', label='Granite, dry')
            ax1.plot(crust_wet, -crust_depth, color='gray', label='Granite, wet')
            ax1.plot(crust_dry, -crust_depth, color='gray', linestyle='--', label='Granite, dry')

        plot_mantle_solidus = False
        if plot_mantle_solidus:
            mantle_depth, mantle_wet, mantle_dry = mantle_solidus()
            mantle_slice = mantle_depth >= params['init_moho_depth']
            # crust_depth_interp = np.linspace(crust_depth.min(), crust_depth.max(), nx)
            # spl = make_interp_spline(crust_depth, crust_wet, k=3)  # type: BSpline
            # power_smooth = spl(crust_depth_interp)
            # ax1.plot(power_smooth, -crust_depth_interp, label='Granite, wet')
            # ax1.plot(mantle_wet[mantle_slice], -mantle_depth[mantle_slice], color='gray', label='Olivine, wet')
            # ax1.plot(mantle_dry[mantle_slice], -mantle_depth[mantle_slice], color='gray', linestyle='--', label='Olivine, dry')
            ax1.plot(mantle_wet, -mantle_depth, color='gray', label='Olivine, wet')
            ax1.plot(mantle_dry, -mantle_depth, color='gray', linestyle='--', label='Olivine, dry')

        ax1.text(20.0, -moho_depth / kilo2base(1) + 1.0, 'Final Moho')
        ax1.text(20.0, -params['init_moho_depth'] - 3.0, 'Initial Moho', color='gray')
        ax1.legend()
        ax1.axis([xmin, xmax, -L / 1000, 0])
        ax1.set_xlabel('Temperature (°C)')
        ax1.set_ylabel('Depth (km)')
        # ax1.grid()

        xmin = 2700
        xmax = 3300
        ax2.plot(rhoTnew, -x / 1000, label='{0:.1f} Myr'.format(t_total / myr2sec(1)), color=colors[-1])
        ax2.plot([xmin, xmax], [-moho_depth / kilo2base(1), -moho_depth / kilo2base(1)], linestyle='--', color='black',
                 lw=0.5)
        ax2.plot([xmin, xmax], [-params['init_moho_depth'], -params['init_moho_depth']], linestyle='--', color='gray', lw=0.5)
        ax2.axis([xmin, xmax, -L / 1000, 0])
        ax2.set_xlabel('Density (kg m$^{-3}$)')
        ax2.set_ylabel('Depth (km)')
        ax2.legend()
        # ax2.grid()

        plt.tight_layout()
        if params['save_plots']:
            plt.savefig(fp + 'png/T_rho_hist.png', dpi=300)
        plt.show()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        # ax1.plot(time_list, elev_list, 'k-')
        ax1.plot(time_list, elev_list)
        ax1.set_xlabel('Time (Myr)')
        ax1.set_ylabel('Elevation (m)')
        ax1.set_xlim(0.0, t_total / myr2sec(1))
        ax1.set_title('Elevation history')
        # plt.axis([0.0, t_total/myr2sec(1), 0, 750])
        # ax1.grid()

        ax2.plot(time_hist / myr2sec(1), vx_hist / mmyr2ms(1))
        ax2.fill_between(time_hist / myr2sec(1), vx_hist / mmyr2ms(1), 0.0, alpha=0.33, color='tab:blue',
                         label='Erosion magnitude: {0:.1f} km'.format(params['init_moho_depth'] - params['final_moho_depth']))
        ax2.set_xlabel('Time (Myr)')
        ax2.set_ylabel('Erosion rate (mm/yr)')
        ax2.set_xlim(0.0, t_total / myr2sec(1))
        ax2.set_ylim(ymin=0.0)
        # plt.axis([0.0, t_total/myr2sec(1), 0, 750])
        # ax2.grid()
        ax2.legend()

        plt.tight_layout()
        if params['save_plots']:
            plt.savefig(fp + 'png/elev_hist.png', dpi=300)
        plt.show()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        # ax1.plot(time_ma, T_hist, 'r-', lw=2)

        # Find effective closure temperatures
        ahe_temp = np.interp(float(corr_ahe_age), np.flip(time_ma), np.flip(T_hist))
        aft_temp = np.interp(float(aft_age), np.flip(time_ma), np.flip(T_hist))
        zhe_temp = np.interp(float(corr_zhe_age), np.flip(time_ma), np.flip(T_hist))

        # Calculate synthetic uncertainties
        ahe_uncert = 0.1
        aft_uncert = 0.2
        zhe_uncert = 0.1
        ahe_min, ahe_max = (1.0 - ahe_uncert) * float(corr_ahe_age), (1.0 + ahe_uncert) * float(corr_ahe_age)
        aft_min, aft_max = (1.0 - aft_uncert) * float(aft_age), (1.0 + aft_uncert) * float(aft_age)
        zhe_min, zhe_max = (1.0 - zhe_uncert) * float(corr_zhe_age), (1.0 + zhe_uncert) * float(corr_zhe_age)
        ax1.plot(time_ma, T_hist)
        ax1.axvspan(ahe_min, ahe_max, alpha=0.33, color='tab:blue',
                    label='AHe age ({0:.2f} Ma ± {1:.0f}% uncertainty; T$_c$ = {2:.1f}°C)'.format(float(corr_ahe_age),
                                                                                                  ahe_uncert * 100.0,
                                                                                                  ahe_temp))
        ax1.axvspan(aft_min, aft_max, alpha=0.33, color='tab:orange',
                    label='AFT age ({0:.2f} Ma ± {1:.0f}% uncertainty; T$_c$ = {2:.1f}°C)'.format(float(aft_age),
                                                                                                  aft_uncert * 100.0,
                                                                                                  aft_temp))
        ax1.axvspan(zhe_min, zhe_max, alpha=0.33, color='tab:green',
                    label='ZHe age ({0:.2f} Ma ± {1:.0f}% uncertainty; T$_c$ = {2:.1f}°C)'.format(float(corr_zhe_age),
                                                                                                  zhe_uncert * 100.0,
                                                                                                  zhe_temp))
        ax1.plot(float(corr_ahe_age), ahe_temp, marker='o', color='tab:blue')
        ax1.plot(float(aft_age), aft_temp, marker='o', color='tab:orange')
        ax1.plot(float(corr_zhe_age), zhe_temp, marker='o', color='tab:green')
        ax1.set_xlim(t_total / myr2sec(1), 0.0)
        ax1.set_ylim(ymin=params['Tsurf'])
        ax1.set_xlabel('Time (Ma)')
        ax1.set_ylabel('Temperature (°C)')
        ax1.set_title('Thermal history for surface sample')
        """
        if echo_tc_ages == True:
            bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5)
            t = ax1.text(0.45*time_ma.max(), 0.12*T_hist.max(),
            'Apatite (U-Th)/He age: {0:6.2f} Ma\nApatite fission-track age: {1:6.2f} Ma\nZircon (U-Th)/He age: {2:6.2f} Ma'.format(float(corr_ahe_age), float(aft_age), float(corr_zhe_age)),
            ha="right", va="center", size=10, fontname='Menlo', bbox=bbox_props)
        """
        # ax1.grid()
        ax1.legend()

        ax2.plot(time_ma, vx_hist / mmyr2ms(1))
        ax2.fill_between(time_ma, vx_hist / mmyr2ms(1), 0.0, alpha=0.33, color='tab:blue',
                         label='Erosion magnitude: {0:.1f} km'.format(params['init_moho_depth'] - params['final_moho_depth']))
        ax2.set_xlabel('Time (Ma)')
        ax2.set_ylabel('Erosion rate (mm/yr)')
        ax2.set_xlim(t_total / myr2sec(1), 0.0)
        ax2.set_ylim(ymin=0.0)
        # plt.axis([0.0, t_total/myr2sec(1), 0, 750])
        # ax2.grid()
        ax2.legend()

        plt.tight_layout()
        if params['save_plots']:
            plt.savefig(fp + 'png/cooling_hist.png', dpi=300)
        plt.show()

    if params['read_temps']:
        load_file = 'py/output_temps.csv'
        data = np.genfromtxt(fp + load_file, delimiter=',', skip_header=1)
        temps = data[:, 1]
        temp_diff = temps[1:] - Tnew[1:]
        pct_diff = temp_diff / temps[1:] * 100.0
        plt.figure(figsize=(12, 6))
        plt.plot(pct_diff, -x[1:] / 1000, 'k-')
        plt.xlabel('Percent temperature difference')
        plt.ylabel('Depth (km)')
        plt.grid()
        plt.title('Percent difference from explicit FD solution')
        plt.show()

    if params['write_temps']:
        print('')
        print('--- Writing temperature output to file ---')
        print('')
        Txout = np.zeros([len(x), 2])
        Txout[:, 0] = x
        Txout[:, 1] = Tnew
        savefile = 'py/output_temps.csv'
        np.savetxt(fp + savefile, Txout, delimiter=',', header="Depth (m),Temperature(deg. C)")
        print('- Temperature output saved to file\n  ' + fp + savefile)

    if params['batch_mode']:
        # Write output to a file
        outfile = 'delam1D_batch_log.csv'

        # Check if file already exists
        try:
            with open(outfile) as f:
                write_header = False
                infile = f.readlines()
                if len(infile) < 1:
                    write_header = True
        except IOError:
            write_header = True

        # Open file for writing
        with open(outfile, 'a+') as f:
            f.write('{0:.4f},{1:.4f},{2:.4f},{3},{4:.4f},{5:.4},{6},{7:.4f},{8:.4f},'
                    '{9},{10:.4f},{11:.4f},{12:.4f},{13:.4f},{14:.4f},{15:.4f},{16:.4f},'
                    '{17:.4f},{18:.4f},{19:.4f},{20:.4f},{21:.4f},{22:.4f},{23:.4f},{24:.4f},'
                    '{25:.4f},{26:.4f},{27:.4f},{28:.4f}'
                    '\n'.format(t_total / myr2sec(1), dt / yr2sec(1), L / kilo2base(1), params['nx'], params['Tsurf'], \
                                params['Tbase'], params['mantle_adiabat'], params['rho_crust'], params['removal_fraction'], \
                                params['erotype'], params['erotype_opt1'], params['erotype_opt2'], params['init_moho_depth'], \
                                init_moho_temp, init_heat_flow, elev_list[1] / kilo2base(1), params['final_moho_depth'], \
                                final_moho_temp, final_heat_flow, elev_list[-1] / kilo2base(1), \
                                params['ap_rad'], params['ap_U'], params['ap_Th'], params['zr_rad'], params['zr_U'], params['zr_Th'], \
                                float(corr_ahe_age), float(aft_age), float(corr_zhe_age)))

    if not params['batch_mode']:
        print('')
        print(30 * '-' + ' Execution complete ' + 30 * '-')


def main():
    parser = argparse.ArgumentParser(description='Calculates transient 1D temperatures and thermochronometer ages',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--echo_inputs', help='Print input values to the screen', default=False, type=bool)
    parser.add_argument('--echo_info', help='Print basic model info to the screen', default=True, type=bool)
    parser.add_argument('--echo_thermal_info', help='Print thermal model info to the screen', default=True, type=bool)
    parser.add_argument('--calc_tc_ages', help='Enable calculation of thermochronometer ages', default=True, type=bool)
    parser.add_argument('--echo_tc_ages', help='Print calculated thermochronometer age(s) to the screen', default=True,
                        type=bool)
    parser.add_argument('--plot_results', help='Plot calculated temperatures and densities', default=True, type=bool)
    parser.add_argument('--save_plots', help='Save plots to a file', default=False, type=bool)
    parser.add_argument('--batch_mode', help='Enable batch mode (no screen output, outputs writen to file)',
                        default=False, type=bool)
    parser.add_argument('--mantle_adiabat', help='Use adiabat for asthenosphere temperature', nargs='+', default=[True],
                        type=bool)
    parser.add_argument('--implicit', help='Use implicit finite-difference calculation', default=True,
                        action='store_true')
    parser.add_argument('--explicit', help='Use explicit finite-difference calculation', dest='implicit',
                        action='store_false')
    parser.add_argument('--read_temps', help='Read temperatures from a file', default=False, type=bool)
    parser.add_argument('--compare_temps', help='Compare model temperatures to those from a file', default=False,
                        type=bool)
    parser.add_argument('--write_temps', help='Save model temperatures to a file', default=False, type=bool)
    parser.add_argument('--madtrax', help='Use MadTrax algorithm for predicting FT ages', default=False, type=bool)
    parser.add_argument('--ketch_aft', help='Use the Ketcham et al. (2007) for predicting FT ages', default=True,
                        type=bool)
    parser.add_argument('--t_plots', help='Output times for temperature plotting (Myrs)', nargs='+',
                        default=[0.1, 1, 5, 10, 20, 30, 50], type=float)
    parser.add_argument('--length', help='Model depth extent (km)', nargs='+', default=[125.0], type=float)
    parser.add_argument('--nx', help='Number of grid points for temperature calculation', nargs='+', default=[251],
                        type=int)
    parser.add_argument('--init_moho_depth', help='Initial depth of Moho (km)', nargs='+', default=[50.0], type=float)
    parser.add_argument('--final_moho_depth', help='Final depth of Moho (km)', nargs='+', default=[35.0], type=float)
    parser.add_argument('--removal_fraction', help='Fraction of lithospheric mantle to remove', nargs='+',
                        default=[1.0], type=float)
    parser.add_argument('--crustal_flux', help='Rate of change of crustal thickness', nargs='+', default=[0.0],
                        type=float)
    parser.add_argument('--erotype', help='Type of erosion model (1, 2, 3 - see GitHub docs)', nargs='+', default=[1],
                        type=int)
    parser.add_argument('--erotype_opt1', help='Erosion model option 1 (see GitHub docs)', nargs='+', default=[0.0],
                        type=float)
    parser.add_argument('--erotype_opt2', help='Erosion model option 2 (see GitHub docs)', nargs='+', default=[0.0],
                        type=float)
    parser.add_argument('--Tsurf', help='Surface boundary condition temperature (C)', nargs='+', default=[0.0],
                        type=float)
    parser.add_argument('--Tbase', help='Basal boundary condition temperature (C)', nargs='+', default=[1300.0],
                        type=float)
    parser.add_argument('--time', help='Total simulation time (Myr)', nargs='+', default=[50.0], type=float)
    parser.add_argument('--dt', help='Time step (years)', nargs='+', default=[5000.0], type=float)
    parser.add_argument('--vx_init', help='Initial steady-state advection velocity (mm/yr)', nargs='+', default=[0.0],
                        type=float)
    parser.add_argument('--rho_crust', help='Crustal density (kg/m^3)', nargs='+', default=[2850.0], type=float)
    parser.add_argument('--Cp_crust', help='Crustal heat capacity (J/kg/K)', nargs='+', default=[800.0], type=float)
    parser.add_argument('--k_crust', help='Crustal thermal conductivity (W/m/K)', nargs='+', default=[2.75], type=float)
    parser.add_argument('--H_crust', help='Crustal heat production (uW/m^3)', nargs='+', default=[0.5], type=float)
    parser.add_argument('--alphav_crust', help='Crustal coefficient of thermal expansion (km)', nargs='+',
                        default=[3.0e-5], type=float)
    parser.add_argument('--rho_mantle', help='Mantle lithosphere density (kg/m^3)', nargs='+', default=[3250.0],
                        type=float)
    parser.add_argument('--Cp_mantle', help='Mantle lithosphere heat capacity (J/kg/K)', nargs='+', default=[1000.0],
                        type=float)
    parser.add_argument('--k_mantle', help='Mantle lithosphere thermal conductivity (W/m/K)', nargs='+', default=[2.5],
                        type=float)
    parser.add_argument('--H_mantle', help='Mantle lithosphere heat production (uW/m^3)', nargs='+', default=[0.0],
                        type=float)
    parser.add_argument('--alphav_mantle', help='Mantle lithosphere coefficient of thermal expansion (km)', nargs='+',
                        default=[3.0e-5], type=float)
    parser.add_argument('--rho_a', help='Mantle asthenosphere density (kg/m^3)', nargs='+', default=[3250.0],
                        type=float)
    parser.add_argument('--k_a', help='Mantle asthenosphere thermal conductivity (W/m/K)', nargs='+', default=[20.0],
                        type=float)
    parser.add_argument('--ap_rad', help='Apatite grain radius (um)', nargs='+', default=[45.0], type=float)
    parser.add_argument('--ap_U', help='Apatite U concentration (ppm)', nargs='+', default=[10.0], type=float)
    parser.add_argument('--ap_Th', help='Apatite Th concentration radius (ppm)', nargs='+', default=[40.0], type=float)
    parser.add_argument('--zr_rad', help='Zircon grain radius (um)', nargs='+', default=[60.0], type=float)
    parser.add_argument('--zr_U', help='Zircon U concentration (ppm)', nargs='+', default=[100.0], type=float)
    parser.add_argument('--zr_Th', help='Zircon Th concentration radius (ppm)', nargs='+', default=[40.0], type=float)

    args = parser.parse_args()

    params = {'cmd_line_call': True, 'echo_inputs': args.echo_inputs, 'echo_info': args.echo_info,
              'echo_thermal_info': args.echo_thermal_info, 'calc_tc_ages': args.calc_tc_ages,
              'echo_tc_ages': args.echo_tc_ages, 'plot_results': args.plot_results,
              'save_plots': args.save_plots, 'batch_mode': args.batch_mode,
              'mantle_adiabat': args.mantle_adiabat, 'implicit': args.implicit,
              'read_temps': args.read_temps, 'compare_temps': args.compare_temps,
              'write_temps': args.write_temps, 'madtrax': args.madtrax,
              'ketch_aft': args.ketch_aft, 't_plots': args.t_plots,
              'L': args.length, 'nx': args.nx, 'init_moho_depth': args.init_moho_depth,
              'final_moho_depth': args.final_moho_depth, 'removal_fraction': args.removal_fraction,
              'crustal_flux': args.crustal_flux, 'erotype': args.erotype,
              'erotype_opt1': args.erotype_opt1, 'erotype_opt2': args.erotype_opt2,
              'Tsurf': args.Tsurf, 'Tbase': args.Tbase, 't_total': args.time,
              'dt': args.dt, 'vx_init': args.vx_init, 'rho_crust': args.rho_crust,
              'Cp_crust': args.Cp_crust, 'k_crust': args.k_crust, 'H_crust': args.H_crust,
              'alphav_crust': args.alphav_crust, 'rho_mantle': args.rho_mantle,
              'Cp_mantle': args.Cp_mantle, 'k_mantle': args.k_mantle, 'H_mantle': args.H_mantle,
              'alphav_mantle': args.alphav_mantle, 'rho_a': args.rho_a, 'k_a': args.k_a,
              'ap_rad': args.ap_rad, 'ap_U': args.ap_U, 'ap_Th': args.ap_Th,
              'zr_rad': args.zr_rad, 'zr_U': args.zr_U, 'zr_Th': args.zr_Th}

    prep_model(params)

if __name__ == "__main__":
    # execute only if run as a script
    main()