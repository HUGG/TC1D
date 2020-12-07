#!/usr/bin/env python3

# Import libaries we need
import numpy as np
import delam1D

# Run a model
"""
run_model(echo_inputs=False, echo_info=True, echo_thermal_info=True,
              calc_tc_ages=True, echo_tc_ages=True, plot_results=True,
              save_plots=False, batch_mode=False, mantle_adiabat=True,
              implicit=True, read_temps=False, compare_temps=False,
              write_temps=False, madtrax=False, ketch_aft=True,
              t_plots=[0.1, 1, 5, 10, 20, 30, 50], L=125.0, nx=251,
              init_moho_depth=50.0, final_moho_depth=35.0, removal_fraction=1.0,
              erotype=1, erotype_opt1=0.0, erotype_opt2=0.0, Tsurf=0.0,
              Tbase=1300.0, t_total=50.0, dt=5000.0, vx_init=0.0, rho_crust=2850,
              Cp_crust=800, k_crust=2.75, H_crust=0.5, alphav_crust=3.0e-5,
              rho_mantle=3250, Cp_mantle=1000, k_mantle=2.5, H_mantle=0.0,
              alphav_mantle=3.0e-5, rho_a=3250.0, k_a=20.0, ap_rad=60.0,
              ap_U=10.0, ap_Th=40.0, zr_rad=60.0, zr_U=100.0, zr_Th=40.0)
"""

# Below we could include the logic to define variable ranges, their values, and
# a list/array to batch process jobs using delam1D
#
# For now, just a simple example using the default parameters
delam1D.run_model(batch_mode=True)