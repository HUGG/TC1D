#!/usr/bin/env python3

# Import libraries we need
import argparse
import TC1D as tc1d


def main():
    parser = argparse.ArgumentParser(
        description="Calculates transient 1D temperatures and thermochronometer ages",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--echo-inputs",
        dest="echo_inputs",
        help="Print input values to the screen",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no-echo-info",
        dest="no_echo_info",
        help="Do not print basic model info to the screen",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no-echo-thermal-info",
        dest="no_echo_thermal_info",
        help="Do not print thermal model info to the screen",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no-calc-ages",
        dest="no_calc_ages",
        help="Disable calculation of thermochronometer ages",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no-echo-ages",
        dest="no_echo_ages",
        help="Do not print calculated thermochronometer age(s) to the screen",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no-plot-results",
        dest="no_plot_results",
        help="Do not plot calculated temperatures and densities",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--save-plots",
        dest="save_plots",
        help="Save plots to a file",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--batch-mode",
        dest="batch_mode",
        help="Enable batch mode (no screen output, outputs writen to file)",
        action="store_true",
        default=False,
    )
    # Does the following option work?
    parser.add_argument(
        "--mantle_adiabat",
        help="Use adiabat for asthenosphere temperature",
        nargs="+",
        default=[True],
        type=bool,
    )
    # Following two options are OK?
    parser.add_argument(
        "--implicit",
        help="Use implicit finite-difference calculation",
        default=True,
        action="store_true",
    )
    parser.add_argument(
        "--explicit",
        help="Use explicit finite-difference calculation",
        dest="implicit",
        action="store_false",
    )
    parser.add_argument(
        "--read-temps",
        dest="read_temps",
        help="Read temperatures from a file",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--compare-temps",
        dest="compare_temps",
        help="Compare model temperatures to those from a file",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--write-temps",
        dest="write_temps",
        help="Save model temperatures to a file",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--ketch-aft",
        dest="ketch_aft",
        help="Use the Ketcham et al. (2007) for predicting FT ages",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--madtrax",
        help="Use MadTrax algorithm for predicting FT ages",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--t-plots",
        dest="t_plots",
        help="Output times for temperature plotting (Myrs)",
        nargs="+",
        default=[0.1, 1, 5, 10, 20, 30, 50],
        type=float,
    )
    parser.add_argument(
        "--length",
        help="Model depth extent (km)",
        nargs="+",
        default=[125.0],
        type=float,
    )
    parser.add_argument(
        "--nx",
        help="Number of grid points for temperature calculation",
        nargs="+",
        default=[251],
        type=int,
    )
    parser.add_argument(
        "--init-moho-depth",
        dest="init_moho_depth",
        help="Initial depth of Moho (km)",
        nargs="+",
        default=[50.0],
        type=float,
    )
    parser.add_argument(
        "--final-moho-depth",
        dest="final_moho_depth",
        help="Final depth of Moho (km)",
        nargs="+",
        default=[35.0],
        type=float,
    )
    parser.add_argument(
        "--removal-fraction",
        dest="removal_fraction",
        help="Fraction of lithospheric mantle to remove",
        nargs="+",
        default=[0.0],
        type=float,
    )
    parser.add_argument(
        "--crustal-flux",
        dest="crustal_flux",
        help="Rate of change of crustal thickness",
        nargs="+",
        default=[0.0],
        type=float,
    )
    parser.add_argument(
        "--erotype",
        help="Type of erosion model (1, 2, 3 - see GitHub docs)",
        nargs="+",
        default=[1],
        type=int,
    )
    parser.add_argument(
        "--erotype-opt1",
        dest="erotype_opt1",
        help="Erosion model option 1 (see GitHub docs)",
        nargs="+",
        default=[0.0],
        type=float,
    )
    parser.add_argument(
        "--erotype-opt2",
        dest="erotype_opt2",
        help="Erosion model option 2 (see GitHub docs)",
        nargs="+",
        default=[0.0],
        type=float,
    )
    parser.add_argument(
        "--temp-surf",
        dest="temp_surf",
        help="Surface boundary condition temperature (C)",
        nargs="+",
        default=[0.0],
        type=float,
    )
    parser.add_argument(
        "--temp_base",
        dest="temp_base",
        help="Basal boundary condition temperature (C)",
        nargs="+",
        default=[1300.0],
        type=float,
    )
    parser.add_argument(
        "--time",
        help="Total simulation time (Myr)",
        nargs="+",
        default=[50.0],
        type=float,
    )
    parser.add_argument(
        "--dt", help="Time step (years)", nargs="+", default=[5000.0], type=float
    )
    parser.add_argument(
        "--vx-init",
        dest="vx_init",
        help="Initial steady-state advection velocity (mm/yr)",
        nargs="+",
        default=[0.0],
        type=float,
    )
    parser.add_argument(
        "--rho-crust",
        dest="rho_crust",
        help="Crustal density (kg/m^3)",
        nargs="+",
        default=[2850.0],
        type=float,
    )
    parser.add_argument(
        "--cp-crust",
        dest="cp_crust",
        help="Crustal heat capacity (J/kg/K)",
        nargs="+",
        default=[800.0],
        type=float,
    )
    parser.add_argument(
        "--k-crust",
        dest="k_crust",
        help="Crustal thermal conductivity (W/m/K)",
        nargs="+",
        default=[2.75],
        type=float,
    )
    parser.add_argument(
        "--heat-prod-crust",
        dest="heat_prod_crust",
        help="Crustal heat production (uW/m^3)",
        nargs="+",
        default=[0.5],
        type=float,
    )
    parser.add_argument(
        "--alphav-crust",
        dest="alphav_crust",
        help="Crustal coefficient of thermal expansion (km)",
        nargs="+",
        default=[3.0e-5],
        type=float,
    )
    parser.add_argument(
        "--rho-mantle",
        dest="rho_mantle",
        help="Mantle lithosphere density (kg/m^3)",
        nargs="+",
        default=[3250.0],
        type=float,
    )
    parser.add_argument(
        "--cp-mantle",
        dest="cp_mantle",
        help="Mantle lithosphere heat capacity (J/kg/K)",
        nargs="+",
        default=[1000.0],
        type=float,
    )
    parser.add_argument(
        "--k-mantle",
        dest="k_mantle",
        help="Mantle lithosphere thermal conductivity (W/m/K)",
        nargs="+",
        default=[2.5],
        type=float,
    )
    parser.add_argument(
        "--heat-prod-mantle",
        dest="heat_prod_mantle",
        help="Mantle lithosphere heat production (uW/m^3)",
        nargs="+",
        default=[0.0],
        type=float,
    )
    parser.add_argument(
        "--alphav-mantle",
        dest="alphav_mantle",
        help="Mantle lithosphere coefficient of thermal expansion (km)",
        nargs="+",
        default=[3.0e-5],
        type=float,
    )
    parser.add_argument(
        "--rho-a",
        dest="rho_a",
        help="Mantle asthenosphere density (kg/m^3)",
        nargs="+",
        default=[3250.0],
        type=float,
    )
    parser.add_argument(
        "--k-a",
        dest="k_a",
        help="Mantle asthenosphere thermal conductivity (W/m/K)",
        nargs="+",
        default=[20.0],
        type=float,
    )
    parser.add_argument(
        "--ap-rad",
        dest="ap_rad",
        help="Apatite grain radius (um)",
        nargs="+",
        default=[45.0],
        type=float,
    )
    parser.add_argument(
        "--ap-uranium",
        dest="ap_uranium",
        help="Apatite U concentration (ppm)",
        nargs="+",
        default=[10.0],
        type=float,
    )
    parser.add_argument(
        "--ap-thorium",
        dest="ap_thorium",
        help="Apatite Th concentration radius (ppm)",
        nargs="+",
        default=[40.0],
        type=float,
    )
    parser.add_argument(
        "--zr-rad",
        dest="zr_rad",
        help="Zircon grain radius (um)",
        nargs="+",
        default=[60.0],
        type=float,
    )
    parser.add_argument(
        "--zr-uranium",
        dest="zr_uranium",
        help="Zircon U concentration (ppm)",
        nargs="+",
        default=[100.0],
        type=float,
    )
    parser.add_argument(
        "--zr-thorium",
        dest="zr_thorium",
        help="Zircon Th concentration radius (ppm)",
        nargs="+",
        default=[40.0],
        type=float,
    )
    # Option below should be fixed.
    parser.add_argument(
        "--pad-thist",
        dest="pad_thist",
        help="Add time at starting temperature in t-T history",
        nargs="+",
        default=[False],
        type=bool,
    )
    parser.add_argument(
        "--pad-time",
        dest="pad_time",
        help="Additional time at starting temperature in t-T history (Myr)",
        nargs="+",
        default=[0.0],
        type=float,
    )
    parser.add_argument(
        "--crust-solidus",
        dest="crust_solidus",
        help="Calculate and plot a crustal solidus",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--crust-solidus-comp",
        dest="crust_solidus_comp",
        help="Crustal composition for solidus",
        default="wet_intermediate",
    )
    parser.add_argument(
        "--mantle-solidus",
        dest="mantle_solidus",
        help="Calculate and plot a mantle solidus",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--mantle-solidus-xoh",
        dest="mantle_solidus_xoh",
        help="Water content for mantle solidus calculation (ppm)",
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--obs-ahe",
        dest="obs_ahe",
        help="Measured apatite (U-Th)/He age(s) (Ma)",
        nargs="+",
        default=[],
        type=float,
    )
    parser.add_argument(
        "--obs-ahe-stdev",
        dest="obs_ahe_stdev",
        help="Measured apatite (U-Th)/He age standard deviation(s) (Ma)",
        nargs="+",
        default=[],
        type=float,
    )
    parser.add_argument(
        "--obs-aft",
        dest="obs_aft",
        help="Measured apatite fission-track age(s) (Ma)",
        nargs="+",
        default=[],
        type=float,
    )
    parser.add_argument(
        "--obs-aft-stdev",
        dest="obs_aft_stdev",
        help="Measured apatite fission-track age standard deviation(s) (Ma)",
        nargs="+",
        default=[],
        type=float,
    )
    parser.add_argument(
        "--obs-zhe",
        dest="obs_zhe",
        help="Measured zircon (U-Th)/He age(s) (Ma)",
        nargs="+",
        default=[],
        type=float,
    )
    parser.add_argument(
        "--obs-zhe-stdev",
        dest="obs_zhe_stdev",
        help="Measured zircon (U-Th)/He age standard deviation(s) (Ma)",
        nargs="+",
        default=[],
        type=float,
    )
    parser.add_argument(
        "--misfit-num-params",
        dest="misfit_num_params",
        help="Number of model parameters to use in misfit calculation",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--misfit-type",
        dest="misfit_type",
        help="Misfit type for misfit calculation",
        default=1,
        type=int,
    )

    args = parser.parse_args()

    # Flip command-line flags to be opposite for function call
    # Function call expects
    # - echo_info = True for basic model info to be displayed
    # - echo_thermal_info = True for thermal model info to be displayed
    # - calc_ages = True if thermochronometer ages should be calculated
    # - echo_ages = True if thermochronometer ages should be displayed on the screen
    # - plot_results = True if plots of temperatures and densities should be created
    echo_info = not args.no_echo_info
    echo_thermal_info = not args.no_echo_thermal_info
    calc_ages = not args.no_calc_ages
    echo_ages = not args.no_echo_ages
    plot_results = not args.no_plot_results

    params = {
        "cmd_line_call": True,
        "echo_inputs": args.echo_inputs,
        "echo_info": echo_info,
        "echo_thermal_info": echo_thermal_info,
        "calc_ages": calc_ages,
        "echo_ages": echo_ages,
        "plot_results": plot_results,
        "save_plots": args.save_plots,
        "batch_mode": args.batch_mode,
        "mantle_adiabat": args.mantle_adiabat,
        "implicit": args.implicit,
        "read_temps": args.read_temps,
        "compare_temps": args.compare_temps,
        "write_temps": args.write_temps,
        "madtrax": args.madtrax,
        "ketch_aft": args.ketch_aft,
        "t_plots": args.t_plots,
        "max_depth": args.length,
        "nx": args.nx,
        "init_moho_depth": args.init_moho_depth,
        "final_moho_depth": args.final_moho_depth,
        "removal_fraction": args.removal_fraction,
        "crustal_flux": args.crustal_flux,
        "erotype": args.erotype,
        "erotype_opt1": args.erotype_opt1,
        "erotype_opt2": args.erotype_opt2,
        "temp_surf": args.temp_surf,
        "temp_base": args.temp_base,
        "t_total": args.time,
        "dt": args.dt,
        "vx_init": args.vx_init,
        "rho_crust": args.rho_crust,
        "cp_crust": args.cp_crust,
        "k_crust": args.k_crust,
        "heat_prod_crust": args.heat_prod_crust,
        "alphav_crust": args.alphav_crust,
        "rho_mantle": args.rho_mantle,
        "cp_mantle": args.cp_mantle,
        "k_mantle": args.k_mantle,
        "heat_prod_mantle": args.heat_prod_mantle,
        "alphav_mantle": args.alphav_mantle,
        "rho_a": args.rho_a,
        "k_a": args.k_a,
        "ap_rad": args.ap_rad,
        "ap_uranium": args.ap_uranium,
        "ap_thorium": args.ap_thorium,
        "zr_rad": args.zr_rad,
        "zr_uranium": args.zr_uranium,
        "zr_thorium": args.zr_thorium,
        "pad_thist": args.pad_thist,
        "pad_time": args.pad_time,
        "crust_solidus": args.crust_solidus,
        "crust_solidus_comp": args.crust_solidus_comp,
        "mantle_solidus": args.mantle_solidus,
        "mantle_solidus_xoh": args.mantle_solidus_xoh,
        "obs_ahe": args.obs_ahe,
        "obs_aft": args.obs_aft,
        "obs_zhe": args.obs_zhe,
        "obs_ahe_stdev": args.obs_ahe_stdev,
        "obs_aft_stdev": args.obs_aft_stdev,
        "obs_zhe_stdev": args.obs_zhe_stdev,
        "misfit_num_params": args.misfit_num_params,
        "misfit_type": args.misfit_type,
    }

    tc1d.prep_model(params)


if __name__ == "__main__":
    # execute only if run as a script
    main()
