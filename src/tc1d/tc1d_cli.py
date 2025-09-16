#!/usr/bin/env python3

# Import libraries we need
import argparse
import sys
import tc1d

# import cProfile
# from gooey import Gooey


# @Gooey(navigation='tabbed', tabbed_groups=True)
def main():
    parser = argparse.ArgumentParser(
        description="Calculates transient 1D temperatures and thermochronometer ages",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    general = parser.add_argument_group(
        "General options", "Options for various general features"
    )
    general.add_argument(
        "--echo-inputs",
        dest="echo_inputs",
        help="Print input values to the screen",
        action="store_true",
        default=False,
    )
    general.add_argument(
        "--no-echo-info",
        dest="no_echo_info",
        help="Do not print basic model info to the screen",
        action="store_true",
        default=False,
    )
    general.add_argument(
        "--no-echo-thermal-info",
        dest="no_echo_thermal_info",
        help="Do not print thermal model info to the screen",
        action="store_true",
        default=False,
    )
    general.add_argument(
        "--no-echo-ages",
        dest="no_echo_ages",
        help="Do not print calculated thermochronometer age(s) to the screen",
        action="store_true",
        default=False,
    )
    general.add_argument(
        "--run-type",
        dest="run_type",
        help="Define type of run: forward, batch, na, or mcmc.",
        default="forward",
        type=str,
    )
    general.add_argument(
        "--batch-mode",
        dest="batch_mode",
        help="Enable batch mode (no screen output, outputs writen to file)",
        action="store_true",
        default=False,
    )
    general.add_argument(
        "--inverse-mode",
        dest="inverse_mode",
        help="Enable inverse mode",
        action="store_true",
        default=False,
    )
    general.add_argument(
        "--debug",
        help="Enable debug output",
        action="store_true",
        default=False,
    )
    geometry = parser.add_argument_group(
        "Geometry and time options", "Options for the model geometry and run time"
    )
    geometry.add_argument(
        "--length",
        help="Model depth extent (km)",
        nargs="+",
        default=[125.0],
        type=float,
    )
    geometry.add_argument(
        "--nx",
        help="Number of grid points for temperature calculation",
        nargs="+",
        default=[251],
        type=int,
    )
    geometry.add_argument(
        "--time",
        help="Total simulation time (Myr)",
        nargs="+",
        default=[50.0],
        type=float,
    )
    geometry.add_argument(
        "--dt", help="Time step (years)", nargs="+", default=[5000.0], type=float
    )
    geometry.add_argument(
        "--init-moho-depth",
        dest="init_moho_depth",
        help="Initial depth of Moho (km)",
        nargs="+",
        default=[50.0],
        type=float,
    )
    geometry.add_argument(
        "--crustal-uplift",
        dest="crustal_uplift",
        help="Uplift only the crust in the thermal model",
        action="store_true",
        default=False,
    )
    geometry.add_argument(
        "--fixed-moho",
        dest="fixed_moho",
        help="Do not update Moho depth",
        action="store_true",
        default=False,
    )
    geometry.add_argument(
        "--removal-fraction",
        dest="removal_fraction",
        help="Fraction of lithospheric mantle to remove",
        nargs="+",
        default=[0.0],
        type=float,
    )
    geometry.add_argument(
        "--removal-start-time",
        dest="removal_start_time",
        help="Time to start removal of lithospheric mantle in Myr",
        nargs="+",
        default=[0.0],
        type=float,
    )
    geometry.add_argument(
        "--removal-end-time",
        dest="removal_end_time",
        help="Time to end removal of lithospheric mantle in Myr",
        nargs="+",
        default=[-1.0],
        type=float,
    )
    materials = parser.add_argument_group(
        "Material options", "Options for the model materials"
    )
    materials.add_argument(
        "--rho-crust",
        dest="rho_crust",
        help="Crustal density (kg/m^3)",
        nargs="+",
        default=[2850.0],
        type=float,
    )
    materials.add_argument(
        "--cp-crust",
        dest="cp_crust",
        help="Crustal heat capacity (J/kg/K)",
        nargs="+",
        default=[800.0],
        type=float,
    )
    materials.add_argument(
        "--k-crust",
        dest="k_crust",
        help="Crustal thermal conductivity (W/m/K)",
        nargs="+",
        default=[2.75],
        type=float,
    )
    materials.add_argument(
        "--heat-prod-crust",
        dest="heat_prod_crust",
        help="Crustal heat production (uW/m^3)",
        nargs="+",
        default=[0.5],
        type=float,
    )
    materials.add_argument(
        "--heat-prod-decay-depth",
        dest="heat_prod_decay_depth",
        help="Crustal heat production exponential decay depth (km)",
        nargs="+",
        default=[-1.0],
        type=float,
    )
    materials.add_argument(
        "--alphav-crust",
        dest="alphav_crust",
        help="Crustal coefficient of thermal expansion (1/K)",
        nargs="+",
        default=[3.0e-5],
        type=float,
    )
    materials.add_argument(
        "--rho-mantle",
        dest="rho_mantle",
        help="Mantle lithosphere density (kg/m^3)",
        nargs="+",
        default=[3250.0],
        type=float,
    )
    materials.add_argument(
        "--cp-mantle",
        dest="cp_mantle",
        help="Mantle lithosphere heat capacity (J/kg/K)",
        nargs="+",
        default=[1000.0],
        type=float,
    )
    materials.add_argument(
        "--k-mantle",
        dest="k_mantle",
        help="Mantle lithosphere thermal conductivity (W/m/K)",
        nargs="+",
        default=[2.5],
        type=float,
    )
    materials.add_argument(
        "--heat-prod-mantle",
        dest="heat_prod_mantle",
        help="Mantle lithosphere heat production (uW/m^3)",
        nargs="+",
        default=[0.0],
        type=float,
    )
    materials.add_argument(
        "--alphav-mantle",
        dest="alphav_mantle",
        help="Mantle lithosphere coefficient of thermal expansion (1/K)",
        nargs="+",
        default=[3.0e-5],
        type=float,
    )
    materials.add_argument(
        "--rho-a",
        dest="rho_a",
        help="Mantle asthenosphere density (kg/m^3)",
        nargs="+",
        default=[3250.0],
        type=float,
    )
    materials.add_argument(
        "--k-a",
        dest="k_a",
        help="Mantle asthenosphere thermal conductivity (W/m/K)",
        nargs="+",
        default=[20.0],
        type=float,
    )
    thermal = parser.add_argument_group(
        "Thermal model options", "Options for the thermal model"
    )
    # TODO: Fix this so it works with gooey
    thermal.add_argument(
        "--explicit",
        help="Use explicit instead of implicit finite-difference calculation",
        dest="implicit",
        action="store_false",
        default=True,
    )
    thermal.add_argument(
        "--temp-surf",
        dest="temp_surf",
        help="Surface boundary condition temperature (C)",
        nargs="+",
        default=[0.0],
        type=float,
    )
    thermal.add_argument(
        "--temp-base",
        dest="temp_base",
        help="Basal boundary condition temperature (C)",
        nargs="+",
        default=[1300.0],
        type=float,
    )
    # Does the following option work?
    thermal.add_argument(
        "--mantle_adiabat",
        help="Use adiabat for asthenosphere temperature",
        nargs="+",
        default=[True],
        type=bool,
    )
    intrusion = parser.add_argument_group(
        "Magmatic intrusion options", "Options for the intrusion model"
    )
    intrusion.add_argument(
        "--intrusion-temperature",
        dest="intrusion_temperature",
        help="Intrusion temperature (deg. C)",
        nargs="+",
        default=[750.0],
        type=float,
    )
    intrusion.add_argument(
        "--intrusion-start-time",
        dest="intrusion_start_time",
        help="Time for when magmatic intrusion becomes active (Myr)",
        nargs="+",
        default=[-1.0],
        type=float,
    )
    intrusion.add_argument(
        "--intrusion-duration",
        dest="intrusion_duration",
        help="Duration for which a magmatic intrusion is active (Myr)",
        nargs="+",
        default=[-1.0],
        type=float,
    )
    intrusion.add_argument(
        "--intrusion-thickness",
        dest="intrusion_thickness",
        help="Thickness of magmatic intrusion (km)",
        nargs="+",
        default=[-1.0],
        type=float,
    )
    intrusion.add_argument(
        "--intrusion-base-depth",
        dest="intrusion_base_depth",
        help="Depth of base of intrusion (km)",
        nargs="+",
        default=[-1.0],
        type=float,
    )
    erosion = parser.add_argument_group(
        "Erosion model options", "Options for the erosion model"
    )
    erosion.add_argument(
        "--vx-init",
        dest="vx_init",
        help="Initial steady-state advection velocity (mm/yr)",
        nargs="+",
        default=[0.0],
        type=float,
    )
    erosion.add_argument(
        "--ero-type",
        dest="ero_type",
        help="Type of erosion model (1-7 - see GitHub docs)",
        nargs="+",
        default=[1],
        type=int,
    )
    erosion.add_argument(
        "--ero-option1",
        dest="ero_option1",
        help="Erosion model option 1 (see GitHub docs)",
        nargs="+",
        default=[0.0],
        type=float,
    )
    erosion.add_argument(
        "--ero-option2",
        dest="ero_option2",
        help="Erosion model option 2 (see GitHub docs)",
        nargs="+",
        default=[0.0],
        type=float,
    )
    erosion.add_argument(
        "--ero-option3",
        dest="ero_option3",
        help="Erosion model option 3 (see GitHub docs)",
        nargs="+",
        default=[0.0],
        type=float,
    )
    erosion.add_argument(
        "--ero-option4",
        dest="ero_option4",
        help="Erosion model option 4 (see GitHub docs)",
        nargs="+",
        default=[0.0],
        type=float,
    )
    erosion.add_argument(
        "--ero-option5",
        dest="ero_option5",
        help="Erosion model option 5 (see GitHub docs)",
        nargs="+",
        default=[0.0],
        type=float,
    )
    erosion.add_argument(
        "--ero-option6",
        dest="ero_option6",
        help="Erosion model option 6 (see GitHub docs)",
        nargs="+",
        default=[0.0],
        type=float,
    )
    erosion.add_argument(
        "--ero-option7",
        dest="ero_option7",
        help="Erosion model option 7 (see GitHub docs)",
        nargs="+",
        default=[0.0],
        type=float,
    )
    erosion.add_argument(
        "--ero-option8",
        dest="ero_option8",
        help="Erosion model option 8 (see GitHub docs)",
        nargs="+",
        default=[0.0],
        type=float,
    )
    erosion.add_argument(
        "--ero-option9",
        dest="ero_option9",
        help="Erosion model option 9 (see GitHub docs)",
        nargs="+",
        default=[0.0],
        type=float,
    )
    erosion.add_argument(
        "--ero-option10",
        dest="ero_option10",
        help="Erosion model option 10 (see GitHub docs)",
        nargs="+",
        default=[0.0],
        type=float,
    )
    prediction = parser.add_argument_group(
        "Age prediction options", "Options for age prediction"
    )
    prediction.add_argument(
        "--no-calc-ages",
        dest="no_calc_ages",
        help="Disable calculation of thermochronometer ages",
        action="store_true",
        default=False,
    )
    prediction.add_argument(
        "--ketch-aft",
        dest="ketch_aft",
        help="Use the Ketcham et al. (2007) model for predicting FT ages",
        action="store_true",
        default=True,
    )
    prediction.add_argument(
        "--madtrax-aft",
        dest="madtrax_aft",
        help="Use the MadTrax algorithm for predicting apatite FT ages",
        action="store_true",
        default=False,
    )
    prediction.add_argument(
        "--madtrax-aft-kinetic-model",
        dest="madtrax_aft_kinetic_model",
        help="Kinetic model to use for AFT age prediction with MadTrax (see GitHub docs)",
        choices=range(1, 4),
        default=1,
        type=int,
    )
    prediction.add_argument(
        "--madtrax-zft-kinetic-model",
        dest="madtrax_zft_kinetic_model",
        help="Kinetic model to use for ZFT age prediction with MadTrax (see GitHub docs)",
        choices=range(1, 3),
        default=1,
        type=int,
    )
    prediction.add_argument(
        "--ap-rad",
        dest="ap_rad",
        help="Apatite grain radius (um)",
        nargs="+",
        default=[45.0],
        type=float,
    )
    prediction.add_argument(
        "--ap-uranium",
        dest="ap_uranium",
        help="Apatite U concentration (ppm)",
        nargs="+",
        default=[10.0],
        type=float,
    )
    prediction.add_argument(
        "--ap-thorium",
        dest="ap_thorium",
        help="Apatite Th concentration radius (ppm)",
        nargs="+",
        default=[40.0],
        type=float,
    )
    prediction.add_argument(
        "--zr-rad",
        dest="zr_rad",
        help="Zircon grain radius (um)",
        nargs="+",
        default=[60.0],
        type=float,
    )
    prediction.add_argument(
        "--zr-uranium",
        dest="zr_uranium",
        help="Zircon U concentration (ppm)",
        nargs="+",
        default=[100.0],
        type=float,
    )
    prediction.add_argument(
        "--zr-thorium",
        dest="zr_thorium",
        help="Zircon Th concentration radius (ppm)",
        nargs="+",
        default=[40.0],
        type=float,
    )
    prediction.add_argument(
        "--pad-time",
        dest="pad_time",
        help="Additional time added at starting temperature in t-T history (Myr)",
        nargs="+",
        default=[0.0],
        type=float,
    )
    prediction.add_argument(
        "--past-age-increment",
        dest="past_age_increment",
        help="Time increment in past (in Myr) at which ages should be calculated",
        default=0.0,
        type=float,
    )
    comparison = parser.add_argument_group(
        "Age comparison options", "Options for age comparison"
    )
    comparison.add_argument(
        "--obs-ahe",
        dest="obs_ahe",
        help="Measured apatite (U-Th)/He age(s) (Ma)",
        nargs="+",
        default=[],
        type=float,
    )
    comparison.add_argument(
        "--obs-ahe-stdev",
        dest="obs_ahe_stdev",
        help="Measured apatite (U-Th)/He age standard deviation(s) (Ma)",
        nargs="+",
        default=[],
        type=float,
    )
    comparison.add_argument(
        "--obs-aft",
        dest="obs_aft",
        help="Measured apatite fission-track age(s) (Ma)",
        nargs="+",
        default=[],
        type=float,
    )
    comparison.add_argument(
        "--obs-aft-stdev",
        dest="obs_aft_stdev",
        help="Measured apatite fission-track age standard deviation(s) (Ma)",
        nargs="+",
        default=[],
        type=float,
    )
    comparison.add_argument(
        "--obs-zhe",
        dest="obs_zhe",
        help="Measured zircon (U-Th)/He age(s) (Ma)",
        nargs="+",
        default=[],
        type=float,
    )
    comparison.add_argument(
        "--obs-zhe-stdev",
        dest="obs_zhe_stdev",
        help="Measured zircon (U-Th)/He age standard deviation(s) (Ma)",
        nargs="+",
        default=[],
        type=float,
    )
    comparison.add_argument(
        "--obs-zft",
        dest="obs_zft",
        help="Measured zircon fission-track age(s) (Ma)",
        nargs="+",
        default=[],
        type=float,
    )
    comparison.add_argument(
        "--obs-zft-stdev",
        dest="obs_zft_stdev",
        help="Measured zircon fission-track age standard deviation(s) (Ma)",
        nargs="+",
        default=[],
        type=float,
    )
    comparison.add_argument(
        "--obs-age-file",
        dest="obs_age_file",
        help="CSV file containing measured ages",
        default="",
        type=str,
    )
    comparison.add_argument(
        "--misfit-num-params",
        dest="misfit_num_params",
        help="Number of model parameters to use in misfit calculation",
        default=0,
        type=int,
    )
    comparison.add_argument(
        "--misfit-type",
        dest="misfit_type",
        help="Misfit type for misfit calculation",
        default=1,
        type=int,
    )
    plotting = parser.add_argument_group("Plotting options", "Options for plotting")
    plotting.add_argument(
        "--no-plot-results",
        dest="no_plot_results",
        help="Do not plot calculated results",
        action="store_true",
        default=False,
    )
    plotting.add_argument(
        "--no-display-plots",
        dest="no_display_plots",
        help="Do not display plots on screen",
        action="store_true",
        default=False,
    )
    plotting.add_argument(
        "--plot-myr",
        dest="plot_myr",
        help="Plot model time in Myr from start rather than Ma (ago)",
        action="store_true",
        default=False,
    )
    plotting.add_argument(
        "--plot-depth-history",
        dest="plot_depth_history",
        help="Plot depth history on plot of thermal history",
        action="store_true",
        default=False,
    )
    plotting.add_argument(
        "--plot-fault-depth-history",
        dest="plot_fault_depth_history",
        help="Plot fault depth history on plot of thermal history",
        action="store_true",
        default=False,
    )
    plotting.add_argument(
        "--invert-tt-plot",
        dest="invert_tt_plot",
        help="Invert temperature/depth on thermal history plot",
        action="store_true",
        default=False,
    )
    plotting.add_argument(
        "--t-plots",
        dest="t_plots",
        help="Output times for temperature plotting (Myrs). Treated as increment if only one value given.",
        nargs="+",
        default=[0.1, 1, 5, 10, 20, 30, 50],
        type=float,
    )
    plotting.add_argument(
        "--crust-solidus",
        dest="crust_solidus",
        help="Calculate and plot a crustal solidus",
        action="store_true",
        default=False,
    )
    plotting.add_argument(
        "--crust-solidus-comp",
        dest="crust_solidus_comp",
        help="Crustal composition for solidus",
        default="wet_intermediate",
    )
    plotting.add_argument(
        "--mantle-solidus",
        dest="mantle_solidus",
        help="Calculate and plot a mantle solidus",
        action="store_true",
        default=False,
    )
    plotting.add_argument(
        "--mantle-solidus-xoh",
        dest="mantle_solidus_xoh",
        help="Water content for mantle solidus calculation (ppm)",
        default=0.0,
        type=float,
    )
    plotting.add_argument(
        "--solidus-ranges",
        dest="solidus_ranges",
        help="Plot ranges for the crustal and mantle solidii",
        action="store_true",
        default=False,
    )
    output = parser.add_argument_group(
        "Output options", "Options for saving output to files"
    )
    output.add_argument(
        "--log-output",
        dest="log_output",
        help="Write model summary info to a csv file",
        action="store_true",
        default=False,
    )
    output.add_argument(
        "--log-file",
        dest="log_file",
        help="CSV filename for log output",
        default="",
    )
    output.add_argument(
        "--model-id",
        dest="model_id",
        help="Model identification character string",
        default="",
    )
    output.add_argument(
        "--write-temps",
        dest="write_temps",
        help="Save model temperatures to a file",
        action="store_true",
        default=False,
    )
    output.add_argument(
        "--write-past-ages",
        dest="write_past_ages",
        help="Write out incremental past ages to csv file",
        action="store_true",
        default=False,
    )
    output.add_argument(
        "--write-age-output",
        dest="write_age_output",
        help="Write out measured and predicted age data to csv file",
        action="store_true",
        default=False,
    )
    output.add_argument(
        "--save-plots",
        dest="save_plots",
        help="Save plots to a file",
        action="store_true",
        default=False,
    )
    advanced = parser.add_argument_group(
        "Advanced options", "Options for advanced users"
    )
    advanced.add_argument(
        "--read-temps",
        dest="read_temps",
        help="Read temperatures from a file",
        action="store_true",
        default=False,
    )
    advanced.add_argument(
        "--compare-temps",
        dest="compare_temps",
        help="Compare model temperatures to those from a file",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    # Display help and exit if no flags are set
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    # Flip command-line flags to be opposite for function call
    # Function call expects
    # - echo_info = True for basic model info to be displayed
    # - echo_thermal_info = True for thermal model info to be displayed
    # - calc_ages = True if thermochronometer ages should be calculated
    # - echo_ages = True if thermochronometer ages should be displayed on the screen
    # - plot_results = True if plots of temperatures and densities should be created
    # - display_plots = True if plots should be displayed on the screen
    # - plot_ma = True if plots should be in millions of years ago (Ma)
    echo_info = not args.no_echo_info
    echo_thermal_info = not args.no_echo_thermal_info
    calc_ages = not args.no_calc_ages
    echo_ages = not args.no_echo_ages
    plot_results = not args.no_plot_results
    display_plots = not args.no_display_plots
    plot_ma = not args.plot_myr

    params = {
        "cmd_line_call": True,
        "echo_inputs": args.echo_inputs,
        "echo_info": echo_info,
        "echo_thermal_info": echo_thermal_info,
        "calc_ages": calc_ages,
        "echo_ages": echo_ages,
        "plot_results": plot_results,
        "save_plots": args.save_plots,
        "display_plots": display_plots,
        "plot_ma": plot_ma,
        "plot_depth_history": args.plot_depth_history,
        "plot_fault_depth_history": args.plot_fault_depth_history,
        "invert_tt_plot": args.invert_tt_plot,
        "run_type": args.run_type,
        "batch_mode": args.batch_mode,
        "inverse_mode": args.inverse_mode,
        "mantle_adiabat": args.mantle_adiabat,
        "implicit": args.implicit,
        "read_temps": args.read_temps,
        "compare_temps": args.compare_temps,
        "write_temps": args.write_temps,
        "write_age_output": args.write_age_output,
        "debug": args.debug,
        "madtrax_aft": args.madtrax_aft,
        "madtrax_aft_kinetic_model": args.madtrax_aft_kinetic_model,
        "madtrax_zft_kinetic_model": args.madtrax_zft_kinetic_model,
        "ketch_aft": args.ketch_aft,
        "t_plots": args.t_plots,
        "max_depth": args.length,
        "nx": args.nx,
        "init_moho_depth": args.init_moho_depth,
        "removal_fraction": args.removal_fraction,
        "removal_start_time": args.removal_start_time,
        "removal_end_time": args.removal_end_time,
        "crustal_uplift": args.crustal_uplift,
        "fixed_moho": args.fixed_moho,
        "ero_type": args.ero_type,
        "ero_option1": args.ero_option1,
        "ero_option2": args.ero_option2,
        "ero_option3": args.ero_option3,
        "ero_option4": args.ero_option4,
        "ero_option5": args.ero_option5,
        "ero_option6": args.ero_option6,
        "ero_option7": args.ero_option7,
        "ero_option8": args.ero_option8,
        "ero_option9": args.ero_option9,
        "ero_option10": args.ero_option10,
        "temp_surf": args.temp_surf,
        "temp_base": args.temp_base,
        "t_total": args.time,
        "dt": args.dt,
        "vx_init": args.vx_init,
        "rho_crust": args.rho_crust,
        "cp_crust": args.cp_crust,
        "k_crust": args.k_crust,
        "heat_prod_crust": args.heat_prod_crust,
        "heat_prod_decay_depth": args.heat_prod_decay_depth,
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
        "pad_time": args.pad_time,
        "past_age_increment": args.past_age_increment,
        "write_past_ages": args.write_past_ages,
        "crust_solidus": args.crust_solidus,
        "crust_solidus_comp": args.crust_solidus_comp,
        "mantle_solidus": args.mantle_solidus,
        "mantle_solidus_xoh": args.mantle_solidus_xoh,
        "solidus_ranges": args.solidus_ranges,
        "obs_ahe": args.obs_ahe,
        "obs_aft": args.obs_aft,
        "obs_zhe": args.obs_zhe,
        "obs_zft": args.obs_zft,
        "obs_ahe_stdev": args.obs_ahe_stdev,
        "obs_aft_stdev": args.obs_aft_stdev,
        "obs_zhe_stdev": args.obs_zhe_stdev,
        "obs_zft_stdev": args.obs_zft_stdev,
        "obs_age_file": args.obs_age_file,
        "misfit_num_params": args.misfit_num_params,
        "misfit_type": args.misfit_type,
        "log_output": args.log_output,
        "log_file": args.log_file,
        "model_id": args.model_id,
        "intrusion_temperature": args.intrusion_temperature,
        "intrusion_start_time": args.intrusion_start_time,
        "intrusion_duration": args.intrusion_duration,
        "intrusion_thickness": args.intrusion_thickness,
        "intrusion_base_depth": args.intrusion_base_depth,
    }

    tc1d.prep_model(params)


if __name__ == "__main__":
    # execute only if run as a script
    # pr = cProfile.Profile()
    # pr.enable()
    main()
    # pr.disable()
    # pr.dump_stats('profile.pstat')
