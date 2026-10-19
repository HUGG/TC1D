#!/usr/bin/env python3

# Import libraries we need
import argparse
from pathlib import Path
import sys
import tc1d
import copy

try:
    from importlib.metadata import version
    __version__ = version("tc1d")
except ImportError:
    __version__ = "dev"

# import cProfile
# from gooey import Gooey

# -----------------------------------------------------------------------------
# YAML configuration schema
# -----------------------------------------------------------------------------
# Defines the YAML sections and keys currently supported by tc1d-cli.
#
# This provides a single reference for checking consistency between:
#   tc1d.py parameters
#   tc1d-cli arguments
#   YAML input files
#
# Keep this mapping updated whenever a new YAML-accessible parameter is added.
YAML_ALLOWED_KEYS = {
    "general": {
        "run_type",
        "batch_mode",
        "inverse_mode",
        "debug",
        "echo_inputs",
        "no_echo_info",
        "no_echo_thermal_info",
        "no_echo_ages",
    },
    "geometry_time": {
        "length",
        "nx",
        "time",
        "dt",
        "init_moho_depth",
        "crustal_uplift",
        "fixed_moho",
        "removal_fraction",
        "removal_start_time",
        "removal_end_time",
    },
    "materials": {
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
    },
    "thermal_model": {
        "solution_type",
        "bc_type",
        "temp_surf",
        "temp_base",
        "flux_base",
        "no_mantle_adiabat",
        "temp_adiabat_ref",
    },
    "intrusion_model": {
        "intrusion_temperature",
        "intrusion_start_time",
        "intrusion_duration",
        "intrusion_thickness",
        "intrusion_base_depth",
    },
    "erosion_model": {
        "vx_init",
        "mantle_velocity",
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
        "ero_stages",
    },
    "age_prediction": {
        "no_calc_ages",
        "ketch_aft",
        "madtrax_aft",
        "madtrax_aft_kinetic_model",
        "madtrax_zft_kinetic_model",
        "ap_rad",
        "ap_uranium",
        "ap_thorium",
        "zr_rad",
        "zr_uranium",
        "zr_thorium",
        "pad_time",
        "past_age_increment",
    },
    "observations": {
        "obs_age_file",
        "obs_ahe",
        "obs_ahe_stdev",
        "obs_aft",
        "obs_aft_stdev",
        "obs_zhe",
        "obs_zhe_stdev",
        "obs_zft",
        "obs_zft_stdev",
        "misfit_num_params",
        "misfit_type",
    },
    "plotting": {
        "no_plot_results",
        "no_display_plots",
        "plot_myr",
        "plot_depth_history",
        "plot_fault_depth_history",
        "plot_density",
        "plot_elevation_history",
        "plot_peclet_number",
        "plot_ft_length_dist",
        "invert_tt_plot",
        "t_plots",
        "watch_it_exhume",
        "crust_solidus",
        "crust_solidus_comp",
        "mantle_solidus",
        "mantle_solidus_xoh",
        "solidus_ranges",
    },
    "output": {
        "log_output",
        "log_file",
        "model_id",
        "write_temps",
        "write_tt_history",
        "write_ttdp_history",
        "write_past_ages",
        "write_age_output",
        "write_ft_lengths",
        "save_plots",
    },
    "advanced": {
        "read_temps",
        "compare_temps",
    },
    "inversion": {
        "neighbourhood_algorithm",
        "mcmc",
    },
}


YAML_INVERSION_ALLOWED_KEYS = {
    "neighbourhood_algorithm": {
        "na_ns",
        "na_nr",
        "na_ni",
        "na_n",
        "na_n_resample",
        "na_n_walkers",
    },
    "mcmc": {
        "mcmc_nwalkers",
        "mcmc_nsteps",
        "mcmc_discard",
        "mcmc_thin",
    },
}


def _validate_yaml_keys(y: dict) -> None:
    """
    Validate YAML section and key names against the supported Tc1D YAML schema.

    Raises
    ------
    ValueError
        If an unknown top-level section or unsupported key is found.
    """

    if not isinstance(y, dict):
        raise ValueError("YAML configuration must be a mapping/dict.")

    # Check top-level sections.
    unknown_sections = set(y) - set(YAML_ALLOWED_KEYS)

    if unknown_sections:
        raise ValueError(
            "Unknown YAML section(s): " + ", ".join(sorted(unknown_sections))
        )

    # Check keys inside each section.
    for section, values in y.items():
        if not isinstance(values, dict):
            raise ValueError(f"YAML section '{section}' must be a mapping/dict.")

        unknown_keys = set(values) - YAML_ALLOWED_KEYS[section]

        if unknown_keys:
            raise ValueError(
                f"Unknown YAML key(s) in section '{section}': "
                + ", ".join(sorted(unknown_keys))
            )

    # Check nested inversion subsections separately.
    inversion = y.get("inversion", {})

    if isinstance(inversion, dict):
        for subsection, values in inversion.items():
            if subsection not in YAML_INVERSION_ALLOWED_KEYS:
                raise ValueError(f"Unknown YAML inversion subsection: '{subsection}'")

            if not isinstance(values, dict):
                raise ValueError(
                    f"YAML inversion subsection '{subsection}' "
                    "must be a mapping/dict."
                )

            unknown_keys = set(values) - YAML_INVERSION_ALLOWED_KEYS[subsection]

            if unknown_keys:
                raise ValueError(
                    f"Unknown YAML key(s) in inversion subsection "
                    f"'{subsection}': " + ", ".join(sorted(unknown_keys))
                )


def _load_yaml_dict(path: str) -> dict:
    """
    Load a YAML input file and return a dictionary.

    Notes
    -----
    - Requires PyYAML (pip install pyyaml).
    - Top-level YAML structure must be a mapping/dict.
    """
    try:
        import yaml  # PyYAML
    except ImportError as e:
        raise ImportError(
            "Missing dependency: PyYAML. Install it with:\n"
            "  pip install pyyaml\n"
            "or add it to your environment/requirements."
        ) from e

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Cannot find input YAML file: {path}")

    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(
            f"Top-level YAML structure must be a mapping/dict, got {type(data)}"
        )

    return data


def _as_list(x):
    """Keep argparse nargs='+' behaviour: store scalars as one-element lists."""
    return x if isinstance(x, list) else [x]


def _as_float_list(x):
    """Return a list[float] from a YAML scalar or list."""
    if isinstance(x, list):
        return [float(v) for v in x]
    return [float(x)]


def _as_bool(x) -> bool:
    """
    Parse booleans from YAML.
    Accepts: bool, 0/1, 'true'/'false' strings (case-insensitive).
    """
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)) and x in (0, 1):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true", "yes", "y", "1", "on"):
            return True
        if s in ("false", "no", "n", "0", "off"):
            return False
    raise ValueError(f"Cannot parse boolean from value: {x!r}")


def _apply_yaml_to_args(args, y: dict) -> None:
    """
    Apply YAML overrides to the argparse Namespace.

    Rules:
    - Only keys present in YAML are applied.
    - YAML values override values provided on the CLI.
    """
    # ---- general
    g = y.get("general", {})
    if isinstance(g, dict):
        for k in (
            "run_type",
            "batch_mode",
            "inverse_mode",
            "debug",
            "echo_inputs",
            "no_echo_info",
            "no_echo_thermal_info",
            "no_echo_ages",
        ):
            if k in g:
                # booleans:
                if k in (
                    "batch_mode",
                    "inverse_mode",
                    "debug",
                    "echo_inputs",
                    "no_echo_info",
                    "no_echo_thermal_info",
                    "no_echo_ages",
                ):
                    setattr(args, k, _as_bool(g[k]))
                else:
                    setattr(args, k, g[k])

    # ---- geometry_time
    gt = y.get("geometry_time", {})
    if isinstance(gt, dict):
        if "length" in gt:
            args.length = _as_list(float(gt["length"]))
        if "nx" in gt:
            args.nx = _as_list(int(gt["nx"]))
        if "time" in gt:
            args.time = _as_list(float(gt["time"]))
        if "dt" in gt:
            args.dt = _as_list(float(gt["dt"]))

        for k in (
            "init_moho_depth",
            "removal_fraction",
            "removal_start_time",
            "removal_end_time",
        ):
            if k in gt:
                setattr(args, k, _as_list(float(gt[k])))

        for k in ("crustal_uplift", "fixed_moho"):
            if k in gt:
                setattr(args, k, _as_bool(gt[k]))

    # ---- materials
    m = y.get("materials", {})
    if isinstance(m, dict):
        list_float_keys = (
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
        )
        for k in list_float_keys:
            if k in m:
                setattr(args, k, _as_list(float(m[k])))

    # ---- thermal_model
    th = y.get("thermal_model", {})
    if isinstance(th, dict):
        if "solution_type" in th:
            args.solution_type = _as_list(int(th["solution_type"]))
        if "no_mantle_adiabat" in th:
            args.no_mantle_adiabat = _as_bool(th["no_mantle_adiabat"])
        if "bc_type" in th:
            args.bc_type = _as_list(int(th["bc_type"]))
        if "temp_surf" in th:
            args.temp_surf = _as_list(float(th["temp_surf"]))
        if "temp_base" in th:
            args.temp_base = _as_list(float(th["temp_base"]))
        if "flux_base" in th:
            args.flux_base = _as_list(float(th["flux_base"]))
        if "temp_adiabat_ref" in th:
            args.temp_adiabat_ref = _as_list(float(th["temp_adiabat_ref"]))

    # ---- intrusion_model
    intr = y.get("intrusion_model", {})
    if isinstance(intr, dict):
        list_float_keys_intr = (
            "intrusion_temperature",
            "intrusion_start_time",
            "intrusion_duration",
            "intrusion_thickness",
            "intrusion_base_depth",
        )
        for k in list_float_keys_intr:
            if k in intr:
                setattr(args, k, _as_float_list(intr[k]))

    # ---- erosion_model
    e = y.get("erosion_model", {})
    if isinstance(e, dict):
        if "vx_init" in e:
            args.vx_init = _as_list(float(e["vx_init"]))
        if "mantle_velocity" in e:
            args.mantle_velocity = _as_list(float(e["mantle_velocity"]))
        if "ero_type" in e:
            args.ero_type = _as_list(int(e["ero_type"]))
        # BG: keep YAML-defined stages in the raw YAML dict 'y',
        # but also attach them to args for convenience/debug (not an argparse option).
        if "ero_stages" in e:
            setattr(args, "ero_stages", e["ero_stages"])

        for i in range(1, 11):
            key = f"ero_option{i}"
            if key in e:
                setattr(args, key, _as_float_list(e[key]))

    # ---- age_prediction
    ap = y.get("age_prediction", {})
    if isinstance(ap, dict):
        for k in ("no_calc_ages", "ketch_aft", "madtrax_aft"):
            if k in ap:
                setattr(args, k, _as_bool(ap[k]))

        for k in ("madtrax_aft_kinetic_model", "madtrax_zft_kinetic_model"):
            if k in ap:
                setattr(args, k, int(ap[k]))

        for k in (
            "ap_rad",
            "ap_uranium",
            "ap_thorium",
            "zr_rad",
            "zr_uranium",
            "zr_thorium",
            "pad_time",
        ):
            if k in ap:
                setattr(args, k, _as_list(float(ap[k])))

        if "past_age_increment" in ap:
            args.past_age_increment = float(ap["past_age_increment"])

    # ---- observations
    obs = y.get("observations", {})
    if isinstance(obs, dict):
        if "obs_age_file" in obs:
            args.obs_age_file = str(obs["obs_age_file"])

        list_float_list_keys = (
            "obs_ahe",
            "obs_ahe_stdev",
            "obs_aft",
            "obs_aft_stdev",
            "obs_zhe",
            "obs_zhe_stdev",
            "obs_zft",
            "obs_zft_stdev",
        )
        for k in list_float_list_keys:
            if k in obs:
                args_list = obs[k] if isinstance(obs[k], list) else [obs[k]]
                setattr(args, k, [float(v) for v in args_list])

        if "misfit_num_params" in obs:
            args.misfit_num_params = int(obs["misfit_num_params"])
        if "misfit_type" in obs:
            args.misfit_type = int(obs["misfit_type"])

    # ---- plotting
    pl = y.get("plotting", {})
    if isinstance(pl, dict):
        for k in (
            "no_plot_results",
            "no_display_plots",
            "plot_myr",
            "plot_depth_history",
            "plot_fault_depth_history",
            "plot_density",
            "plot_elevation_history",
            "plot_peclet_number",
            "plot_ft_length_dist",
            "invert_tt_plot",
            "watch_it_exhume",
            "crust_solidus",
            "mantle_solidus",
            "solidus_ranges",
        ):
            if k in pl:
                setattr(args, k, _as_bool(pl[k]))

        if "crust_solidus_comp" in pl:
            args.crust_solidus_comp = str(pl["crust_solidus_comp"])
        if "mantle_solidus_xoh" in pl:
            args.mantle_solidus_xoh = float(pl["mantle_solidus_xoh"])
        if "t_plots" in pl:
            args.t_plots = list(pl["t_plots"])

    # ---- output
    out = y.get("output", {})
    if isinstance(out, dict):
        for k in (
            "log_output",
            "write_temps",
            "write_tt_history",
            "write_ttdp_history",
            "write_past_ages",
            "write_age_output",
            "write_ft_lengths",
            "save_plots",
        ):
            if k in out:
                setattr(args, k, _as_bool(out[k]))
        for k in ("log_file", "model_id"):
            if k in out:
                setattr(args, k, str(out[k]))

    # ---- advanced
    adv = y.get("advanced", {})
    if isinstance(adv, dict):
        for k in ("read_temps", "compare_temps"):
            if k in adv:
                setattr(args, k, _as_bool(adv[k]))

    # ---- inversion hyperparameters
    inv = y.get("inversion", {})
    if isinstance(inv, dict):
        na = inv.get("neighbourhood_algorithm", {})
        if isinstance(na, dict):
            for k in (
                "na_ns",
                "na_nr",
                "na_ni",
                "na_n",
                "na_n_resample",
                "na_n_walkers",
            ):
                if k in na:
                    setattr(args, k, int(na[k]))

        mc = inv.get("mcmc", {})
        if isinstance(mc, dict):
            for k in ("mcmc_nwalkers", "mcmc_nsteps", "mcmc_discard", "mcmc_thin"):
                if k in mc:
                    setattr(args, k, int(mc[k]))


def _warn_yaml_cli_conflicts(parser, cli_args, default_args, y: dict) -> None:
    """
    Warn if user provided CLI options that are also set in YAML.
    In Tc1D, YAML has priority => those CLI options will be ignored.
    """
    if not isinstance(y, dict):
        return

    # Map: (yaml_section, yaml_key) -> argparse dest
    mapping = {
        ("general", "run_type"): "run_type",
        ("general", "batch_mode"): "batch_mode",
        ("general", "inverse_mode"): "inverse_mode",
        ("general", "debug"): "debug",
        ("general", "echo_inputs"): "echo_inputs",
        ("general", "no_echo_info"): "no_echo_info",
        ("general", "no_echo_thermal_info"): "no_echo_thermal_info",
        ("general", "no_echo_ages"): "no_echo_ages",
        ("erosion_model", "ero_type"): "ero_type",
        ("plotting", "no_plot_results"): "no_plot_results",
        ("plotting", "no_display_plots"): "no_display_plots",
        ("plotting", "plot_myr"): "plot_myr",
        ("plotting", "plot_depth_history"): "plot_depth_history",
        ("plotting", "plot_fault_depth_history"): "plot_fault_depth_history",
        ("plotting", "invert_tt_plot"): "invert_tt_plot",
        ("plotting", "t_plots"): "t_plots",
        ("plotting", "watch_it_exhume"): "watch_it_exhume",
        ("plotting", "crust_solidus"): "crust_solidus",
        ("plotting", "crust_solidus_comp"): "crust_solidus_comp",
        ("plotting", "mantle_solidus"): "mantle_solidus",
        ("plotting", "mantle_solidus_xoh"): "mantle_solidus_xoh",
        ("plotting", "solidus_ranges"): "solidus_ranges",
        ("output", "log_output"): "log_output",
        ("output", "log_file"): "log_file",
        ("output", "model_id"): "model_id",
        ("output", "write_temps"): "write_temps",
        ("output", "write_tt_history"): "write_tt_history",
        ("output", "write_ttdp_history"): "write_ttdp_history",
        ("output", "write_past_ages"): "write_past_ages",
        ("output", "write_age_output"): "write_age_output",
        ("output", "write_ft_lengths"): "write_ft_lengths",
        ("output", "save_plots"): "save_plots",
        ("advanced", "read_temps"): "read_temps",
        ("advanced", "compare_temps"): "compare_temps",
    }

    # Add ero_option1..10
    for i in range(1, 11):
        mapping[("erosion_model", f"ero_option{i}")] = f"ero_option{i}"

    conflicts = []

    for (section, key), dest in mapping.items():
        sec = y.get(section, {})
        if not isinstance(sec, dict):
            continue
        if key not in sec:
            continue

        cli_val = getattr(cli_args, dest, None)
        def_val = getattr(default_args, dest, None)

        # If CLI differs from default, user likely set it explicitly on CLI.
        if cli_val != def_val:
            conflicts.append((dest, cli_val, sec[key]))

    if not conflicts:
        return

    msg_lines = [
        "YAML input file is active: YAML values override CLI flags for the same keys.",
        "Ignored CLI flags (also defined in YAML):",
    ]
    for dest, cli_val, y_val in conflicts:
        msg_lines.append(f"  - {dest}: CLI={cli_val!r} ignored, YAML={y_val!r} used")

    msg_lines.append(
        "Tip: remove the key from YAML if you want to control it from the command line."
    )

    print("tc1d-cli: warning: " + msg_lines[0], file=sys.stderr)
    for line in msg_lines[1:]:
        print(line, file=sys.stderr)


def validate_args(args, parser, y=None) -> None:
    """
    Validate logical consistency of final args (after YAML overrides).
    Use parser.error(...) for user-facing CLI/YAML config errors.
    """

    # -------------------------
    # 1) run_type / inverse_mode / batch_mode
    # -------------------------
    run_type = str(args.run_type).strip().lower()
    valid_run_types = {"forward", "batch", "na", "mcmc"}
    if run_type not in valid_run_types:
        parser.error(
            f"--run-type must be one of {sorted(valid_run_types)}, got: {args.run_type!r}"
        )

    # Define "truth" from run_type
    # - forward/batch => inverse_mode should be False
    # - na/mcmc       => inverse_mode should be True
    if run_type in {"na", "mcmc"} and not args.inverse_mode:
        parser.error(
            f"Inconsistent config: run_type='{run_type}' requires inverse_mode=true"
        )
    if run_type in {"forward", "batch"} and args.inverse_mode:
        parser.error(
            f"Inconsistent config: run_type='{run_type}' is not compatible with inverse_mode=true"
        )

    # batch_mode: batch_mode only meaningful for batch runs
    if args.batch_mode and run_type != "batch":
        parser.error(
            "Inconsistent config: batch_mode=true is only valid when run_type='batch'"
        )

    # -------------------------
    # 2) Erosion model consistency (including YAML inversion ranges)
    # -------------------------
    # ero_type
    ero_type = (
        args.ero_type[0] if isinstance(args.ero_type, list) else int(args.ero_type)
    )
    if ero_type < 0 or ero_type > 7:
        parser.error(f"ero_type must be in [0..7], got: {ero_type}")

    # helper: check if an ero_option is a range [min,max] (inversion) or scalar [value]
    def _is_range(opt_list):
        return isinstance(opt_list, list) and len(opt_list) == 2

    def _is_scalar_list(opt_list):
        return isinstance(opt_list, list) and len(opt_list) == 1

    # BG: ero_type=0 is YAML-only: require erosion_model.ero_stages in the YAML input file.
    if ero_type == 0:
        has_yaml_stages = False
        if isinstance(y, dict):
            em = y.get("erosion_model", {})
            if isinstance(em, dict):
                stages = em.get("ero_stages", None)
                has_yaml_stages = isinstance(stages, list) and len(stages) > 0

        if not has_yaml_stages:
            parser.error(
                "ero_type=0 requires a YAML input file with:\n"
                "  erosion_model: ero_stages: [...]\n"
                "Use: tc1d-cli --input-file <file.yaml>\n"
            )


# @Gooey(navigation='tabbed', tabbed_groups=True)
def main():
    parser = argparse.ArgumentParser(
        description="Calculates transient 1D temperatures and thermochronometer ages",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-v", "--version", action="version", version=__version__)
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
    general.add_argument(
        "--input-file",
        dest="input_file",
        help=(
            "YAML input file to override CLI values "
            "(YAML has priority when both YAML and CLI provide a value)."
        ),
        default="",
        type=str,
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
        "--solution-type",
        dest="solution_type",
        help="Finite-difference solution type (1 = explicit, 2 = implicit, 3 = Crank-Nicolson)",
        nargs="+",
        default=[3],
        type=int,
    )
    thermal.add_argument(
        "--bc-type",
        dest="bc_type",
        help="Basal boundary condition type (1=constant temperature, 2=heat flux)",
        nargs="+",
        default=[1],
        type=int,
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
    thermal.add_argument(
        "--flux-base",
        dest="flux_base",
        help="Basal boundary condition heat flux (mW/m^2)",
        nargs="+",
        default=[20.0],
        type=float,
    )
    # Does the following option work?
    thermal.add_argument(
        "--no-mantle-adiabat",
        dest="no_mantle_adiabat",
        help="Do not use adiabat for asthenosphere temperature",
        action="store_true",
        default=False,
    )
    thermal.add_argument(
        "--temp-adiabat-ref",
        dest="temp_adiabat_ref",
        help="Reference temperature (C) for mantle adiabat calculation",
        nargs="+",
        default=[1300.0],
        type=float,
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
        help="Type of erosion model (0-7 - see GitHub docs). Use 0 with YAML erosion_model.ero_stages.",
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
    erosion.add_argument(
        "--mantle-velocity",
        dest="mantle_velocity",
        help="Velocity for mantle movement in fixed-Moho models (mm/yr)",
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
        default=[60.0],
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
    # BG: Neighbourhood Algorithm (NA) options
    na_group = parser.add_argument_group(
        "Neighbourhood Algorithm options",
        "Options controlling the Neighbourhood Algorithm inverse search",
    )

    na_group.add_argument(
        "--na-ns",
        dest="na_ns",
        type=int,
        default=24,
        help="NA: number of new samples per iteration",
    )
    na_group.add_argument(
        "--na-nr",
        dest="na_nr",
        type=int,
        default=12,
        help="NA: number of Voronoi cells to resample per iteration",
    )
    na_group.add_argument(
        "--na-ni",
        dest="na_ni",
        type=int,
        default=50,
        help="NA: size of initial random search",
    )
    na_group.add_argument(
        "--na-n",
        dest="na_n",
        type=int,
        default=6,
        help="NA: number of NA iterations after the initial search",
    )
    na_group.add_argument(
        "--na-n-resample",
        dest="na_n_resample",
        type=int,
        default=2000,
        help="NA appraiser: number of new samples to draw for PDF estimation",
    )
    na_group.add_argument(
        "--na-n-walkers",
        dest="na_n_walkers",
        type=int,
        default=5,
        help="NA appraiser: number of parallel walkers",
    )
    # BG: MCMC options (emcee)
    mcmc = parser.add_argument_group(
        "MCMC options", "Options for MCMC inversion (emcee)"
    )
    mcmc.add_argument(
        "--mcmc-nwalkers",
        dest="mcmc_nwalkers",
        help="MCMC: number of walkers in the ensemble",
        type=int,
        default=8,
    )
    mcmc.add_argument(
        "--mcmc-nsteps",
        dest="mcmc_nsteps",
        help="MCMC: number of steps per walker",
        type=int,
        default=50,
    )
    mcmc.add_argument(
        "--mcmc-discard",
        dest="mcmc_discard",
        help="MCMC: number of burn-in steps to discard",
        type=int,
        default=5,
    )
    mcmc.add_argument(
        "--mcmc-thin",
        dest="mcmc_thin",
        help="MCMC: thinning factor for chains",
        type=int,
        default=3,
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
        "--plot-density",
        dest="plot_density",
        help="Plot density beside geotherms plot",
        action="store_true",
        default=False,
    )
    plotting.add_argument(
        "--plot-elevation-history",
        dest="plot_elevation_history",
        help="Plot surface elevation history",
        action="store_true",
        default=False,
    )
    plotting.add_argument(
        "--plot-peclet-number",
        dest="plot_peclet_number",
        help="Plot peclet number on erosion history plot",
        action="store_true",
        default=False,
    )
    plotting.add_argument(
        "--plot-ft-length-dist",
        dest="plot_ft_length_dist",
        help="Plot apatite fission-track length distribution",
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
        "--watch-it-exhume",
        dest="watch_it_exhume",
        help="Use animation plot for thermal history rather than static plotting",
        action="store_true",
        default=False,
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
        "--write-tt-history",
        dest="write_tt_history",
        help="Write time-temperature history to a csv file",
        action="store_true",
        default=False,
    )
    output.add_argument(
        "--write-ttdp-history",
        dest="write_ttdp_history",
        help="Write time-temperature-depth-pressure history to a csv file",
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
        "--write-ft-lengths",
        dest="write_ft_lengths",
        help="Write apatite fission track-length distribution to a file",
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

    # Parse twice: defaults vs actual CLI, to detect what the user explicitly set.
    default_args = parser.parse_args([])
    cli_args = parser.parse_args()

    args = cli_args

    y = None
    if args.input_file:
        y = _load_yaml_dict(args.input_file)

        # Validate YAML sections and keys before applying overrides.
        _validate_yaml_keys(y)

        # Warn about YAML/CLI conflicts BEFORE applying YAML overrides
        _warn_yaml_cli_conflicts(parser, cli_args, default_args, y)

        # Apply YAML overrides
        _apply_yaml_to_args(args, y)

    validate_args(args, parser, y=y)

    # Display help and exit if no flags are set
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    # Flip command-line flags to be opposite for function call
    # Function call expects
    # - echo_info = True for basic model info to be displayed
    # - echo_thermal_info = True for thermal model info to be displayed
    # - mantle_adiabat = True if mantle adiabat should be used for asthenosphere temperatures
    # - calc_ages = True if thermochronometer ages should be calculated
    # - echo_ages = True if thermochronometer ages should be displayed on the screen
    # - plot_results = True if plots of temperatures and densities should be created
    # - display_plots = True if plots should be displayed on the screen
    # - plot_ma = True if plots should be in millions of years ago (Ma)
    echo_info = not args.no_echo_info
    echo_thermal_info = not args.no_echo_thermal_info
    mantle_adiabat = not args.no_mantle_adiabat
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
        "plot_density": args.plot_density,
        "plot_elevation_history": args.plot_elevation_history,
        "plot_peclet_number": args.plot_peclet_number,
        "plot_ft_length_dist": args.plot_ft_length_dist,
        "invert_tt_plot": args.invert_tt_plot,
        "watch_it_exhume": args.watch_it_exhume,
        "run_type": args.run_type,
        "batch_mode": args.batch_mode,
        "inverse_mode": args.inverse_mode,
        "mantle_adiabat": mantle_adiabat,
        "solution_type": args.solution_type,
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
        "ero_stages": getattr(
            args, "ero_stages", None
        ),  # BG: YAML-defined stages for ero_type=0
        "ero_stages_template": copy.deepcopy(
            getattr(args, "ero_stages", None)
        ),  # BG: raw YAML template for NA duration inversion
        "mantle_velocity": args.mantle_velocity,
        "bc_type": args.bc_type,
        "temp_surf": args.temp_surf,
        "temp_base": args.temp_base,
        "flux_base": args.flux_base,
        "temp_adiabat_ref": args.temp_adiabat_ref,
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
        "write_tt_history": args.write_tt_history,
        "write_ttdp_history": args.write_ttdp_history,
        "write_ft_lengths": args.write_ft_lengths,
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
        "na_ns": args.na_ns,  # BG: NA - samples per iteration
        "na_nr": args.na_nr,  # BG: NA - number of cells to resample
        "na_ni": args.na_ni,  # BG: NA - size of initial random search
        "na_n": args.na_n,  # BG: NA - number of NA iterations
        "na_n_resample": args.na_n_resample,  # BG: NAAppraiser - total new samples
        "na_n_walkers": args.na_n_walkers,  # BG: NAAppraiser - parallel walkers
        "mcmc_nwalkers": args.mcmc_nwalkers,  # BG: MCMC - number of walkers in the ensemble
        "mcmc_nsteps": args.mcmc_nsteps,  # BG: MCMC - number of steps per walker
        "mcmc_discard": args.mcmc_discard,  # BG: MCMC - number of burn-in steps to discard
        "mcmc_thin": args.mcmc_thin,  # BG: MCMC - thinning factor for chains
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
