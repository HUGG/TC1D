# T<sub>c</sub>1D - A 1D thermal and thermochronometer age prediction model

[![DOI](https://zenodo.org/badge/314177994.svg)](https://zenodo.org/badge/latestdoi/314177994)
[![PyPI - Version](https://img.shields.io/pypi/v/tc1d)](https://pypi.org/project/tc1d/)
[![Documentation Status](https://readthedocs.org/projects/tc1d/badge/?version=latest)](https://tc1d.readthedocs.io/en/latest/?badge=latest)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/HUGG/TC1D/HEAD?labpath=notebooks%2Fexplore_tc1d.ipynb)

T<sub>c</sub>1D is a one-dimensional thermal and thermochronometer age prediction model that can simulate the effects of various geodynamic and geomorphic processes on thermochronometer ages.
It currently supports prediction of apatite and zircon (U-Th)/He and fission-track ages.
Below you will find some essential details about using the code and detailed documentation can be found on the [T<sub>c</sub>1D documentation site](https://tc1d.readthedocs.io).

## Getting started

The easiest way to get started using T<sub>c</sub>1D is via the **launch binder** button above. This will open an interactive demo of T<sub>c</sub>1D using [Binder](https://mybinder.readthedocs.io/).
Instructions for how to install T<sub>c</sub>1D are below.

## Installation

You can install T<sub>c</sub>1D in your Python environment using `pip`.

```bash
pip install tc1d
```

**Note**: In order for T<sub>c</sub>1D to work properly you will also need to install the thermochronometer age prediction programs available in the [T<sub>c</sub>core package](https://github.com/HUGG/Tc_core/).

## Usage

An example model with 10 km of exhumation and default values can be run from the command line as follows:

```bash
tc1d-cli --ero-option1 10.0
```

A full list of options that can be used with T<sub>c</sub>1D can be found by running the code with the `--help` flag (or no specified flags):

```bash
tc1d-cli --help
```

This will return a usage statement and list of flags the code accepts.

## References

- Flowers, R. M., Ketcham, R. A., Shuster, D. L., & Farley, K. A. (2009). Apatite (U-Th)/He thermochronometry using a radiation damage accumulation and annealing model. Geochimica et Cosmochimica Acta, 73(8), 2347--2365.
- Guenthner, W. R., Reiners, P. W., Ketcham, R. A., Nasdala, L., & Giester, G. (2013). Helium diffusion in natural zircon: Radiation damage, anisotropy, and the interpretation of zircon (U-Th)/He thermochronology. American Journal of Science, 313(3), 145–198. https://doi.org/10.2475/03.2013.01
- Ketcham, R. A., Donelick, R. A., & Carlson, W. D.: Variability of apatite fission-track annealing kinetics III: Extrapolation to geological time scales. American Mineralogist, 84, 1235-1255, doi: [10.2138/am-1999-0903](https://doi.org/10.2138/am-1999-0903), 1999.
- Ketcham, R. A., Carter, A., Donelick, R. A., Barbarand, J., & Hurford, A. J. (2007). Improved modeling of fission-track annealing in apatite. American Mineralogist, 92(5–6), 799--810.
- Ketcham, R. A., Mora, A., and Parra, M.: Deciphering exhumation and burial history with multi-sample down-well thermochronometric inverse modelling, Basin Res., 30, 48-64, [10.1111/bre.12207](https://doi.org/10.1111/bre.12207), 2018.

## License

The T<sub>c</sub>1D is licensed under the GNU General Public License version 3: [T<sub>c</sub>1D software license](LICENSE)
