# Erosion models

## Overview

There are several options for how erosion can be defined in the T<sub>c</sub>1D thermal models.
Options for the erosion rate calculation include:

1. Constant erosion rate
2. Constant rate with a step-function change at a specified time
3. Exponential decay
4. Emplacement and erosional removal of a thrust sheet
5. Tectonic exhumation and erosion
6. Linear increase in erosion rate from a specified starting time
7. Extensional tectonics

Below is a general description of how erosion is implemented in the code as well as details about how each option works.

## General implementation

The calculation of erosion rates in T<sub>c</sub>1D is done in a function titled `calculate_erosion_rate()`. The function definition statement is below, to give you a sense of the values that can be passed to the function:

```python
def calculate_erosion_rate(params, dt, t_total, current_time, x, vx_array, fault_depth, moho_depth):
    """Defines the way in which erosion should be applied."""
    ...
    return vx_array, vx_surf, vx_max, fault_depth
```

The function expects the following values to be passed:

- `params`: The Tc1D model parameters dictionary. Relevant parameters include:
    - `params["ero_type"]`: The type of erosion model to be used
        - `1` = Constant erosion rate
        - `2` = Constant rate with a step-function change at a specified time
        - `3` = Exponential decay
        - `4` = Thrust sheet emplacement/erosion
        - `5` = Tectonic exhumation and erosion
        - `6` = Linear rate change
        - `7` = Extensional tectonics
    - `params["ero_option1"]`, `params["ero_option2"]`, `...`: Optional parameters depending on the selected erosion model
- `dt`: The model time step
- `t_total`: The total model run time
- `current_time`: The current time in the model
- `x`: The model spatial coordinates (depths)
- `vx_array`: The array of velocities across the model depth range
- `fault_depth`: The depth of the fault in erosion model 7 (ignored for other erosion models)
- `moho_depth`: The current depth to the model Moho

The function returns the following values:

- `vx_array`: The array of velocities across the model depth range
- `vx_surf`: The velocity at the model surface
- `vx_max`: The magnitude of the maximum velocity in the model
- `fault_depth`: The depth of the fault in erosion model 7 (ignored for other erosion models)

Details about the implementation of the erosion model options can be found below.

### Type 1: Constant erosion rate (`params["ero_type"] = 1`)

![Constant erosion rate model example](png/cooling_hist_erotype1.png)<br/>
*Example cooling history for the constant erosion rate erosion model.*

The constant erosion rate case is the simplest option.
It is defined using one parameter:

- `params["ero_option1"]`: The erosion magnitude in km

The calculated value for the erosion rate is simply the erosion magnitude divided by the simulation time.

### Type 2: Constant rate with a step-function change at a specified time

![Step-function change in erosion rate model example](png/cooling_hist_erotype2.png)<br/>
*Example cooling history for the constant rate with a step-function change at a specified time erosion model.*

The constant rate with a step-function change at a specified time model is designed to have a first phase of exhumation followed by a second.
The magnitude of exhumation for each phase is specified separately, as is the transition time.
The parameters used in this case are:

- `ero_option1`: the exhumation magnitude (in km) for the first phase
- `ero_option2`: the time of the transition in erosion rate
- `ero_option3`: the exhumation magnitude (in km) for the second phase

The code for this implementation can be found below.

```python
    # Constant erosion rate with a step-function change at a specified time
    elif ero_type == 2:
        rate_change_time = myr2sec(ero_option2)
        init_rate = kilo2base(ero_option1) / rate_change_time
        final_rate = kilo2base(ero_option3) / (t_total - rate_change_time)
        # First stage of erosion
        if current_time < rate_change_time:
            vx = init_rate
        # Second stage of erosion
        else:
            vx = final_rate
```

### Type 3: Exponential decay

![Exponential decay in erosion rate model example](png/cooling_hist_erotype3.png)<br/>
*Example cooling history for the exponential decay erosion model.*

The exponential decay erosion model works by calculating a maximum erosion rate based on the magnitude of exhumation and the characteristic time of exponential decay.
The user inputs the exhumation magnitude and the time over which the erosion rate should decay exponentially to $1/e$ times the original value, and the code determines the erosion rate that will result.
Two erosion model options are used for this case:

- `ero_option1`: the exhumation magnitude (in km)
- `ero_option2`: the characteristic time (in Myr)

The code for this implementation can be found below.

```python
    # Exponential erosion rate decay with a set characteristic time
    elif ero_type == 3:
        erosion_magnitude = kilo2base(ero_option1)
        decay_time = myr2sec(ero_option2)
        max_rate = erosion_magnitude / (
            decay_time * (np.exp(0.0 / decay_time) - np.exp(-t_total / decay_time))
        )
        vx = max_rate * np.exp(-current_time / decay_time)
```

### Type 4: Emplacement and erosional removal of a thrust sheet

Coming soon :)

### Type 5: Tectonic exhumation and erosion

Coming soon :)

### Type 6: Linear increase in erosion rate from a specified time

This model is designed to have a linear increase in erosion rate from a starting rate to a final rate over a specified time window.


### Elevation-dependent erosion

Elevation-dependent erosion has not yet been implemented.

## Notes

1. It would be good to ensure that in the step model the initial erosion phase doesn't result in erosion of the entire difference in Moho height. There should at least be a warning printed to the screen in these cases.
