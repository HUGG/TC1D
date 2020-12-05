# Erosion rates in the delamination 1D model

## Overview

There are several options for how erosion can be defined in the delamination 1D thermal models. In most cases these options use the difference in initial and final Moho depth (along with the total simulation time) to determine the magnitude of erosion. From there, the distribution of erosion rates with time can be determined. Planned (and implemented) options for the erosion rate calculation include:

1. Constant erosion rate
2. Constant rate with a step-function change at a specified time
3. Exponential decay
4. Elevation-dependent erosion (does not use Moho depths!)

Below is a general description of how erosion is implemented in the code as well as details about how each option works.

## General implementation

The calculation of erosion rates in the delamination 1D model is done in a function titled `calculate_erosion_rate`, which returns a single value `vx`, the erosion rate to be used for the given time step. The function definition statement is below, to give you a sense of the values that can be passed to the function:

```python
def calculate_erosion_rate(t_total, current_time, magnitude, type, erotype_opt1, erotype_opt2):
    """Defines the way in which erosion should be applied."""
```

The function expects the following values to be passed:

- `t_total`: The total model run time
- `current_time`: The current time in the model
- `magnitude`: The erosion magnitude (difference in Moho depths)
- `type`: The type of erosion model to be used.
  - `1` = Constant erosion rate
  - `2` = Constant rate with a step-function change at a specified time
  - `3` = Exponential decay
  - `4` = Elevation-dependent erosion
- `erotype_opt1`, `erotype_opt2`: Optional parameters depending on the selected erosion model

Details about the implementation of the erosion model options can be found below.

### Constant erosion rate

![Constant erosion rate model example](https://github.com/HUGG/delamination-1D/blob/main/png/cooling_hist_erotype1.png)<br/>
*Example cooling history for the constant erosion rate erosion model.*

The constant erosion rate case is the simplest option. Here, the return value `vx` is simply the erosion magnitude divided by the simulation time. In Python this is

```python
    # Constant erosion rate
    if type == 1:
        vx = magnitude / t_total
```

`erotype_opt1` and `erotype_opt2` are not used in this erosion model.

### Constant rate with a step-function change at a specified time

![Step-function change in erosion rate model example](https://raw.githubusercontent.com/HUGG/delamination-1D/main/png/cooling_hist_erotype2.png)<br/>
*Example cooling history for the constant rate with a step-function change at a specified time erosion model.*

The constant rate with a step-function change at a specified time model works similarly to the constant erosion case above, except that the user specifies the initial erosion rate and duration, while the second phase of erosion will be calculated based on the remaining erosion magnitude. Thus, there are some additional options in this model:

- `erotype_opt1` is the initial erosion rate
- `erotype_opt2` is the time of the transition in erosion rates

The code for this implementation can be found below.

```python
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
```

### Exponential decay

![Exponential decay in erosion rate model example](https://raw.githubusercontent.com/HUGG/delamination-1D/main/png/cooling_hist_erotype3.png)<br/>
*Example cooling history for the exponential decay erosion model.*

The exponential decay erosion model works by calculating a maximum erosion rate based on the characteristic time of decay, magnitude of erosion, and model run time. The user inputs the time over which the erosion rate should decay exponentially to $1/e$ times the original value and the code determines the erosion rate that will result in erosion of the difference in Moho depths over the simulation time. One erosion model option is used for this case:

- `erotype_opt1` is the characteristic time

The code for this implementation can be found below.

```python
    # Exponential erosion rate decay with a set characteristic time
    elif erotype == 3:
        decay_time = myr2sec(erotype_opt1)
        # Calculate max erosion rate for exponential
        max_rate = magnitude / (decay_time * (np.exp(0.0 / decay_time) - np.exp(-t_total / decay_time)))
        vx = max_rate * np.exp(-current_time / decay_time)
```

### Elevation-dependent erosion

Elevation-dependent erosion has not yet been implemented.

## Notes

1. It would be good to ensure that in the step model the initial erosion phase doesn't result in erosion of the entire difference in Moho height. There should at least be a warning printed to the screen in these cases.
