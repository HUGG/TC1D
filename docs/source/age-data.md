# Dealing with age data

T<sub>c</sub>1D supports the option to both [read in measured ages from a data file]() and [write summary age output to a file]().

## Reading age data from a file

The age data file format and how to read in a data file are described below.

### Age data file format

T<sub>c</sub>1D expects the age data file you use to have the following format:

- It should be a plain text file with commas separating the values in the file
- It should contain 3–6 columns:

    1. The age types (AHe, AFT, ZHe, or ZFT)
    2. The ages in Ma
    3. The age uncertainties (standard deviations) in Myrs
    4. (*optional*) The eU concentration in parts per million
    5. (*optional*) The effective spherical grain radius in micrometers
    6. (*optional*) A sample ID 

- The text file should include a header as the first line

An example of the file format can be found in the file `sample_data.csv` in the [T<sub>c</sub>1D GitHub repository](https://github.com/HUGG/TC1D).
The contents of that file are also shown below, for convenience.

```text
Age type, Age (Ma), Standard deviation (Ma), eU concentration (ppm), Grain radius (um), Sample ID
AHe, 9.0, 0.5, 40.0, 60.0, 2025-DW001
AFT, 18.0, 1.5, , , 2025-DW001
ZHe, 28.0, 1.5, 900.0, 60.0, 2025-DW001
ahe, 12.0, 1.0, 120.0, 90.0, 2025-DW002
ZHe, 33.0, 1.5, 2000.0, 80.0, 2025-DW002
ZHe, 35.0, 1.5, 3200.0, 55.0, 2025-DW003
ZFT, 45.0, 2.5
```

#### Notes about the age file

- The eU and grain radius values are currently only supported for AHe and ZHe ages.
- Any He age lacking either an eU or radius value will use the defaults for those values.
- The sample ID is not required.
- If a sample ID is listed for a fission-track sample, it should be preceded by three commas, as the expectation is that the sample ID is in column 6 of the comma-separated values data file. Line 3 in the data file example above shows the required format.

### How to use your own data file

To use your own text file you should ensure a copy of it is in the directory where you are running T<sub>c</sub>1D (or you give the complete path to the file).
You can enable reading of your age data file using the `obs_age_file` parameter if using the `init_params()` function or using the `--obs-age-file` command-line flag.
Examples of both can be found below.

#### Reading an age data file using the init_params() function

```python
init_params(obs_age_file="sample_data.csv")
```

#### Reading an age data file from the command line

```text
./tc1d_cli.py --obs-age-file sample_data.csv
```

## Writing summary age data to a file

A summary of measured and predicted ages can be written to a file if either age data are read from a file or observed ages are passed to T<sub>c</sub>1D from the command line or in the `init_params()` function.
