# Dealing with age data

T<sub>c</sub>1D supports the option to both [read in measured ages from a data file](#reading-age-data-from-a-file) and [write summary age output to a file](#writing-summary-age-data-to-a-file).

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

A summary of measured and predicted ages can be written to a file if either (1) age data are read from a file or (2) observed ages are passed to T<sub>c</sub>1D from the command line or in the `init_params()` function.
The output file will contain all of the information passed to T<sub>c</sub>1D (observed ages, uncertainties, eU/radius, sample ID) as well as the predicted age(s) and eU/radius values used for age prediction ((U-Th)/He ages only).
This can be handy to have in case you want to tweak plotting of the age data or further explore your results.
The file will be written to `csv/age_summary.csv`.

An example of the output that is written to the age data file can be found below:

```text
Age type, Observed age (Ma), Observed age stdev (Ma), Observed age eU (ppm), Observed age grain radius (um), Sample ID, Predicted age (Ma), Predicted age eU (ppm), Predicted age grain radius (um)
AHe,9.0,0.5,40.0,60.0,2025-DW001,7.45,40.0,60.0
AHe,12.0,1.0,120.0,90.0,2025-DW001,8.49,120.0,90.0
AFT,18.0,1.5,,,2025-DW001,13.03,,
ZHe,28.0,1.5,900.0,60.0,2025-DW002,22.0,900.0,60.0
ZHe,33.0,1.5,2000.0,80.0,2025-DW002,24.4,2000.0,80.0
ZHe,35.0,1.5,3200.0,55.0,2025-DW003,24.24,3200.0,55.0
ZFT,45.0,2.5,,,,43.95,,
```

**Note**: The ZFT age on the last line was missing a sample ID when the model was run. This is not a typo, this is the expected behavior.

### Nicer looking output

It is perhaps easier to read the output in table format, as it might look in [pandas](https://pandas.pydata.org/) or a spreadsheet program. An example is shown below:

| Age type | Observed age (Ma) | Observed age stdev (Ma) | Observed age eU (ppm) | Observed age grain radius (µm) | Sample ID  | Predicted age (Ma) | Predicted age eU (ppm) | Predicted age grain radius (µm) |
|----------|-------------------|-------------------------|-----------------------|--------------------------------|------------|--------------------|------------------------|---------------------------------|
| AHe      | 9.0               | 0.5                     | 40.0                  | 60.0                           | 2025-DW001 | 7.45               | 40.0                   | 60.0                            |
| AHe      | 12.0              | 1.0                     | 120.0                 | 90.0                           | 2025-DW001 | 8.49               | 120.0                  | 90.0                            |
| AFT      | 18.0              | 1.5                     |                       |                                | 2025-DW001 | 13.03              |                        |                                 |
| ZHe      | 28.0              | 1.5                     | 900.0                 | 60.0                           | 2025-DW002 | 22.0               | 900.0                  | 60.0                            |
| ZHe      | 33.0              | 1.5                     | 2000.0                | 80.0                           | 2025-DW002 | 24.4               | 2000.0                 | 80.0                            |
| ZHe      | 35.0              | 1.5                     | 3200.0                | 55.0                           | 2025-DW003 | 24.24              | 3200.0                 | 55.0                            |
| ZFT      | 45.0              | 2.5                     |                       |                                |            | 43.95              |                        |                                 |

### How to enable output of the summary age data file

#### Writing summary age output using the init_params() function

```python
init_params(write_age_output=True)
```

#### Writing summary age output from the command line

```text
./tc1d_cli.py --write-age-output
```