Getting started
===============

.. _installation:

Installation (for most users)
-----------------------------

You can install T\ :sub:`c`\ 1D in your Python environment using ``pip``.

.. code-block:: console

   pip install tc1d

.. warning::

   If you choose to install T\ :sub:`c`\ 1D versions older than 0.5.0, you will also need to install the thermochronometer age prediction programs available in the https://github.com/HUGG/Tc_core in order for T\ :sub:`c`\ 1D to work properly. Installing T\ :sub:`c`\ 1D >= 0.5.0 is strongly recommended.

Installation (for parallel MCMC inversions)
-------------------------------------------

You can include support for parallel Markov-Chain Monte Carlo (MCMC) inversions in T\ :sub:`c`\ 1D by installing using the following options in ``pip``.

.. code-block:: console

   pip install 'tc1d[mpi]'

.. note::

   This should work in Linux and macOS, but you may need to install OpenMPI or MPICH for MPI support. In Windows, MPI is available via ms-mpi or IntelMPI, but you will need to install that software manually for MPI support.

Running a basic model
---------------------

After installation, an example model with 10 km of exhumation and default values for all other parameters can be run from the command line as follows:

.. code-block:: console

   tc1d-cli --ero-option1 10.0

.. note::

   It is also possible to define parameters and run T\ :sub:`c`\ 1D using Python function calls or an input file. These cases are described in the user guide.

Model options
-------------

A full list of options that can be used with T\ :sub:`c`\ 1D can be found by running the code with the ``--help`` flag (or no specified flags):

.. code-block:: console

   tc1d-cli --help

This will return a usage statement and list of flags the code accepts.
