User guide
==========

.. _installation:

Installation
------------

You can install T\ :sub:`c`\ 1D in your Python environment using ``pip``.

.. code-block:: console

   pip install tc1d

.. note::

   In order for T\ :sub:`c`\ 1D to work properly you will also need to install the thermochronometer age prediction programs available at https://github.com/HUGG/Tc_core.

Running a model
---------------

An example model with 10 km of exhumation and default values can be run from the command line as follows:

.. code-block:: console

   tc1d-cli --ero-option1 10.0

Configuring a model
-------------------

A full list of options that can be used with T\ :sub:`c`\ 1D can be found by running the code with the ``--help`` flag (or no specified flags):

.. code-block:: console

   tc1d-cli --help

This will return a usage statement and list of flags the code accepts.

Additional details about code options
-------------------------------------

.. toctree::
   :maxdepth: 1
   :caption: Detailed usage instructions

   age-data
   erosion-models