User guide
==========

.. _installation:

Installation
------------

1. To get started using T\ :sub:`c`\ 1D you can either clone or download the source code from https://github.com/HUGG/TC1D.

2. In order to use the code, you should first compile the thermochronometer age prediction codes in the ``c`` and ``cpp`` directories. From the base code directory you can do the following in a terminal:

   .. code-block:: console

      cd c
      make && make install
      cd ..

      cd cpp
      make && make install
      cd ..

   This will build the age prediction programs and install them in the ``bin`` directory. Note that you may need to edit the ``Makefile`` in the ``c`` and ``cpp`` subdirectories to specify your compilers.

Running a model
---------------

An example model with 10 km of exhumation and default values can be run from the command line as follows:

.. code-block:: console

   cd py
   ./tc1d_cli.py --ero-option1 10.0

Configuring a model
-------------------

A full list of options that can be used with T\ :sub:`c`\ 1D can be found by running the code with no specified flags:

.. code-block:: console

   ./tc1d_cli.py

This will return a usage statement and list of flags the code accepts.

Additional details about code options
-------------------------------------

.. toctree::
   :maxdepth: 1
   :caption: Detailed usage instructions

   age-data
   erosion-models