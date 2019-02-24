correlationMatrix
=========================

correlationMatrix is a Python powered library for the statistical analysis and visualization of correlation phenomena.
It can be used to analyze any dataset that captures timestamped correlations in a discrete state space.
Use cases include credit rating correlations, system state event logs and more.

* Author: `Open Risk <http://www.openriskmanagement.com>`_
* License: Apache 2.0
* Code Documentation: `Read The Docs <https://correlationmatrix.readthedocs.io/en/latest/>`_
* Mathematical Documentation: `Open Risk Manual <https://www.openriskmanual.org/wiki/correlation_Matrix>`_
* Training: `Open Risk Academy <https://www.openriskacademy.com/login/index.php>`_
* Development Website: `Github <https://github.com/open-risk/correlationMatrix>`_
* Discussion: `Gitter <https://gitter.im/open-risk/correlationMatrix>`_

Functionality
-------------

You can use correlationMatrix to

- Estimate correlation matrices from historical event data using a variety of estimators
- Visualize event data and correlation matrices
- Characterise correlation matrices
- Manipulate correlation matrices (derive generators, perform comparisons, stress correlation rates etc.)
- Access standardized datasets for testing

**NB: correlationMatrix is still in active development. If you encounter issues please raise them in our
github repository**

Architecture
------------

* transitioMatrix supports file input/output in json and csv formats
* it has a powerful API for handling event data (based on pandas)
* provides intuitive objects for handling correlation matrices individually and as sets (based on numpy)
* supports visualization using matplotlib

Links to other open source software
-----------------------------------

- Duration based estimators are similar to etm, an R package for estimating empirical correlation matrices
- There is some overlap with lower dimensionality (survival) models like lifelines

Installation
=======================

You can install and use the correlationMatrix package in any system that supports the `Scipy ecosystem of tools <https://scipy.org/install.html>`_

Dependencies
-----------------

- correlationMatrix requires Python 3
- It depends on numerical and data processing Python libraries (Numpy, Scipy, Pandas)
- The Visualization API depends on Matplotlib
- The precise dependencies are listed in the requirements.txt file.
- correlationMatrix may work with earlier versions of these packages but this has not been tested.

From PyPi
-------------

.. code:: bash

    pip3 install pandas
    pip3 install matplotlib
    pip3 install correlationMatrix

From sources
-------------

Download the sources to your preferred directory:

.. code:: bash

    git clone https://github.com/open-risk/correlationMatrix


Using virtualenv
----------------

It is advisable to install the package in a virtualenv so as not to interfere with your system's python distribution

.. code:: bash

    virtualenv -p python3 tm_test
    source tm_test/bin/activate

If you do not have pandas already installed make sure you install it first (will also install numpy)

.. code:: bash

    pip3 install pandas
    pip3 install matplotlib
    pip3 install -r requirements.txt

Finally issue the install command and you are ready to go!

.. code:: bash

    python3 setup.py install

File structure
-----------------
The distribution has the following structure:

| correlationMatrix         The library source code
|    model.py              Main data structures
|    estimators            Estimator methods
|    utils                 Helper classes and methods
|    thresholds            Algorithms for calibrating AR(n) process thresholds to input correlation rates
|    portfolio_model_lib   Collection of portfolio analytic solutions
| examples                 Usage examples
| datasets                 Contains a variety of datasets useful for getting started with correlationMatrix
| tests                    Testing suite

Testing
----------------------

It is a good idea to run the test-suite. Before you get started:

- Adjust the source directory path in correlationMatrix/__init__ and then issue the following in at the root of the distribution
- Unzip the data files in the datasets directory

.. code:: bash

    python3 test.py

Getting Started
=======================

Check the Usage pages in this documentation

Look at the examples directory for a variety of typical workflows.

For more in depth study, the Open Risk Academy has courses elaborating on the use of the library

- Analysis of Credit Migration using Python correlationMatrix: https://www.openriskacademy.com/course/view.php?id=38

