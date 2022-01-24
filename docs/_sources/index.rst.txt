.. role:: python(code)
   :language: python

3G3 Lab
=======

In this lab, you will explore sparse coding in the primary visual cortex (V1).

Getting started
===============


The entry point for this lab is `3g3lab.ipynb`.
You can run it either locally or on Google Colab.
Running it locally has the advantage that you will not need internet access to run the code.
But for those of you who do not have `python3` installed, or find the process of installing the relevant packages difficult, running the notebook on Google Colab might be easier.

1. Running the notebook locally
-------------------------------

To run the notebook locally, you will need to install `git`, `python3`, and the relevant python packages if you haven't already. 
Then you can type the following into your command line to open the jupyter notebook

.. code-block:: sh

   git clone https://github.com/ghennequin/3g3lab
   cd 3g3lab
   jupyter notebook

The python packages that you will need are:

1. `numpy <https://numpy.org/install/>`_
2. `scipy <https://scipy.org/install.html>`_
3. `jupyter <https://jupyter.org/install>`_
4. `matplotlib <https://matplotlib.org/users/installing.html>`_

You can install all of these packages with:

.. code-block:: sh

   pip install numpy scipy jupyter matplotlib

if you are using pip, or

.. code-block:: sh

   conda install numpy scipy jupyter matplotlib

if you are using conda.

2. Running the notebook on Google Colab
---------------------------------------

To run the notebook on Google Colab, you can simply click on this `link <https://colab.research.google.com/github/ghennequin/3g3lab/blob/master/3g3lab.ipynb>`_ to open up the notebook in your browser. 
When you start running the notebook, you will be prompted to log in to your Google account. 
In section 0 of the notebook `Getting started`, make sure to change the python variable :python:`mode` from :python:`'local'` to :python:`'colab'`:

.. code-block:: python

   mode = 'colab'


Happy coding!

Feedback
========

If you run into any issues, spot any typos, find parts of the documentation that is unclear, or just have nay feedback in general, please do let us know (by email or during the office hours).
Your comments and suggestions will be most appreciated!


LAB API
=======

In the notebook, you will be using a python package called `lab`, specifically written for 3G3. 
You can find its API below.
To view the source code for each of these functions, click on the link `[source]` on the side of each function.

.. automodule:: lab
   :members:
