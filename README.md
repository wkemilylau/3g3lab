# 3G3 Lab

## Building the documentation for `lab.py` locally

To build the documentation for `lab.py`, you need to install a few python packages:

```sh
pip install sphinx sphinx_rtd_theme numpydoc
```

Once you have these packages installed, you can simply run

```sh
sphinx-build -b html . doc
````

and open the documentation at `doc/index.html` with your browser.



