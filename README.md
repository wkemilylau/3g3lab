# 3G3 Lab

The documentation can be found [here](https://ghennequin.github.io/3g3lab).

## Building the documentation for `lab.py` locally

To build the documentation for `lab.py`, you need to install a few python packages:

```sh
pip install sphinx sphinx_rtd_theme numpydoc
```

Once you have these packages installed, you can simply run

```sh
sphinx-build -b html . docs
````

and open the documentation at `docs/index.html` with your browser.



