# Ironman

Ironman is a Python package for jointly fitting in-transit and out-of-transit radial velocities with photometric data. Its main objective is measuring the stellar obliquity of the orbit of a planet using the canonical Rossiter-McLaughlin effect.

# Dependencies

- numpy (pip install numpy)
- scipy (pip install scipy)
- pandas (pip install pandas)
- batman-package (pip install batman-package)
- rmfit (and its dependencies, install from here: https://github.com/gummiks/rmfit)
- dynesty (pip install dynesty)
- astropy (pip install astropy)

# Installation

You can install ironman with pip

```
pip install ironman-package
```
Or cloning the repository

```
git clone https://github.com/jiespinozar/ironman
cd ironman
pip install .
```

# Examples

To see examples of the usage of this module, see the example notebooks in the Examples folder.

- Example 1: How to fit data with ironman
- Example 2: How to simulate data (e.g., for proposals)

# Citation

If you make use of this code, please cite [Espinoza-Retamal et al. 2024](https://ui.adsabs.harvard.edu/abs/2023ApJ...958L..20E/abstract)
