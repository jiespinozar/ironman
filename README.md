# Ironman

Ironman is a Python package for jointly fitting in-transit and out-of-transit radial velocities with photometric data, with the objective of measuring the obliquity of the orbit of a planet using the canonical Rossiter-McLaughlin effect.

# Dependencies

Required packages:

- numpy (pip install numpy)
- scipy (pip install scipy)
- pandas (pip install pandas)
- batman-package (pip install batman-package)
- rmfit (and its dependencies, install from here: https://github.com/gummiks/rmfit)
- dynesty (pip install dynesty)
- contextlib 
- multiprocessing
- astropy (pip install astropy)
- itertools

# Installation

You should be able to do the following:

```
git clone https://github.com/jiespinozar/ironman
cd ironman
python setup.py install
```

# Examples

To see examples of the usage of this module, see the example notebooks.

- Example 1: How to fit data with ironman
- Example 2: How to simulate data (e.g., for proposals)

# Citation

If you make use of this code, please cite [Espinoza-Retamal et al 2024](https://ui.adsabs.harvard.edu/abs/2023ApJ...958L..20E/abstract)
