# Ironman

Ironman is a Python package for jointly fitting in-transit and out-of-transit radial velocities with photometric data. Its main objective is measuring the stellar obliquity of the orbit of a planet using the canonical Rossiter-McLaughlin effect.

# Dependencies

- rmfit (and its dependencies, install from here: https://github.com/gummiks/rmfit)
- numpy (pip install numpy)
- scipy (pip install scipy)
- pandas (pip install pandas)
- batman-package (pip install batman-package)
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

- Example 1: How to fit data with ironman (sky-projected obliquity)
- Example 2: How to simulate data (e.g., for proposals)
- Example 3: How to fit data with ironman (true 3D obliquity)

# Citation

If you use this code, please cite [Espinoza-Retamal et al. 2024](https://ui.adsabs.harvard.edu/abs/2023ApJ...958L..20E/abstract)

If you use the RM fitting capability, please cite [Stefansson et al. 2022](https://ui.adsabs.harvard.edu/abs/2022ApJ...931L..15S/abstract) and [Hirano et al. 2011](https://ui.adsabs.harvard.edu/abs/2011ApJ...742...69H/abstract)

If you use the transit fitting capability, please cite [Kreidberg et al. 2015](https://ui.adsabs.harvard.edu/abs/2015ascl.soft10002K/abstract)

If you use the RV fitting capability, please cite: [Fulton et al. 2018](https://ui.adsabs.harvard.edu/abs/2018PASP..130d4504F/abstract)

