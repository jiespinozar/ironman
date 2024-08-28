from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='ironman-package',
    version='1.0.3',
    author='Juan Ignacio Espinoza-Retamal',
    author_email='jiespinozar@uc.cl',
    description='Joint Fit Rossiter McLaughlin Data with photometry and out-of-transit radial velocities',
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/jiespinozar/ironman',
    packages=find_packages(),
    install_requires=['numpy','scipy','pandas','batman-package','dynesty','astropy','rmfit>=1.0.2','tqdm'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
    keywords='Astronomy, Obliquities, Radial Velocities',
    license='MIT',
    include_package_data=True,
    python_requires='>=3.6',
)
