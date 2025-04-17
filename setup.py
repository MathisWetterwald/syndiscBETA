from setuptools import setup, find_packages

setup(name="syndiscBETA",
        packages=find_packages(),
        provides=["syndiscBETA"],
        version="0.1",
        author="Mathis Wetterwald and Pedro Mediano",
        url="https://www.information-dynamics.org",
        install_requires=[
            "dit>=1.5",
            "pypoman",
            "cvxopt",
            "numpy",
            "scipy",
            "networkx>=3.1",
            "matplotlib",   # Actually required by pypoman
            ],
        long_description=open("README.md").read(),
        license="BSD",
        classifiers=[
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python :: 3.3",
            "Programming Language :: Python :: 3.4",
            "Programming Language :: Python :: 3.5",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Mathematics",
            "Topic :: Scientific/Engineering :: Physics",
            ],
        )
