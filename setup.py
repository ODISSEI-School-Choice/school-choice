#!/usr/bin/env python

from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="compass-school",
    version="v1.0.1",
    description="This Python implementation tries to model school choice and resulting school segregation based on the work of Schelling (1971) and Stoica & Flache (2014).",
    author="Eric Digum, Jisk Attenma, Ji Qi",
    author_email="e.p.n.dignum@uva.nl, j.attema@esciencecenter.nl, j.qi@esciencecenter.nl",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["compass"],
    python_requires=">=3.10",
    url="https://github.com/ODISSEI-School-Choice/school-choice",
    download_url="https://github.com/ODISSEI-School-Choice/school-choice/archive/refs/tags/v1.0.0.tar.gz",
    install_requires=[
        "bokeh",
        "hypothesis",
        "ijson",
        "Mesa",
        "pytest",
        "scikit_learn",
        "scipy",
        "seaborn",
        "dask",
        "distributed",
        "geopandas",
        "shapely"
    ],
)
