import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cc_mpi_py",
    version="0.0.1",
    author="Katie Biegel, Genevieve Savard",
    author_email="katherine.biegel@ucalgary.ca, genevieve.savard@ucalgary.ca",
    description="MPI4PY Cross-correlation Code for Pre-Downloaded Waveform file",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/savardge/cc_mpi_py",
    project_urls={
        "Bug Tracker": "https://github.com/savardge/cc_mpi_py/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "cc_mpi"},
    packages=setuptools.find_packages(where=""),
    python_requires=">=3.6",
)
