# Blending Pipeline Analysis Tool for Hydrogen

[![DOI 10.11578/dc.20240216.1](https://img.shields.io/badge/DOI-10.11578%2Fdc.20240216.1-brightgreen?link=https://doi.org/10.11578/dc.20240216.1)](https://www.osti.gov/biblio/2309692)
[![DOI 10.11578/dc.20240216.1](https://img.shields.io/badge/DOI-10.2172%2F2583524-brightgreen?link=https://doi.org/10.2172/2583524)](https://www.osti.gov/biblio/2583524)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python](https://img.shields.io/badge/python-3.10_|_3.11_|_3.12_|_3.13-blue)]()

[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

Welcome to the Blending Pipeline Analysis Tool for Hydrogen (BlendPATH) [SWR-24-10](https://www.osti.gov/doecode/biblio/117221). 

# User Manual
Please refer to the [BlendPATH User Manual](https://docs.nlr.gov/docs/fy25osti/87790.pdf) for in-depth documentation. The file is also available in the repository linked [here](user_manual/BlendPATH_User_Manual.pdf).

# Installation
This is a Python-based tool. The instructions below assumes that the user possesses background knowledge in Python and has Anaconda installed. Below is a quick-start guide:

The BlendPATH repository can by cloned to your local computer using:
```console
git clone https://github.com/natlabrockies/BlendPATH.git
```

## Create environment
A conda environoment can useful for satisfying the Python requirement. First, open an anaconda powershell. Within the powershell, we will first create a conda environment to install BlendPATH in. This command below creates a conda environment called `BlendPATH_env`. The second argument is the Python version. The BlendPATH team recommends Python 3.10 or 3.11.  Note: you may swap `BlendPATH_env` with whatever you would like to name the environment

```console
conda create -n BlendPATH_env python=3.10
```

Within the same powershell, enter into the conda environment using

```console
conda activate BlendPATH_env
```

## Installation
BlendPATH is built as a Python package that can be installed with pip. Once the conda environment is activated, navigate the directory where the BlendPATH repository was cloned. In most cases this will be in a designated GitHub folder.

```console
cd \Users\MY_USERNAME\Documents\GitHub\BlendPATH
```

Then install BlendPATH using the command below. Be sure to include the period at the end.

```console
pip install .
```

# VS Code setup
We recommend using VS Code for your Python IDE. Within VS Code, open your working directory for BlendPATH.

Within VS Code you will need to activate the conda environment that was created earlier. This can be achieved by pressing `Ctrl + Shift + P`
which brings up a menu window. Here type and select 
`Python: Select Interpreter`

This will bring up a list of your available environments. Select the environment that was just created. Check the file path matches what you expect.

# Quick-Start guide

BlendPATH can be run in a Python script by importing the package. There is also an example script to show usage in the [examples](examples) directory. Within this directory is a [template.py](examples/template.py) file. This uses the data files in the [wangetal2018](examples/wangetal2018) directory,

# License terms - ProFAST
By downloading and using BlendPATH, the user also agrees to the [BlendPATH license terms](LICENSE) as well as the license terms of [ProFAST](https://github.com/natlabrockies/ProFAST/blob/main/LICENSE).

# NOTE: 
BlendPATH continues to see active development and maintenance. If you observe issues with the code, please inform the authors of said issues for continuous improvement. Lastly, please include version number when referencing or citing BlendPATH. 

# Citation

```bibtex
   @techreport{kee2024BlendPATH,
     title = {{BlendPATH (Blending Pipeline Analysis Tool for Hydrogen) [SWR-24-10]}},
     author = {Kee, Jamie and Reznicek, Evan and Chung, Mark and Topolski, Kevin},
     year = {2024},
     month = jan,
     number = {BlendPATH},
     institution = {{National Renewable Energy Laboratory (NREL), Golden, CO (United States)}},
     doi = {10.11578/dc.20240216.1},
     url = {https://www.osti.gov/biblio/2309692},
    }
```
