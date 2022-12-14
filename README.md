# PX915_Reproducible_Result

Repo for PX915 summer project reproducible result. Full code for summer project is housed in the repo epoch_surra:

https://github.com/gosling123/epoch_surra.git

## Note :
The jupyter notebook will intialise 10 epoch runs simultaneously,
each running on 3 cores (30 in total). This will require you to use
significant computational resources (I.e hetmathsys node ) and can take several 
hours to complete.

Due to this run time, please make sure you keep the set intensity in the notebook 
at the default value, or change it before running the epoch simulations, otherwise if you want to run it 
again at a different intensity, the sample runs will take a further several hours.

## Python package dependencies:
#### (NOTE: sdf package comes with EPOCH. Instructions to install package are given in the Using EPOCH section, please do not do pip3 install sdf as this is a seprate package.)

* sdf (Warwick's own scientific data format - comes with EPOCH)
* time
* fileinput
* os
* sys
* multiprocessing
* GPy (1.10.0)
* matplotlib (3.1.1)
* numpy (1.17.3)
* scipy (1.3.1)
* sklearn (1.0.2)
* json (2.0.9)

## Reproducible Result Exercise
The jupyter notebook contained within this repository provides the details of the reproducible result exercise. Once you have set up EPOCH (instructions below) on your machine and cloned the software within this repository, you should be able to run the notebook. It'll take you through the various stages of the reproducible result. The notebook will execute the required epoch samples to run to extract the mean and variance of the laser reflectivity and then use the software within this repo to perform Gaussian Process regression on the training data supplied. At the end of the notebook, a simple UQ exercise is performed in which the predicted variance from the GP model is compared to the variance of the EPOCH samples ran via this notebook. 

## Acessing EPOCH
This project used EPOCH version 4.17.16, the latest releases can by found here https://github.com/Warwick-Plasma/epoch/releases.

#### Downloading
To download EPOCH using the tarball file, simply click the version you want to download it. Once you have moved it into your chosen directory, you can unpack the code using the command:
`tar xzfv epoch-4.17.16.tar.gz`, creating a directory called epoch-4.17.16.

## How to use this Repository in epoch

Clone this repository inside the epoch1d directory once you have a working copy of EPOCH on your machiene. The jupyter notebook will guide you through how the reflectivity is calculated and the GP process. If you want to see the Gaussian process code in more detail, then please look in the `Py_scripts folder`.

To make full use of the scripts, please add these paths to your `.bashrc`:

* `export EPOCH1D="$HOME/epoch-4.17.16/epoch1d"`
* `export PX915_RR="$EPOCH1D/PX915_Reproducible_Result"`
* `export EPOCH1D_EXE="$EPOCH1D/bin/epoch1d"`

### Using EPOCH
(More in depth information is found in the README of https://github.com/Warwick-Plasma/epoch and the epoch documentation https://epochpic.github.io/)
#### Compiling EPOCH
This project only uses the 1D version of the code, so this example will be for compiling epoch1d, however to do so for 2D and 3D is analagous to that of 1D.
To compile the 1D version of the code, you must first change to the correct directory (`epoch1d`) by typing `cd epoch-4.17.16/epoch1d`
The code is compiled using make, however a compiler must be specified. EPOCH is written using Fortran so a common compiler used is `gfortran`.
* To compile the code you type in the command `make COMPILER=gfortran`.
* To compile the code in parallel (for example using 4 processors) which saves a bit of time, you can use the command `make COMPILER=gfortran -j4` instead.
* You can also save typing by editing your ~/.bashrc file and adding the line export COMPILER=gfortran at the top of the file. Then the command would just be `make -j4`.

#### Compiling SDF
To access the SDF python package used in this repo, type the command `make sdfutils` in the chosen version of epoch you have previously compiled/want to use.
To check everything is set up corrcetly after it has compiled, type in the following commands in the terminal:
* `$ python3`
* `>>> import sdf`
* `>>> sdf.__file__`
Hopefully this will print out a path to your site-packages directory with the filename being of the form `sdf.cpython-37m-x86_64-linux-gnu.so`.

#### Running EPOCH
Once you have built the version of EPOCH you want (e.g 1D), you can simply run it using the command `echo <directory> | mpiexec -n <npro> ./bin/epoch1d`, where you simply replace `<directory>` with the name of the directory that houses the required `input.deck` file and is where the outputted data will be, and `<npro>` to the number of processors you want it to run on.

This reepo houses a shell script which will do this for you named `epoch.sh` which in which the directory and number of processors to use are controlled by an arg parser, as well as an optional to print the result to the file `run.log` rather than the terminal. An exmple is shown below for how to use the shell script to run epoch:
* `$ chmod u+x epoch.sh` (only required the first time)
* `$ ./epoch.sh <directory> <npro>` (prints to terminal)
* `$ ./epoch.sh <directory> <npro> log` (terminal print goes to run.log file)

#### input.deck file
To run epoch, you must have a suitable input file that epoch can read. This must be saved as 'input.deck' and an example of such a file is given in this repo under the same name. This file must always be in the directory you specify in command line, for if the input.deck file is not found epoch won't run. To edit the given input.deck file or create a new one, please read the epoch user manual for gidence.
https://github.com/Warwick-Plasma/EPOCH_manuals/releases/
