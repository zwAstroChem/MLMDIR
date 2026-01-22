# MLMDIR
A machine-learning molecular dynamics (MLMD) code to compute anharmonic infrared spectra of PAHs

The anharmonic IR spectra are computed using a machine-learning-based molecular dynamics approach:

1. Molecular dynamics (MD) simulations are performed using atomic forces predicted by a neural-network force field.

2. Molecular dipole moments are evaluated along the MD trajectory using a machine-learning dipole model.

3. The dipole time-autocorrelation function is Fourier-transformed to obtain the IR absorption cross-section spectrum.

# Machine-Learning Models

The MLMD framework employs two neural-network models:

1. Neural Network Force Field (NNFF) 
   Used to construct the potential energy surface and predict atomic forces.  
   DOI: 10.1021/acs.jcim.1c01380

2. Electron Passing Neural Network (EPNN) 
   Used to predict molecular dipole moments along the MD trajectory.  
   DOI: 10.1021/acs.jcim.0c01071
   
Both machine-learning models are pre-trained and can be used directly without additional training.

# Installation

We recommend using Anaconda to manage the environment.

1. Create and activate the environment

conda create -n MLMD_env python=3.10.13  # Create a virtual environment
conda activate MLMD_env  # Activate the virtual environment

2. Install required packages

The machine-learning components required for the MLMD framework are provided as three compressed ZIP archives located in the ./lib/ directory.
These three packages together provide all model implementations and core dependencies required for MLMD simulations.

Please extract each archive first, then install the packages locally in the following order.
(1) Learned optimizer (required dependency)
	cd lib
	unzip learned_optimization.zip
	cd learned_optimization
	pip install -e .
(2) Neural Network Force Field
	cd lib
	unzip bessel-nn-potentials-velo.zip
	cd bessel-nn-potentials-velo
	pip install -e .
(3) Electron Passing Neural Network
	cd lib
	unzip epnn-main.zip
	cd epnn-main
	pip install -e .

3. Install additional dependencies

pip install flax==0.7.4

pip install optax==0.1.7

pip install orbax-checkpoint==0.4.1

pip install numpy==1.26.1

pip install scipy==1.11.3

pip install chex==0.1.84

pip install oryx==0.2.7
	
Note: You may encounter version incompatibility warnings, which can be safely ignored.

4. Install JAX and JAXLib (CPU version):

pip install jax[cpu]==0.4.19 -f https://storage.googleapis.com/jax-releases/jax_releases.html

pip install jaxlib==0.4.19 -f https://storage.googleapis.com/jax-releases/jax_releases.html
	
Note: If version incompatibilities occur, you can ignore them. For specific library versions, refer to the environment.yaml file. After installation, use pip list to confirm JAX and JAXLib are both version 0.4.19. If not, repeat the installation Step 4.

# Running the Program

1. Data Preparation:

Place the .xyz files for the molecules you wish to compute in the ./inputs/XYZ/ directory.
The .xyz file format consists of:
Line 1: Total number of atoms.
Line 2: Comment line.
Lines 3 and beyond: Each line contains the atom type (e.g., C for carbon) and its Cartesian coordinates (x, y, z) in Angstrom.
Example: C10H8_330.xyz.

2. MLMD Calculations

On Linux:

Modify the TEMPERATURE variable on line 4 of code/run_calc.sh to set the desired simulation temperature.

Run the script: 
./code/run_calc.sh

This script performs the following: 

a) MD Simulations: The script runs code/1_MD_calc_position.py to perform MD simulations, generating atomic trajectories at the specified temperature. Progress bars will indicate the current molecule being processed and the simulation phase (NVT and NVE). The simulation for molecules like C10H8 takes around 16 hours on a single core.

b) Dipole Moment Calculation: The script runs code/2_calc_dipole.py to calculate time-evolved dipole moments based on atomic trajectories. Progress bars will display the progress for 400,000 time steps. For molecules like C10H8, this step takes approximately 32 hours on a single core.

c) IR Spectrum Calculation: The script runs code/3_calc_IR.py to compute the IR spectrum using a Fourier transform of the dipole time-autocorrelation function. Each molecule's IR spectrum calculation typically completes in a few seconds on a single core.

On Windows:

Navigate to the code/ directory and run the following commands:

python 1_MD_calc_position.py 50  # Set temperature to 50 K

python 2_calc_dipole.py 50 

python 3_calc_IR.py 50 --qm 0 #qm: Quantum correction level (0â€“3) for normalized IR absorption cross-section intensity (DOI:10.1021/jp034788u). Default: 0 (no correction).

Each python program will process each .xyz file in the ./inputs/XYZ/ directory one by one.

# Output Files

1. IR Spectrum:
   
The computed IR spectrum will be saved in the ./outputs/IR_txt/ folder.

Example: ./outputs/IR_txt/C10H8_330_qm0.txt contains two columns:
Column 1: Wavenumber (Freq_MD) in cm^-1
Column 2: Normalized IR absorption cross-section intensity (Inten_MD), where all spectral line intensities are scaled by the maximum intensity so that the peak intensity equals 1 (dimensionless).
A visualization of the IR spectrum is saved as C10H8_330_qm0.jpg, with the X-axis representing the wavenumber in cm^-1 and the Y-axis representing the normalized IR absorption cross-section intensity (dimensionless), scaled so that the maximum intensity equals 1.

2. Intermediate Results
   
Atomic Trajectories: Saved in intermediate_results/equilibration/ (for NVT) and intermediate_results/VelocityVerlet/ (for NVE). Each line includes: Step Number: The timestep number. Total Energy: The total energy of the molecule at this timestep, in electron volts (eV). Temperature: The temperature at this timestep, in Kelvin (K). Atomic positions (X, Y, Z) for each atom.

Dipole Moments: Saved in intermediate_results/Dipole_txt_mass/, with each row containing the dipole moment components (dipole_x, dipole_y, dipole_z) at a specific time step. The dipole moment components are given in units of e*Angstrom.

# Limitations

The model is currently trained only on neutral PAHs and does not support charged molecules or isotopologues. Predictions for molecules that significantly differ from the training dataset may have increased uncertainty. 

#Citation

If you use this code, please cite:
'Computing Anharmonic Infrared Spectra of Polycyclic Aromatic Hydrocarbons Using Machine-Learning Molecular Dynamics', DOI: https://doi.org/10.1093/mnras/staf1156

#License

The code is licensed under the Apache License 2.0, allowing for free use, modification, and distribution, including for commercial purposes, while requiring proper attribution and inclusion of the original license.
