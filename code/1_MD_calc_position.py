#!/usr/bin/env python

import argparse
import pickle
import flax
import flax.core
import jax
import operator
import os
import time
from tqdm import tqdm

import ase
from ase.io import read
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

from neuralil.ase_integration_gai import NeuralILASECalculator
from neuralil.bessel_descriptors import PowerSpectrumGenerator
from neuralil.model_older import ResNetCore, NeuralILwithMorse
from neuralil.plain_ensembles.model import PlainEnsemblewithMorse

time_start=time.time()
print("Start：1_MD_calc_position.py")  # Print message indicating the script has started
PICKLE_FILE =f"../model/model_enery.pkl" # Path to the trained model for energy and force predictions
MAX_NEIGHBORS = 25  # Set the maximum number of neighbors to consider in the simulation
N_ensembles=5   # Define the number of ensemble runs for the simulation

# Load the machine learning model information from the pickle file
model_info = pickle.load(open(PICKLE_FILE, "rb"))

# Initialize the argument parser for handling command-line inputs
parser = argparse.ArgumentParser(
    description="Parameters for molecular dynamics simulations."
    )
parser.add_argument("target_t", help="target temperature [K]", type=float)
parser.add_argument(
    "-m",
    "--minimizer_tolerance",
    help="tolerance for the forces during the minimization",
    type=float,
    default=0.1,
)
parser.add_argument(
    "-e",
    "--equilibration_nsteps",
    help="number of steps for the NVT equilibration run",
    type=int,
    default=400000,
)
parser.add_argument(
    "-p",
    "--production_nsteps",
    help="number of steps for the NVE production run",
    type=int,
    default=400000,
)
parser.add_argument(
    "-f",
    "--friction",
    help="friction parameter for Langevin dynamics",
    type=float,
    default=2e-2,
)
parser.add_argument(
    "-d",
    "--delta_t",
    help="time step for the integrators [fs]",
    type=float,
    default=0.5,
)
parser.add_argument(
    "-i",
    "--trajectory_interval",
    help="number of steps between writes to the trajectory file",
    type=int,
    default=2,
)
parser.add_argument(
    "-n",
    "--name",
    help="directory of running structure",
    type=str,
    default='../inputs/XYZ',
)
args = parser.parse_args()

# Define paths for storing results of the equilibration and VelocityVerlet runs
equilibration_path = f'../intermediate_results/equilibration'
VelocityVerlet_path = f'../intermediate_results/VelocityVerlet'
if not os.path.exists(equilibration_path):
    os.makedirs(equilibration_path) 
if not os.path.exists(VelocityVerlet_path):
    os.makedirs(VelocityVerlet_path) 

# "Extract a list of individual parameters sets from the ensemble parameters."        
def unpack_params(model_params):
    n_models = N_ensembles
    nruter = []
    for i_model in range(n_models):
        subparams = jax.tree_map(operator.itemgetter(i_model), model_params)
        individual_params = dict(params=subparams["params"]["neuralil"])
        nruter.append(flax.core.freeze(individual_params))
    return nruter
in_params=unpack_params(model_info.params)

# Create the object used to compute descriptors...   <|
descriptor_generator = PowerSpectrumGenerator(
    model_info.n_max,
    model_info.r_cut,
    len(model_info.sorted_elements),
    MAX_NEIGHBORS,
)

# Define the immutable model object for the individual neural network
individual_model = NeuralILwithMorse(
    len(model_info.sorted_elements),
    model_info.embed_d,
    model_info.r_cut,
    descriptor_generator.process_data,
    descriptor_generator.process_some_data,
    ResNetCore(model_info.core_widths),
)

# Combine the individual models into an ensemble model
dynamics_model = PlainEnsemblewithMorse(
    individual_model, N_ensembles
)
for root, path, files in os.walk(str(args.name)):
  for file_one in tqdm(files, desc="Performing MD simulation for molecules"):
    time_start_one=time.time()
    if not file_one.endswith('.xyz'): continue
    # Read input file and define the output file name suffix
    file_i = os.path.join(root, file_one)    
    tail=f'{float(args.target_t)}K_{str(os.path.basename(file_i).split(".")[0])}'
    print('\n run file_i:',file_i)

    # Time step in atomic simulation units (converted from femtoseconds)
    dt = args.delta_t * units.fs

    # Initialize the ASE (Atomic Simulation Environment) calculator
    calculator = NeuralILASECalculator(dynamics_model, model_info, MAX_NEIGHBORS)

    # Attach the calculator to the atomic structure object
    atoms = read(file_i)
    #atoms.set_calculator(calculator) # Old method: Set the calculator to the atoms system, deprecated (for older JAX versions)
    atoms.calc = calculator

    # Initialize the velocities at random.
    MaxwellBoltzmannDistribution(atoms, temperature_K=float(args.target_t))

    # Create a Langevin integrator to take the system to the desired temperature.
    integrator = Langevin(
        atoms, dt, temperature_K=float(args.target_t), friction=args.friction
    )

    # Open a file to write equilibration trajectory data
    fw = open(f'{equilibration_path}/equilibration_{tail}.xyz', 'w')

    # Define a callback function to print and record MD information during equilibration
    def print_md_info(current_atoms=atoms, current_integrator=integrator):
        """
        Callback to log molecular dynamics (MD) information and write it to the trajectory file.

        Args:
            current_atoms: ASE Atoms object, represents the current atomic structure.
            current_integrator: ASE integrator object, manages the MD simulation.
        """
        progress_bar.update(args.trajectory_interval)
        
        n_atoms = len(current_atoms)
        e_total=current_atoms.get_potential_energy()
        e_pot_per_atom = e_total / n_atoms
        e_kin_per_atom = current_atoms.get_kinetic_energy() / n_atoms
        kinetic_temperature = e_kin_per_atom / (1.5 * units.kB)

        formula = current_atoms.symbols
        str_type = current_atoms.get_chemical_symbols()
        coordinates = current_atoms.get_positions()
        
        # Write atomic data and MD information to the equilibration file
        fw.write(str(n_atoms) + '\n')
        fw.write(f'Step #{current_integrator.nsteps + 1}: {str(formula)},  E_tot={e_total} eV, T={kinetic_temperature} K \n')
        for i in range(len(str_type)):
            fw.write(str(str_type[i]) + ' ' + ' '.join('%.5f' % c for c in coordinates[i]) + '\n')
        
        # Check for unphysical temperature values to avoid simulation instability
        if abs(kinetic_temperature) > 30000:    
            print(f'\n {file_i} is dispersion')
            print(f'Machine learning model cannot compute {file_i}')
            exit()
    
    print(f"Starting NVT equilibration simulation for molecule {file_i} with {args.equilibration_nsteps} steps...")
    # Add a progress bar to track the NVT equilibration progress for the molecule
    progress_bar = tqdm(total=args.equilibration_nsteps, desc=f" {file_i}: NVT Equilibration Progress", unit="steps")    
    # Attach the callback function to the integrator with a specific interval
    integrator.attach(print_md_info, interval=args.trajectory_interval)
    # Run the equilibration simulation for the specified number of steps
    integrator.run(args.equilibration_nsteps)
    progress_bar.close()
    fw.close()

    # Set up and launch the production run using the VelocityVerlet integrator
    integrator = VelocityVerlet(atoms, dt)
    n_species = len(atoms.calc.symbol_map)
    flux_file = open(f"{VelocityVerlet_path}/VelocityVerlet_{tail}.xyz", "w") 

    # Define a callback function to record MD information during the production run
    def calc_Position(current_atoms=atoms, current_integrator=integrator):
        
        progress_bar.update(args.trajectory_interval)
        n_atoms = len(current_atoms)
        e_total=current_atoms.get_potential_energy()
        e_pot_per_atom = e_total / n_atoms
        e_kin_per_atom = current_atoms.get_kinetic_energy() / n_atoms
        kinetic_temperature = e_kin_per_atom / (1.5 * units.kB)
        
        formula = current_atoms.symbols
        str_type = current_atoms.get_chemical_symbols()
        coordinates = current_atoms.get_positions()

        flux_file.write(str(n_atoms) + '\n')
        flux_file.write(f'Step #{current_integrator.nsteps + 1}: {str(formula)},  E_tot={e_total} eV, T={kinetic_temperature} K \n')
        for i in range(len(str_type)):
            flux_file.write(str(str_type[i]) + ' ' + ' '.join('%.5f' % c for c in coordinates[i]) + '\n')
        
        if abs(kinetic_temperature) > 30000:    
            print(f'\n {file_i} is dispersion')
            print(f'Machine learning model cannot compute {file_i}')
            exit()    
        
    # Run the production simulation for the specified number of steps
    print(f"Starting NVE simulation for molecule {file_i} with {args.production_nsteps} steps...")
    progress_bar = tqdm(total=args.production_nsteps, desc=f" {file_i}: NVE Simulation Progress", unit="steps")
    integrator.attach(calc_Position, interval=args.trajectory_interval)
    integrator.run(args.production_nsteps)
    progress_bar.close()
    flux_file.close()
    
    time_end_one=time.time() 
    time_diff = time_end_one - time_start_one
    hours = time_diff // 3600
    minutes = (time_diff % 3600) // 60
    seconds = time_diff % 60 
    print(f"MD Simulation for {file_i} completed",
      "time cost: ",f"{int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")

time_end=time.time() 
time_diff = time_end - time_start
hours = time_diff // 3600
minutes = (time_diff % 3600) // 60
seconds = time_diff % 60 
print("End：1_MD_calc_position.py",
      "time cost: ",f"{int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")  # Indicate that the script has completed