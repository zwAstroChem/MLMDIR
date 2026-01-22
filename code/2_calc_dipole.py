#!/usr/bin/env python

import argparse
import os
import re
import pickle
import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple
from neuralil.bessel_descriptors import PowerSpectrumGenerator
from neuralil.model import  ResNetCore                      
from epnn.model import EPNN, MLP
from epnn.preprocessing import create_batched_graph_list
from epnn.utils import get_nodes_from_graph
import sys
import time as TIME
from tqdm import tqdm

# Disable pre-allocation of GPU memory in JAX to avoid memory issues.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Function to read molecular coordinates from a given file.
def read_coord(file):
    fp = open(file,'r') 
    lines = fp.readlines()
    
    # Define sorted element types and map each type to a unique index.
    sorted_elements = sorted(['C', 'H','O','N'])
    symbol_map = {s: i for i, s in enumerate(sorted_elements)}
    
    # Read the number of atoms and initialize cell and charge arrays.
    natom = int(lines[0].strip().split()[0])
    cell  = jnp.zeros((3,3),dtype=jnp.float32)
    init_charges = jnp.zeros(natom, dtype=float)
    init_charges = jnp.expand_dims(init_charges,axis=-1) 
    type_str = []
    type_num = []
    mol_weight = []
    
    # Parse atomic types and properties from the file.
    for line in lines[2:2+natom]:
        line = line.strip().split()
        at = line[0]
        type_str.append(at)
               
        if at == 'C':
            type_num.append(6)
            mol_weight.append(12.011)
        elif at == 'H':
            type_num.append(1)
            mol_weight.append(1.008)
        elif at == 'O':
            type_num.append(8)
            mol_weight.append(15.999)
        elif at == 'N':
            type_num.append(7)
            mol_weight.append(14.007)
        elif at == 'Si':
            type_num.append(14)
            mol_weight.append(28.085)
        elif at == 'Mg':
            type_num.append(12)
            mol_weight.append(24.305)
        elif at == 'Fe':
            type_num.append(26)
            mol_weight.append(55.845)
        else:
            print(file, f'unexpected atom type! {at}')
            
    type_train = [symbol_map[s] for s in type_str] 
    Tol_step = len(lines) // (2+natom)
    cell_list = []
    type_train_list = []
    position_list = []
    mol_weight_list = []
    init_charges_list = []
    mol_weight0 = jnp.array(mol_weight)
    
    for step in range(0,Tol_step):
        position = []
        for line in lines[2+(2+natom)*step:2+natom+(2+natom)*step]:
            line = line.strip().split()
            position.append(list(map(float, line[1:4])))
        
        # Convert positions to JAX array and compute center of mass.
        position0 = jnp.array(position)
        center_of_mass = jnp.sum(position0.T * mol_weight0, axis=1) / jnp.sum(mol_weight0)
        position_center = position0 - center_of_mass
        cell_list.append(cell)
        type_train_list.append(type_train)
        position_list.append(position_center)
        mol_weight_list.append(mol_weight)
        init_charges_list.append(init_charges)
    
    fp.close()
    return (cell_list,type_train_list,position_list,init_charges_list)

# Function to check if a simulation file has completed successfully.
def juge_file(filename,group=200000,T_max=50000):
    
    flag = False
    fr = open(filename, 'r')
    lines = fr.readlines()
    row_count = int(lines[0]) + 2
    group_count = int(len(lines) // row_count)
    
    # Loop through each group to check the temperature and group count.
    for group_i in range(0,group_count):
        index_i = 1 + row_count*group_i
        line_T = lines[index_i].strip()
        match = re.search(r"T=(\d+\.\d+)", line_T)
        if match:
            T_i = float(match.group(1))
            if abs(T_i) > T_max:
               flag = True 
               break
        else:
            continue
    
    # If the number of groups is less than the expected value, mark as incomplete.
    if group_count < group:
        flag = True

    fr.close()
    if flag:
        print(filename,'is not finish')
    
    return flag

time_start=TIME.time()
print("Start：2_calc_dipole.py")  # Print message indicating the script has started                   
# Filepath for the pre-trained machine learning model for dipole moment prediction.
PICKLE_dipole = f"../model/model_dipole.pkl"

# Argument parser for handling command-line inputs.
parser = argparse.ArgumentParser(
    description="Run a file up to a specified temperature to perform calculations."
)

# Required argument: target temperature (integer in Kelvin).
parser.add_argument("target_t", help="Target temperature [K]", type=float)
parser.add_argument(
    "-n",
    "--name",
    help="directory of running structure",
    type=str,
    default='../inputs/XYZ',
)
parser.add_argument(
    "-p",
    "--production_nsteps",
    help="number of steps for the NVE production run",
    type=int,
    default=400000,
)
parser.add_argument(
    "-i",
    "--trajectory_interval",
    help="number of steps between writes to the trajectory file",
    type=int,
    default=2,
)
args = parser.parse_args()

Sum_step = int(args.production_nsteps // args.trajectory_interval)  # Total number of steps for file saving
N_MAX_D=6
N_BATCH_D=1
max_neighbors =  25  # Maximum number of neighboring atoms considered in calculations
count_one = 500    # Number of time steps processed by each core
# Define paths for reading molecular dynamics simulation files and saving outputs.
xyzpath = f'../intermediate_results/VelocityVerlet'
save_Dipole = f'../intermediate_results/Dipole_txt_mass'
if not os.path.exists(save_Dipole):
    os.makedirs(save_Dipole)

for root, path, files in os.walk(str(args.name)):
  for file_one in tqdm(files, desc="Calculating the time-evolved dipole moments for molecules"):
    time_start_one=TIME.time()
    if not file_one.endswith('.xyz'): continue  
    file_i = os.path.join(root, file_one)
    file_i = os.path.basename(file_i).split(".")[0] 
    file_name = os.path.join(xyzpath, f'VelocityVerlet_{float(args.target_t)}K_{file_i}.xyz')
    # Check if the MD trajectory file exists
    if not os.path.exists(file_name):
        print(f"\n Error: Molecular Dynamics simulation file not found: {file_name}")
        print("Ensure that the MD simulation was completed successfully.")
        sys.exit(1)  # Exit the program with an error status    
     
    print('\n run file_i:',file_i)   
    model_info_dipole = pickle.load(open(PICKLE_dipole, "rb"))
    sorted_elements = sorted(['C', 'H','O','N'])
    symbol_map = {s: i for i, s in enumerate(sorted_elements)}
    n_types = len(sorted_elements)
    descriptor_generator = PowerSpectrumGenerator(N_MAX_D, model_info_dipole.r_cut, n_types, max_neighbors).process_data

    @jax.jit
    def get_bessel_descriptor(pos,t,c):
        descriptor=descriptor_generator(pos, t, c)
        return descriptor

    # Create model and optimizer, initialize
    update_model_dipole = MLP(model_info_dipole.update_model_widths)

    message_generation_models_diople = tuple([
            ResNetCore(model_info_dipole.message_generator_widths)
                for _ in range(model_info_dipole.n_passes)
                ])

    electron_pass_generation_models_diople = tuple([
            ResNetCore(model_info_dipole.pass_generator_widths)
                for _ in range(model_info_dipole.e_passes)
                ])

    # Create the EPNN model.
    model_dipole = EPNN(
                n_types,
                model_info_dipole.embed_dim,
                update_model_dipole,
                message_generation_models_diople,
                electron_pass_generation_models_diople,
                                )
    '''                            
    # Check if the output file already exists.                            
    if os.path.exists(f'{save_Dipole}/{os.path.basename(file_name)}_dipole.txt'):
        print("Dipole file already exists. Program terminated.")
        sys.exit()   # exit program
    '''
    
    # Read atomic coordinates and other input data from the specified file.
    cell_list0,type_train_list0,position_list0,init_charges_list0 = read_coord(file_name)
    Time_list0 = jnp.arange(0, len(type_train_list0), 1)

    if juge_file(file_name,group=Sum_step,T_max=50000):
       print("The file has issues or does not meet the required time condition. Program exiting.")
       sys.exit(1)   # exit program
          
    # Avoid running out of memory by calculating a portion of the data at a time.
    progress_bar = tqdm(total=args.production_nsteps, desc=f" {file_i}: Calculating the time-evolved dipole moments", unit="steps")
    count_num = int(len(type_train_list0) // count_one)
    for count_i in range(0,count_num,1):
         
        # Calculate the indices for the data slice this process will handle
        progress_bar.update(count_one*args.trajectory_interval)
        start_index = count_i * count_one
        end_index = start_index + count_one if count_i < count_num - 1 else len(type_train_list0)
        cell_list = jnp.array(cell_list0[start_index:end_index])
        type_train_list = jnp.array(type_train_list0[start_index:end_index])
        position_list = jnp.array(position_list0[start_index:end_index])
        init_charges_list = jnp.array(init_charges_list0[start_index:end_index])
        Time_list = jnp.array(Time_list0[start_index:end_index])
        
        # Generate descriptors for each frame    
        descriptors = []
        for i in range(0,len(type_train_list)):
            descriptor = get_bessel_descriptor(position_list[i], type_train_list[i], cell_list[i])
            descriptors.append(descriptor)
        descriptors = jnp.asarray(descriptors)
        descriptors_list = descriptors.reshape(*descriptors.shape[:2],-1)
        
        # Create batched graphs for the ML model
        test_graphs = create_batched_graph_list(
            jnp.array(position_list),
            jnp.array(cell_list),
            descriptors_list,
            jnp.array(type_train_list),
            jnp.array(init_charges_list),
            model_info_dipole.r_cut,
            model_info_dipole.r_switch,#R_SWITCH,
            model_info_dipole.node_state_dim,#NODE_STATE_DIM,
            N_BATCH_D,
            )

        # Predict charges using the ML model for each graph
        pred_charges = np.asarray(jnp.squeeze(jnp.vstack(list(map(
                lambda graph: get_nodes_from_graph(model_dipole.apply(model_info_dipole.params, graph))[0], test_graphs
                )))))
        pred_charges = jnp.asarray(pred_charges) # calculated chagres
        pred_charges = jnp.expand_dims(pred_charges,axis=-1)

        # Compute dipole moments for each frame
        dipole_monment_pred=[]
        for i in range(pred_charges.shape[0]):
            dipole_monment_pred.append(sum(pred_charges[i]*position_list[i])
                )
        dipole_monment_pred=jnp.array(dipole_monment_pred)

        # Extract and save dipole components over time
        time = []
        dipole_x, dipole_y, dipole_z = [],[],[]
        for i in range(0,len(dipole_monment_pred)): 
            time.append(Time_list[i])
            dipole_x.append(dipole_monment_pred[i][0])
            dipole_y.append(dipole_monment_pred[i][1])
            dipole_z.append(dipole_monment_pred[i][2])

        time = np.array(time)
        dipole_x = np.array(dipole_x)
        dipole_y = np.array(dipole_y)
        dipole_z = np.array(dipole_z)
        np.savetxt(f'{save_Dipole}/{os.path.basename(file_name)}_dipole-data_{count_i}.txt', 
                   np.column_stack((time, dipole_x, dipole_y, dipole_z)),fmt='%d %.10f %.10f %.10f',
                       header='time dipole_x dipole_y dipole_z')

    #gathers all partial results and merges them    
    Time_tol = []
    Diople_x_tol = []
    Diople_y_tol = []
    Diople_z_tol = []
    for i in range(0,count_num):
        time, dipole_x, dipole_y, dipole_z = np.loadtxt(f'{save_Dipole}/{os.path.basename(file_name)}_dipole-data_{i}.txt', 
                                             skiprows=1, usecols=(0,1,2,3), unpack=True)
        Time_tol.extend(time)
        Diople_x_tol.extend(dipole_x)
        Diople_y_tol.extend(dipole_y)
        Diople_z_tol.extend(dipole_z)
        os.remove(f'{save_Dipole}/{os.path.basename(file_name)}_dipole-data_{i}.txt')

    # Save the final combined results
    np.savetxt(f'{save_Dipole}/{os.path.basename(file_name)}_dipole.txt', 
               np.column_stack((Time_tol, Diople_x_tol, Diople_y_tol, Diople_z_tol)),
               fmt='%d %.10f %.10f %.10f',header='time dipole_x dipole_y dipole_z')
    
    progress_bar.close()
    time_end_one=TIME.time() 
    time_diff = time_end_one - time_start_one
    hours = time_diff // 3600
    minutes = (time_diff % 3600) // 60
    seconds = time_diff % 60   
    print(f"\n Calculating the time-evolved dipole moments for {file_i} completed",
      "time cost: ",f"{int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")
    
time_end=TIME.time() 
time_diff = time_end - time_start
hours = time_diff // 3600
minutes = (time_diff % 3600) // 60
seconds = time_diff % 60     
print("End: 2_calc_dipole.py",
      "time cost: ",f"{int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")  # Indicate that the script has completed        