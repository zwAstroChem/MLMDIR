#!/usr/bin/env python

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
import sys
from scipy import fftpack
from scipy import signal
import math
from scipy.constants import physical_constants

np.set_printoptions(threshold=np.inf)
 
def calc_MD_IR(file_txt):
    """
    Calculate the IR spectrum from molecular dynamics (MD) dipole moment data.
    Args:
        file_txt: Input file containing time and dipole moment data.
    Returns:
        Freq_MD_cauchy: Frequency axis of the IR spectrum with Cauchy broadening.
        Inten_MD_cauchy: Intensities corresponding to the broadened spectrum.
    """
   
    # Load time and dipole moment data from the input file
    time, dipole_x, dipole_y, dipole_z = np.loadtxt(file_txt, skiprows=1, usecols=(0, 1, 2, 3),
                                                    unpack=True)
    
    # Constants
    boltz = physical_constants["Boltzmann constant"][0] #1.38064852E-23  # m^2 kg s^-2 K^-1
    lightspeed = physical_constants["speed of light in vacuum"][0] #299792458.  # m s^-1
    reduced_planck = physical_constants["reduced Planck constant"][0] #1.05457180013E-34  # kg m^2 s^-1
    timestep = (time[1] - time[0]) * 1.E-15  # converts time from femtoseconds to seconds
    e_c = physical_constants["elementary charge"][0] # 1.602176634e-19 C
    N_A = physical_constants["Avogadro constant"][0]  # 6.02214076e+23 mol^-1
    AA_0 = physical_constants["Angstrom star"][0]  # 1.00e-10 m
    epsilo_0 =physical_constants["vacuum electric permittivity"][0]  # 8.8541878188e-12 F m^-1  C^2 N^-1 m^-2
    
    dipole_x = dipole_x * e_c * AA_0    #C·m
    dipole_y = dipole_y * e_c * AA_0
    dipole_z = dipole_z * e_c * AA_0
    
    T = float(args.target_t)  # K
    fraction_autocorrelation_function_to_fft = 0.07987  # since you only want to fft the part that has meaningful statistics,
 
    # Calculate autocorrelation function for dipole moments
    if len(time) % 2 == 0:
        dipole_x_shifted = np.zeros(len(time) * 2)
        dipole_y_shifted = np.zeros(len(time) * 2)
        dipole_z_shifted = np.zeros(len(time) * 2)
    else:
        dipole_x_shifted = np.zeros(len(time) * 2 - 1)
        dipole_y_shifted = np.zeros(len(time) * 2 - 1)
        dipole_z_shifted = np.zeros(len(time) * 2 - 1)
    
    # Embed original dipole data into shifted arrays for correlation calculation
    dipole_x_shifted[len(time) // 2:len(time) // 2 + len(time)] = dipole_x
    dipole_y_shifted[len(time) // 2:len(time) // 2 + len(time)] = dipole_y
    dipole_z_shifted[len(time) // 2:len(time) // 2 + len(time)] = dipole_z
    
    # Compute autocorrelation for each dipole moment component
    autocorr_x_full = (signal.fftconvolve(dipole_x_shifted, dipole_x[::-1], mode='same')[(-len(time)):]
                       / np.arange(len(time), 0, -1))
    autocorr_y_full = (signal.fftconvolve(dipole_y_shifted, dipole_y[::-1], mode='same')[(-len(time)):]
                       / np.arange(len(time), 0, -1))
    autocorr_z_full = (signal.fftconvolve(dipole_z_shifted, dipole_z[::-1], mode='same')[(-len(time)):]
                       / np.arange(len(time), 0, -1))
    autocorr_full = autocorr_x_full + autocorr_y_full + autocorr_z_full
    
    # Truncate the autocorrelation array to the meaningful fraction
    autocorr = autocorr_full[:int(len(time) * fraction_autocorrelation_function_to_fft)]
   
    # Compute the lineshape
    lineshape = fftpack.dct(autocorr, type=1)[1:] * timestep
    
    lineshape_frequencies_freq = np.linspace(0, 0.5 / timestep, len(autocorr))[1:]    
    lineshape_frequencies_wn = lineshape_frequencies_freq / (100*lightspeed)  # converts to wavenumbers (cm^-1)
    
    lineshape_frequencies = lineshape_frequencies_freq * 2 * np.pi        # Convert to angular frequency
    ratio_0 = (10000 / (6*reduced_planck*lightspeed*epsilo_0))       # Convert to cm^2  
    
    # Calculate spectra
    field_description = lineshape_frequencies * (
                1. - np.exp(-reduced_planck * lineshape_frequencies / (boltz * T)))
    
    if args.qm == 0:
        quantum_correction = 1
    elif args.qm == 1:
        quantum_correction = 2 / (
                1. + np.exp(-reduced_planck * lineshape_frequencies / (boltz * T)))
    elif args.qm == 2:
        quantum_correction = (reduced_planck * lineshape_frequencies / (boltz * T)) /(
                    2 * np.tanh(reduced_planck * lineshape_frequencies / (2*boltz * T)))
    elif args.qm == 3:
        quantum_correction = (reduced_planck * lineshape_frequencies / (boltz * T)) / (
                1. - np.exp(-reduced_planck * lineshape_frequencies / (boltz * T)))
    else:
        print("Error: qm parameter can only be 0, 1, 2, or 3. Other values are invalid, please re-enter!")
        exit(1)
    
    spectra = lineshape * field_description * ratio_0 * quantum_correction   # cm^2 
    spectra[spectra<0] = 0.0
    
    # Filter spectrum to the desired wavenumber range
    lineshape_frequencies_wn = lineshape_frequencies_wn * 0.9578   #a scaling factor of 0.9578 to vibrational frequencies
    mask = (lineshape_frequencies_wn >= 300) & (lineshape_frequencies_wn <= 3800)
    Freq0 = lineshape_frequencies_wn[mask]
    Inten0 = spectra[mask]  
    if args.no_normalize:
       Inten0 = Inten0/np.max(Inten0)
    
    return Freq0,Inten0      

def plot_one(x, y,save_name,Label):
    """
    Plots the IR spectrum and saves it as a PNG image.
    Args:
        x (list or array): Frequencies to plot on the x-axis.
        y (list or array): Intensities to plot on the y-axis.
        save_name (str): Path to save the figure.
        Label (str): Label for the plot legend.
    """
    font = {'size': 20}
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fig.subplots_adjust(right=0.98, top=0.98)
    ax.tick_params(axis='x', labelsize=font['size'])  
    ax.tick_params(axis='y', labelsize=font['size']) 
    
    # Plot data with a black line and specified label.
    ax.plot(x, y, color='black',linewidth=2.0,label=Label)
    
    # Add x and y axis labels
    ax.set_xlabel('Wavenumber (cm^-1)', fontdict=font)
    ax.set_ylabel('Normalized intensity', fontdict=font)
    ax.set_xlim(300, 3800)
    ax.set_xticks([1000,2000,3000], ['1000','2000','3000'])
    ax.legend()
    plt.savefig(save_name, dpi=300, format='jpg', bbox_inches='tight')
    plt.close()
        
if __name__ == "__main__":
    """
    Main script to calculate the IR spectrum based on dipole data, 
    and save both the computed data and the plot.
    """
    time_start=time.time()
    print("Start：3_calc_IR.py")  # Print message indicating the script has started
    # Argument parser for handling command-line inputs.
    parser = argparse.ArgumentParser(
        description="Run a file up to a specified temperature to perform calculations."
    )

    # Required argument: target temperature (integer in Kelvin).
    parser.add_argument("target_t", help="Target temperature [K]", type=float)
    parser.add_argument(
            "--qm",
            help="Quantum correction of absorption cross-section intensity. More details at DOI:10.1021/jp034788u. Default: no correction.",
            type=int,
            default=0
        )
    parser.add_argument(
            "--no_normalize",
            help="Whether to normalize intensities so that total intensity sums to 1. Default is True (normalization enabled).",
            action="store_false",
            default=True
        )
    parser.add_argument(
        "-n",
        "--name",
        help="directory of running structure",
        type=str,
        default='../inputs/XYZ',        
    )
    args = parser.parse_args()
    
    # Construct paths for input and output files based on the target temperature and structure name.
    xyzpath = f'../intermediate_results/VelocityVerlet'
    # Directory to save IR data.
    save_IR_path = f'../outputs/IR_txt'
    if not os.path.exists(save_IR_path):
        os.makedirs(save_IR_path)
    # Directory to save IR figure.
    save_IR_figure = f'../outputs/IR_txt/figure'
    if not os.path.exists(save_IR_figure):
        os.makedirs(save_IR_figure) 
    
    for root, path, files in os.walk(str(args.name)):
      for file_one in tqdm(files, desc="Calculating the infrared spectra for molecules"):
        if not file_one.endswith('.xyz'): continue 
        file_i = os.path.join(root, file_one)
        file_i = os.path.basename(file_i).split(".")[0] 
        file_name = os.path.join(xyzpath, f'VelocityVerlet_{float(args.target_t)}K_{file_i}.xyz')
        save_Dipole = f'../intermediate_results/Dipole_txt_mass'
        filetxt = f'{save_Dipole}/{os.path.basename(file_name)}_dipole.txt'
        # Check if the dipole moment file exists
        if not os.path.exists(filetxt):
            print(f"\n Error: Dipole moment calculation file not found: {filetxt}")
            print("Ensure that the time-evolved dipole moments calculation was completed successfully.")
            sys.exit(1)  # Exit the program with an error status
               
        # Calculating and Saving the IR Spectrum 
        print('\n run file_i:',file_i)
        Freq_MD,Inten_MD = calc_MD_IR(filetxt)
        np.savetxt(f'{save_IR_path}/{file_i}_qm{args.qm}.txt', 
                                np.column_stack((Freq_MD,Inten_MD)),
                                fmt='%.1f %.4e',header='Freq_MD (cm^-1) Inten_MD (Normalized intensity)')
        save_figure = os.path.join(save_IR_figure, f'{file_i}_qm{args.qm}.jpg')
        plot_one(Freq_MD,Inten_MD,save_figure,file_i)
    
    time_end=time.time() 
    time_diff = time_end - time_start
    hours = time_diff // 3600
    minutes = (time_diff % 3600) // 60
    seconds = time_diff % 60 
    print("End: 3_calc_IR.py",
          "time cost: ",f"{int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")  # Indicate that the script has completed
    
            