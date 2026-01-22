#!/bin/bash

# Set the target temperature for the calculations
TEMPERATURE=50
# Directory containing .xyz files for molecular structures
XYZ_DIR="../inputs/XYZ"

# Check if the XYZ_DIR exists; exit with an error message if not
if [ ! -d "$XYZ_DIR" ]; then
    echo " $XYZ_DIR is not exited"
    exit 1
fi
       
# Run the Python script to perform MD position calculations
echo " Run the Python script to perform MD position calculations "
python 1_MD_calc_position.py "$TEMPERATURE" 
# Run the Python script to calculate dipole moments
echo " Run the Python script to calculate dipole moments "
python 2_calc_dipole.py "$TEMPERATURE" 
# After processing all files, calculate the IR spectrum using the dipole data
echo " Calculate the IR spectrum "
python 3_calc_IR.py "$TEMPERATURE" --qm 0
		
