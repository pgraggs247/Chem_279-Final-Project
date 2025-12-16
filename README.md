Determining and Visualizing the Normal Modes of Various Molecular Systems
Description
This project implements a complete workflow for molecular vibrational analysis and visualization:

Geometry Optimization - Optimizes molecular geometry via steepest descent with backtracking line search.
Mass-weighted Hessian calculation - Calculates the mass-weighted Hessian matrix (force constants) using numerical differentiation of analytical force gradients.
Normal Mode Analysis - Solves the Eigenvalue problem to obtain vibrational frequencies (eigenvalues) and normal mode displacements (eigenvectors).
Visualization Output - Produces PDB trajectory files for animating vibrational modes in Avogadro &/ PyMol.
Supported Elements
Hydrogen (H)
Carbon (C)
Nitrogen (N)
Oxygen (O)
Fluorine (F)
Available Tested Molecules
H₂ (Hydrogen)
HF (Hydrogen Fluoride)
HO (Hydroxyl radical)
H₂O (Water)
NH₃ (Ammonia)
CH₄ (Methane)
Implementation Details
Core Components
File	Description
hw3.cpp	Implements Gaussian overlap integrals; constructs basis functions from STO-3G parameters
hw4.cpp	Implements CNDO/2 Self-Consistent Field (SCF) procedure including Fock matrix construction, density matrix calculation, and convergence checking
hw5.cpp	Computes analytical gradients of electronic and nuclear repulsion energies; performs geometry optimization via steepest descent with adaptive step size
analyzer.cpp	Calculates Hessian matrix via central finite differences; computes mass-weighted Hessian and solves eigenvalue problem for frequencies
writter.cpp	Generates multi-frame PDB trajectory files for visualizing normal mode animations
Project Structure
Below is a directory tree to help navigate the codebase:

├── atoms/           # XYZ molecular geometry files  
├── basis/           # STO-3G basis set parameters (JSON)  
├── include/         # Header files  
├── sample_input     # Configuration JSON files  
├── sample_output    # Generated results  
└── src/             # Source code

In addition, the repo contains:
1. CHEM-279-FINAL-PROJECT.pptx: The group's final project powerpoint presentation.
Building & Running the Project
A pre-configured development environment is available; launch the interactive Docker container: ./interactive.sh

Inside the container, build the project: ./build.sh (you might need to chmod +x to gain permission access). To remove the build subdirectory and output files to start fresh, run ./clean.sh

To run the program, in the root directory, run ./run.sh <molecule_name>. To execute all tested molecules at once, run ./run.sh.

The program will output results saved to sample_output/Normal_Modes/<molecule>/:

- frequencies.txt: list all vibrational frequencies in cm⁻¹. The first 5 modes (linear) or 6 modes (non-linear) correspond to translations and rotations, and should have frequencies near zero.

- mode_animations: subfolder which contains PDB trajectories files containing 20 frames that animate each vibrational mode.

Example Output
Running ./run.sh H2O produces in the terminal:

Initial Energy: -538.021 eV

=== Optimization Complete ===
Final energy: -538.451 eV

Final optimized positions (Bohr):
Atom 0 (8): -3.65939e-17, 8.33392e-11, 0.268639
Atom 1 (1): -5.89235e-17, 1.51054, -0.910969
Atom 2 (1): 9.55174e-17, -1.51054, -0.910969
The frequencies have been saved to the file: "sample_output/Normal_Modes/H2O/frequencies.txt"
The normal modes have been created and saved to the directory: "sample_output/Normal_Modes/H2O/mode_animations"
Visualization with Avogadro & PyMol
Avogadro
Open Avogadro
File → Open and select a .pdb file from sample_output/Normal_Modes/<molecule>/mode_animations/
The animation will load automatically with multiple frames
Use the Animation toolbar (Extensions → Animations) or the timeline slider to play the vibration
Press Play to animate the normal mode
PyMol
Open PyMol
Run this multi-line commands in the PyMol command line
reinitialize
hide everything
set sphere_scale, 0.4
set stick_radius, 0.35
set connect_cutoff, 2.0
bg_color white
load sample_output/Normal_Modes/H2O/mode_animations/H2O_mode_7.pdb
rebuild
bond id 1, id 
show spheres
show sticks
center
count_states
mset 1 -20
frame 1
mpng frame_, mode=2
set ray_trace_frames, 1
set ray_opaque_background, on
set antialias, 2
set ray_shadows, on
set ray_trace_mode, 1
set sphere_quality, 4
set stick_quality, 16
frame 1
mpng frame_, mode=2
mplay
Note: The animations for the test molecules were produced, and can be viewed in the final powerpoint presentation in this repo.
