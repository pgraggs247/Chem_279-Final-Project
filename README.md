# Determining and Visualizing the Normal Modes of Various Molecular Systems

## Description

This project implements a computational chemistry tool that determines normal modes for different molecular systems, calculates vibrational frequencies using semi-empirical quantum mechanics (CNDO/2) and generates animations for visualizations through these major steps:

1. **Geometry Optimization** - Optimizes molecular geometry via steepest descent with backtracking line search.
2. **Mass-weighted Hessian calculation** - Calculates the mass-weighted Hessian matrix (force constants) using numerical differentiation of analytical force gradients.
3. **Normal Mode Analysis** - Solves the Eigenvalue problem to obtain vibrational frequencies (eigenvalues) and normal mode displacements (eigenvectors).
4. **Visualization Output** - Produces PDB trajectory files for animating vibrational modes in Avogadro &/ PyMol.

### Supported Elements
- Hydrogen (H)
- Carbon (C)
- Nitrogen (N)
- Oxygen (O)
- Fluorine (F)

### Available Tested Molecules
- H₂ (Hydrogen)
- HF (Hydrogen Fluoride)
- HO (Hydroxyl radical)
- H₂O (Water)
- NH₃ (Ammonia)
- CH₄ (Methane)

## Implementation Details

### Core Components


| File | Description |
|------|-------------|
| `hw3.cpp` | Implements Gaussian overlap integrals; constructs basis functions from STO-3G parameters |
| `hw4.cpp` | Implements CNDO/2 Self-Consistent Field (SCF) procedure including Fock matrix construction, density matrix calculation, and convergence checking |
| `hw5.cpp` | Computes analytical gradients of electronic and nuclear repulsion energies; performs geometry optimization via steepest descent with adaptive step size |
| `analyzer.cpp` | Calculates Hessian matrix via central finite differences; computes mass-weighted Hessian and solves eigenvalue problem for frequencies |
| `writer.cpp` | Generates multi-frame PDB trajectory files for visualizing normal mode animations |


### Project Structure

Below is a directory tree to help navigate the codebase:
```
├── atoms/           # XYZ molecular geometry files  
├── basis/           # STO-3G basis set parameters (JSON)  
├── include/         # Header files  
├── sample_input     # Configuration JSON files  
├── sample_output    # Generated results  
└── src/             # Source code

In addition, the repo contains:
1. CHEM-279-Final-Project-Presentation.pptx/pdf: The group's final project powerpoint presentation
2. CHEM-279-Final-Project-Report: The group's final project report
3. misc: folder containing a PyMol macro script to automate & produce animations for molecular vibrations at a particular mode
```

## Building & Running the Project

1. A pre-configured development environment is available; launch the interactive Docker container: `./interactive.sh`

2. Inside the container, run `./clean.sh` to remove any build or output artefacts. To build the project, run `./build.sh`. 
Note: you might need to `chmod +x` to gain permission access.

3. To run the program, in the root directory, run `./run.sh <molecule_name>`. In addition, to execute all tested molecules at once, run `./run.sh`. 

4. The program will output results saved to `sample_output/Normal_Modes/<molecule>/`:

    `- frequencies.txt`: list all vibrational frequencies in cm⁻¹. The first 5 modes (linear) or 6 modes (non-linear) correspond to translations and rotations, and should have frequencies near zero.

    `- mode_animations`: subfolder which contains PDB trajectories files containing 20 frames that animate each vibrational mode.

### Example Output
Running `./run.sh H2O` produces in the terminal:

```
Initial Energy: -538.021 eV

=== Optimization Complete ===
Final energy: -538.451 eV

Final optimized positions (Bohr):
Atom 0 (8): -3.65939e-17, 8.33392e-11, 0.268639
Atom 1 (1): -5.89235e-17, 1.51054, -0.910969
Atom 2 (1): 9.55174e-17, -1.51054, -0.910969
The frequencies have been saved to the file: "sample_output/Normal_Modes/H2O/frequencies.txt"
The normal modes have been created and saved to the directory: "sample_output/Normal_Modes/H2O/mode_animations"
```

## Visualization with Avogadro & PyMol

### Avogadro

1. Open Avogadro
2. File → Open and select a .pdb file from `sample_output/Normal_Modes/<molecule>/mode_animations/`
3. The animation will load automatically with multiple frames
4. Use the Animation toolbar (Extensions → Animations) or the timeline slider to play the vibration
5. Press Play to animate the normal mode


### PyMol
1. In the repo, navigate to `misc` folder, and copy the entire multi-line commands in the `pymol_script.txt` file. 
1. Open PyMol
2. In PyMol terminal, paste the command, and execute the macro. 
The interface will display animations of molecular vibrations at a specific mode. To visualize other .pdb files, the user needs to point to different files which are located in `sample_output/Normal_Modes/<Molecule_name>/mode_animations`.

`Note:` The animations for the test molecules were produced, and can be viewed in the `CHEM-279-Final-Project-Presentation.pptx` file in this repo.