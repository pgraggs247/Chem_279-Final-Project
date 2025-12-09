#include "hw_5.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <armadillo>
#include <nlohmann/json.hpp>

using namespace arma;
using json = nlohmann::json;

// Function to create a mode animation of the molecule from the frequencies and eigenvectors
void create_mode_animation(std::string filename,
                           const std::vector<double>& equilibrium_coords,
                           const std::vector<int>& atomic_nums,
                           const std::map<int, std::string>& atomic_symbols,
                           const mat& eigenvectors,
                           int mode_index,
                           int num_frames)
{

    // Check if the mode index is valid:
    if (mode_index < 0 || mode_index >= eigenvectors.n_cols) {
        std::cerr << "Error: Invalid mode_index " << mode_index << std::endl;
        return;
    }

    // Check if the equilibrium coordinates and atomic numbers match:
    if (equilibrium_coords.size() != 3 * atomic_nums.size()) {
        std::cerr << "Error: Coordinate/atom count mismatch" << std::endl;
        return;
    }

    // Open the file for writing:
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Failed to open file for writing:" << filename << std::endl;
        return;
    }

    // Proceed with the animation:
    int num_atoms = atomic_nums.size();
    //double amplitude = 0.5;
    double amplitude = 0.1;

    for (int i = 0; i < num_frames; i++)
    {
        // create a sine wave phase (0 to 2*PI)
        double phase = 2.0 * M_PI * i / (num_frames);
        double displacement_factor = amplitude * std::sin(phase);

        // XYZ Header:
        file << num_atoms << std::endl;
        file << "Mode " << mode_index + 1 << " Frame " << i << std::endl;;
        for (int j = 0; j < num_atoms; j++)
        {
            // Determine the equilibrium coordinates
            double x0 = equilibrium_coords[3*j];
            double y0 = equilibrium_coords[3*j + 1];
            double z0 = equilibrium_coords[3*j + 2];

            // Get the vibrational vector:
            double dx = eigenvectors(3*j, mode_index);
            double dy = eigenvectors(3*j + 1, mode_index);
            double dz = eigenvectors(3*j + 2, mode_index);

            // Calculate the new coordinates:
            double x = x0 + displacement_factor * dx;
            double y = y0 + displacement_factor * dy;
            double z = z0 + displacement_factor * dz;

            // Write the new coordinates:
            std::string symbol = atomic_symbols.at(atomic_nums[j]).c_str();
            file << symbol << " " << x << " " << y << " " << z << std::endl;
        }
    }
    file.close();
}

void create_mode_animation_pdb(std::string filename,
                            const std::vector<double>& equilibrium_coords,
                            const std::vector<int>& atomic_nums,
                            const std::map<int, std::string>& atomic_symbols,
                            const mat& eigenvectors,
                            int mode_index,
                            int num_frames)
{
    // Check if the mode index is valid:
    if (mode_index < 0 || mode_index >= eigenvectors.n_cols) {
        std::cerr << "Error: Invalid mode_index " << mode_index << std::endl;
        return;
    }

    // Check if the equilibrium coordinates and atomic numbers match:
    if (equilibrium_coords.size() != 3 * atomic_nums.size()) {
        std::cerr << "Error: Coordinate/atom count mismatch" << std::endl;
        return;
    }

    // Change extension to .pdb
    std::ofstream file(filename);
    if (!file.is_open()) 
    {
        std::cerr << "Error: Failed to open file for writing:" << filename << std::endl;
        return;
    }

    int num_atoms = atomic_nums.size();
    double amplitude = 0.5;

    for (int i = 0; i < num_frames; i++)
    {
        double phase = 2.0 * M_PI * i / num_frames;
        double displacement_factor = amplitude * std::sin(phase);

        // PDB MODEL header
        file << "MODEL     " << std::setw(4) << (i + 1) << std::endl;

        for (int j = 0; j < num_atoms; j++)
        {
            double x0 = equilibrium_coords[3*j];
            double y0 = equilibrium_coords[3*j + 1];
            double z0 = equilibrium_coords[3*j + 2];

            double dx = eigenvectors(3*j, mode_index);
            double dy = eigenvectors(3*j + 1, mode_index);
            double dz = eigenvectors(3*j + 2, mode_index);

            double x = x0 + displacement_factor * dx;
            double y = y0 + displacement_factor * dy;
            double z = z0 + displacement_factor * dz;

            std::string symbol = atomic_symbols.at(atomic_nums[j]);

            // PDB HETATM format (fixed-width columns)
            file << "HETATM"                                      // 1-6:   Record name
                << std::setw(5) << (j + 1)                       // 7-11:  Atom serial number
                << " "                                           // 12:    Space
                << std::setw(2) << std::left << symbol           // 13-14: Atom name
                << std::right                                    
                << "   "                                         // 15-17: Padding
                << "MOL"                                         // 18-20: Residue name
                << " "                                           // 21:    Space
                << "A"                                           // 22:    Chain ID
                << std::setw(4) << 1                             // 23-26: Residue seq number
                << "    "                                        // 27-30: Padding
                << std::fixed << std::setprecision(3)
                << std::setw(8) << x                             // 31-38: X coordinate
                << std::setw(8) << y                             // 39-46: Y coordinate
                << std::setw(8) << z                             // 47-54: Z coordinate
                << std::setw(6) << std::setprecision(2) << 1.00  // 55-60: Occupancy
                << std::setw(6) << 0.00                          // 61-66: Temp factor
                << "          "                                  // 67-76: Padding
                << std::setw(2) << std::right << symbol          // 77-78: Element symbol
                << std::endl;
        }
        file << "ENDMDL" << std::endl;
    }
    file << "END" << std::endl;
    file.close();
}