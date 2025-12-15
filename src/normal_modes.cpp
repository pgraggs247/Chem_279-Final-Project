#include "normal_modes.hpp"
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include <armadillo>
#include <nlohmann/json.hpp>


namespace fs = std::filesystem;
using json = nlohmann::json;

int main(int argc, char **argv)
{
    // check that a config file is supplied
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <molecule_name>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string molecule_name = argv[1];

    // parse the config file
    fs::path config_file_path = fs::path("sample_input/hw_5_2/" + molecule_name + ".json");
    if (!fs::exists(config_file_path))
    {
        std::cerr << "Config file not found: " << config_file_path << std::endl;
        return EXIT_FAILURE;
    }

    std::ifstream config_file(config_file_path);
    json config = json::parse(config_file);

    // extract the important info from the config file
    fs::path atoms_file_path = config["atoms_file_path"];
    int num_alpha_electrons = config["num_alpha_electrons"];
    int num_beta_electrons = config["num_beta_electrons"];

    // read the atoms from the file
    std::vector<Atom> atoms = read_xyz(atoms_file_path.string());
    int num_atoms = atoms.size();

    // load the basis set shells
    std::vector<ContractedShell> basis_set_shells = load_basis_set_shells();

    // create the molecule
    Molecule molecule = create_molecule(atoms, num_alpha_electrons, num_beta_electrons, basis_set_shells);

    // optimize the geometry
    geometry_optimization(molecule);

    // calculate the Hessian matrix from the optimized geometry
    mat hessian = calculate_hessian(molecule);

    // calculate the weighted Hessian matrix
    mat weighted_hessian = calculate_weighted_hessian(hessian, molecule);

    // solve the eigenvalue problem
    std::pair<vec, mat> eigenvalues_and_eigenvectors = calculate_eigenvalues_and_eigenvectors(weighted_hessian);

    // calculate the frequencies
    std::vector<double> frequencies = calculate_frequencies(eigenvalues_and_eigenvectors.first);

    // Create the directory for results:
    fs::path results_dir = fs::path("sample_output/Normal_Modes/" + molecule_name);
    fs::create_directories(results_dir);

    // Save the frequencies to the file:
    fs::path frequencies_file_path = results_dir / "frequencies.txt";
    std::ofstream frequencies_file(frequencies_file_path);
    frequencies_file << "=== Corrected Frequencies ===\n";
    for (size_t i = 0; i < frequencies.size(); i++)
    {
        frequencies_file << "Mode " << i + 1 << ": " << frequencies[i] << " cm^-1" << std::endl;
    }
    frequencies_file.close();
    std::cout << "The frequencies have been saved to the file: " << frequencies_file_path << std::endl;


    // create a vector of equilibrium coordinates
    std::vector<double> equilibrium_coords;
    for (const auto &atom : molecule.atoms)
    {
        equilibrium_coords.push_back(atom.x);
        equilibrium_coords.push_back(atom.y);
        equilibrium_coords.push_back(atom.z);
    }

    // create a vector of atomic numbers
    std::vector<int> atomic_nums;
    for (const auto &atom : molecule.atoms)
    {
        atomic_nums.push_back(atom.atomic_number);
    }

    // create the directory for the mode animations
    fs::path mode_animations_dir = results_dir / "mode_animations";
    fs::create_directories(mode_animations_dir);

    int start_index = (num_atoms <= 2) ? 5 : 6;
    for (int i = start_index; i < 3 * num_atoms; i++)
    {
        create_mode_animation_pdb(mode_animations_dir / (molecule_name + "_mode_" + std::to_string(i + 1) + ".pdb"), equilibrium_coords, atomic_nums, atomic_symbols, eigenvalues_and_eigenvectors.second, i);
    }

    std::cout << "The normal modes have been created and saved to the directory: " << mode_animations_dir << std::endl;

    return 0;
}