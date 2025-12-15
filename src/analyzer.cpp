#include "normal_modes.hpp"
#include <iostream>
#include <armadillo>
#include <nlohmann/json.hpp>


using namespace arma;

// Function to update the basis function centers:
void update_basis_centers(Molecule &molecule)
{
    // Iterate through all basis functions
    for (auto &bf : molecule.basis_functions)
    {
        // Find which atom this basis function belongs to
        int atom_idx = bf.atom_index;

        // Update the basis function's coordinates to match the atom's current coordinates
        bf.x = molecule.atoms[atom_idx].x;
        bf.y = molecule.atoms[atom_idx].y;
        bf.z = molecule.atoms[atom_idx].z;
    }
}

// Function to calculate the Hessian matrix:
mat calculate_hessian(Molecule molecule)
{
    // 1. Instantiate the Hessian matrix:
    int N = molecule.atoms.size();
    mat hessian(3 * N, 3 * N, fill::zeros);
    int index = 0;

    // 2. Establish the step size:
    double step_size = 0.01;

    Molecule mol_pertubed = molecule;
    // 3. Iterate through every coordinate i from 1 to 3N:
    for (int i = 0; i < N; ++i)
    {
        // Create a an array of pointers to the coordinates:
        // double* coords[3] = { &molecule.atoms[i].x, &molecule.atoms[i].y, &molecule.atoms[i].z };
        double *coords[3] = {&mol_pertubed.atoms[i].x, &mol_pertubed.atoms[i].y, &mol_pertubed.atoms[i].z};

        // Iterate through each coordinate:
        for (int j = 0; j < 3; ++j)
        {
            // Store original value
            double original_value = *coords[j];

            // Positive displacement:
            *coords[j] = original_value + step_size;
            update_basis_centers(mol_pertubed);
            mat pos_gradient = calculate_overall_gradient(mol_pertubed);
            vec flat_pos_gradient = vectorise(pos_gradient);

            // Negative displacement:
            *coords[j] = original_value - step_size;
            update_basis_centers(mol_pertubed);
            mat neg_gradient = calculate_overall_gradient(mol_pertubed);
            vec flat_neg_gradient = vectorise(neg_gradient);

            // Restore the original coordinate:
            *coords[j] = original_value;
            update_basis_centers(mol_pertubed);

            // Compute the central difference approximation:
            vec diff = (flat_pos_gradient - flat_neg_gradient) / (2 * step_size);
            hessian.col(index) = diff;

            // Increment the index:
            index++;
        }
    }
    // 4. Post-process the Hessian matrix: (make it symmetric --> hessian(i, j) = hessian(j, i) = (hessian(i, j) + hessian(j, i)) / 2)
    hessian = (hessian + hessian.t()) / 2;

    // 5. Return the Hessian matrix into ha:
    return hessian;
}

// Function to calculate the weighted Hessian matrix:
mat calculate_weighted_hessian(const mat &hessian, const Molecule &molecule)
{
    // 1. Create a weighted Hessian Matrix: (H_mw(i, j) = H(i, j) * sqrt(m_i * m_j))
    int N = molecule.atoms.size();
    mat weighted_hessian(3 * N, 3 * N, fill::zeros);
    for (int i = 0; i < 3 * N; ++i)
    {
        for (int j = 0; j < 3 * N; ++j)
        {
            int atom_i_idx = i / 3;
            int atom_j_idx = j / 3;
            double mass_i = atomic_masses.at(molecule.atoms[atom_i_idx].atomic_number) * AMU_TO_AU;
            double mass_j = atomic_masses.at(molecule.atoms[atom_j_idx].atomic_number) * AMU_TO_AU;
            weighted_hessian(i, j) = hessian(i, j) / std::sqrt(mass_i * mass_j);
        }
    }
    // 2. Return the weighted Hessian matrix:
    return weighted_hessian;
}

// Function to solve the eigenvalue problem:
std::pair<vec, mat> calculate_eigenvalues_and_eigenvectors(const mat &weighted_hessian)
{
    // 1. Solve the eigenvalue problem:
    vec eigenvalues;
    mat eigenvectors;
    eig_sym(eigenvalues, eigenvectors, weighted_hessian);
    return std::make_pair(eigenvalues, eigenvectors);
}

// Function to calculate the frequencies from the eigenvalues:
std::vector<double> calculate_frequencies(const vec &eigenvalues)
{
    // 1. Create a vector of frequencies:
    std::vector<double> frequencies;

    // 2. Calculate the frequencies:
    for (int i = 0; i < eigenvalues.n_elem; ++i)
    {
        double frequency = eigenvalues(i);
        double freq_cm = 0.0;

        if (std::abs(frequency) < 1e-5)
        {
            freq_cm = 0.0;
        }
        else if (frequency > 0)
        {
            freq_cm = std::sqrt(frequency) * AU_TO_WAVENUMBER;
        }
        else
        {
            freq_cm = 0.0;
        }
        frequencies.push_back(freq_cm);
    }

    // 4. Return the frequencies:
    return frequencies;
}