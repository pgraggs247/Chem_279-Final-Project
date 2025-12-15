#include "normal_modes.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <utility>
#include <armadillo>
#include <nlohmann/json.hpp>


using namespace arma;

// Function to read in the information from the xyz file:
std::vector<Atom> read_xyz(std::string file_path)
{
    // Open up a file for input:
    std::ifstream infile(file_path);

    // Check if the file was successfully opened:
    if (!infile.is_open())
    {
        throw std::runtime_error("The filepath does not exist.");
    }

    // Defining the number of atoms, and a vector for atoms:
    int num_atoms;
    std::vector<Atom> atomic_system;

    // Pipe the number of atoms in the first line:
    infile >> num_atoms;

    for (int i = 0; i < num_atoms; i++)
    {
        // Initialize an atom struct:
        Atom atom;

        // Now build the atom struct for each atom in the file:
        infile >> atom.atomic_number >> atom.x >> atom.y >> atom.z;

        // Append the atom to the vector:
        atomic_system.push_back(atom);
    }
    return atomic_system;
}

// Function to determine the number of basis functions & number of electrons:
std::pair<int, int> calculate_basis_fun_electrons_pair(std::vector<Atom> atom_system)
{
    // Instantiate counters for the carbon and hydrogen atoms;
    int carbon_counter = 0, hydrogen_counter = 0;

    for (const auto &atom : atom_system)
    {
        if (atom.atomic_number == CARBON_ATOMIC_NUMBER)
        {
            carbon_counter++;
        }
        else if (atom.atomic_number == HYDROGEN_ATOMIC_NUMBER)
        {
            hydrogen_counter++;
        }
    }

    // Calculate the number of basis functions:
    int num_basis_functions = 4 * carbon_counter + hydrogen_counter;

    // Calculate the number of electrons:
    double num_electrons = 2 * carbon_counter + (hydrogen_counter / 2);

    if (std::fmod(num_electrons, 1.0) != 0.0)
    {
        throw std::runtime_error("num_electrons is not an integer");
    }

    // Return the pair of basis functions, and number of electrons:
    return std::make_pair(num_basis_functions, num_electrons);
}

// Function to read in the information from Json and construct the gaussian:
ContractedShell construct_contracted_g(const json &data)
{
    // Initialize an empty Contracted_G gaussian:
    ContractedShell shell;

    // Read through the JSON file:
    if (data.is_object())
    {
        if (data.contains("atomic_number"))
        {
            shell.atomic_num = data.at("atomic_number");
        }

        if (data.contains("shell_momentum"))
        {
            shell.momentum = data.at("shell_momentum");
        }

        if (data.contains("contracted_gaussians"))
        {
            const auto &cg = data.at("contracted_gaussians");
            if (cg.is_array())
            {
                for (const auto &it : cg)
                {
                    double exp = it.at("exponent");
                    double coeff = it.at("contraction_coefficient");
                    shell.exponent_coeff.emplace_back(exp, coeff);
                }
            }
        }
    }
    return shell;
}

// Function to return a list of basis functions:
std::vector<BasisFunction> return_basis_functions(const std::vector<Atom> &atoms,
                                                  const std::vector<ContractedShell> &basis_set_shells)
{
    std::vector<BasisFunction> basis_functions;

    for (size_t i = 0; i < atoms.size(); ++i)
    {
        const Atom &atom = atoms[i];
        int atom_index = i;

        std::vector<const ContractedShell *> s_shells;
        std::vector<const ContractedShell *> p_shells;

        for (const auto &shell : basis_set_shells)
        {
            if (shell.atomic_num != atom.atomic_number)
                continue;

            if (shell.momentum == 0)
                s_shells.push_back(&shell);
            else if (shell.momentum == 1)
                p_shells.push_back(&shell);
        }

        for (const ContractedShell *shell : s_shells)
        {
            basis_functions.push_back({atom.x, atom.y, atom.z, 0, 0, 0, atom.atomic_number, shell, atom_index});
        }

        for (const ContractedShell *shell : p_shells)
        {
            basis_functions.push_back({atom.x, atom.y, atom.z, 1, 0, 0, atom.atomic_number, shell, atom_index});
            basis_functions.push_back({atom.x, atom.y, atom.z, 0, 1, 0, atom.atomic_number, shell, atom_index});
            basis_functions.push_back({atom.x, atom.y, atom.z, 0, 0, 1, atom.atomic_number, shell, atom_index});
        }
    }

    return basis_functions;
}

// Function to return the first factor of the overlap matrix
double calculate_factor(const PrimitiveGaussian &gA, const PrimitiveGaussian &gB, double coordA, double coordB)
{
    double coordDiff = coordA - coordB;
    double expProduct = gA.exponent * gB.exponent;
    double expSum = gA.exponent + gB.exponent;
    double sqrtFactor = std::sqrt(M_PI / expSum);
    return (std::exp(-(expProduct * coordDiff * coordDiff) / expSum)) * sqrtFactor;
}

// Function to return the center:
double calculate_center(const PrimitiveGaussian &gA, const PrimitiveGaussian &gB, double coordA, double coordB)
{
    double expSum = gA.exponent + gB.exponent;
    double center = ((gA.exponent * coordA) + (gB.exponent * coordB)) / expSum;
    return center;
}

// Function to calculate the binomial coefficient:
double calculate_binomial_coefficient(int n, int k)
{
    if (k > n || k < 0)
    {
        return 0.0;
    }

    if (k == 0 || k == n)
    {
        return 1.0;
    }

    double result = 1.0;
    for (int i = 0; i < k; ++i)
    {
        result *= (n - i);
        result /= (i + 1);
    }

    return result;
}

// Function to calculate the double factorial:
double calculate_double_factorial(int n)
{
    if (n <= 1)
    {
        return 1.0;
    }

    double result = 1.0;
    for (int i = n; i > 0; i -= 2)
    {
        result *= i;
    }

    return result;
}

// Function to calculate the summation term of the overlap interval:
double calculate_summation(int la, int lb, double PA, double PB, double expSum)
{
    double sum = 0.0;

    for (int i = 0; i <= la; ++i)
    {
        for (int j = 0; j <= lb; ++j)
        {
            if ((i + j) % 2 == 0)
            {
                double binomialCoeff = calculate_binomial_coefficient(la, i) * calculate_binomial_coefficient(lb, j);
                double powerTerm = std::pow(PA, la - i) * std::pow(PB, lb - j);
                double factorialTerm = calculate_double_factorial(i + j - 1);
                double denominator = std::pow(2 * expSum, static_cast<double>(i + j) / 2.0);

                sum += binomialCoeff * powerTerm * factorialTerm / denominator;
            }
        }
    }
    return sum;
}

// Function calculate the normalization_constant
double calculate_normalization_constant(double exponent, int l, int m, int n)
{
    double term1 = std::pow(2.0, 2.0 * (l + m + n) + 1.5);
    double term2 = std::pow(exponent, (l + m + n) + 1.5);
    double term3 = calculate_double_factorial(2 * l - 1) * calculate_double_factorial(2 * m - 1) * calculate_double_factorial(2 * n - 1) * std::pow(M_PI, 1.5);
    return std::sqrt(term1 * term2 / term3);
}

// Function to calculate the complete integral between two 3D gaussians:
double calculate_single_overlap(const PrimitiveGaussian &gA, const PrimitiveGaussian &gB, int lx_a, int ly_a, int lz_a, int lx_b, int ly_b, int lz_b)
{
    // Calculate the factors for each coordinate:
    double factor_x = calculate_factor(gA, gB, gA.xa, gB.xa);
    double factor_y = calculate_factor(gA, gB, gA.ya, gB.ya);
    double factor_z = calculate_factor(gA, gB, gA.za, gB.za);

    // Calculate the centers for each coordinate:
    double center_x = calculate_center(gA, gB, gA.xa, gB.xa);
    double center_y = calculate_center(gA, gB, gA.ya, gB.ya);
    double center_z = calculate_center(gA, gB, gA.za, gB.za);

    // Calculate the exponent sum:
    double exponent_sum = gA.exponent + gB.exponent;

    // Define the PA and PB distances for each coordinate:
    double PA_x = center_x - gA.xa;
    double PB_x = center_x - gB.xa;
    double PA_y = center_y - gA.ya;
    double PB_y = center_y - gB.ya;
    double PA_z = center_z - gA.za;
    double PB_z = center_z - gB.za;

    // Calculate the summation terms for each coordinate:
    double sum_x = calculate_summation(lx_a, lx_b, PA_x, PB_x, exponent_sum);
    double sum_y = calculate_summation(ly_a, ly_b, PA_y, PB_y, exponent_sum);
    double sum_z = calculate_summation(lz_a, lz_b, PA_z, PB_z, exponent_sum);

    // Calculate the overlap as the product of the terms:
    double overlap = factor_x * factor_y * factor_z * sum_x * sum_y * sum_z;

    return overlap;
}

// Function to calculate the overall contracted overlap:
double calculate_contracted_overlap(const BasisFunction &bf1, const BasisFunction &bf2)
{
    // Instantiate a total_overlap variable
    double total_overlap = 0.0;

    for (const auto &pair1 : bf1.shell_info->exponent_coeff)
    {
        for (const auto &pair2 : bf2.shell_info->exponent_coeff)
        {
            PrimitiveGaussian pg1 = {bf1.x, bf1.y, bf1.z, pair1.first};
            PrimitiveGaussian pg2 = {bf2.x, bf2.y, bf2.z, pair2.first};

            double coeff1 = pair1.second;
            double coeff2 = pair2.second;

            // Calculate the primitive overlap:
            double primitive_overlap = calculate_single_overlap(pg1, pg2, bf1.lx, bf1.ly, bf1.lz, bf2.lx, bf2.ly, bf2.lz);

            // Calculate the normalization constants according to the angular momentum:
            double norm1 = calculate_normalization_constant(pg1.exponent, bf1.lx, bf1.ly, bf1.lz);
            double norm2 = calculate_normalization_constant(pg2.exponent, bf2.lx, bf2.ly, bf2.lz);

            // Calculate the total overlap:
            total_overlap += coeff1 * coeff2 * norm1 * norm2 * primitive_overlap;
        }
    }
    return total_overlap;
}

// Function to return the overlap matrix S:
mat return_overlap_matrix(std::vector<BasisFunction> basis_functions)
{
    // Calculate the size of basis functions:
    int N = basis_functions.size();

    // Instantiate an empty matrix of N * N
    mat overlap_matrix(N, N);

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            overlap_matrix(i, j) = calculate_contracted_overlap(basis_functions[i], basis_functions[j]);
        }
    }
    return overlap_matrix;
}

// Function to return the Huckel-Hamiltonian matrix
mat return_hamiltonian_matrix(const std::vector<BasisFunction> &basis_functions, const mat &overlap_matrix)
{
    // Calculate the size of basis functions:
    int N = basis_functions.size();

    // Instantiate an empty matrix of N * N
    mat hamiltonian_matrix(N, N);

    // Calculate the diagonal elements:
    for (int i = 0; i < N; ++i)
    {
        const auto &bf = basis_functions[i];
        if (bf.atomic_number == HYDROGEN_ATOMIC_NUMBER)
        {
            hamiltonian_matrix(i, i) = h_H_1s;
        }
        else if (bf.atomic_number == CARBON_ATOMIC_NUMBER)
        {
            if (bf.shell_info->momentum == 0)
            {
                hamiltonian_matrix(i, i) = h_C_2s;
            }
            else
            {
                hamiltonian_matrix(i, i) = h_C_2p;
            }
        }
    }

    // Calculate the off-diagonal elements:
    for (int i = 0; i < N; ++i)
    {
        for (int j = i + 1; j < N; ++j)
        {
            double hamiltonian_ij = 0.5 * K * (hamiltonian_matrix(i, i) + hamiltonian_matrix(j, j)) * overlap_matrix(i, j);
            hamiltonian_matrix(i, j) = hamiltonian_ij;
            hamiltonian_matrix(j, i) = hamiltonian_ij;
        }
    }
    return hamiltonian_matrix;
}

// Function to return the eigenvalues(E) and eigenvectors(C)
std::pair<mat, vec> solve_eigenvalue_problem(const mat &S, const mat &H)
{
    // 1. Get the orthogonal transformation:
    // a. find the eigenvalues & eigenvectors of S = U * L * U^T
    vec eig_val_S;
    mat eig_vec_S;
    eig_sym(eig_val_S, eig_vec_S, S);

    // b. Form L^(-1/2) by taking 1/sqrt of each eigenvalue:
    mat L_inv_sqrt = diagmat(1.0 / sqrt(eig_val_S));

    // C. Now reconstruct the matrix X = U * L ^ (1/2) * U ^ T
    mat X = eig_vec_S * L_inv_sqrt * eig_vec_S.t();

    // 2. Form the hamiltonian in the Orthogonalized basis H'
    mat H_prime = X.t() * H * X;

    // 3. Solve the eigenvalue problem for H'
    vec E; // vector to store eigenvalues
    mat C_prime;
    eig_sym(E, C_prime, H_prime);

    // 4. Back transform the eigenvectors to get the final coefficients C:
    mat C = X * C_prime;

    // 5. Return a pair of eigenvector and eigenvalues:
    return std::make_pair(C, E);
}

// Function to calculate the energy:
double calculate_energy(vec ei, int num_electrons)
{
    // 1. Instantiate the energy variable
    double total_energy = 0.0;

    for (int i = 0; i < num_electrons; ++i)
    {
        total_energy += 2.0 * ei[i];
    }

    return total_energy;
}