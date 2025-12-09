#include "hw_3_code.cpp"
#include "hw_4_code.cpp"
#include "hw_5a_code.cpp"
#include "hessian.cpp"
#include "writter.cpp"
#include <iostream>
#include <armadillo>
#include <nlohmann/json.hpp>
#include "hw_5.hpp"
#include <iomanip>

using namespace arma;
using json = nlohmann::json;

int main()
{
    // 1. Initialize the std::vector of atoms:
    std::vector<Atom> atoms = read_xyz("../atoms/H2O.xyz");

    // 2. Load the basis set shells:
    std::vector<ContractedShell> basis_set_shells = load_basis_set_shells();

    // 3. Return a list of basis functions:
    std::vector<BasisFunction> basis_functions = return_basis_functions(atoms, basis_set_shells);
    int N = basis_functions.size();

    // 4. Build the orbital to atom map:
    std::vector<int> orbital_to_atom = build_orbital_to_atom_map(atoms, basis_functions);
    int num_alpha_e = 4;
    int num_beta_e = 4;

    // 5. Create the molecule:
    Molecule molecule = create_molecule(atoms, num_alpha_e, num_beta_e, basis_set_shells);

    // 6. Run the SCF calculation:
    SCFResults results = run_scf_calculation(molecule);

    // 7. Build the weighted overlap matrix:
    mat X = build_energy_weighted_matrix(results.epsilon_alpha, results.C_alpha, num_alpha_e) + build_energy_weighted_matrix(results.epsilon_beta, results.C_beta, num_beta_e);

    // 8. Build the Y matrix:
    mat Y = build_Y_matrix(atoms, results.P_total, molecule.basis_functions, cndo_params);

    // 8.b. Build the P_beta matrix:
    mat P_beta = build_P_beta_matrix(results.P_total, molecule.basis_functions, cndo_params);

    // 9. Build the Suv_RA matrix efficiently:
    mat Suv_RA = build_Suv_RA_matrix_efficient(basis_functions);
    Suv_RA.print("Suv_RA");

    // 10. Build the gamma matrix derivative:
    mat gamma_derivative = build_gamma_matrix_derivative_efficient(atoms, orbital_to_atom, basis_functions);
    gamma_derivative.print("gammaAB_RA");

    // 11. Calculate the gradient of the nuclear repulsion energy:
    mat gradient_nuclear_repulsion = calculate_gradient_nuclear_repulsion(atoms, cndo_params);
    gradient_nuclear_repulsion.t().print("gradient_nuclear");

    // 12. Build the electronic energy gradient:
    mat gradient_electronic = build_electronic_energy_gradient(atoms, basis_functions, orbital_to_atom, P_beta, X, Y);
    gradient_electronic.print("gradient_electronic");

    // 13. Calculate the total gradient:
    mat total_gradient = gradient_electronic + gradient_nuclear_repulsion.t();
    total_gradient.print("Gradient");

    // 14. Optimize the geometry:
    geometry_optimization_2(molecule);
    std::cout << "Geometry optimized" << std::endl;

    // 15. Calculate the Hessian matrix from the optimized geometry:
    mat hessian = calculate_hessian(molecule);
    hessian.print("Hessian");

    // 16. Calculate the weighted Hessian matrix:
    mat weighted_hessian = calculate_weighted_hessian(hessian, molecule);
    weighted_hessian.print("Weighted Hessian");

    // 17. Solve the eigenvalue problem:
    std::pair<vec, mat> eigenvalues_and_eigenvectors = calculate_eigenvalues_and_eigenvectors(weighted_hessian);
    eigenvalues_and_eigenvectors.first.print("Eigenvalues");
    eigenvalues_and_eigenvectors.second.print("Eigenvectors");

    // 18. Calculate the frequencies:
    std::vector<double> frequencies = calculate_frequencies(eigenvalues_and_eigenvectors.first);

    // 19. a. create a vector of equilibrium coordinates:
    std::vector<double> equilibrium_coords;
    for (const auto& atom : molecule.atoms)
    {
        equilibrium_coords.push_back(atom.x);
        equilibrium_coords.push_back(atom.y);
        equilibrium_coords.push_back(atom.z);
    }
    std::cout << "Equilibrium coordinates:" << std::endl;
    for (const auto& coord : equilibrium_coords)
    {
        std::cout << coord << " ";
    }
    std::cout << std::endl;

    // 19. b. create a vector of atomic numbers:
    std::vector<int> atomic_nums;
    for (const auto& atom : molecule.atoms)
    {
        atomic_nums.push_back(atom.atomic_number);
    }
    std::cout << "Atomic numbers:" << std::endl;
    for (const auto& atomic_num : atomic_nums)
    {
        std::cout << atomic_num << " ";
    }
    std::cout << std::endl;

    // 19. Create the mode animation:
    create_mode_animation_pdb("mode_animation.pdb", equilibrium_coords, atomic_nums, atomic_symbols, eigenvalues_and_eigenvectors.second, 6);

    return 0;
}