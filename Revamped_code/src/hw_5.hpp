#ifndef HW_5_HPP
#define HW_5_HPP

#include <iostream>
#include <cmath>
#include <armadillo>
#include <variant>
#include <filesystem>
#include <nlohmann/json.hpp>


namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace arma;

// Declaring some constants:
const int CARBON_ATOMIC_NUMBER = 6;
const int HYDROGEN_ATOMIC_NUMBER = 1;
const double K = 1.75;
const double h_H_1s = -13.6;
const double h_C_2s = -21.4;
const double h_C_2p = -11.4;
const double eV_AU = 27.211324570273;
const double eV_to_Ha = 1.0 / eV_AU;
const double ANGSTROM_TO_BOHR = 1.889726125456;
// To Atomic Units
const double AMU_TO_AU = 1822.888486;
const double AU_TO_WAVENUMBER = 219474.63; //cm^(-1) for wavenumber

struct CNDOParameters
{
    double Is_plus_As;
    double Ip_plus_Ap;
    double beta;
    int Z_star;
};

const std::map<int, CNDOParameters> cndo_params
{
    {1, {7.176, 0.0, -9, 1}},      // Keep in eV!
    {6, {14.051, 5.572, -21.0, 4}},
    {7, {19.316, 7.275, -25.0, 5}},
    {8, {25.390, 9.111, -31.0, 6}},
    {9, {32.272, 11.080, -39.0, 7}}
};

// Create a map of atomic masses:
const std::map<int, double> atomic_masses = {
    {1, 1.00784}, // Hydrogen
    {6, 12.011}, // Carbon
    {7, 14.007}, // Nitrogen
    {8, 15.999}, // Oxygen
    {9, 18.998}, // Fluorine
};

// Create a map of atomic symbols:
const std::map<int, std::string> atomic_symbols = {
    {1, "H"},
    {6, "C"},
    {7, "N"},
    {8, "O"},
    {9, "F"},
};

// Create a struct Atom
struct Atom
{
    int atomic_number;
    double x, y, z;
};

// Create a struct Contracted Shell
struct ContractedShell
{
    int atomic_num;
    int momentum;
    std::vector<std::pair<double, double>> exponent_coeff; 
};

// Create a struct for Primitive gaussian
struct PrimitiveGaussian
{
    double xa, ya, za;
    double exponent;
};

// Create a struct for Basis function
struct BasisFunction
{
    double x, y, z;
    int lx, ly, lz;
    int atomic_number;
    const ContractedShell * shell_info;
    int atom_index;
};

// Struct to return the results of the SCF calculation
struct SCFResults {
    mat P_total;
    mat F_alpha;
    mat F_beta;
    vec epsilon_alpha;
    vec epsilon_beta;
    mat C_alpha;
    mat C_beta;
    double E_electron;
    double E_total;
    double E_nuclear;
};

// Struct to create a molecule
struct Molecule
{
    std::vector<Atom> atoms;
    std::vector<BasisFunction> basis_functions;
    int N;
    int num_alpha_e;
    int num_beta_e;
    std::vector<int> orbital_to_atom;
};

// ---------------------------------------------------------------------------------          For Homework 3         ---------------------------------------------------------------------------------------------------------------------- //


// Function to read in the information from the xyz file:
std::vector<Atom> read_xyz (std::string file_path);

// Function to determine the number of basis functions & number of electrons:
std::pair<int,int> calculate_basis_fun_electrons_pair(std::vector<Atom>atom_system);

// Function to read in the information from Json and construct the gaussian:
ContractedShell construct_contracted_g(const json &data);

// Function to return a list of basis functions:
std::vector<BasisFunction> return_basis_functions(const std::vector<Atom> &atoms, const std::vector<ContractedShell> &basis_set_shells);

// Function to return the first factor of the overlap matrix
double calculate_factor(const PrimitiveGaussian& gA, const PrimitiveGaussian& gB, double coordA, double coordB);

// Function to return the center:
double calculate_center(const PrimitiveGaussian& gA, const PrimitiveGaussian& gB, double coordA, double coordB);

// Function to calculate the binomial coefficient:
double calculate_binomial_coefficient(int n, int k);

// Function to calculate the double factorial:
double calculate_double_factorial(int n);

// Function to calculate the summation term of the overlap interval:
double calculate_summation(int la, int lb, double PA, double PB, double expSum);

// Function calculate the normalization_constant
double calculate_normalization_constant(double exponent, int l, int m, int n);

// Function to calculate the complete integral between two 3D gaussians:
double calculate_single_overlap(const PrimitiveGaussian& gA, const PrimitiveGaussian& gB, int lx_a, int ly_a, int lz_a, int lx_b, int ly_b, int lz_b);

// Function to calculate the overall contracted overlap:
double calculate_contracted_overlap(const BasisFunction &bf1, const BasisFunction &bf2);

// Function to return the overlap matrix S:
mat return_overlap_matrix (std::vector<BasisFunction> basis_functions);

// Function to return the Huckel-Hamiltonian matrix
mat return_hamiltonian_matrix (const std::vector<BasisFunction>& basis_functions, const mat& overlap_matrix);

// Function to return the eigenvalues(E) and eigenvectors(C)
std::pair<mat,vec> solve_eigenvalue_problem(const mat& S, const mat& H);

// Function to calculate the energy:
double calculate_energy(vec ei, int num_electrons);

// ---------------------------------------------------------------------------------          For Homework 4         ---------------------------------------------------------------------------------------------------------------------- //

// Function to calculate the density matrix elements for alpha and beta electrons
void calculate_density_matrices(const mat& C_alpha, const mat& C_beta, int num_alpha_e, int num_beta_e, mat& p_alpha, mat& p_beta);

// Function to calculate the gamma
double calculate_primitive_ERI(double alpha, const vec& A, double beta, const vec& B, double gamma, const vec& C, double delta, const vec& D);

// Function to normalize the s-type gaussian primitives:
double gaussian_norm_s(double exp);

// Function to calculate contracted ERIs:
double contracted_ERI(const BasisFunction& A, const BasisFunction& B, const BasisFunction& C, const BasisFunction& D);

// Helper function to return the Z_charge of the atom
int get_Z_charge(int atomic_num, const std::map<int, CNDOParameters>& parameters);

// Function to build the orbital to atom map:
std::vector<int> build_orbital_to_atom_map(const std::vector<Atom>& atoms, const std::vector<BasisFunction>& basis_functions);

// Function to build the gamma matrix:
mat build_gamma_matrix(const std::vector<Atom>& atoms, const std::vector<int>& orbital_to_atom, const std::vector<BasisFunction>& basis_functions);

// Function to build_core_hamiltonian
mat build_core_hamiltonian(const mat& S, const mat& gamma, const std::vector<int>& orbital_to_atom, const std::vector<Atom> & atoms, const std::vector<BasisFunction>& basis_functions);

// Function to build Fock matrices:
void build_fock_matrices(const mat& H_core, const mat& P_alpha, const mat& P_beta, const mat& gamma, mat& F_alpha, mat& F_beta, const std::vector<int>& orbital_to_atom, const std::vector<Atom>& atoms, const mat& S, const std::vector<BasisFunction>& basis_functions);

// Function to solve the eigenvalue problem:
std::pair<mat, vec> solve_eigenvalue_problem_normal(const mat& F);

// Function to calculate the nuclear repulsion energy:
double calculate_nuclear_repulsion(const std::vector<Atom>& atoms, const std::map<int, CNDOParameters>& cndo_parameters);

// Function to run the SCF function to determine the energy:
SCFResults run_scf(const mat& S, const mat& H_core, const mat& gamma, int num_alpha_e, int num_beta_e, mat& P_alpha, mat& P_beta, double E_nuclear, const std::vector<int>& orbital_to_atom, const std::vector<Atom>& atoms, const std::vector<BasisFunction>& basis_functions);


// ---------------------------------------------------------------------------------          For Homework 5         ---------------------------------------------------------------------------------------------------------------------- //

// Function to calculate the gradient of the nuclear repulsion energy:
mat calculate_gradient_nuclear_repulsion(const std::vector<Atom>& atoms, const std::map<int, CNDOParameters>& cndo_parameters);

// Function to calculate the gradient of the overlap matrix:
mat calculate_overlap_matrix_derivative(const std::vector<BasisFunction>& basis_functions, int target_atom_index, int coordinate_index);

// Function to get the first s orbitals:
std::vector<int> get_first_s_orbitals(const std::vector<Atom>& atoms, const std::vector<BasisFunction>& basis_functions, const std::vector<int>& orbital_to_atom);

// Function to build the Suv_RA matrix efficiently:
mat build_Suv_RA_matrix_efficient(const std::vector<BasisFunction>& basis_functions);

// Function to calculate boys function:
void calculate_FO_F1(double T, double& F0, double& F1);

// Function to calculate the primitive ERIs:
double calculate_primitive_ERI_refactored(double alpha, const vec& A, double beta, const vec& B, double gamma, const vec& C, double delta, const vec& D);

// Function to calculate the derivative of the primitive ERIs:
vec calculate_primitive_ERI_derivative(double alphaA, const BasisFunction& A, double alphaB, const BasisFunction& B, double alphaC, const BasisFunction& C, double alphaD, const BasisFunction& D, int target_atom_index);

// Function to calculate contracted ERIs:
vec calculate_contracted_ERI_derivative(const BasisFunction& A, const BasisFunction& B, const BasisFunction& C, const BasisFunction& D, int target_atom_index);

// Function to return the S orbitals:
std::vector<int> get_first_s_orbitals(const std::vector<Atom>& atoms, const std::vector<BasisFunction>& basis_functions, const std::vector<int>& orbital_to_atom);

// Function to build the gamma matrix derivative:
mat build_gamma_matrix_derivative_efficient(const std::vector<Atom>& atoms, const std::vector<int>& orbital_to_atom, const std::vector<BasisFunction>& basis_functions);

// Function to build the gamma matrix derivative:
mat build_gamma_matrix_derivative(const std::vector<Atom>& atoms, const std::vector<int>& orbital_to_atom, const std::vector<BasisFunction>& basis_functions, int target_atom_index, int coordinate_index);

// Function to build the weighted overlap matrix:
mat build_energy_weighted_matrix(const vec& orbital_energies, const mat& C_matrix, int num_occupied_orbitals);

// Function to build the P_AA vector:
vec calculate_P_AA_vector(int num_atoms, const mat& P_matrix, const std::vector<BasisFunction>& basis_functions);

// Function to build the exchange:
double calculate_P_exch_AB(int atom_A_index, int atom_B_index, const mat& P_matrix, const std::vector<BasisFunction>& basis_functions);

// Function to build the Y matrix:
mat build_Y_matrix(const std::vector<Atom>& atoms, const mat& P_matrix, const std::vector<BasisFunction>& basis_functions, const std::map<int, CNDOParameters>& cndo_parameters);

// Function to build the P_beta matrix:
mat build_P_beta_matrix(const mat& P_matrix, const std::vector<BasisFunction>& basis_functions, const std::map<int, CNDOParameters>& cndo_parameters);

// Function to build the electronic energy gradient:
mat build_electronic_energy_gradient(const std::vector<Atom>& atoms, const std::vector<BasisFunction>& basis_functions, const std::vector<int>& orbital_to_atom, const mat& P_beta, const mat& X, const mat& Y);

// ---------------------------------------------------------------------------------          For Homework 4 Part 2         ---------------------------------------------------------------------------------------------------------------------- //
// Helper function to create the molecules:
Molecule create_molecule(const std::vector<Atom>& atoms,int num_alpha_e, int num_beta_e, const std::vector<ContractedShell>& basis_set_shells);

// Helper function to load all the basis set shells:
std::vector<ContractedShell> load_basis_set_shells();

// Run the SCF calculation for a molecule:
SCFResults run_scf_calculation(const Molecule& molecule);

// Function to calculate the optimized geometry:
void geometry_optimization(Molecule& molecule);

// Function to calculate the optimized geometry:
void geometry_optimization_2(Molecule& molecule);

// Function to perform golden section search for line minimization:
double golden_section_line_search(Molecule& mol, const arma::mat& original_coords, const arma::mat& direction, double a, double b, double c);

// Function to calculate the energy at a given step size along the search direction:
double phi_of_alpha(Molecule& mol, const arma::mat& original_coords, const arma::mat& direction, double alpha);

// Function to calculate the overall gradient (electronic and nuclear):
mat calculate_overall_gradient(const Molecule& molecule);

// ---------------------------------------------------------------------------------          For FINAL PROJECT        ---------------------------------------------------------------------------------------------------------------------- //

// Function to calculate the Hessian matrix:
mat calculate_hessian(Molecule molecule);

// Function to update the basis function centers:
void update_basis_centers(Molecule& molecule);

// Function to calculate the weighted Hessian matrix:
mat calculate_weighted_hessian(const mat& hessian, const Molecule& molecule);

// Function to solve the eigenvalue problem:
std::pair<vec, mat> calculate_eigenvalues_and_eigenvectors(const mat& weighted_hessian);

// Function to calculate the frequencies from the eigenvalues:
std::vector<double> calculate_frequencies(const vec& eigenvalues);

// Function to create a mode animation of the molecule from the frequencies and eigenvectors:
void create_mode_animation(std::string filename, const std::vector<double>& equilibrium_coords, const std::vector<int>& atomic_nums, const std::map<int, std::string>& atomic_symbols, const mat& eigenvectors, int mode_index, int num_frames = 20);

// Function to create a mode animation of the molecule from the frequencies and eigenvectors:
void create_mode_animation_pdb(std::string filename, const std::vector<double>& equilibrium_coords, const std::vector<int>& atomic_nums, const std::map<int, std::string>& atomic_symbols, const mat& eigenvectors, int mode_index, int num_frames = 20);




// Function to calculate the frequencies:
//std::vector<double> calculate_frequencies(const mat& hessian, const Molecule& molecule);

// Function to create a mode animation of the molecule from the frequencies and eigenvectors:
//void create_mode_animation(std::string filename, const std::vector<double>& equilibrium_coords, const std::vector<int>& atomic_nums, const vec& eigenvectors, int num_frames = 20);

#endif