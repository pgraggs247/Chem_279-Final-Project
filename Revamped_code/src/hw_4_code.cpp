#include <iostream>
#include <cmath>
#include <armadillo>
#include <nlohmann/json.hpp>
#include "hw_5.hpp"

using namespace arma;

// Function to calculate the density matrices for alpha and beta electrons:
void calculate_density_matrices(const mat& C_alpha, 
    const mat& C_beta, 
    int num_alpha_e, 
    int num_beta_e, 
    mat& p_alpha, 
    mat& p_beta)
{
    int N = C_alpha.n_rows;
    p_alpha.zeros(N, N);
    p_beta.zeros(N, N);

    // Calculate p_alpha:
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            for (int k = 0; k < num_alpha_e; ++k)
            {
                p_alpha(i, j) += C_alpha(i, k) * C_alpha(j, k);
            }
        }
    }

    // Calculate p_beta:
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            for (int k = 0; k < num_beta_e; ++k)
            {
                p_beta(i, j) += C_beta(i, k) * C_beta(j, k);
            }
        }
    }
}

// Function to calculate the gamma
double calculate_primitive_ERI(double alpha, 
    const vec& A, 
    double beta, 
    const vec& B, 
    double gamma, 
    const vec& C, 
    double delta, 
    const vec& D)
{
    // Add the exponents of same orbitals:
    double p = alpha + beta;
    double q = gamma + delta;

    // Calculate the Kab prefactor
    double Rab2 = std::pow(norm(A - B), 2.0);
    double kab = std::exp(-(alpha * beta /p) * Rab2);

    double Rcd2 = std::pow(norm(C - D), 2.0);
    double kcd = std::exp(-(gamma * delta /q) * Rcd2);

    // Calculate the center for each orbitals:
    vec P = (alpha * A + beta * B) / p;
    vec Q = (gamma * C + delta * D) / q;

    // Calculate oA and oB, Ua and UB, V2, U and T values:
    double oA = 1.0 / p;
    double oB = 1.0 / q;

    double Ua = std::pow(M_PI * oA, 1.5);
    double Ub = std::pow(M_PI * oB, 1.5);
    double U = Ua * Ub;


    double v2 = 1.0 / (oA + oB);
    double R_PQ_squared = std::pow(norm(P - Q), 2.0);
    double T = v2 * R_PQ_squared;

    
    constexpr double epsilon = 1e-12;
    double result;

    if (std::abs(T) < epsilon)
    {
        result = U * std::sqrt(2.0 * v2) * std::sqrt(2.0 / M_PI);
    }
    else
    {
        result = U * (1.0 / std::sqrt(R_PQ_squared)) * std::erf(std::sqrt(T));
    }

    return kab * kcd * result;

}

// Function to normalize the s-type gaussian primitives:
double gaussian_norm_s(double exp)
{
    return std::pow(2 * exp / M_PI, 0.75);
}

// Function to calculate contracted ERIs:
double contracted_ERI(const BasisFunction& A, 
    const BasisFunction& B, 
    const BasisFunction& C, 
    const BasisFunction& D)
{
    // Instantiate the sum as zero:
    double sum = 0.0;

    // Loop over basis functions:
    for(auto& primA: A.shell_info->exponent_coeff)
    {
        double alphaA = primA.first;
        double coeffA = primA.second * gaussian_norm_s(alphaA);
    
        for(auto& primB: B.shell_info->exponent_coeff)
        {
            double alphaB = primB.first;
            double coeffB = primB.second * gaussian_norm_s(alphaB);
        

            for(auto& primC: C.shell_info->exponent_coeff)
            {
                double alphaC = primC.first;
                double coeffC = primC.second * gaussian_norm_s(alphaC);
            

                for(auto& primD: D.shell_info->exponent_coeff)
                {
                    double alphaD = primD.first;
                    double coeffD = primD.second * gaussian_norm_s(alphaD);
                
                    vec RA = {A.x, A.y, A.z};
                    vec RB = {B.x, B.y, B.z};
                    vec RC = {C.x, C.y, C.z};
                    vec RD = {D.x, D.y, D.z};

                    double result = calculate_primitive_ERI(alphaA, RA, alphaB, RB, alphaC, RC, alphaD, RD);

                    sum += coeffA * coeffB * coeffC * coeffD * result;
                }
            }
        }
    }
    return sum;
}

// Function to build the gamma matrix:
mat build_gamma_matrix(const std::vector<Atom>& atoms, 
    const std::vector<int>& orbital_to_atom, 
    const std::vector<BasisFunction>& basis_functions)
{

    int N = atoms.size();
    mat gamma(N, N, fill::zeros);

    // For each pair of atoms, use the gamma value from their first orbitals:
    std::vector<int> first_orbital_on_atom(N, -1);
    for (size_t i = 0; i < basis_functions.size(); ++i)
    {
        int atom_idx = orbital_to_atom[i];
        if (first_orbital_on_atom[atom_idx] == -1)
        {
            first_orbital_on_atom[atom_idx] = i;
        }
    } 

    // Calculate gamma_AB for each atom pair 
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j <= i; ++j)
        {
            int i_A = first_orbital_on_atom[i];
            int j_B = first_orbital_on_atom[j];
            double val = contracted_ERI(basis_functions[i_A], basis_functions[i_A], basis_functions[j_B], basis_functions[j_B]);
            val *= eV_AU;
            gamma(i, j) = val;
            gamma(j, i) = val;
        }
    }
    return gamma;
}

// Function to build the core Hamiltonian matrix:
mat build_core_hamiltonian(const mat& S, 
    const mat& gamma, 
    const std::vector<int>& orbital_to_atom, 
    const std::vector<Atom>& atoms, 
    const std::vector<BasisFunction>& basis_functions) 
{
    // Calculate the number of basis functions and atoms
    int N = basis_functions.size();
    int num_atoms = atoms.size();
    mat H_core(N, N, fill::zeros);

    // Find the first orbital on each atom
    std::vector<int> first_orbital_on_atom(num_atoms, -1);
    for (int i = 0; i < N; ++i) {
        int atom_idx = orbital_to_atom[i];
        if (first_orbital_on_atom[atom_idx] == -1) {
            first_orbital_on_atom[atom_idx] = i;
        }
    }

    for (int i = 0; i < N; ++i) {
        // Get information for orbital i on Atom A
        int atom_A = orbital_to_atom[i];
        int type_A = atoms[atom_A].atomic_number;
        const CNDOParameters& params_A = cndo_params.at(type_A);
        int Z_A = params_A.Z_star;

        // Get first orbital on atom A for gamma indexing
        int i_on_A = first_orbital_on_atom[atom_A];

        // 1. Check if this is an S-shell:
        bool is_S = (basis_functions[i].shell_info->momentum == 0);
        double I_plus_A;
        if (is_S) {
            I_plus_A = params_A.Is_plus_As;
        } else {
            I_plus_A = params_A.Ip_plus_Ap;
        }
        double U_ii = -I_plus_A;

        // 2. Calculate the (Z_A - 1/2)*gamma_AA (atom-level gamma)
        //U_ii -= (Z_A - 0.5) * gamma(i_on_A, i_on_A); --> You made a change here 1.
        U_ii -= (Z_A - 0.5) * gamma(atom_A, atom_A);

        // 3. Calculate -sum_{B != A} (Z_B * gamma_AB)
        for (int atom_B = 0; atom_B < num_atoms; ++atom_B) {
            if (atom_A == atom_B) continue;

            int type_B = atoms[atom_B].atomic_number;
            int Z_B = cndo_params.at(type_B).Z_star;

            // Find gamma_AB = gamma(atom_A_orbital, atom_B_orbital)
            int j_on_B = first_orbital_on_atom[atom_B];
            //double gamma_AB = gamma(i_on_A, j_on_B); --> You made a change here 2.
            double gamma_AB = gamma(atom_A, atom_B);

            U_ii -= Z_B * gamma_AB;
        }
        H_core(i, i) = U_ii;

        // Building the off-diagonal elements (i != j)
        for (int j = 0; j < i; ++j) {
            int atom_B = orbital_to_atom[j];

            if (atom_A == atom_B) {
                H_core(i, j) = 0.0;
                H_core(j, i) = 0.0;
            } else {
                int type_B = atoms[atom_B].atomic_number;
                const CNDOParameters& params_B = cndo_params.at(type_B);

                double beta_sum = params_A.beta + params_B.beta;
                H_core(i, j) = 0.5 * beta_sum * S(i, j);
                H_core(j, i) = H_core(i, j);
            }
        }
    }
    return H_core;
}

// Function to build the Fock matrices -- Referenced heavilyfrom Gemini:
void build_fock_matrices(const mat& H_core, const mat& P_alpha, const mat& P_beta,
    const mat& gamma, 
    mat& F_alpha, mat& F_beta,
    const std::vector<int>& orbital_to_atom,
    const std::vector<Atom>& atoms,
    const mat& S,
    const std::vector<BasisFunction>& basis_functions)
{
    int N = P_alpha.n_rows; // N = 8
    int num_atoms = atoms.size(); // num_atoms = 2

    // 1. Calculate the total density matrix
    mat P_total = P_alpha + P_beta;

    // 2. Compute p_AA^tot (total density on each atom)
    std::vector<double> p_AA_tot(num_atoms, 0.0);
    for (int mu = 0; mu < N; ++mu)
    {
        int atom_mu = orbital_to_atom[mu];
        p_AA_tot[atom_mu] += P_total(mu, mu);
    }

    // 3. Initialize Fock matrices
    F_alpha.zeros(N, N);
    F_beta.zeros(N, N);

    for (int mu = 0; mu < N; ++mu)
    {
        int atom_A = orbital_to_atom[mu]; 
        int type_A = atoms[atom_A].atomic_number;
        const CNDOParameters& params_A = cndo_params.at(type_A);
        int Z_A = params_A.Z_star;

        // Determine I_mu + A_mu based on orbital type (s or p)
        bool is_S = (basis_functions[mu].shell_info->momentum == 0);
        double I_plus_A = is_S ? params_A.Is_plus_As : params_A.Ip_plus_Ap;

        for (int nu = 0; nu < N; ++nu)
        {
            int atom_B = orbital_to_atom[nu]; 

            if (mu == nu)
            {
                // Use the ATOM index atom_A
                double gamma_AA = gamma(atom_A, atom_A);
                
                // Term 1: -1/2(I_μ + A_μ)
                double term1_from_Hcore = H_core(mu, mu);

                // Term 2: [(p_AA^tot) - (p_μμ^α)]γ_AA
                double term2_alpha = (p_AA_tot[atom_A] - P_alpha(mu, mu)) * gamma_AA;
                double term2_beta  = (p_AA_tot[atom_A] - P_beta(mu, mu)) * gamma_AA;
                
                // Term 3: Σ_{C≠A} p_CC^tot * γ_AC
                double term3 = 0.0;
                for (int atom_C = 0; atom_C < num_atoms; ++atom_C)
                {
                    if (atom_C == atom_A) continue;
                    
                    // Use the ATOM indices atom_A and atom_C
                    double gamma_AC = gamma(atom_A, atom_C);
                    term3 += p_AA_tot[atom_C] * gamma_AC;
                }
                
                // F_mu_mu = H_core + two-electron terms
                // The H_core part already contains all Z* and (I+A) terms
                F_alpha(mu, mu) = H_core(mu, mu) + term2_alpha + term3;
                F_beta(mu, mu)  = H_core(mu, mu) + term2_beta + term3;
            }
            else
            {
                // --- Off-diagonal element (Formula 1.5) ---
                
                if (atom_A == atom_B)
                {
                    // One-center, off-diagonal (μ != ν)
                    
                    // Use the ATOM index atom_A
                    double gamma_AA = gamma(atom_A, atom_A);
                    
                    F_alpha(mu, nu) = -P_alpha(mu, nu) * gamma_AA;
                    F_beta(mu, nu)  = -P_beta(mu, nu) * gamma_AA;
                }
                else
                {
                    // Two-center, off-diagonal (H_core part + exchange part)
                    
                    // Use the ATOM indices atom_A and atom_B
                    double gamma_AB = gamma(atom_A, atom_B);
                    
                    // Term 1: H_core part
                    double beta_term = H_core(mu, nu); // This is 0.5 * (B_A + B_B) * S_uv
                    
                    // Term 2: -p_μν^α γ_AB
                    double exchange_alpha = P_alpha(mu, nu) * gamma_AB;
                    double exchange_beta  = P_beta(mu, nu) * gamma_AB;
                    
                    F_alpha(mu, nu) = beta_term - exchange_alpha;
                    F_beta(mu, nu)  = beta_term - exchange_beta;
                }
            }
        }
    }
}

// Helper function to return the Z_charge of the atom
int get_Z_charge(int atomic_num, 
    const std::map<int, CNDOParameters>& parameters)
{
    auto it = parameters.find(atomic_num);
    if (it != parameters.end())
    {
        return it->second.Z_star;
    }
    else
    {
        throw std::runtime_error("Could not find the Z_charge for atomic number" + std::to_string(atomic_num));
    }
}

// Function to calculate the nuclear repulsion energy:
double calculate_nuclear_repulsion(const std::vector<Atom>& atoms, 
        const std::map<int, CNDOParameters>& cndo_parameters)
{
    // 1. Initialize energy to zero:
    double Energy_nuclear = 0.0;

    // 2. Loop over all unique pairs of atoms
    for (size_t i = 0; i < atoms.size(); ++i)
    {
        const Atom& A = atoms[i];
        int Z_A = get_Z_charge(A.atomic_number, cndo_parameters);

        for (size_t j = i + 1; j < atoms.size(); ++j)
        {
            const Atom& B = atoms[j];
            int Z_B = get_Z_charge(B.atomic_number, cndo_parameters);

            double dx = A.x - B.x;
            double dy = A.y - B.y;
            double dz = A.z - B.z;
            double distance = std::sqrt(dx*dx + dy*dy + dz*dz);

            Energy_nuclear += Z_A * Z_B / distance;
        }
    }

    return Energy_nuclear;
}

//function to return the results of the SCF calculation in a vector of variants
SCFResults run_scf(const mat& S, 
    const mat& H_core, 
    const mat& gamma, 
    int num_alpha_e, 
    int num_beta_e, 
    mat& P_alpha, 
    mat& P_beta, 
    double E_nuclear, 
    const std::vector<int>& orbital_to_atom, 
    const std::vector<Atom>& atoms, 
    const std::vector<BasisFunction>& basis_functions)
{
    SCFResults results;
   
    int N = H_core.n_rows;

    // 1. Set the initial density matrices to zero:
    P_alpha.zeros(N, N);
    P_beta.zeros(N, N);

    mat F_alpha(N, N);
    mat F_beta(N, N);
    mat P_alpha_old(N, N);
    mat P_beta_old(N, N);

    double tolerance = 1e-6;
    int max_iterations = 100;
    bool converged = false;

    for (int iter = 0; iter < max_iterations; ++iter)
    {
        //std::cout << "Iteration: " << iter << std::endl;

        // 2. Build the Fock matrices:
        build_fock_matrices(H_core, P_alpha, P_beta, gamma, F_alpha, F_beta, orbital_to_atom, atoms, S, basis_functions);

        //F_alpha.print("Fa");
        //F_beta.print("Fb");

        if (iter == 0) {
            mat F_alpha_initial = F_alpha;
            mat F_beta_initial = F_beta;
            results.F_alpha = F_alpha_initial;
            results.F_beta = F_beta_initial;
        }
        // 3. Solve the eigenvalue problem:
        vec epsilon_alpha, epsilon_beta;
        mat C_alpha, C_beta;

        std::pair<mat, vec> results_alpha = solve_eigenvalue_problem_normal(F_alpha);
        std::pair<mat, vec> results_beta = solve_eigenvalue_problem_normal(F_beta);

        C_alpha = results_alpha.first;
        epsilon_alpha = results_alpha.second;
        C_beta = results_beta.first;
        epsilon_beta = results_beta.second;
        
        //std::cout << "after solving Eigen equation: " << iter << std::endl;
        
        //C_alpha.print("Ca");
        //C_beta.print("Cb");
        
        // 4. Copy old density matrices:
        P_alpha_old = P_alpha;
        P_beta_old = P_beta;

        // 5. Calculate the new density matrices:
        calculate_density_matrices(C_alpha, C_beta, num_alpha_e, num_beta_e, P_alpha, P_beta);
        //std::cout << " p = " << num_alpha_e << " q = " << num_beta_e << std::endl;
        //P_alpha.print("Pa_new");
        //P_beta.print("Pb_new");

        mat P_total = P_alpha + P_beta;

        // Sum populations by atom
        std::vector<double> P_atom(atoms.size(), 0.0);
        for (int mu = 0; mu < N; ++mu) 
        {
            int atom_idx = orbital_to_atom[mu];
            P_atom[atom_idx] += P_total(mu, mu);
        }

        // Print as a column vector
        vec P_t_atoms(atoms.size());
        for (int i = 0; i < atoms.size(); ++i) 
        {
            P_t_atoms(i) = P_atom[i];
        }
        //P_t_atoms.print("P_t");


        // 6. Check for convergence:
        double max_change_alpha = abs(P_alpha - P_alpha_old).max();
        double max_change_beta = abs(P_beta - P_beta_old).max();
        double max_change = std::max(max_change_alpha, max_change_beta);

        if (max_change < tolerance)
        {
            // Print final eigenvalues and eigenvectors AFTER convergence
            converged = true;
            //P_total.print("P_total");
            //epsilon_alpha.print("Ea");
            //epsilon_beta.print("Eb");
            //C_alpha.print("Ca");
            //C_beta.print("Cb");
            results.P_total = P_total;
            results.epsilon_alpha = epsilon_alpha;
            results.epsilon_beta = epsilon_beta;
            results.C_alpha = C_alpha;
            results.C_beta = C_beta;
            break;
        }
    }

    if (!converged)
    {
        std::cerr << "SCF did not converge in within "
        << max_iterations << " iterations" << std::endl;
        throw std::runtime_error("SCF did not converge in within " + std::to_string(max_iterations) + " iterations");
    }

    // 7. Rebuild the final Fock matrices with converged densities:
    build_fock_matrices(H_core, P_alpha, P_beta, gamma, F_alpha, F_beta, orbital_to_atom, atoms, S, basis_functions);

    // 8. Calculate the energy:
    double E_electron = 0.0;
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            E_electron += P_alpha(i, j) * (H_core(i, j) + F_alpha(i, j));
            E_electron += P_beta(i, j) * (H_core(i, j) + F_beta(i, j));
        }
    }

    E_electron *= 0.5;

    double E_total = E_electron + E_nuclear;
    results.E_nuclear = E_nuclear;
    results.E_electron = E_electron;
    results.E_total = E_total;

    //std::cout << "Nuclear repulsion energy is " << E_nuclear << " eV" << std::endl;
    //std::cout << "Electron energy: " << E_electron << " eV" << std::endl;
    //std::cout << "The molecule in file has energy " << E_total << " eV" << std::endl;
    
    return results;
}

// Function to build the orbital to atom map:
std::vector<int> build_orbital_to_atom_map(const std::vector<Atom>& atoms,
    const std::vector<BasisFunction>& basis_functions)
{
    std::vector<int> orbital_to_atom;
    orbital_to_atom.reserve(basis_functions.size());

    const double tol = 1e-6;   // positional tolerance

    for (const auto& bf : basis_functions)
    {
        int owning_atom = -1;

        for (std::size_t idx = 0; idx < atoms.size(); ++idx)
        {
            const Atom& atom = atoms[idx];
            if (std::fabs(bf.x - atom.x) < tol &&
                std::fabs(bf.y - atom.y) < tol &&
                std::fabs(bf.z - atom.z) < tol)
            {
                owning_atom = static_cast<int>(idx);
                break;
            }
        }

        if (owning_atom == -1)
        {
            throw std::runtime_error("Could not match basis function to any atom.");
        }

        orbital_to_atom.push_back(owning_atom);
    }

    return orbital_to_atom;
}

// Function to solve the eigenvalue problem:
std::pair<mat, vec> solve_eigenvalue_problem_normal(const mat& F)
{
    vec epsilon;
    mat C;
    bool status = eig_sym(epsilon, C, F);

    if (!status) {
        throw std::runtime_error("Eigenvalue Diagonalization Failed.");
    }

    return std::make_pair(C, epsilon);
}

// Helper function to create the molecules:
Molecule create_molecule(const std::vector<Atom>& atoms,int num_alpha_e, int num_beta_e, const std::vector<ContractedShell>& basis_set_shells)
{
    std::vector<BasisFunction> basis_functions = return_basis_functions(atoms, basis_set_shells);
    int N = basis_functions.size();
    std::vector<int> orbital_to_atom = build_orbital_to_atom_map(atoms, basis_functions);
    return Molecule{atoms, basis_functions, N, num_alpha_e, num_beta_e, orbital_to_atom};
}

// Helper function to load all the basis set shells:
std::vector<ContractedShell> load_basis_set_shells()
{
    std::vector<ContractedShell> basis_set_shells;
    std::vector<std::string> filepaths = {
        "../basis/C_s_STO3G.json",
        "../basis/C_p_STO3G.json",  
        "../basis/F_s_STO3G.json",
        "../basis/F_p_STO3G.json",
        "../basis/H_s_STO3G.json",  
        "../basis/N_s_STO3G.json",
        "../basis/N_p_STO3G.json",
        "../basis/O_s_STO3G.json",
        "../basis/O_p_STO3G.json"
    };

    for (const auto &filepath : filepaths)
    {
        std::ifstream file(filepath);
        json data = json::parse(file);
        ContractedShell gs = construct_contracted_g(data);
        basis_set_shells.push_back(gs);
    }

    return basis_set_shells;
}

// Run the SCF calculation for a molecule:
SCFResults run_scf_calculation(const Molecule& molecule)
{

    // 1. Calculate the gamma overlap matrix:
    mat gamma = build_gamma_matrix(molecule.atoms, molecule.orbital_to_atom, molecule.basis_functions);

    // 2. Calculate the overlap matrix:
    mat S = return_overlap_matrix(molecule.basis_functions);

    // 3. Build the core Hamiltonian matrix:
    mat H_core = build_core_hamiltonian(S, gamma, molecule.orbital_to_atom, molecule.atoms, molecule.basis_functions);

    // 4. Calculate the nuclear repulsion energy:
    double E_nuclear = calculate_nuclear_repulsion(molecule.atoms, cndo_params);
    E_nuclear *= eV_AU;

    // 5. Initialize the density matrices:
    mat P_alpha(molecule.N, molecule.N, fill::zeros);
    mat P_beta(molecule.N, molecule.N, fill::zeros);

    // 6. Run the SCF calculation:
    SCFResults results = run_scf(S, H_core, gamma, molecule.num_alpha_e, molecule.num_beta_e, P_alpha, P_beta, E_nuclear, molecule.orbital_to_atom, molecule.atoms, molecule.basis_functions);
    return results;
}