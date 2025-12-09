#include <iostream>
#include <armadillo>
#include <nlohmann/json.hpp>
#include "hw_5.hpp"

using namespace arma;

// ---------------------------------------------------------------------------------          For Homework 5 Part A         ---------------------------------------------------------------------------------------------------------------------- //

// 1. Function to calculate the derivative of the nuclear repulsion energy:
mat calculate_gradient_nuclear_repulsion(const std::vector<Atom>& atoms, 
                                         const std::map<int, 
                                         CNDOParameters>& cndo_parameters)
{
    // 1. Find the size of the matrix:
    size_t num_atoms = atoms.size();
    mat gradient_nuclear(num_atoms, 3, fill::zeros);

    // 2. Loop over all unique pairs of atoms
    for (size_t i = 0; i < num_atoms; ++i)
    {
        const Atom& A = atoms[i];
        int Z_A = get_Z_charge(A.atomic_number, cndo_parameters);
        vec r_A = {A.x, A.y, A.z};

        for (size_t j = i+1; j < num_atoms; ++j)
        {
            const Atom& B = atoms[j];
            int Z_B = get_Z_charge(B.atomic_number, cndo_parameters);
            vec r_B = {B.x, B.y, B.z};

            // 3. Make the distance vector:
            vec distance_vector = r_A - r_B;
            double distance = norm(distance_vector);

            // 4. Calculate the distance cubed:
            double distance_cubed = std::pow(distance, 3);

            // 5. Calculate the gradient of the nuclear repulsion energy in eV/Å:
            vec gradient_term = -(Z_A * Z_B * distance_vector) / distance_cubed;
            gradient_nuclear.row(i) += gradient_term.t() * eV_AU;
            gradient_nuclear.row(j) -= gradient_term.t() * eV_AU;
        }
    }
    return gradient_nuclear;
}

// 2. Function to calculate the gradient of the overlap matrix:
// 2.a. Helper function to calculate_1D_overlap:
double calculate_1D_overlap(const PrimitiveGaussian& gA, 
                            const PrimitiveGaussian& gB, 
                            double coordA, 
                            double coordB, 
                            double la, 
                            double lb)
{
    // 1. calculate the factor:
    double factor = calculate_factor(gA, gB, coordA, coordB);

    // 2. calculate the center:
    double center = calculate_center(gA, gB, coordA, coordB);

    // 3. calculate the PA and PB distances:
    double PA = center - coordA;
    double PB = center - coordB;

    // 4. calculate the exponent sum:
    double exponent_sum = gA.exponent + gB.exponent;

    // 5. calculate the summation term:
    double sum = calculate_summation(la, lb, PA, PB, exponent_sum);

    // 6. return the overlap:
    return factor * sum;
}

// 2.b. Function to calculate the primitive overlap derivative:
void calculate_primitive_overlap_derivative(const PrimitiveGaussian& gA, const PrimitiveGaussian& gB, 
                                            int lx_a, int ly_a, int lz_a, 
                                            int lx_b, int ly_b, int lz_b, 
                                            vec& grad_A, vec& grad_B)
{
    //1. Calculate the overlap needed:
    double Sx_00 = calculate_1D_overlap(gA, gB, gA.xa, gB.xa, lx_a, lx_b);
    double Sx_upA = calculate_1D_overlap(gA, gB, gA.xa, gB.xa, lx_a+1, lx_b);
    double Sx_downA = (lx_a > 0) ? calculate_1D_overlap(gA, gB, gA.xa, gB.xa, lx_a-1, lx_b) : 0.0;
    double Sx_upB = calculate_1D_overlap(gA, gB, gA.xa, gB.xa, lx_a, lx_b+1);
    double Sx_downB = (lx_b > 0) ? calculate_1D_overlap(gA, gB, gA.xa, gB.xa, lx_a, lx_b-1) : 0.0;

    // 2. Calculate the derivative of the overlap for the y coordinate:
    double Sy_00 = calculate_1D_overlap(gA, gB, gA.ya, gB.ya, ly_a, ly_b);
    double Sy_upA = calculate_1D_overlap(gA, gB, gA.ya, gB.ya, ly_a+1, ly_b);
    double Sy_downA = (ly_a > 0) ? calculate_1D_overlap(gA, gB, gA.ya, gB.ya, ly_a-1, ly_b) : 0.0;
    double Sy_upB = calculate_1D_overlap(gA, gB, gA.ya, gB.ya, ly_a, ly_b+1);
    double Sy_downB = (ly_b > 0) ? calculate_1D_overlap(gA, gB, gA.ya, gB.ya, ly_a, ly_b-1) : 0.0;

    // 3. Calculate the derivative of the overlap for the z coordinate:
    double Sz_00 = calculate_1D_overlap(gA, gB, gA.za, gB.za, lz_a, lz_b);
    double Sz_upA = calculate_1D_overlap(gA, gB, gA.za, gB.za, lz_a+1, lz_b);
    double Sz_downA = (lz_a > 0) ? calculate_1D_overlap(gA, gB, gA.za, gB.za, lz_a-1, lz_b) : 0.0;
    double Sz_upB = calculate_1D_overlap(gA, gB, gA.za, gB.za, lz_a, lz_b+1);
    double Sz_downB = (lz_b > 0) ? calculate_1D_overlap(gA, gB, gA.za, gB.za, lz_a, lz_b-1) : 0.0;

    // 4. Calculate the derivative of the overlap:
    // d/dAx =
    double dSx_dAx = 2.0 * gA.exponent * Sx_upA - (lx_a > 0 ? lx_a * Sx_downA : 0.0);
    double dSy_dAy = 2.0 * gA.exponent * Sy_upA - (ly_a > 0 ? ly_a * Sy_downA : 0.0);
    double dSz_dAz = 2.0 * gA.exponent * Sz_upA - (lz_a > 0 ? lz_a * Sz_downA : 0.0);

    // d/dBx =
    double dSx_dBx = 2.0 * gB.exponent * Sx_upB - (lx_b > 0 ? lx_b * Sx_downB : 0.0);
    double dSy_dBy = 2.0 * gB.exponent * Sy_upB - (ly_b > 0 ? ly_b * Sy_downB : 0.0);
    double dSz_dBz = 2.0 * gB.exponent * Sz_upB - (lz_b > 0 ? lz_b * Sz_downB : 0.0);

    // 5. Calculate the gradient of the overlap:
    grad_A(0) = dSx_dAx + dSy_dAy + dSz_dAz;
    grad_A(1) = dSy_dAy + dSy_dBy + dSz_dAz;

    // Fill in the gradient for Atom A:
    grad_A(0) = dSx_dAx * Sy_00 * Sz_00;
    grad_A(1) = Sx_00 * dSy_dAy * Sz_00;
    grad_A(2) = Sx_00 * Sy_00 * dSz_dAz;

    // Fill in the gradient for Atom B:
    grad_B(0) = dSx_dBx * Sy_00 * Sz_00;
    grad_B(1) = Sx_00 * dSy_dBy * Sz_00;
    grad_B(2) = Sx_00 * Sy_00 * dSz_dBz;

    // 6. Return the gradient:
    return;
}

// 2.c. Function to calculate the contracted overlap derivative:
vec calculate_contracted_overlap_derivative(const BasisFunction& bf1, 
                                            const BasisFunction& bf2, 
                                            int target_atom_index)
{
    //1. Instantiate a vector to store the gradient:
    vec tot_gradient(3, fill::zeros);

    //2. Check if the target atom index is the same as the first basis function:
    bool bf1_on_target = bf1.atom_index == target_atom_index;
    bool bf2_on_target = bf2.atom_index == target_atom_index;
    
    if (!bf1_on_target && !bf2_on_target)
    {
        return tot_gradient;
    }

    //2. Loop over the exponent coefficients:
    for (const auto& pair1 : bf1.shell_info->exponent_coeff)
    {
        for (const auto& pair2 : bf2.shell_info->exponent_coeff)
        {
            PrimitiveGaussian pg1 = {bf1.x, bf1.y, bf1.z, pair1.first};
            PrimitiveGaussian pg2 = {bf2.x, bf2.y, bf2.z, pair2.first};

            double coeff1 = pair1.second;
            double coeff2 = pair2.second;

            // Calculate the normalization constants:
            double norm1 = calculate_normalization_constant(pg1.exponent, bf1.lx, bf1.ly, bf1.lz);
            double norm2 = calculate_normalization_constant(pg2.exponent, bf2.lx, bf2.ly, bf2.lz);

            // Calculate the primitive overlap:
            vec grad_A(3);
            vec grad_B(3);
            calculate_primitive_overlap_derivative(pg1, pg2, bf1.lx, bf1.ly, bf1.lz, bf2.lx, bf2.ly, bf2.lz, grad_A, grad_B);

            // Calculate the total gradient:
            double derivative = coeff1 * coeff2 * norm1 * norm2;
            if (bf1_on_target)
            {
                tot_gradient += derivative * grad_A;
            }
            if (bf2_on_target)
            {
                tot_gradient += derivative * grad_B;
            }
        }
    }

    // 3. Return the total gradient:
    return tot_gradient;
}

// 2.d. Function to calculate the overlap matrix derivative:
mat calculate_overlap_matrix_derivative(const std::vector<BasisFunction>& basis_functions, 
                                        int target_atom_index, 
                                        int coordinate_index)
{
    //1. Instantiate a matrix to store the gradient:
    size_t N = basis_functions.size();
    mat tot_gradient(N, N, fill::zeros);

    //2. Loop over the basis functions:
    for (size_t i = 0; i < N; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            vec gradient = calculate_contracted_overlap_derivative(basis_functions[i], basis_functions[j], target_atom_index);
            tot_gradient(i, j) = gradient(coordinate_index);
        }
    }
    return tot_gradient;
}

// 2.e. Function to build the Suv_RA matrix efficiently:
mat build_Suv_RA_matrix_efficient(const std::vector<BasisFunction>& basis_functions)
{
    size_t N = basis_functions.size();
    mat Suv_RA(3, N * N);

    // Loop through all (i, j) pairs
    for (size_t i = 0; i < N; ++i)
    {
        const BasisFunction& bf1 = basis_functions[i];
        
        // Get the atom index that basis function i is centered on
        int atom_of_i = bf1.atom_index; 

        for (size_t j = 0; j < N; ++j)
        {
            const BasisFunction& bf2 = basis_functions[j];

            // 1. Get the flattened column index for this (i, j) pair
            int col_idx = i * N + j;

            // 2. Calculate the 3D gradient vector [dS/dRx, dS/dRy, dS/dRz]
            vec gradient_3d = calculate_contracted_overlap_derivative(
                bf1, bf2, atom_of_i
            );

            // 3. Assign the 3D vector directly to the column
            Suv_RA.col(col_idx) = gradient_3d;
        }
    }

    return Suv_RA;
}

// 3. Function to calculate boys function:
void calculate_FO_F1(double T, double& F0, double& F1)
{
    constexpr double epsilon = 1e-12;
    if (std::abs(T) < epsilon)
    {
        F0 = 1.0;
        F1 = 1.0 / 3.0;
    }
    else
    {
        double sqrt_T = std::sqrt(T);
        F0 = (0.5 * std::sqrt(M_PI / T)) * std::erf(sqrt_T);
        F1 = (F0 - std::exp(-T)) / (2.0 * T);
    }
}

// 4. Refactored code to calculate the primitive ERIs:
double calculate_primitive_ERI_refactored(double alpha, const vec& A, 
                                          double beta, const vec& B, 
                                          double gamma, const vec& C, 
                                          double delta, const vec& D)
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

    double v2 = (p * q) / (p + q);
    double R_PQ_squared = std::pow(norm(P - Q), 2.0);
    double T = v2 * R_PQ_squared;

    constexpr double epsilon = 1e-12;
    double result;

    double F0, F1;
    calculate_FO_F1(T, F0, F1);

    double prefactor = (2.0 * std::pow(M_PI, 2.5)) / (p * q * std::sqrt(p + q));

    result = prefactor * F0;

    return kab * kcd * result;
}

// 5. Function to calculate the derivative of the primitive ERIs:
vec calculate_primitive_ERI_derivative(double alphaA, const BasisFunction& A, 
                                        double alphaB, const BasisFunction& B, 
                                        double alphaC, const BasisFunction& C, 
                                        double alphaD, const BasisFunction& D,
                                        int target_atom_index)

{
    // instantiate vectors:
    vec RA = {A.x, A.y, A.z};
    vec RB = {B.x, B.y, B.z};
    vec RC = {C.x, C.y, C.z};
    vec RD = {D.x, D.y, D.z};

    double p = alphaA + alphaB;
    double q = alphaC + alphaD;

    // Calculate the Kab prefactor
    double Rab2 = std::pow(norm(RA - RB), 2.0);
    double kab = std::exp(-(alphaA * alphaB /p) * Rab2);

    double Rcd2 = std::pow(norm(RC - RD), 2.0);
    double kcd = std::exp(-(alphaC * alphaD /q) * Rcd2);
    
    vec P = (alphaA * RA + alphaB * RB) / p;
    vec Q = (alphaC * RC + alphaD * RD) / q;
    vec PQ = P - Q;

    double v2 = (p * q) / (p + q);
    double R_PQ_squared = std::pow(norm(P - Q), 2.0);
    double T = v2 * R_PQ_squared;

    double F0, F1;
    calculate_FO_F1(T, F0, F1);
    
    double prefactor = (2.0 * std::pow(M_PI, 2.5)) / (p * q * std::sqrt(p + q));
    double dERI_dT = kab * kcd * prefactor * (-F1);

    double change_ak = (A.atom_index == target_atom_index) ? 1.0 : 0.0;
    double change_bk = (B.atom_index == target_atom_index) ? 1.0 : 0.0;
    double change_ck = (C.atom_index == target_atom_index) ? 1.0 : 0.0;
    double change_dk = (D.atom_index == target_atom_index) ? 1.0 : 0.0;

    double Ck = (1.0 / p) * (change_ak * alphaA + change_bk * alphaB) - 
                (1.0 / q) * (change_ck * alphaC + change_dk * alphaD);

    vec gradient_R_PQ_squared = 2.0 * Ck * PQ;
    vec gradient_T = v2 * gradient_R_PQ_squared;

    return dERI_dT * gradient_T;
}

// 6. Function to calculate contracted ERIs:
vec calculate_contracted_ERI_derivative(const BasisFunction& A, 
    const BasisFunction& B, 
    const BasisFunction& C, 
    const BasisFunction& D,
    int target_atom_index)
{
    // Instantiate the sum as zero:
    vec tot_gradient(3, fill::zeros);

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
                
                    // calculate the gradient for this set of primitives:
                    vec primitive_gradient = calculate_primitive_ERI_derivative(alphaA, A, alphaB, B, alphaC, C, alphaD, D, target_atom_index);

                    // add the gradient to the total gradient:
                    tot_gradient += coeffA * coeffB * coeffC * coeffD * primitive_gradient;
                }
            }
        }
    }
    return tot_gradient;
}

// 7. Function to get the S orbitals:
std::vector<int> get_first_s_orbitals(const std::vector<Atom>& atoms, 
                                      const std::vector<BasisFunction>& basis_functions,
                                      const std::vector<int>& orbital_to_atom)
{
    int N = atoms.size();
    std::vector<int> first_s_orbitals(N, -1);
    for (size_t i = 0; i < basis_functions.size(); ++i)
    {
        int atom_idx = orbital_to_atom[i];
        bool is_s_orbital = basis_functions[i].lx == 0 
                            && basis_functions[i].ly == 0 
                            && basis_functions[i].lz == 0;
        if (is_s_orbital && first_s_orbitals[atom_idx] == -1)
        {
            first_s_orbitals[atom_idx] = i;
        }
    }
    return first_s_orbitals;
}

// 8. Function to build the gamma matrix derivative:
mat build_gamma_matrix_derivative_efficient(const std::vector<Atom>& atoms,
                                  const std::vector<int>& orbital_to_atom, 
                                  const std::vector<BasisFunction>& basis_functions)
{
    int N = atoms.size();
    mat gamma_derivative(3, N * N);

    // For each pair of atoms, use the gamma value from their first orbitals:
    std::vector<int> first_s_orbitals = get_first_s_orbitals(atoms, basis_functions, orbital_to_atom);

    // Loop over the basis functions and get the first s orbital for each atom:
    for (size_t i = 0; i < N; ++i)
    {
        int target_atom_idx = i;
        const BasisFunction& bf_i = basis_functions[first_s_orbitals[i]];

        for (size_t j = 0; j < N; ++j)
        {
            const BasisFunction& bf_j = basis_functions[first_s_orbitals[j]];

            int col_idx = i * N + j;
            vec gradient = calculate_contracted_ERI_derivative(bf_i, bf_i, bf_j, bf_j, target_atom_idx);
            gamma_derivative.col(col_idx) = gradient * eV_AU;
        }
    } 
    return gamma_derivative;
}

// 9. Function to build the gamma matrix derivative:
mat build_gamma_matrix_derivative(const std::vector<Atom>& atoms,
                                  const std::vector<int>& orbital_to_atom, 
                                  const std::vector<BasisFunction>& basis_functions,
                                  int target_atom_index,
                                  int coordinate_index)
{
    int N = atoms.size();
    mat gamma_derivative(N, N, fill::zeros);

    std::vector<int> first_orbital_on_atom(N, -1);
    for (size_t i = 0; i < basis_functions.size(); ++i)
    {
        int atom_idx = orbital_to_atom[i];
        if (first_orbital_on_atom[atom_idx] == -1 && basis_functions[i].lx == 0)
        {
            first_orbital_on_atom[atom_idx] = i;
        }
    }
    // calculate dgamma_AB/dR_A for each atom pair:
    for (size_t i = 0; i < N; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            int i_A = first_orbital_on_atom[i];
            int j_B = first_orbital_on_atom[j];

            // calculate the gradient of the contracted ERIs for the target atom and coordinate:
            vec gradient = calculate_contracted_ERI_derivative(basis_functions[i_A], basis_functions[i_A], basis_functions[j_B], basis_functions[j_B], target_atom_index);
            
            // get the value of the gradient for the target atom and coordinate:
            double value = gradient(coordinate_index);
            value *= eV_AU;
            gamma_derivative(i, j) = value;
            gamma_derivative(j, i) = value;
        }
    }
    return gamma_derivative;
}

// 10. Function to build the weighted overlap matrix:
mat build_energy_weighted_matrix(const vec& orbital_energies,
                                 const mat& C_matrix, 
                                 int num_occupied_orbitals)
{
    int N = C_matrix.n_rows;
    mat X(N, N, fill::zeros); // return the X matrix

    for (size_t i = 0; i < N; ++i)
    {   
        for (size_t j = 0; j < N; ++j)
        {
            double sum = 0.0;
            for (size_t k = 0; k < num_occupied_orbitals; ++k)
            {
                double epsilon_k = orbital_energies(k);
                double C_ik = C_matrix(i, k);
                double C_jk = C_matrix(j, k);
                sum += 2.0 * epsilon_k * C_ik * C_jk;
            }
            X(i, j) = sum;
        }
    }
    return X;
}

// 11. Function to build the P_AA vector:
vec calculate_P_AA_vector(int num_atoms, 
                          const mat& P_matrix, 
                          const std::vector<BasisFunction>& basis_functions)
{
    // 1. Instantiate the vector:
    vec P_AA(num_atoms, fill::zeros);
    int N = basis_functions.size();

    // 2. Loop over all diagonal elements of the density matrix:
    for (size_t i = 0; i < N; ++i)
    {
        int atom_idx = basis_functions[i].atom_index;
        P_AA(atom_idx) += P_matrix(i, i);
    }
    return P_AA;
}

// 12. Function to build the exchange:
double calculate_P_exch_AB(int atom_A_index,
                           int atom_B_index,
                           const mat& P_matrix,
                           const std::vector<BasisFunction>& basis_functions)
{
    // 1. Instantiate the sum:
    double P_exch_AB = 0.0;
    size_t N = basis_functions.size();

    std::vector<int> A_indices;
    std::vector<int> B_indices;
    for (size_t i = 0; i < N; ++i)
    {
        if (basis_functions[i].atom_index == atom_A_index)
        {
            A_indices.push_back(i);
        }
        if (basis_functions[i].atom_index == atom_B_index)
        {
            B_indices.push_back(i);
        }
    }

    for (int i : A_indices)
    {
        for (int j : B_indices)
        {
            P_exch_AB += std::pow(P_matrix(i, j), 2.0);
        }
    }
    return P_exch_AB;
}

// 13. Function to build the Y matrix:
mat build_Y_matrix(const std::vector<Atom>& atoms,
                   const mat& P_matrix, 
                   const std::vector<BasisFunction>& basis_functions,
                   const std::map<int, CNDOParameters>& cndo_parameters)
{
    // 1. Instantiate the matrix:
    int N = atoms.size();
    mat Y_matrix(N, N, fill::zeros);

    // 2. Pre calculate the P_AA values:
    vec P_AA = calculate_P_AA_vector(N, P_matrix, basis_functions);

    // 3. Loop over all pairs of atoms:
    for (size_t i = 0; i < N; ++i)
    {
        int Z_A = cndo_parameters.at(atoms[i].atomic_number).Z_star;
        double P_AA_i = P_AA(i);

        for (size_t j = i; j < N; ++j)
        {
            double P_exch_AB = calculate_P_exch_AB(i, j, P_matrix, basis_functions);
            
            if (i == j)
            {
                double Y_ii = 0.5 * (P_AA_i * P_AA_i - 0.5 * P_exch_AB);
                Y_matrix(i, i) = Y_ii;
            }
            else
            {
                double P_BB_j = P_AA(j);
                int Z_B = cndo_parameters.at(atoms[j].atomic_number).Z_star;
                double Y_ij = (P_AA_i * P_BB_j - 0.5 * P_exch_AB);
                double Y_ji = -(P_AA_i * Z_B + P_BB_j * Z_A);

                double Y_ij_total = Y_ij + Y_ji;
                Y_matrix(i, j) = Y_ij_total;
                Y_matrix(j, i) = Y_ij_total;
            }
        }
    }
    return Y_matrix;
}

// 14. Function to calculate the P_beta matrix:
mat build_P_beta_matrix(const mat& P_matrix, 
                        const std::vector<BasisFunction>& basis_functions,
                        const std::map<int, CNDOParameters>& cndo_parameters)
{
    int N = basis_functions.size();
    mat P_beta(N, N, fill::zeros);

    for (size_t i = 0; i < N; ++i)
    {
        const auto& bf_i = basis_functions[i];
        int atom_A_number = bf_i.atomic_number;

        double beta_A = cndo_parameters.at(atom_A_number).beta;

        for (size_t j = i + 1; j < N; ++j)
        {
            const auto& bf_j = basis_functions[j];
            if (bf_i.atom_index == bf_j.atom_index) continue;

            int atom_B_number = bf_j.atomic_number;
            double beta_B = cndo_parameters.at(atom_B_number).beta;

            double beta_AB = 0.5 * (beta_A + beta_B);

            P_beta(i, j) = P_matrix(i, j) * beta_AB;
            P_beta(j, i) = P_matrix(j, i) * beta_AB;            
        }
    }
    return P_beta;
}

// 15. Function to build electronic energy gradient:
mat build_electronic_energy_gradient(const std::vector<Atom>& atoms,
                                    const std::vector<BasisFunction>& basis_functions,
                                    const std::vector<int>& orbital_to_atom,
                                    const mat& P_beta,
                                    const mat& X,
                                    const mat& Y)
{
    // 1. Instantiate the gradient matrix:
    int N = atoms.size();
    mat gradient_electronic(3, N, fill::zeros); 
    
    // 2. Loop over all atoms and coordinates:
    for (size_t i = 0; i < N; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            // Calculate the overall S matrix derivative:
            mat S_der = calculate_overlap_matrix_derivative(basis_functions, i, j);
            double gradient_term_1 = arma::accu(P_beta % S_der);

            // Calculate the overall gamma matrix derivative:
            mat gamma_der = build_gamma_matrix_derivative(atoms, orbital_to_atom, basis_functions, i, j);
            double gradient_term_2 = 0.5 * arma::accu(Y % gamma_der);

            gradient_electronic(j,i) = gradient_term_1 + gradient_term_2;
        }
    }
    return gradient_electronic;
}

// 16. Helper function to evaluate the energy at a given step size along the search direction:
double phi_of_alpha(Molecule& mol, const arma::mat& original_coords, const arma::mat& direction, double alpha)
{
    // 1. Update the molecule coordinates temporarily
    for (size_t i = 0; i < mol.atoms.size(); ++i)
    {
        mol.atoms[i].x = original_coords(i, 0) + alpha * direction(0, i);
        mol.atoms[i].y = original_coords(i, 1) + alpha * direction(1, i);
        mol.atoms[i].z = original_coords(i, 2) + alpha * direction(2, i);
    }

    // 2. Update basis function coordinates to match the new atomic positions
    for (size_t i = 0; i < mol.basis_functions.size(); ++i)
    {
        int atom_idx = mol.basis_functions[i].atom_index;
        mol.basis_functions[i].x = mol.atoms[atom_idx].x;
        mol.basis_functions[i].y = mol.atoms[atom_idx].y;
        mol.basis_functions[i].z = mol.atoms[atom_idx].z;
    }

    // 3. Run SCF calculation at the new geometry
    SCFResults results = run_scf_calculation(mol);

    // 4. Return the total energy
    return results.E_total;
}

// 17. Function to perform golden section search for line minimization:
// 16. Function to perform golden section search for line minimization
double golden_section_line_search(Molecule& mol, const arma::mat& original_coords, 
                                   const arma::mat& direction, double a, double b, double c)
{
    // 1. Define constants for the golden section search:
    const double R = 0.618033988; // Golden ratio constant (1/phi)
    const double tolerance = 1e-4; // Convergence tolerance for step size

    // 2. Initialize the test points:
    double x1 = c - R * (c - a); // Interior point closer to a
    double x2 = a + R * (c - a); // Interior point closer to c

    // 3. Evaluate the energy at the initial test points:
    double f1 = phi_of_alpha(mol, original_coords, direction, x1);
    double f2 = phi_of_alpha(mol, original_coords, direction, x2);

    // 4. Iteratively narrow the bracket until convergence:
    while (std::abs(c - a) > tolerance)
    {
        if (f1 < f2)
        {
            // Minimum is in [a, x2], so update the bracket:
            c = x2;
            x2 = x1;
            f2 = f1;
            x1 = c - R * (c - a);
            f1 = phi_of_alpha(mol, original_coords, direction, x1);
        }
        else
        {
            // Minimum is in [x1, c], so update the bracket:
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = a + R * (c - a);
            f2 = phi_of_alpha(mol, original_coords, direction, x2);
        }
    }

    // 5. Return the midpoint of the final bracket as the optimal step size:
    return (a + c) / 2.0;
}

// 18. Function to calculate the optimized geometry:
// 17. Function to calculate the optimized geometry:
void geometry_optimization(Molecule& molecule)
{
    // 1. Set optimization parameters:
    double gradient_threshold = 1e-4;
    double initial_step = 0.5;
    int max_iterations = 100;

    // 2. Run initial SCF calculation:
    SCFResults results = run_scf_calculation(molecule);
    double E_initial = results.E_total;

    // 3. Store original positions:
    mat original_positions(molecule.atoms.size(), 3, fill::zeros);
    for (size_t i = 0; i < molecule.atoms.size(); ++i)
    {
        original_positions(i, 0) = molecule.atoms[i].x;
        original_positions(i, 1) = molecule.atoms[i].y;
        original_positions(i, 2) = molecule.atoms[i].z;
    }

    // 4. Main optimization loop:
    int iter = 0;
    for (iter = 0; iter < max_iterations; ++iter)
    {
        // 4.a. Build the necessary matrices for gradient calculation:
        mat X = build_energy_weighted_matrix(results.epsilon_alpha, results.C_alpha, molecule.num_alpha_e) + 
                build_energy_weighted_matrix(results.epsilon_beta, results.C_beta, molecule.num_beta_e);
        mat Y = build_Y_matrix(molecule.atoms, results.P_total, molecule.basis_functions, cndo_params);
        mat P_beta = build_P_beta_matrix(results.P_total, molecule.basis_functions, cndo_params);
        
        // 4.b. Calculate the total gradient:
        mat gradient_electronic = build_electronic_energy_gradient(molecule.atoms, molecule.basis_functions, 
                                                                    molecule.orbital_to_atom, P_beta, X, Y);
        mat gradient_nuclear = calculate_gradient_nuclear_repulsion(molecule.atoms, cndo_params).t();
        mat gradient = gradient_electronic + gradient_nuclear;

        // 4.c. Check for convergence:
        double force_magnitude = norm(gradient);
        if (force_magnitude < gradient_threshold)
        {
            std::cout << "Optimization converged! Gradient magnitude: " << force_magnitude << std::endl;
            break;
        }

        // 4.d. Determine the search direction (normalized):
        mat direction = -gradient / force_magnitude;

        // 4.e. Initialize bracket for line search:
        double a = 0.0;
        double b = initial_step;
        double c = b + 1.618 * (b - a);
        bool within_bracket = false;

        // 4.f. Bracketing phase - find a bracket containing the minimum:
        int max_bracket_iterations = 20;
        int bracket_counter = 0;
        
        while (!within_bracket && bracket_counter < max_bracket_iterations)
        {
            // Evaluate energies at the three bracket points:
            double E_a = phi_of_alpha(molecule, original_positions, direction, a);
            double E_b = phi_of_alpha(molecule, original_positions, direction, b);
            double E_c = phi_of_alpha(molecule, original_positions, direction, c);

            // Check if the minimum is bracketed (E_b should be lowest):
            if (E_b < E_a && E_b < E_c)
            {
                within_bracket = true;
            }
            else
            {
                // Expand the bracket in the direction of decreasing energy:
                if (E_a < E_c)
                {
                    // Minimum is to the left, expand leftward:
                    c = b;
                    b = a;
                    a = b - 1.618 * (c - b);
                }
                else
                {
                    // Minimum is to the right, expand rightward:
                    a = b;
                    b = c;
                    c = b + 1.618 * (b - a);
                }
            }
            bracket_counter++;
        }

        // 4.g. Perform golden section search to find optimal step size:
        double optimal_step = golden_section_line_search(molecule, original_positions, direction, a, b, c);

        // 4.h. Update positions with optimal step:
        mat current_positions(molecule.atoms.size(), 3, fill::zeros);
        for (size_t i = 0; i < molecule.atoms.size(); ++i)
        {
            current_positions(i, 0) = original_positions(i, 0) + optimal_step * direction(0, i);
            current_positions(i, 1) = original_positions(i, 1) + optimal_step * direction(1, i);
            current_positions(i, 2) = original_positions(i, 2) + optimal_step * direction(2, i);
        }

        // 4.i. Update molecule atomic coordinates:
        for (size_t i = 0; i < molecule.atoms.size(); ++i)
        {
            molecule.atoms[i].x = current_positions(i, 0);
            molecule.atoms[i].y = current_positions(i, 1);
            molecule.atoms[i].z = current_positions(i, 2);
        }

        // 4.j. Update basis function coordinates:
        for (size_t i = 0; i < molecule.basis_functions.size(); ++i)
        {
            int atom_idx = molecule.basis_functions[i].atom_index;
            molecule.basis_functions[i].x = molecule.atoms[atom_idx].x;
            molecule.basis_functions[i].y = molecule.atoms[atom_idx].y;
            molecule.basis_functions[i].z = molecule.atoms[atom_idx].z;
        }

        // 4.k. Run SCF at new geometry and update original positions:
        results = run_scf_calculation(molecule);
        E_initial = results.E_total;
        original_positions = current_positions;

        // 4.l. Print iteration information:
        std::cout << "Iteration " << iter + 1 << ": Energy = " << E_initial 
                  << " eV, Gradient = " << force_magnitude << std::endl;
    }

    // 5. Print final results:
    if (iter >= max_iterations)
    {
        std::cout << "Warning: Maximum iterations reached without convergence!" << std::endl;
    }
    std::cout << "\n=== Optimization Complete ===" << std::endl;
    std::cout << "Total iterations: " << iter << std::endl;
    std::cout << "Final energy: " << results.E_total << " eV" << std::endl;
    std::cout << "\nFinal optimized positions (Bohr):" << std::endl;
    for (size_t i = 0; i < molecule.atoms.size(); ++i)
    {
        std::cout << "Atom " << i << " (" << molecule.atoms[i].atomic_number << "): "
                  << molecule.atoms[i].x << ", " 
                  << molecule.atoms[i].y << ", " 
                  << molecule.atoms[i].z << std::endl;
    }
}


// 17. Function to calculate the optimized geometry:
void geometry_optimization_2(Molecule& molecule)
{
    // 1. Set optimization parameters:
    double gradient_threshold = 1e-4; // Convergence criteria
    double current_step_size = 0.5;   // Start conservative (0.2 Bohr is safer than 0.5)
    int max_iterations = 100;
    
    // Ensure basis functions match initial atoms
    update_basis_centers(molecule); 

    // 2. Run initial SCF:
    SCFResults results = run_scf_calculation(molecule);
    double current_energy = results.E_total;

    std::cout << "Initial Energy: " << current_energy << " eV" << std::endl;

    // 3. Main Loop
    int iter = 0;
    for (iter = 0; iter < max_iterations; ++iter)
    {
        // --- A. Calculate Gradient ---
        // (Using your existing helper function is cleaner than re-writing the matrix logic here)
        mat gradient_mat = calculate_overall_gradient(molecule);
        
        // Flatten gradient for easier math (optional, but convenient)
        vec gradient = vectorise(gradient_mat); 
        double force_magnitude = norm(gradient);

        // --- B. Check Convergence ---
        if (force_magnitude < gradient_threshold)
        {
            std::cout << "Optimization converged! Gradient magnitude: " << force_magnitude << std::endl;
            break;
        }

        // --- C. Save Current Positions ---
        // We need these in case we have to backtrack (reject a step)
        std::vector<double> saved_x, saved_y, saved_z;
        for(const auto& atom : molecule.atoms) {
            saved_x.push_back(atom.x);
            saved_y.push_back(atom.y);
            saved_z.push_back(atom.z);
        }

        // --- D. Backtracking Line Search ---
        bool step_accepted = false;
        
        // Loop until we find a step that lowers energy
        while (!step_accepted)
        {
            // 1. Move Atoms: New = Old - (Step * Gradient_Direction)
            // Normalized direction d = -g / |g|
            // Delta = Step * d = -Step * g / |g|
            double scaling_factor = current_step_size / force_magnitude;

            // Iterate through atoms and update coordinates
            int grad_idx = 0;
            for (size_t i = 0; i < molecule.atoms.size(); ++i)
            {
                molecule.atoms[i].x = saved_x[i] - gradient(grad_idx)   * scaling_factor;
                molecule.atoms[i].y = saved_y[i] - gradient(grad_idx+1) * scaling_factor;
                molecule.atoms[i].z = saved_z[i] - gradient(grad_idx+2) * scaling_factor;
                grad_idx += 3;
            }

            // 2. CRITICAL: Update Basis Functions to match new atoms
            update_basis_centers(molecule);

            // 3. Calculate New Energy
            SCFResults new_results = run_scf_calculation(molecule);
            double new_energy = new_results.E_total;

            // 4. Accept or Reject?
            if (new_energy < current_energy) 
            {
                // --- SUCCESS ---
                current_energy = new_energy;
                step_accepted = true;
                
                // Slight speedup: If step worked, try a slightly larger one next time 
                // to move faster across flat regions.
                current_step_size *= 1.1; 
            }
            else 
            {
                // --- FAILURE (Overshot) ---
                // Energy went UP. We stepped too far.
                // 1. Restore old coordinates
                for (size_t i = 0; i < molecule.atoms.size(); ++i) {
                    molecule.atoms[i].x = saved_x[i];
                    molecule.atoms[i].y = saved_y[i];
                    molecule.atoms[i].z = saved_z[i];
                }
                
                // 2. Shrink step size
                current_step_size *= 0.5;

                // 3. Safety check: If step becomes tiny, we are stuck
                if (current_step_size < 1e-6) {
                    std::cout << "Step size too small. Forcing convergence." << std::endl;
                    goto end_optimization; // Jump out of loops
                }
                
                // Loop repeats with smaller step...
            }
        }

        std::cout << "Iteration " << iter + 1 
                  << ": Energy = " << current_energy 
                  << " Gradient = " << force_magnitude 
                  << " Step = " << current_step_size << std::endl;
    }

    end_optimization:

    // 5. Print final results:
    if (iter >= max_iterations) {
        std::cout << "Warning: Maximum iterations reached without convergence!" << std::endl;
    }

    std::cout << "\n=== Optimization Complete ===" << std::endl;
    std::cout << "Final energy: " << current_energy << " eV" << std::endl;
    std::cout << "\nFinal optimized positions (Bohr):" << std::endl;
    for (size_t i = 0; i < molecule.atoms.size(); ++i)
    {
        std::cout << "Atom " << i << " (" << molecule.atoms[i].atomic_number << "): "
                  << molecule.atoms[i].x << ", " 
                  << molecule.atoms[i].y << ", " 
                  << molecule.atoms[i].z << std::endl;
    }
}


// 19. Function to calculate overall gradient (electronic and nuclear):
mat calculate_overall_gradient(const Molecule& molecule)
{
    // 1. Instantiate the gradient matrix:
    int N = molecule.atoms.size();
    mat gradient(3, N, fill::zeros);

    // 2. Run SCF calculation:
    SCFResults results = run_scf_calculation(molecule);

    // 3. Build the weighted overlap matrix:
    mat X = build_energy_weighted_matrix(results.epsilon_alpha, results.C_alpha, molecule.num_alpha_e) + build_energy_weighted_matrix(results.epsilon_beta, results.C_beta, molecule.num_beta_e);

    // 4. Build the Y matrix:
    mat Y = build_Y_matrix(molecule.atoms, results.P_total, molecule.basis_functions, cndo_params);

    // 5. Build the P_beta matrix:
    mat P_beta = build_P_beta_matrix(results.P_total, molecule.basis_functions, cndo_params);

    // 6. Build the electronic energy gradient:
    mat gradient_electronic = build_electronic_energy_gradient(molecule.atoms, molecule.basis_functions, molecule.orbital_to_atom, P_beta, X, Y);
    mat gradient_nuclear = calculate_gradient_nuclear_repulsion(molecule.atoms, cndo_params).t();
    gradient = gradient_electronic + gradient_nuclear;

    // 7. Return the gradient:
    return gradient * eV_to_Ha;
}