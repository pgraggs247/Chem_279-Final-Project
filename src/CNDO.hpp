#pragma once 

#include "new_molecule_fixed.hpp"
#include "3D_gaussian.hpp"
#include <iostream>
#include <armadillo>
#include <vector>

using std::cout;
using std::endl;


class CNDO{


    private:

        using Atom = NewMolecule::Atom;
        using Primitive = NewMolecule::Primitive;
        using BasisFunction = NewMolecule::BasisFunction;


        NewMolecule& molecule;
        int num_atoms;
        int num_basis;

        const double conversion_factor = 27.2114;

    public:

    CNDO(NewMolecule& molecule)
        : molecule(molecule)
    {
        num_atoms = molecule.atoms_of_molecule.size();
        num_basis = molecule.num_basis_funcs;
    }
        
    

        double get_Suv_Ra(int u, int v, int atom_idx, int derivative_component){
            double result{0.0};

            // Implementation
            NewMolecule::BasisFunction& basis_U = molecule.basis_functions.at(u);
            NewMolecule::BasisFunction& basis_V = molecule.basis_functions.at(v);

            // Let's extract the center, we will need this to build the Gaussian Objects for each primitive later
            arma::vec center_U = basis_U.center;
            arma::vec center_V = basis_V.center;

            // Build the Gaussian Objects, we will need this for the Skl function below
            int momentum_U = arma::accu(basis_U.quantum_numbers);
            int momentum_V = arma::accu(basis_V.quantum_numbers);

            

            NewMolecule::Atom& atom_moving = molecule.atoms_of_molecule.at(atom_idx);

            bool u_is_element_of_A;
            bool v_is_element_of_A;

            // Is the center of the basis the same as the atom we've indexed
            if(basis_U.center.at(0) == atom_moving.x && basis_U.center.at(1) == atom_moving.y && basis_U.center.at(2) == atom_moving.z){
                u_is_element_of_A = true;
            } else { u_is_element_of_A = false;}

            if(basis_V.center.at(0) == atom_moving.x && basis_V.center.at(1) == atom_moving.y && basis_V.center.at(2) == atom_moving.z){
                v_is_element_of_A = true;
            } else {v_is_element_of_A = false;} 
                

            if (!(u_is_element_of_A && !v_is_element_of_A)){
                return 0.0;
            }
                for(int k = 0; k < basis_U.primitives.size(); k++){
                    for(int l = 0; l < basis_V.primitives.size(); l++){


                        const Primitive& prime_k = basis_U.primitives.at(k);
                        const Primitive& prime_l = basis_V.primitives.at(l);

                        double Dku = prime_k.coefficient;
                        double Dlv = prime_l.coefficient;
                        double Nku = prime_k.normalization;
                        double Nlv = prime_l.normalization;

                        // Now that we have the Primitives, we can build the Gaussian. The Gaussian needs the exponent from the Primitive.
                        Gaussian_3D GA(center_U.at(0), center_U.at(1), center_U.at(2), prime_k.exponent, momentum_U);
                        Gaussian_3D GB(center_V.at(0), center_V.at(1), center_V.at(2), prime_l.exponent, momentum_V);

                        double dir_skl = get_DR_SKL(GA, GB, basis_U.quantum_numbers, basis_V.quantum_numbers, derivative_component);

                        double prelim_res = Dku * Dlv * Nku * Nlv * dir_skl;

                        result += prelim_res;

                        
                    }

                }
            
            
            return result;
        }

        double get_DR_SKL(Gaussian_3D GA, Gaussian_3D GB, arma::vec La, arma::vec Lb, int coord){

            int Lk = La.at(coord); 
            int Ll = Lb.at(coord);

            int other_coord_1;
            int other_coord_2;

            if(coord == 0){
                other_coord_1 = 1;
                other_coord_2 = 2;
            } else if( coord == 1){
                other_coord_1 = 0;
                other_coord_2 = 2;
            } else {
                other_coord_1 = 0;
                other_coord_2 = 1;
            }

            // 
            double S1D_res_1 = Skl_1D(GA, GB, Lk - 1, Ll, coord);
            double S1D_res_2 = Skl_1D(GA, GB, Lk + 1, Ll, coord);

            double other_1 = Skl_1D(GA, GB, La.at(other_coord_1), Lb.at(other_coord_1), other_coord_1);
            double other_2 = Skl_1D(GA, GB, La.at(other_coord_2), Lb.at(other_coord_2), other_coord_2);

            double alpha_K = GA.get_exponent(); 
            

            double first_term = -Lk * S1D_res_1;
            double second_term = 2*alpha_K*S1D_res_2;

            double prelim_result = first_term + second_term;

            return prelim_result * other_1 * other_2;


        }

        // A new version of SAB_1D, except design to deal with the negative momentums produced by derivatives
        double Skl_1D(Gaussian_3D G1, Gaussian_3D G2, int lA, int lB, int coord){
        
            // Ensure the user has passed in appropriate values
            if(coord > 2 || coord <0){
                 throw std::invalid_argument("Coordinate number must be between zero and two.");

            }

            if(lA < 0 || lB < 0){
                return 0.0;
            }

            if(lA > 2 || lB > 2){
                std::cout << "No Support For Momentum values greater than two.";
                return 0.0;
            }


            // Get the product center at the appropriate coordinate
            arma::vec product_center = get_product_center(G1, G2);
            double Xp = product_center.at(coord);  // Extract the appropriate coordinate from the product center {PCx, PCy, PCz}

            // Get the full centroid vector and extract the proper coordinate
            arma::vec centroid_G1 = G1.get_centroid();
            double XA = centroid_G1.at(coord);

            // Repeat above for the second gaussian
            arma::vec centroid_G2 = G2.get_centroid();
            double XB = centroid_G2.at(coord);

            // Get alpha and beta
            double alpha = G1.get_exponent();
            double beta = G2.get_exponent();

            // Get Xp - XA and Xp - XB
            double Xp_minus_XA = Xp - XA;
            double Xp_minus_XB = Xp - XB;

            // Get alpha + beta, 2 * (alpha + beta), and alpha * beta
            double alpha_plus_beta = alpha + beta;
            double two_times_alpha_plus_beta = 2 * alpha_plus_beta;
            double alpha_times_beta = alpha * beta;


            /* Write the summation loop: Be careful this is the critical component */

            // Declare variable outside of loop to sum the total
            double total_value {0};
            // The outer loop
            for(int i = 0; i <= lA; i++){

                // The inner loop
                for(int j = 0; j <= lB; j++){


                    if( (i + j) % 2 == 1){
                        continue;
                    }

                    // Get the two binomials using n choose k formula (we named in m choose n)
                    double lA_choose_i = m_choose_n(lA, i);
                    double lB_choose_j = m_choose_n(lB, j);

                    // Get the i + j - 1 term
                    double i_plus_j_minus_one = i + j - 1;

                    // Get the double factorial of i + j - 1
                    double double_fac = double_factorial(i_plus_j_minus_one);

                    // Get lA - i and lB - j exponents
                    double lA_minus_i = lA - i;
                    double lB_minus_j = lB - j;

                    // Get the i+j / 2 exponent
                    double i_plus_j_divided_by_two = (i + j) / 2.0;

                    // Raised the Xp - XA & Xp - XB terms to the above exponents
                    double Xp_minus_XA_raised = std::pow(Xp_minus_XA, lA_minus_i);
                    double Xp_minus_XB_raised = std::pow(Xp_minus_XB, lB_minus_j);

                    // Compute the numerator 
                    double numerator = double_fac * Xp_minus_XA_raised * Xp_minus_XB_raised;

                    // compute the denominator
                    double denominator = std::pow(two_times_alpha_plus_beta, i_plus_j_divided_by_two);

                    // Incorporate the n_choose_k components
                    double binomial_component = lA_choose_i * lB_choose_j;

                    // compute the result
                    double result =  binomial_component * (numerator / denominator);

                    // Sum the result
                    total_value += result;

            }
        }  

        double leading_term = get_leading_term(G1, G2, coord);


        return leading_term * total_value;
    }

    arma::mat build_full_Suv_Ra(){

        arma::cube res(3, num_basis*num_basis, num_atoms, arma::fill::zeros);

        for(int atom_idx = 0; atom_idx < num_atoms; atom_idx++){

            for(int basis_u = 0; basis_u < num_basis; basis_u++){

                for(int basis_v = 0; basis_v < num_basis; basis_v++){

                    for(int dir_coord = 0; dir_coord < 3; dir_coord++){

                        double value = get_Suv_Ra(basis_u, basis_v, atom_idx, dir_coord);

                        int uv_index = basis_u * num_basis + basis_v;

                        res(dir_coord, uv_index, atom_idx ) = value;
                    }
                }
            }
        }

        // Res is a cube, we want a matrix
        arma::mat result(3, num_basis*num_basis, arma::fill::zeros);

        for(int dir = 0; dir < 3; dir++){
            for(int uv = 0; uv < num_basis*num_basis; uv++){
                for(int atom = 0; atom < num_atoms; atom++){
                    result(dir, uv) += res(dir, uv, atom);
                }
            }
        }

        result.print("The Overlap Integrals (Printed In Accord With Rubric)");

        return result;

    
    }

    int get_S_basis_index(int atom_idx){

        for(int idx = 0; idx < molecule.basis_functions.size(); idx++){

            BasisFunction basis = molecule.basis_functions.at(idx);

            if(basis.atom_idx == atom_idx){
                if(arma::accu(basis.quantum_numbers) == 0){
                    // This is the s shell basis
                    return idx;
                }
            }

        }

        throw std::runtime_error("No S shell basis for this atom.");

    }

    std::vector<int> get_basis_indices(int atom_idx){

        std::vector<int> indices;

        for(int i = 0; i < num_basis; i++){
            BasisFunction basis = molecule.basis_functions.at(i);
            if(basis.atom_idx == atom_idx){
                indices.push_back(i);
            }

        }

        return indices;
    }



    double dir_of_zero_zero(int k, int kp, int l, int lp, BasisFunction basis_1, BasisFunction basis_2, int dir){

        

        double Uk;
        double Ul;
        double sigma_k;
        double sigma_l;

        Primitive prime_k = basis_1.primitives.at(k);
        Primitive prime_kp = basis_1.primitives.at(kp);

        Primitive prime_l = basis_2.primitives.at(l);
        Primitive prime_lp = basis_2.primitives.at(lp);

        sigma_k = 1 / (prime_k.exponent + prime_kp.exponent);
        sigma_l = 1 / (prime_l.exponent + prime_lp.exponent);

        Uk = std::pow(M_PI * sigma_k, 1.5);
        Ul = std::pow(M_PI * sigma_l, 1.5);

        



        double distance = molecule.get_distance_between_basis_centers(basis_1.center, basis_2.center);

        if(distance < 1e-12){
            return 0.0;
        }
        double dx = basis_1.center.at(0) - basis_2.center.at(0);
        double dy = basis_1.center.at(1) - basis_2.center.at(1);
        double dz = basis_1.center.at(2) - basis_2.center.at(2);

        arma::vec displacement_vec ={dx, dy, dz};

        double numerator_first_term = (Uk * Ul) * (displacement_vec.at(dir));
        double denominator_first_term = distance * distance;
        double first_term = numerator_first_term / denominator_first_term;

        // Recall from homework 4 T = V^2 (Ra - Rb)^2, where V = (sigmaA + sigmaB)^-1. (and some other definitions)

        double v_squared = 1.0/(sigma_k + sigma_l);
        double V = std::sqrt(v_squared);
        double T = v_squared * (distance * distance);
        double sq_root_T = std::sqrt(T);
        double erf_T = std::erf(sq_root_T);


        double firstPart_second_term = -erf_T / distance;
        double secondPart_second_term = (2*V / std::sqrt(M_PI)) * exp(-T);
        double second_term = firstPart_second_term + secondPart_second_term;

        double result = first_term * second_term;

        return result;


    }

    
    double gamma_AB_Ra(int atomA_idx, int atomB_idx, int dir){
    
        BasisFunction basis_1;
        BasisFunction basis_2;
        
        int b1_idx = get_S_basis_index(atomA_idx);
        int b2_idx = get_S_basis_index(atomB_idx);

        basis_1 = molecule.basis_functions.at(b1_idx);
        basis_2 = molecule.basis_functions.at(b2_idx);

        double result{0.0};
        for(int k = 0; k < basis_1.primitives.size(); k++){
            for(int kp = 0; kp < basis_1.primitives.size(); kp++){
                for(int l = 0; l < basis_2.primitives.size(); l++){
                    for(int lp =0; lp < basis_2.primitives.size(); lp++){

                        Primitive prime_k = basis_1.primitives.at(k);
                        Primitive prime_kp = basis_1.primitives.at(kp);
                        Primitive prime_l = basis_2.primitives.at(l);
                        Primitive prime_lp = basis_2.primitives.at(lp);

                        double temp_res{0};
                        temp_res = prime_k.d_prime * prime_kp.d_prime * prime_l.d_prime * prime_lp.d_prime * dir_of_zero_zero(k, kp, l, lp, basis_1, basis_2, dir);

                        
                        result += temp_res * conversion_factor;

                        
                    }
                }
            }
        }

        return result;
    }


            arma::mat build_full_gammaAB_Ra(){

            // AI suggestion to use cubes and how to properly initialize them (This technique is repeated for gradient nuclear and gradient electronic)
            arma::cube res(3, num_atoms * num_atoms, num_atoms, arma::fill::zeros);

            for(int atomA = 0; atomA < num_atoms; atomA++){

                for(int atomB = 0; atomB < num_atoms; atomB++){

                    // AI line showing how to properly create the index to use with cubes
                    int AB_index = atomA * num_atoms + atomB;

                    for(int dir = 0; dir < 3; dir++){

                        double value = gamma_AB_Ra(atomA, atomB, dir);

                        res(dir, AB_index, atomA) = value;

                    }
                }
            }

            // AI code to revert cube to matrix
            arma::mat result(3, num_atoms * num_atoms, arma::fill::zeros);
            for(int dir = 0; dir < 3; dir++){
                for(int AB = 0; AB < num_atoms * num_atoms; AB++){
                    for(int atomA = 0; atomA < num_atoms; atomA++){
                        result(dir, AB) += res(dir, AB, atomA);
                    }
                }
            }

            return result;
        }

        double gradient_nuclear_element(int atom_idx_1, int atom_idx_2, int dir_component){

            NewMolecule::EmpiricalFactors factors;
            // BasisFunction basis_1 = molecule.basis_functions.at(basis_idx_1);
            // BasisFunction basis_2 = molecule.basis_functions.at(basis_idx_2);

            if(atom_idx_1 == atom_idx_2){
                return 0;
            }

            Atom atomA = molecule.atoms_of_molecule.at(atom_idx_1);
            Atom atomB = molecule.atoms_of_molecule.at(atom_idx_2);

           

            double zetaA = factors.zeta_starA.at(atomA.Z);
            double zetaB = factors.zeta_starA.at(atomB.Z);

            double dx = atomA.x - atomB.x;
            double dy = atomA.y - atomB.y;
            double dz = atomA.z - atomB.z;

            arma::vec displacement = {dx, dy, dz};

            double distance = std::sqrt(dx*dx + dy*dy + dz*dz);
            double distance_cubed = std::pow(distance, 3);

            double zetaA_times_zetaB = zetaA * zetaB;

            double displacement_componenet = displacement.at(dir_component);

            double result = - zetaA_times_zetaB * (displacement_componenet/ distance_cubed);

            return result * conversion_factor;

        }

       

        arma::mat gradient_nuclear() {
            arma::mat grad(3, num_atoms, arma::fill::zeros);

            for (int A = 0; A < num_atoms; A++) {
                for (int B = 0; B < num_atoms; B++) {

                    if (A == B) continue;

                    for (int dir = 0; dir < 3; dir++) {
                        grad(dir, A) += gradient_nuclear_element(A, B, dir);
                    }
                }
            }
            grad.print("Nuclear Repulsion Energy (Printed In Accordance With Rubric)");
            return grad;
        }

        double get_Xuv(int basis_idx_1, int basis_idx_2){

            if(basis_idx_1 == basis_idx_2){
                return 0.0;
            }

            Atom atom_A = molecule.basis_functions.at(basis_idx_1).atom;
            Atom atom_B = molecule.basis_functions.at(basis_idx_2).atom;

            NewMolecule::EmpiricalFactors factors;

            double P_tot_uv = molecule.P_total(basis_idx_1, basis_idx_2);

            double betaA = - factors.negative_beta.at(atom_A.Z);
            double betaB = - factors.negative_beta.at(atom_B.Z);

            double betaA_plus_betaB = betaA + betaB;

            return betaA_plus_betaB * P_tot_uv;

        }

        arma::mat build_x_matrix(){

            arma::mat res(molecule.num_basis_funcs, molecule.num_basis_funcs, arma::fill::zeros);

            for(int u = 0; u < molecule.num_basis_funcs; u++){
                for(int v = 0; v < molecule.num_basis_funcs; v++){
                    
                    res(u,v) = get_Xuv(u,v);
                }
            }

            return res;
        }

        double Yab(int atom_idx_1, int atom_idx_2){

            if (atom_idx_1 == atom_idx_2) {
                return 0.0;
            }

            NewMolecule::EmpiricalFactors factors;
            Atom atomA = molecule.atoms_of_molecule.at(atom_idx_1);
            Atom atomB = molecule.atoms_of_molecule.at(atom_idx_2);

            double zeta_A = factors.zeta_starA.at(atomA.Z);
            double zeta_B = factors.zeta_starA.at(atomB.Z);

            std::vector<int> atomA_basis_idxs = get_basis_indices(atom_idx_1);
            std::vector<int> atomB_basis_idxs = get_basis_indices(atom_idx_2);
            double pAA_tot{0.0};
            double pBB_tot{0.0};
            for(int u = 0; u < atomA_basis_idxs.size(); u++){
                pAA_tot += molecule.P_total(atomA_basis_idxs.at(u), atomA_basis_idxs.at(u));
            }
            for(int v = 0; v < atomB_basis_idxs.size(); v++){
                pBB_tot += molecule.P_total(atomB_basis_idxs.at(v),atomB_basis_idxs.at(v));
            }

            double summation_term{0.0};
            for(int u = 0; u < atomA_basis_idxs.size(); u++){
                for(int v = 0; v < atomB_basis_idxs.size(); v++){

                double P_alpha_uv = molecule.P_alpha(atomA_basis_idxs.at(u),atomB_basis_idxs.at(v));
                double P_alpha_vu = molecule.P_alpha(atomB_basis_idxs.at(v), atomA_basis_idxs.at(u));    

                double P_beta_uv = molecule.P_beta(atomA_basis_idxs.at(u), atomB_basis_idxs.at(v));
                double P_beta_vu = molecule.P_beta(atomB_basis_idxs.at(v), atomA_basis_idxs.at(u)); 

                double temp_res = (P_alpha_uv * P_alpha_vu) + (P_beta_uv * P_beta_vu);

                summation_term += temp_res;
                }
            }

            double pAA_times_pBB = pAA_tot * pBB_tot;
            double zetaB_times_pAA = zeta_B * pAA_tot;
            double zetaA_times_pBB = zeta_A * pBB_tot;

            double first_three = pAA_times_pBB - zetaB_times_pAA - zetaA_times_pBB;

            double result = first_three - summation_term;

            return result;

        }


        arma::mat build_y_matrix(){

            arma::mat result(num_atoms, num_atoms, arma::fill::zeros);

            for(int A = 0; A < num_atoms; A++){
                for(int B = 0; B < num_atoms; B++){
                    result(A, B) = Yab(A, B);
                    }
                }

            return result;    
            }

        double get_E_component(int atom_idx, int coord){
            

            arma::mat X_mat = build_x_matrix();
            arma::mat Y_mat = build_y_matrix();
            arma::mat Suv_Ra = build_full_Suv_Ra();
            arma::mat Gamma_mat = build_full_gammaAB_Ra();
            arma::mat Nuclear_mat = gradient_nuclear();
            double result{0};
            double first_sum{0};
            double second_sum{0};

            std::vector<int> basisA_indices = get_basis_indices(atom_idx);

            for(int u = 0; u < basisA_indices.size(); u++){
                int u_index = basisA_indices.at(u);
                for(int v = 0; v < molecule.num_basis_funcs; v++){
                    if(u_index == v){continue;}

                    double X = X_mat(u_index, v);

                    // Must index it the same way we built matrix
                    int uv_index = u_index * num_basis + v;
                    double SRa = Suv_Ra(coord, uv_index);

                    first_sum += (X*SRa);
                }
            }

            
            for(int B = 0; B < molecule.atoms_of_molecule.size(); B++){

                if(B == atom_idx){continue;}
                double Y = Y_mat(atom_idx,B);

                int AB_index = atom_idx * num_atoms + B;
                double gammaAB = Gamma_mat(coord, AB_index);
                

                second_sum += (Y * gammaAB);
            }


            double nuc = Nuclear_mat(coord, atom_idx);

            result = first_sum + second_sum + nuc;


            return result;

        }

        arma::mat build_electronic_gradient(){
            arma::mat result(3, num_atoms, arma::fill::zeros);

            arma::mat V_nuc_mat = gradient_nuclear();

            for(int A = 0; A < num_atoms; A++){
                for(int coord = 0; coord < 3; coord++){

                    double E_component = get_E_component(A, coord);
                    double V_nuc = V_nuc_mat(coord, A);
                    
                    result(coord, A) = E_component - V_nuc;

                }
            }
            
            result.print("Two-Electron Integrals (Printed In Accordance With Rubric)");
            return result;
        }

        
        arma::mat compute_gradient(){
            arma::mat res(3, num_atoms, arma::fill::zeros);

            for(int A = 0; A < num_atoms; A++){

                res(0, A) = get_E_component(A, 0);
                res(1, A) = get_E_component(A, 1);
                res(2, A) = get_E_component(A, 2); 
            }

            res.print("Total Gradient (Printed In Accordance With Rubric)");
            return res;
        }


        /* 
        Below is some additional code to complete the extra credit. Since I'm doing this portion on the due date I will use
        AI to write quickly refactor some my old helper functions. 
        */

        // Gemini Refactored Helper Functions

        // ---------------------------------------------------------
        // 1. The "Phi" Function: Energy as a function of Step Size (alpha)
        //    Returns E( R_old + alpha * direction )
        // ---------------------------------------------------------
        double phi_of_alpha(NewMolecule& mol, const arma::mat& original_coords, const arma::mat& direction, double alpha) {
            // Calculate temporary coordinates
            arma::mat temp_coords = original_coords + (alpha * direction);
            
            // Update molecule geometry (rebuilds basis and integrals)
            mol.update_geometry(temp_coords);
            
            // Run SCF to get energy
            auto [e_elec, e_nuc] = mol.SCF_algorithm();
            return e_elec + e_nuc;
        }

        // ---------------------------------------------------------
        // 2. Golden Section Search (The "Minimize Bracket" function)
        //    Finds the alpha that minimizes Energy between a and c
        // ---------------------------------------------------------
        double minimize_bracket(NewMolecule& mol, const arma::mat& original_coords, const arma::mat& direction, double a, double b, double c) {
            const double golden_ratio = 1.618033988;
            const double R = 0.618033988; // 1 / golden_ratio
            const double C = 1.0 - R;
            const double tol = 1e-4; // Tolerance for alpha precision

            double x1 = c - R * (c - a); // Test point 1
            double x2 = a + R * (c - a); // Test point 2
            
            double f1 = phi_of_alpha(mol, original_coords, direction, x1);
            double f2 = phi_of_alpha(mol, original_coords, direction, x2);

            // Loop until the bracket is small enough
            while (std::abs(c - a) > tol) {
                if (f1 < f2) {
                    c = x2;
                    x2 = x1;
                    f2 = f1;
                    x1 = c - R * (c - a);
                    f1 = phi_of_alpha(mol, original_coords, direction, x1);
                } else {
                    a = x1;
                    x1 = x2;
                    f1 = f2;
                    x2 = a + R * (c - a);
                    f2 = phi_of_alpha(mol, original_coords, direction, x2);
                }
            }

            // Return the midpoint of the final bracket
            return (a + c) / 2.0;
        }


        // This function I will write. I'll copy it as much as possible directly from the HW_1_4 implementation
        void geometry_optimization(NewMolecule &molecule){

            double threshold = 0.00001;
            double initial_step = 0.5;

            auto[electronic_energy, nuclear_energy] = molecule.SCF_algorithm();
            double initial_energy = electronic_energy + nuclear_energy;
            double final_energy;

            arma::mat org_positions(molecule.atoms_of_molecule.size(), 3, arma::fill::zeros);
            
            // Fill up the postion matrix with the atomic positions
            for(int i = 0; i < molecule.atoms_of_molecule.size(); i++){
                org_positions(i, 0) = molecule.atoms_of_molecule.at(i).x;
                org_positions(i, 1) = molecule.atoms_of_molecule.at(i).y;
                org_positions(i, 2) = molecule.atoms_of_molecule.at(i).z;
            }

        
            molecule.update_geometry(org_positions);

            int max_iters = 100;
            for(int iter=0; iter < max_iters; iter++){

                CNDO cndo(molecule);
                arma::mat forces = cndo.compute_gradient();
                double force_magnitude  = arma::norm(forces);
                arma::mat direction = -forces.t() / force_magnitude;

                if(force_magnitude < threshold){
                    std::cout << "Force Less Than Threshold. Energy Has COnverged." << std::endl;
                    break;
                }

                bool within_bracket = false;
                double a = 0.0; double b = initial_step; double c = b + 1.618 * (b - a);

                // Line Search
                int max_bracket_iters = 100;
                int counter = 0;
                while(!within_bracket && counter < max_bracket_iters){
                    double E_a = phi_of_alpha(molecule, org_positions, direction, a);
                    double E_b = phi_of_alpha(molecule, org_positions, direction, b);
                    double E_c = phi_of_alpha(molecule, org_positions, direction, c);

                    // Check if the optimal alpha is in the bracket
                    if(E_b < E_a && E_b < E_c){
                        within_bracket = true;
                    }
                    else {
                        // expand outward but alternate directions
                        if(E_a < E_c){
                            c = b;  // shift right bound inward
                            b = a;
                            a = b - 1.618 * (c - b); // expand left
                        } else {
                            a = b;  // shift left bound inward
                            b = c;
                            c = b + 1.618 * (b - a); // expand right
                        }
                    }
                    counter ++;
                }

                initial_step = minimize_bracket(molecule, org_positions, direction, a, b, c );

                org_positions = org_positions + (initial_step * direction);

                molecule.update_geometry(org_positions);

                auto[e_elec, e_nuc] = molecule.SCF_algorithm();
                double current_energy = e_elec + e_nuc;
                
            
                std::cout << "Current Iteration: " << iter << std::endl;

                std::cout << "Current Step Size: " << initial_step << std::endl;
                std::cout << "Current Energy: " << current_energy << std::endl;

                final_energy = current_energy;
                
            } 

            
            std::cout << "Initial Energy:" << initial_energy << std::endl;
            std::cout << "Final Energy: " << final_energy << std::endl;
            org_positions.print("Final Geometry.");

        } 


        
        void print_summary(){
            molecule.print_summary();
        }
        
};
