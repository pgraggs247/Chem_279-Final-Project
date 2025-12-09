#pragma once
#include <iostream>
#include <vector>
#include <armadillo>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <stdexcept>


using std::cout;
using std::cin;
using std::endl;
using std::ifstream;

#include "3D_gaussian.hpp"

// Sticking with object oriented programming. New class for a new strategy
class NewMolecule{

    // Public worked for me previously. Probably need to think about when to make class methods/attributes private
    public:

    // A structure to represent the atoms
    struct Atom{
        int Z;            // The atomic number
        int Beta {0};
        double x , y, z;  // The coordinates
        
    };

    // A structure to represent the shells
    struct Shells{
        arma::vec S_shell = {0, 0, 0};        
        std::vector<arma::vec> P_shells {{1,0,0}, {0,1,0}, {0,0,1}};
    };

    // A structure to represent the primitive gaussian
    struct Primitive{
        double exponent;
        double coefficient;
        double normalization;
        double d_prime;
    };

    // A structure to represent the basis functions
    struct BasisFunction{
        int atom_idx;
        Atom atom;
        arma::vec center;
        arma::vec quantum_numbers;
        std::vector<Primitive> primitives;
    };

    // A structure to store the basis information
    struct BasisInfo{

        // Hydrogen Information
        const arma::vec hydrogen_exponents {3.42525091, 0.62391373, 0.16885540};
        const arma::vec hydrogen_contractions {0.15432897, 0.53532814, 0.44463454};

        // Carbon Information
        const arma::vec carbon_exponents {2.94124940, 0.68348310, 0.22228990};
        const arma::vec carbon_S_shell_contractions {-0.09996723, 0.39951283, 0.70011547};
        const arma::vec carbon_P_shell_contractions {0.15591627, 0.60768372, 0.39195739};

        // Fluorine Information
        const arma::vec fluorine_exponents {0.6464803249E+01, 0.1502281245E+01, 0.4885884864E+00};
        const arma::vec fluorine_S_shell_contractions {-0.9996722919E-01, 0.3995128261E+00, 0.7001154689E+00};
        const arma::vec fluorine_P_shell_contractions {0.1559162750E+00, 0.6076837186E+00, 0.3919573931E+00};

        // Nitrogen Information
        const arma::vec nitrogen_exponents {0.3780455879E+01, 0.8784966449E+00, 0.2857143744E+00};
        const arma::vec nitrogen_S_shell_contractions {-0.9996722919E-01, 0.3995128261E+00, 0.7001154689E+00};
        const arma::vec nitrogen_P_shell_contractions {0.1559162750E+00, 0.6076837186E+00, 0.3919573931E+00};

        // Oxygen Information
        const arma::vec oxygen_exponents {5.033151319, 1.169596125, 0.3803889600};
        const arma::vec oxygen_S_shell_contractions {-0.09996722919, 0.3995128261, 0.7001154689};
        const arma::vec oxygen_P_shell_contractions {0.1559162750, 0.6076837186, 0.3919573931};

    };

    // A structure to store the empirical factors we will need
    struct EmpiricalFactors{

        std::map<int, double> one_half_Is_plus_As {
            {1, 7.176},
            {6, 14.051},
            {7, 19.316},
            {8, 25.390},
            {9, 32.272}
        };
        std::map<int, double> one_half_Ip_plus_Ap {
            {6, 5.572},
            {7, 7.275},
            {8, 9.111},
            {9, 11.080}
        };
        std::map<int, int> negative_beta {
            {1, 9},
            {6, 21},
            {7, 25},
            {8, 31},
            {9, 39}
        };
        std::map<int, int> zeta_starA {
            {1, 1},
            {6, 4},
            {7, 5},
            {8, 6},
            {9, 7}
        };
    };


    // Attributes from previous molecule class
    const double Bohr_radius{0.52917706};
    const double Conversion_factor{27.211324570273};
    int num_basis_funcs{};
    int num_electrons{};
    int num_alpha_e{};
    int num_beta_e{};
    arma::mat molecular_coordinates;
    std::vector<Atom> atoms_of_molecule;
    std::vector<BasisFunction> basis_functions;
    arma::vec carbon_S_shell_norm;
    arma::vec carbon_P_shell_norm;
    arma::vec fluorine_S_shell_norm;
    arma::vec fluorine_P_shell_norm;
    arma::vec oxygen_S_shell_norm;
    arma::vec oxygen_P_shell_norm;
    arma::vec nitrogen_S_shell_norm;
    arma::vec nitrogen_P_shell_norm;
    arma::vec h2_norm;
    

    // New attributes
    arma::mat overlap_matrix;
    arma::mat P_alpha, P_beta, P_total;
    arma::mat fock_matrix;
    arma::mat Fock_alpha;  // To store Fock alpha after convergence
    arma::mat Fock_beta;   // To store Fock beta after convergence
    arma::mat gamma_mat;

    double P_alpha_uv_element;
    double P_beta_uv_element;

    // Diagonalization Factors (for hamilitonian)
    const double hydro_factor = -13.6;
    const double carbon_2S_factor = -21.4;
    const double carbon_2P_factor = -11.4;
    const double nitro_2S_factor = -26.0;
    const double nitro_2P_factor = -13.4;
    const double oxy_2S_factor = -32.3;
    const double oxy_2P_factor = -14.8;
    const double flu_2S_factor = 40;
    const double flu_2P_factor = -17.3;


     // Constructor
    NewMolecule(std::filesystem::path atoms_file_path, int num_alpha_electrons, int num_beta_electrons){

        BasisInfo basis_info;
        EmpiricalFactors empirical_factors;
        num_alpha_e = num_alpha_electrons;
        num_beta_e = num_beta_electrons;
        

        read_in_atoms(atoms_file_path);
        read_coordinates();
        assign_empirical_factors();
        

         // For updated normalization computation
        h2_norm = compute_normalization_vector(0, basis_info.hydrogen_exponents);

        // Carbon normalization
        carbon_S_shell_norm = compute_normalization_vector(0, basis_info.carbon_exponents);
        carbon_P_shell_norm = compute_normalization_vector(1, basis_info.carbon_exponents);

        // Fluorine normalization
        fluorine_S_shell_norm = compute_normalization_vector(0, basis_info.fluorine_exponents);
        fluorine_P_shell_norm = compute_normalization_vector(1, basis_info.fluorine_exponents);

        // Nitrogen normalization
        nitrogen_S_shell_norm = compute_normalization_vector(0, basis_info.nitrogen_exponents);
        nitrogen_P_shell_norm = compute_normalization_vector(1, basis_info.nitrogen_exponents);

        // Oxygen normalization
        oxygen_S_shell_norm = compute_normalization_vector(0, basis_info.oxygen_exponents);
        oxygen_P_shell_norm = compute_normalization_vector(1, basis_info.oxygen_exponents);

        // carbon_S_shell_norm = normalize_S_shell(basis_info.carbon_exponents);
        // carbon_P_shell_norm = normalize_P_shells(basis_info.carbon_exponents);
        // h2_norm = normalize_S_shell(basis_info.hydrogen_exponents);

    
        get_number_of_basis_funcs();
        // get_number_of_electrons();  Don't need to know the number of electrons for now
        build_basis_functions();
        add_dprime_to_primitive();
        overlap_matrix = final_overlap();
        gamma_mat = gamma_matrix();
        P_alpha = initialize_density_matrices();
        P_beta = initialize_density_matrices();
        

    }


    // Methods both old and new
    void read_in_atoms(const std::filesystem::path atoms_file_path){

        // This read atoms will need to handle the new atoms we have
        std::ifstream atoms(atoms_file_path);

        // Check that the above line worked properly
        if (!atoms.is_open()) {
            throw std::runtime_error("Could not open file: " + atoms_file_path.string());
        }   

        // The first line will be the number of atoms
        int num_atoms {0};
        atoms >> num_atoms;
        Atom The_Atoms;

         // For the number of atoms store info inside an Atom struct
        for(int i = 0; i < num_atoms; i++){
            atoms >> The_Atoms.Z >> The_Atoms.x >> The_Atoms.y >> The_Atoms.z;
            
            // Add each atom to the vector of atoms
            atoms_of_molecule.push_back(The_Atoms);
        }
    }


    // Using the vector atoms get the coordinate matrix we've gotten accustomed to 
    void read_coordinates(){

        int num_atoms = atoms_of_molecule.size();

        arma::mat coords(num_atoms, 4);

        for(int i = 0; i < num_atoms; i++){
            Atom molecular_atom = atoms_of_molecule.at(i);
            coords(i, 0) = molecular_atom.Z;
            coords(i, 1) = molecular_atom.x;
            coords(i, 2) = molecular_atom.y;
            coords(i, 3) = molecular_atom.z;
        }

    molecular_coordinates = coords;
        
    }

    // Get the number of basis functions: Temporary logic, works for now but not precise
    void get_number_of_basis_funcs(){

        // Here N = 4a + b, where a = number of C's & b = number of H's
        int non_hydrogen{0};
        int hydrogens{0};
        int num_atoms = atoms_of_molecule.size();

        for(int i = 0; i < num_atoms; i++){
            if(atoms_of_molecule.at(i).Z == 1){
                hydrogens += 1;
            }
            else{ non_hydrogen += 1;}
            
        }

        num_basis_funcs = 4 * non_hydrogen + hydrogens;

    }

    // Get the number of electrons
    void get_number_of_electrons(){

        // 2n = 4a + b, there for n = 2a + b/2 (Error out if n != integer)
        if(num_basis_funcs % 2 != 0){
            throw std::runtime_error("Number of electrons must be an integer.");
        }

        num_electrons = num_basis_funcs / 2;
    }

    double double_factorial(int N){

    if(N == - 1){return 1;}
    if(N == 0){return 1;}
    int total = N;
    
    for(int i = 2; i < N; i = i + 2){
        total = total * (N - i);
        // std::cout << "Total: " << " i: " << i << std::endl;
    }
    return total;
    }

    // Get Angular Momentum
    int angular_momentum(BasisFunction Base){
        int res {0};
        for(int i = 0; i < 3; i++){
            res += Base.quantum_numbers.at(i);
        }
        return res;
    }


    // We need the overlap matrix we computed in the previous molecule class
    // Therefore we also need all of that functions dependencies

     arma::vec normalize_shells(Gaussian_3D gauss){

        // Compute the self overlap using code from homework 2
        arma::mat self_overlap = build_overlap_matrix(gauss, gauss);

        // Vector to store normalization constants
        arma::vec norm_const(self_overlap.n_rows);


        for(int i = 0; i < self_overlap.n_rows; i++){
            // The self overlap for element i
            double SAA = self_overlap(i, i);

            // square root of SAA
            double root_SAA = std::sqrt(SAA);

            // 1 / square root of SAA
            double result = 1 / root_SAA;

            // Store in the normalization vector
            norm_const.at(i) = result;

        }

        return norm_const;

    }

    arma::vec compute_normalization_vector(int momentum, arma::vec exponent_vec){
        arma::vec normalization_vector(exponent_vec.n_elem);

        for(int i = 0; i < exponent_vec.n_elem; i++){
            // Build gaussian to compute overlap
            Gaussian_3D Gauss(0,0,0, exponent_vec.at(i), momentum);

            // Call the above helper function get the normalization vector for the above exponent
            arma::vec norms = normalize_shells(Gauss);

            // The values in this vector are the same, we extract the first one so we are not out of bounds for S shells
            normalization_vector.at(i) = norms.at(0);
            
        }

        return normalization_vector;
    }


    void build_basis_functions(){

        // This is a critical component of the code.

        // Getting the number of atoms in the molecule
        int num_atoms = atoms_of_molecule.size();

        // Create shell & basis info objects to extract info from
        Shells the_shells;
        BasisInfo the_basis_info;

        // For each atom in the molecule we will build the basis and push it
        // On to the attribute basis_functions
        for(int i = 0; i < num_atoms; i++){

            // Get the atom
            Atom the_atom = atoms_of_molecule.at(i);

            // Is this atom carbon or hydrogen

            // Hydrogen atoms
            if(the_atom.Z == 1){
                BasisFunction hydrogen_basis;
                hydrogen_basis.atom = the_atom;

                // Here we store the atom index, we will need this later
                hydrogen_basis.atom_idx = i;

                // Hydrogen is s_shell
                arma::vec hydrogen_shell = the_shells.S_shell;


                // The center is the atomic coordinates
                hydrogen_basis.center = {the_atom.x, the_atom.y, the_atom.z};

                // This is hydrogen so the quantum numbers are (0, 0, 0)
                hydrogen_basis.quantum_numbers = the_shells.S_shell;

                for(int j = 0; j < hydrogen_shell.size(); j++ ){
                    Primitive prime;
                    prime.coefficient = the_basis_info.hydrogen_contractions.at(j);
                    prime.exponent = the_basis_info.hydrogen_exponents.at(j);
                    prime.normalization = h2_norm.at(j);

                    hydrogen_basis.primitives.push_back(prime);

                }

                basis_functions.push_back(hydrogen_basis);
            }

        // The code for building the carbon basis
        if(the_atom.Z==6){

            std::vector<BasisFunction> carbons_four_basis_functions(4);

            // S shell basis
            BasisFunction b1;
            b1.atom = the_atom;
            b1.atom_idx = i;

            // Px shell basis
            BasisFunction b2;
            b2.atom = the_atom;
            b2.atom_idx = i;
            
            // Py shell basis
            BasisFunction b3;
            b3.atom = the_atom;
            b3.atom_idx = i;

            // Pz shell basis
            BasisFunction b4;
            b4.atom = the_atom;
            b4.atom_idx = i;

            // Assign Basis Function Centers
            b1.center = {the_atom.x, the_atom.y, the_atom.z};
            b2.center = {the_atom.x, the_atom.y, the_atom.z};
            b3.center = {the_atom.x, the_atom.y, the_atom.z};
            b4.center = {the_atom.x, the_atom.y, the_atom.z};
            

            // Carbon has an S_shell & three P_shells
            arma::vec carbon_S_shell = the_shells.S_shell;
            arma::vec carbon_P_shell_1 = the_shells.P_shells.at(0);
            arma::vec carbon_P_shell_2 = the_shells.P_shells.at(1);
            arma::vec carbon_P_shell_3 = the_shells.P_shells.at(2);

            // The shells contain the quantum numbers
            b1.quantum_numbers = carbon_S_shell;
            b2.quantum_numbers = carbon_P_shell_1;
            b3.quantum_numbers = carbon_P_shell_2;
            b4.quantum_numbers = carbon_P_shell_3;

            // Get the S shell primitives
            for(int i = 0; i < 3; i++){
                Primitive prime;
                prime.coefficient = the_basis_info.carbon_S_shell_contractions.at(i);
                prime.exponent = the_basis_info.carbon_exponents.at(i);
                prime.normalization = carbon_S_shell_norm.at(i);

                b1.primitives.push_back(prime);
            }

            // Get the P shell primitives
            for(int i = 0; i < 3; i++){
                Primitive prime;
                prime.coefficient = the_basis_info.carbon_P_shell_contractions.at(i);
                prime.exponent = the_basis_info.carbon_exponents.at(i);
                prime.normalization = carbon_P_shell_norm.at(i);

                /* 
                In this case the primitives will be the same for each basis function
                This is because the normalization is depends on the quantum numbers
                Here each combination is equivalent so the normalization vector will be
                Identical. The contraction coefficient and exponents don't change, so
                The primitives are the same.
                */

                // Therefore we can assign the primitives to each basis function
                b2.primitives.push_back(prime);
                b3.primitives.push_back(prime);
                b4.primitives.push_back(prime);


            }

            


            carbons_four_basis_functions.at(0) = b1;
            carbons_four_basis_functions.at(1) = b2;
            carbons_four_basis_functions.at(2) = b3;
            carbons_four_basis_functions.at(3) = b4;

            // For clarity, the above vector is not strictly necessary
            for(int i = 0; i < carbons_four_basis_functions.size(); i++){
                basis_functions.push_back(carbons_four_basis_functions.at(i));
            }
        }


        // The code for building the fluorine basis
        if(the_atom.Z==9){

            std::vector<BasisFunction> fluorines_four_basis_functions(4);

            // S shell basis
            BasisFunction b1;
            b1.atom = the_atom;
            b1.atom_idx = i;

            // Px shell basis
            BasisFunction b2;
            b2.atom = the_atom;
            b2.atom_idx = i;
            
            // Py shell basis
            BasisFunction b3;
            b3.atom = the_atom;
            b3.atom_idx = i;
            // Pz shell basis
            BasisFunction b4;
            b4.atom = the_atom;
            b4.atom_idx = i;

            // Assign Basis Function Centers
            b1.center = {the_atom.x, the_atom.y, the_atom.z};
            b2.center = {the_atom.x, the_atom.y, the_atom.z};
            b3.center = {the_atom.x, the_atom.y, the_atom.z};
            b4.center = {the_atom.x, the_atom.y, the_atom.z};
            

            // Fluorine has an S_shell & three P_shells
            arma::vec fluorine_S_shell = the_shells.S_shell;
            arma::vec fluorine_P_shell_1 = the_shells.P_shells.at(0);
            arma::vec fluorine_P_shell_2 = the_shells.P_shells.at(1);
            arma::vec fluorine_P_shell_3 = the_shells.P_shells.at(2);



            // The shells contain the quantum numbers
            b1.quantum_numbers = fluorine_S_shell;
            b2.quantum_numbers = fluorine_P_shell_1;
            b3.quantum_numbers = fluorine_P_shell_2;
            b4.quantum_numbers = fluorine_P_shell_3;



            // Get the S shell primitives
            for(int i = 0; i < 3; i++){
                Primitive prime;
                prime.exponent = the_basis_info.fluorine_exponents.at(i);
                prime.coefficient = the_basis_info.fluorine_S_shell_contractions.at(i);
                prime.normalization = fluorine_S_shell_norm.at(i);

                b1.primitives.push_back(prime);
            }

            // Get the P shell primitives
            for(int i = 0; i < 3; i++){
                Primitive prime;
                prime.coefficient = the_basis_info.fluorine_P_shell_contractions.at(i);
                prime.exponent = the_basis_info.fluorine_exponents.at(i);
                prime.normalization = fluorine_P_shell_norm.at(i);

                /* 
                In this case the primitives will be the same for each basis function
                This is because the normalization is depends on the quantum numbers
                Here each combination is equivalent so the normalization vector will be
                Identical. The contraction coefficient and exponents don't change, so
                The primitives are the same.
                */

                // Therefore we can assign the primitives to each basis function
                b2.primitives.push_back(prime);
                b3.primitives.push_back(prime);
                b4.primitives.push_back(prime);


            }

            

            fluorines_four_basis_functions.at(0) = b1;
            fluorines_four_basis_functions.at(1) = b2;
            fluorines_four_basis_functions.at(2) = b3;
            fluorines_four_basis_functions.at(3) = b4;

            // For clarity, the above vector is not strictly necessary
            for(int i = 0; i < fluorines_four_basis_functions.size(); i++){
                basis_functions.push_back(fluorines_four_basis_functions.at(i));
            }
        }

        // The code for building the oxygen basis
        if(the_atom.Z==8){

            std::vector<BasisFunction> oxygens_four_basis_functions(4);

            // S shell basis
            BasisFunction b1;
            b1.atom = the_atom;
            b1.atom_idx = i;

            // Px shell basis
            BasisFunction b2;
            b2.atom = the_atom;
            b2.atom_idx = i;
            
            // Py shell basis
            BasisFunction b3;
            b3.atom = the_atom;
            b3.atom_idx = i;

            // Pz shell basis
            BasisFunction b4;
            b4.atom = the_atom;
            b4.atom_idx = i;

            // Assign Basis Function Centers
            b1.center = {the_atom.x, the_atom.y, the_atom.z};
            b2.center = {the_atom.x, the_atom.y, the_atom.z};
            b3.center = {the_atom.x, the_atom.y, the_atom.z};
            b4.center = {the_atom.x, the_atom.y, the_atom.z};
            

            // Fluorine has an S_shell & three P_shells
            arma::vec oxygen_S_shell = the_shells.S_shell;
            arma::vec oxygen_P_shell_1 = the_shells.P_shells.at(0);
            arma::vec oxygen_P_shell_2 = the_shells.P_shells.at(1);
            arma::vec oxygen_P_shell_3 = the_shells.P_shells.at(2);



            // The shells contain the quantum numbers
            b1.quantum_numbers = oxygen_S_shell;
            b2.quantum_numbers = oxygen_P_shell_1;
            b3.quantum_numbers = oxygen_P_shell_2;
            b4.quantum_numbers = oxygen_P_shell_3;



            // Get the S shell primitives
            for(int i = 0; i < 3; i++){
                Primitive prime;
                prime.exponent = the_basis_info.oxygen_exponents.at(i);
                prime.coefficient = the_basis_info.oxygen_S_shell_contractions.at(i);
                prime.normalization = oxygen_S_shell_norm.at(i);

                b1.primitives.push_back(prime);
            }

            // Get the P shell primitives
            for(int i = 0; i < 3; i++){
                Primitive prime;
                prime.coefficient = the_basis_info.oxygen_P_shell_contractions.at(i);
                prime.exponent = the_basis_info.oxygen_exponents.at(i);
                prime.normalization = oxygen_P_shell_norm.at(i);

                /* 
                In this case the primitives will be the same for each basis function
                This is because the normalization is depends on the quantum numbers
                Here each combination is equivalent so the normalization vector will be
                Identical. The contraction coefficient and exponents don't change, so
                The primitives are the same.
                */

                // Therefore we can assign the primitives to each basis function
                b2.primitives.push_back(prime);
                b3.primitives.push_back(prime);
                b4.primitives.push_back(prime);


            }

            
            oxygens_four_basis_functions.at(0) = b1;
            oxygens_four_basis_functions.at(1) = b2;
            oxygens_four_basis_functions.at(2) = b3;
            oxygens_four_basis_functions.at(3) = b4;

            // For clarity, the above vector is not strictly necessary
            for(int i = 0; i < oxygens_four_basis_functions.size(); i++){
                basis_functions.push_back(oxygens_four_basis_functions.at(i));
            }
        }

        }
    }


    double Suv(BasisFunction basis_1, BasisFunction basis_2){

        // Need a variable to store the final result
        double res {0};

        // Step 1 is to extract information required to build the Gaussian objects
        arma::vec center_of_b1 = basis_1.center;
        arma::vec center_of_b2 = basis_2.center;

        // Recall lax, lay, laz from previous problem set. Here for each basis there constant

        // The momentum numbers for basis 1
        int b1_lax = basis_1.quantum_numbers.at(0);
        int b1_lay = basis_1.quantum_numbers.at(1);
        int b1_laz = basis_1.quantum_numbers.at(2);

        // The momentum numbers for basis 2
        int b2_lbx = basis_2.quantum_numbers.at(0);
        int b2_lby = basis_2.quantum_numbers.at(1);
        int b2_lbz = basis_2.quantum_numbers.at(2);

        // The total momentum for creating gaussian objects
        int momentum_of_b1 = angular_momentum(basis_1);
        int momentum_of_b2 = angular_momentum(basis_2);

        for(int i = 0; i < 3; i++){
            // Build a Gaussian object from basis 1 primitives
            Primitive prime_of_b1 = basis_1.primitives.at(i);
            Gaussian_3D basis_1_gaussian(center_of_b1.at(0), center_of_b1.at(1), center_of_b1.at(2), prime_of_b1.exponent, momentum_of_b1);

            for(int j = 0; j < 3; j++){

                // Build Gaussians to compute Skl using code from previous assignment
                Primitive prime_of_b2 = basis_2.primitives.at(j);
                Gaussian_3D basis_2_gaussian(center_of_b2.at(0), center_of_b2.at(1), center_of_b2.at(2), prime_of_b2.exponent, momentum_of_b2);

                double x_component = SAB_1D(basis_1_gaussian, basis_2_gaussian, b1_lax, b2_lbx, 0);
                double y_component = SAB_1D(basis_1_gaussian, basis_2_gaussian, b1_lay, b2_lby, 1);
                double z_component = SAB_1D(basis_1_gaussian, basis_2_gaussian, b1_laz, b2_lbz, 2);

                // Skl
                double total = x_component * y_component * z_component;

                // Now we the the contracted coeff & Normalization constants from each primitive
                double contract_coefficients = prime_of_b1.coefficient * prime_of_b2.coefficient;
                double norm_factors = prime_of_b1.normalization * prime_of_b2.normalization;

                res += contract_coefficients * norm_factors * total;
          
            }

        }

        return res;
    }

    arma::mat final_overlap(){

        arma::mat res_mat(basis_functions.size(), basis_functions.size());
        
        for(int i = 0; i < basis_functions.size(); i++){

            // Get the basis function for the outer loop
            BasisFunction outer_basis = basis_functions.at(i);

            for(int j = 0; j < basis_functions.size(); j++){

                // Get the basis function for the inner loop
                BasisFunction inner_basis = basis_functions.at(j);

                res_mat(i, j) = Suv(outer_basis, inner_basis);

            }
        }

        // res_mat.print("This Is The Overlap Matrix.");
        cout << endl;

        return res_mat;
    }



    /*
    -----------------------------------------------------------------------------------
        Above Is Mostly The Code From The Previous Molecule Class Implementation. 
        Parts of the code required refactoring but it retains it main functionality.
        May attempt to use inheritance if this class get to big.
    -----------------------------------------------------------------------------------
    */
    
    /* 
        Here is a list of functions that will be needed:
        1. Gamma AB - Needed for Fock Non - digonals elements
        2. Puv - Guess Zero at first then compute for subsequent iterations
        3. Get Beta Values - helper to extract BetaA & BetaB from EmpiricalFactors structure
        4. Fuv - Compute Fock Non - diagonal elements
        5. Fuu - Compute Fock Diagonal elements
        6. Helper functions to compute gamma Ab
    */
        

        // Assign beta values directly to each atom
        void assign_empirical_factors(){
            // Called by constructor to add Beta values to Atoms
            EmpiricalFactors empirical_factors;

            // Loop through all atoms and assign Negative Beta (As a positive number)
            for(auto &atom : atoms_of_molecule){
                atom.Beta = empirical_factors.negative_beta.at(atom.Z);
            }

        }

        // Below are a series of helper function to eventually compute gamma AB
        double get_distance_between_atoms(){
            // This code is specific to the known input and may not work for larger systems

            Atom A = atoms_of_molecule.at(0);
            Atom B = atoms_of_molecule.at(1);

            double x_sq_diff = (A.x - B.x) * (A.x - B.x);
            double y_sq_diff = (A.y - B.y) * (A.y - B.y);
            double z_sq_diff = (A.z - B.z) * (A.z - B.z);

            double summed = x_sq_diff + y_sq_diff + z_sq_diff;

            double res = std::sqrt(summed);

            return res;
        }

        double get_distance_between_basis_centers(const arma::vec center_1, const arma::vec center_2){
            // More general version. Can pass in any coordinate vectors and get the distance between them
            double dx = center_1.at(0) - center_2.at(0);
            double dy = center_1.at(1) - center_2.at(1);
            double dz = center_1.at(2) - center_2.at(2);

            double sum_of_squares = (dx*dx) + (dy*dy) + (dz*dz);
            double result = std::sqrt(sum_of_squares);
            return result;
        }

        // Implement equation 3.3
        void add_dprime_to_primitive(){

            for(auto &basis : basis_functions){
                for(auto &primitive : basis.primitives){
                    primitive.d_prime = primitive.coefficient * primitive.normalization;
                }
            }
        }

        // Start the density matrices as zero matrices
        arma::mat initialize_density_matrices(){
            arma::mat density(num_basis_funcs, num_basis_funcs, arma::fill::zeros);

            return density;
        }

        // Implementing equations provided in the appendix (the result is equation 3.15 for R > epsilon and equation 3.16 for R < epsilon)
        double primitive_gamma(Primitive &pk, Primitive &pkp, Primitive &pl, Primitive &plp, double R){


            const double epsilon = 0.00000000000001;
            double sigmaA = 1.0 / (pk.exponent + pkp.exponent);  // The exponents of primitive for s shell of atom 1
            double sigmaB = 1.0 / (pl.exponent + plp.exponent);  // The exponents of primitive for s shell of atom 2

            double UA = std::pow(M_PI * sigmaA, 1.5);       // UA = (pi * sigmaA)^3/2 from the appendix
            double UB = std::pow(M_PI * sigmaB, 1.5);       // UB = (pi * sigmaB)^3/2 from the appendix
            double U = UA * UB;         

            double denominator_of_V_sq = sigmaA + sigmaB;

            double V_sq = 1 / denominator_of_V_sq;

            // If R = 0, follow T = 0 equation (3.16)
            if(R < epsilon){
                double two_times_v_sq = 2 * V_sq;
                double two_divided_by_pi = 2/M_PI;

                double root_two_v_sq = std::sqrt(two_times_v_sq);
                double root_pi_divided_by_two = std::sqrt(two_divided_by_pi);

                double res = U * root_two_v_sq * root_pi_divided_by_two;

                return res;
            }

            double T = V_sq * (R * R);  // Note R is the distance between two centers
            

            double one_over_dist_sq = 1 / (R * R);

        
            double root_one_over_dist_sq = std::sqrt(one_over_dist_sq);
            double erf_root_T = std::erf(std::sqrt(T));

            double result = U * root_one_over_dist_sq * erf_root_T;

            return result;

        }

        // // Implementation of eq 3.4 (choose four nested loops against the advice of the TA)
        // double compute_gamma_scalar(bool diagonals = false, int atom_index = 0){

        //     BasisFunction B1 = basis_functions.at(atom_index);  // S shell basis of the first atom
        //     BasisFunction B2;  // S shell basis of the second atom    

        //     double dist_ij;
        //     double res {0};

        //     // For the diagonal terms basis functions comes from the same atom
        //     if(diagonals == true){
        //         B2 = basis_functions.at(atom_index);

        //         // One Atom, in the same location as itself
        //         dist_ij = 0;
        //     } 
            
        //     // For non-diagonal terms basis functions come from different atoms
        //     else {

        //         // Non-diagonal terms, need to compute the distance between atoms
        //         dist_ij = get_distance_between_atoms();
        //         if (atom_index == 0){
        //             B2 = basis_functions.at(1);
        //         } else {
        //             B2 = basis_functions.at(0);
        //         }
        //     }

            
        //     for (int k = 0; k < 3; k++){
        //         for (int kp = 0; kp < 3; kp++){
        //             for (int l = 0; l < 3; l++){
        //                 for (int lp =0; lp < 3; lp++){
        //                     Primitive pk = B1.primitives.at(k);
        //                     Primitive pkp = B1.primitives.at(kp);
        //                     Primitive pl = B2.primitives.at(l);
        //                     Primitive plp = B2.primitives.at(lp);

        //                     // Notation follow the k, k', l, l' Notation used with the summation signs in eq 3.4 (The p stands for primitive)
        //                     res += pk.d_prime * pkp.d_prime * pl.d_prime * plp.d_prime * primitive_gamma(pk, pkp, pl, plp, dist_ij);
                            
        //                 }
        //             }
        //         }
        //     }
            
        //     return res;
            
        // }

        double compute_gamma_updated(int basis_idx_1, int basis_idx_2){

        /* 
            The version is a modified version of the method above We will probably leave the one above
            since its working but ideally once this version is properly implemented we should delete the
            above version and use this one
        */

        BasisFunction B1 = basis_functions.at(basis_idx_1);
        BasisFunction B2 = basis_functions.at(basis_idx_2);
        const double dist_ij = get_distance_between_basis_centers(B1.center, B2.center);

        double res = 0.0;

        for (int k = 0; k < 3; k++){
                for (int kp = 0; kp < 3; kp++){
                    for (int l = 0; l < 3; l++){
                        for (int lp =0; lp < 3; lp++){
                            Primitive pk = B1.primitives.at(k);
                            Primitive pkp = B1.primitives.at(kp);
                            Primitive pl = B2.primitives.at(l);
                            Primitive plp = B2.primitives.at(lp);

                            // Notation follow the k, k', l, l' Notation used with the summation signs in eq 3.4 (The p stands for primitive)
                            res += pk.d_prime * pkp.d_prime * pl.d_prime * plp.d_prime * primitive_gamma(pk, pkp, pl, plp, dist_ij);
                            
                        }
                    }
                }
            }
            
            return res;


            
        }

        // Old only worked for two atom case
        // arma::mat gamma_matrix(){

        //     // const double Hartree_to_eV = 27.211324570273;

        //     double gammaAA = compute_gamma_scalar(true, 0) * Conversion_factor;
        //     double gammaBB = compute_gamma_scalar(true, 1) * Conversion_factor;
        //     double gammaAB = compute_gamma_scalar(false) * Conversion_factor;

        //     arma::mat gamma_mat(2, 2);

        //     gamma_mat(0,0) = gammaAA;
        //     gamma_mat(0,1) = gammaAB;
        //     gamma_mat(1,0) = gammaAB;
        //     gamma_mat(1,1) = gammaBB;

        //     return gamma_mat;
        // }

        arma::mat gamma_matrix(){

            int num_atoms = atoms_of_molecule.size();
            arma::mat result(num_atoms, num_atoms, arma::fill::zeros);

            // For each atom, find the index of its S-type basis function
            std::vector<int> s_basis_index(num_atoms, -1);

            for(int b = 0; b < num_basis_funcs; b++){
                const BasisFunction &bf = basis_functions.at(b);

                // S-type if lx+ly+lz == 0
                if(arma::accu(bf.quantum_numbers) == 0){
                    int atom_idx = bf.atom_idx;
                    if(s_basis_index.at(atom_idx) == -1){
                        s_basis_index.at(atom_idx) = b;
                    }
                }
            }

            // (Optional) sanity check: make sure every atom has an S basis
            for(int A = 0; A < num_atoms; A++){
                if(s_basis_index[A] == -1){
                    throw std::runtime_error("No S-type basis function found for atom index " + std::to_string(A));
                }
            }

            // Fill gamma matrix in atom space using S-shell basis functions
            for(int A = 0; A < num_atoms; A++){
                for(int B = A; B < num_atoms; B++){
                    int idxA = s_basis_index.at(A);
                    int idxB = s_basis_index.at(B);

                    double gammaAB = compute_gamma_updated(idxA, idxB) * Conversion_factor; // in eV
                    result(A, B) = gammaAB;
                    result(B, A) = gammaAB;
                }
            }

            return result;
        }



        double Huu(int index){

            // Instantiate the empirical factors class
            EmpiricalFactors empirical_factors;

            // The index is for basis functions
            // We need to get the appropriate atom

            BasisFunction basis;
            Atom atom; 
            basis = basis_functions.at(index);
            atom = basis.atom;

        
            // -1/2 * (Iu + Au)
            double neg_one_half_Iu_plus_Au = empirical_factors.one_half_Is_plus_As.at(atom.Z) * -1;
            
            // For the p-shells we have -1/2 * (Ip + Ap)
            double neg_one_half_Ip_plus_Ap {0.0};
            
            // If the atom it a hydrogen there is no P shell
            if(atom.Z != 1){
                neg_one_half_Ip_plus_Ap = empirical_factors.one_half_Ip_plus_Ap.at(atom.Z) * -1;
            }
            

            // Z_starA
            double Z_starA = empirical_factors.zeta_starA.at(atom.Z);


            // Z_starA - 1/2
            double Z_star_A_minus_one_half = Z_starA - 0.5;

            

            int s_idx = basis.atom_idx;

            
            double gammaAA = gamma_mat(s_idx, s_idx);

            // Get the sum of Z_starC * gammaAC
            double sum {0};

            for(int i = 0; i < atoms_of_molecule.size(); i++){

                
                // Skip the case whether the atoms are the same
                if(basis.atom_idx == i){continue;}

                // Skip the case whether the atoms are the same (OLD Broke with new inputs)
                // if(index == 0 && i == 0){continue;}
                // if(index != 0 && i == 1){continue;}

                Atom atom_B = atoms_of_molecule.at(i);
                double zeta = empirical_factors.zeta_starA.at(atom_B.Z);

                // Old line, broke with new inputs
                // double gammaAC = compute_gamma_scalar(false) * Conversion_factor;
                // double gammaAC = compute_gamma_updated(basis.atom_idx, i) * Conversion_factor;
                double gammaAC = gamma_mat(basis.atom_idx, i);

                sum += zeta * gammaAC;
            }
        

            double first_two_terms_combined_S_shell = neg_one_half_Iu_plus_Au - Z_star_A_minus_one_half * gammaAA;
            double first_two_terms_combined_P_shell = neg_one_half_Ip_plus_Ap - Z_star_A_minus_one_half * gammaAA;

            double result{0};


            if(arma::accu(basis.quantum_numbers) == 0){
                result = first_two_terms_combined_S_shell - sum;
            }
            else{
                result = first_two_terms_combined_P_shell - sum;
            }
            

            return result;
        }

        double Huv(int idx1, int idx2){

            // Build diagonals
            if(idx1 == idx2){
                return Huu(idx1);
            }

            Atom atom_A;
            Atom atom_B;


            // OLD: Broke with new inputs
            // if(idx1 == 0){
            //     atom_A = atoms_of_molecule.at(idx1);
            // }
            // else {atom_A = atoms_of_molecule.at(1);}

            // if(idx2 == 0){
            //     atom_B = atoms_of_molecule.at(idx2);
            // }
            // else {atom_B = atoms_of_molecule.at(1);}

            atom_A = basis_functions.at(idx1).atom;
            atom_B = basis_functions.at(idx2).atom;


           
            

            double betaA_plus_betaB = -atom_A.Beta - atom_B.Beta;
            double half_betaA_plus_betaB = betaA_plus_betaB * 0.5;

            
            double Suv = overlap_matrix(idx1, idx2);

            double result = half_betaA_plus_betaB * Suv;

            return result;

            
        }

        // Unused function, H_core is actually built with code below.
        arma::mat build_H_core_diagonals(){
            
            arma::mat result(num_basis_funcs, num_basis_funcs, arma::fill::zeros);

            for(int idx = 0; idx < atoms_of_molecule.size(); idx++){
                double value = Huu(idx);
                result(idx, idx) = value;
            }

            return result;
        }

        arma::mat build_H_core(){

            // The matrices have the following dimensions: num_basis_functions by num_basis_functions
            
            arma::mat result(num_basis_funcs, num_basis_funcs, arma::fill::zeros);

            for(int i = 0; i < num_basis_funcs; i++){
                for(int j = 0; j < num_basis_funcs; j++){
                    result(i, j) = Huv(i, j);               
                }
            }

            return result;
        }

    
        /*-------------------- Building The Fock Matrix ---------------------*/

        // AI Assistance Debugging this function

        double Fuv(int basis_idx_1, int basis_idx_2, bool alpha=true){


           // If the two indices are the same return Fuu
           if(basis_idx_1 == basis_idx_2){
               return Fuu(basis_idx_1, alpha);
           }


           // Instantiate atom objects
           Atom atom_1;
           Atom atom_2;

           // Get Basis Objects
           BasisFunction basis_1 = basis_functions.at(basis_idx_1);
           BasisFunction basis_2 = basis_functions.at(basis_idx_2);


           // The first basis in the list is always hydrogen, the second and beyond the other atom (This is no longer true, We need to modify)
           //    if(basis_idx_1 == 0){
           //        atom_1 = atoms_of_molecule.at(0);
           //    } else { atom_1 = atoms_of_molecule.at(1); }

           // This logic replaces both the logic directly above and directly below
           atom_1 = atoms_of_molecule.at(basis_1.atom_idx);
           atom_2 = atoms_of_molecule.at(basis_2.atom_idx);


           // Repeat the above logic for atom 2
           //    if(basis_idx_2 == 0){
           //        atom_2 = atoms_of_molecule.at(0);
           //    } else { atom_2 = atoms_of_molecule.at(1); }


           // For clarity Extract BetaA and BetaB from atom 1 and atom 2
           double BetaA = -atom_1.Beta;
           double BetaB = -atom_2.Beta;


           // Build the first term 1/2 * (BetaA + BetaB)
           double one_half_betaA_plus_betaB = (BetaA + BetaB) * 0.5;


           
           // The first basis index is associated with the first atom, hydrogen, the rest are associated with the second atom (No Longer True)
           int atom_idx_1;
           int atom_idx_2;

           // if(basis_idx_1 == 0){atom_idx_1 = 0;} 
           // else { atom_idx_1 = 1; }

           atom_idx_1 = basis_1.atom_idx;
           atom_idx_2 = basis_2.atom_idx;

           

            // int atom_idx_2;
            // if(basis_idx_2 == 0){ atom_idx_2 = 0;}
            // else {atom_idx_2 = 1; } 
           
           
           // Extract gammaAB from the correct location in the gamma matrix (If the atom indices are the same gammaAB is one of the gammaAAs, but choose the correct one)
           double gammaAB;
           if(atom_idx_1 == atom_idx_2){gammaAB = gamma_mat(atom_idx_1, atom_idx_1); }
           else { gammaAB = gamma_mat(atom_idx_1, atom_idx_2); }


           // Get Suv
           double Suv = overlap_matrix(basis_idx_1, basis_idx_2);


           // Get Puv
           double Puv_alpha = P_alpha(basis_idx_1, basis_idx_2);
           double Puv_beta  = P_beta(basis_idx_1, basis_idx_2);


           // Compute the result ( 1/2 * (BetaA + BetaB) * Suv ) - (Puv * gamma)
           double result_alpha = ( one_half_betaA_plus_betaB * Suv ) - (Puv_alpha * gammaAB);
           double result_beta  = ( one_half_betaA_plus_betaB * Suv ) - (Puv_beta  * gammaAB);


           if(alpha==true){
               return result_alpha;
           } else {
               return result_beta;
           }
       }



       // AI Assistance Debugging this function

       double Fuu(int basis_idx, bool alpha=true){


               // Testing incorporation of Atom structure into BasisFunction structure
               BasisFunction basis_func = basis_functions.at(basis_idx);
               Atom atom = basis_func.atom;


               // Instantiating an empirical factor object
               EmpiricalFactors empirical_factors;


               // Getting the empirical factor based on which shell
               double one_half_Ix_plus_Ax;

               // New Line, was previously hard coded to previous assignments inputs
               if(arma::accu(basis_func.quantum_numbers) == 0){
                    one_half_Ix_plus_Ax = -1 * empirical_factors.one_half_Is_plus_As.at(atom.Z);
               } else {
                    one_half_Ix_plus_Ax = -1 * empirical_factors.one_half_Ip_plus_Ap.at(atom.Z);
               }


                // get gammaAA
                int s_idx = basis_func.atom_idx;
                // double gammaAA = compute_gamma_updated(s_idx, s_idx) * Conversion_factor;
                double gammaAA = gamma_mat(s_idx, s_idx);
               
                // Select the population for the atom μ is on
                arma::mat P_tot = P_alpha + P_beta;
                double pop_atom = 0.0;
                for(int i = 0; i < num_basis_funcs; i++){
                    if(basis_functions[i].atom_idx == basis_func.atom_idx){
                        pop_atom += P_tot(i,i);
                    }
                }
                

                // get Puu_spin
                double Puu_spin;
                if(alpha == true){Puu_spin = P_alpha(basis_idx, basis_idx); }
                else {Puu_spin = P_beta(basis_idx, basis_idx); }


               // for clarity we will extract ZetaA
               double ZetaA = empirical_factors.zeta_starA.at(atom.Z);


               /*---- Build Term by Term ----*/

               double term_atom_charge = (pop_atom - ZetaA);
               double spin_term = (Puu_spin - 0.5);
               double inside_parenthesis = term_atom_charge - spin_term;
               double first_term = one_half_Ix_plus_Ax + (inside_parenthesis * gammaAA);


               // Calculate the summation term
               double sum {0.0};


               for(int idx = 0; idx < atoms_of_molecule.size(); idx++){


                    if(idx == basis_func.atom_idx){continue;}
    
                    // Atom to extract ZetaC
                    Atom atom_for_indexing = atoms_of_molecule.at(idx);

                   int this_atom = basis_func.atom_idx;
                   int other_atom = idx;

                   // Get gammaAC
                   double gammaAC = gamma_mat(this_atom, other_atom);
                   
                   // Total population on atom C
                   double pCC_total = 0.0;
                    for(int i = 0; i < num_basis_funcs; i++){
                        if(basis_functions[i].atom_idx == other_atom){
                            pCC_total += P_tot(i,i);
                        }
                    }
                   

                   // Extracting ZetaC
                   double ZetaC = empirical_factors.zeta_starA.at(atom_for_indexing.Z);


                   // sum += (pCC - ZetaC) * gammaAC;
                   sum += (pCC_total - ZetaC) * gammaAC;
                }


               double result = first_term + sum;

               return result;
           }


        
        arma::mat build_fock_matrix(bool alpha=true){

            arma::mat fock_mat(num_basis_funcs, num_basis_funcs, arma::fill::zeros);

            for(int i = 0; i < num_basis_funcs; i++){
                for(int j = 0; j < num_basis_funcs; j++){
                    fock_mat(i, j) = Fuv(i, j, alpha);
                }
            }

            return fock_mat;
        }
        

        void construct_P_total(){
            // We will choose when to call this function to construct P_total from P_alpha & P_beta
            P_total = P_alpha + P_beta;
        }

        /*---------- Solving the Eigenvalue Problem ----------*/

        // This class is getting long, a new class that inherits above functionality would probably be preferable, but I'll just continue instead

        // Below we will follow the same work flow we used for EHF (code below largely copied from HW 3 implementation)

        arma::mat get_S_inverse(){
            // Here X * H * X^T becomes X * F * X^T, where X = S^-1/2.

            // Store the result in a matrix
            arma::mat s_inverse;

            // We already have the overlap matrix store as a class attribute so let's get the eigenvectors & values
            arma::vec eigenvalues;
            arma::mat eigenvectors;

            arma::eig_sym(eigenvalues, eigenvectors, overlap_matrix);

            arma::mat diagonal_matrix = arma::diagmat(eigenvalues);
            // diagonal_matrix.print("Diagonal Inside S_inverse before Loop.");

            // Loop through diagonal matrix and replace with 1 / root(eigenvalue)
            for(int i = 0; i < diagonal_matrix.n_rows; i++){
                int j = i;

                // Get the eigenvalue at that location
                double value = diagonal_matrix(i,j);

                // Take the square root of the value
                value = std::sqrt(value);

                // Replace with 1/value
                diagonal_matrix(i,j) = 1/value;
            }

            // diagonal_matrix.print("Diagonal Matrix inside S_inverse after Loop.");

            s_inverse = eigenvectors * diagonal_matrix * eigenvectors.t();

            // s_inverse.print("S_inverse.");
            


            return s_inverse;


        }

        arma::mat orthogonalized_fock_matrix(bool alpha=true){

            // We need the Fock matrix and the S inverse we computed above
            arma::mat Fock_mat = build_fock_matrix(alpha);
            arma::mat S_inv = get_S_inverse();

            // Get Fock Ortho
            arma::mat Fock_ortho = S_inv * Fock_mat * S_inv;

            return Fock_ortho;
        }

        std::pair<arma::mat, arma::vec> get_eigenvectors_and_eigenvalues(bool alpha=true){

        
            // Without orthogonalizing the fock matrix
            arma::mat Fock_mat = build_fock_matrix(alpha);
            arma::mat eigenvectors;
            arma::vec eigenvalues;
            arma::eig_sym(eigenvalues, eigenvectors, Fock_mat);


            // // Orthogonalizing the fock matrix
            // arma::mat Fock_ortho = orthogonalized_fock_matrix(alpha);
            // arma::mat eigenvectors;
            // arma::vec eigenvalues;
            // arma::eig_sym(eigenvalues, eigenvectors, Fock_ortho);

            
            return std::make_pair(eigenvectors, eigenvalues);
        }

        
        arma::mat get_molecular_coefficients(bool alpha=true){


            // Get eigenvectors
            auto [eigenvectors, eigenvalues] = get_eigenvectors_and_eigenvalues(alpha);

            // Create coefficient matrix
            arma::mat molecular_coeff = eigenvectors;

            
            return molecular_coeff;
        }

        

        arma::mat build_density_matrix(arma::mat molecular_coefficients_matrix, int num_electrons){

            // This is a key piece of functionality. Let's be explicit for clarity
            int num_rows_and_cols = num_basis_funcs;

            // This is what we what to build: The dimensions of the coefficient matrix should be num_basis_funcs by num_basis_funcs
            arma::mat density(molecular_coefficients_matrix.n_rows, molecular_coefficients_matrix.n_cols);

            // Here we loop through the rows and columns of the matrix
            
            // For every row in the matrix
            for(int i = 0; i < num_rows_and_cols; i++){   
                
                // For every column in the matrix
                for(int j = 0; j < num_rows_and_cols; j++){

                    // We will replace the current value of the density matrix with a new value
                    double value{0.0};

                    // Here is where the parameters alpha and beta electrons come into play, which we will pass in as parameter
                    for(int k = 0; k < num_electrons; k++){
                        
                        // We get the appropriate molecular coefficient
                        double mol_coeff_wrt_i = molecular_coefficients_matrix(i, k);
                        double mol_coeff_wrt_j = molecular_coefficients_matrix(j, k);

                        value += (mol_coeff_wrt_i * mol_coeff_wrt_j);
                    }

                    // Once out of the inner summation loop we can add to the density matrix
                    density(i, j) = value;
                }

            }

            // Once properly built let's return it
            return density;
            
        }

        double get_electronic_energy(){
            // We will call this once the SCF algorithm has completed
            double sum_of_alpha_portion{0.0};
            for(int i = 0; i < num_basis_funcs; i++){
                for(int j = 0; j < num_basis_funcs; j++){
                    double Huv_val = Huv(i, j);  // Function call
                    double Fuv_alpha_val = Fuv(i, j, true);  // Function call
                    double Puv = P_alpha(i, j); // Value extraction from class attribute
                    double result = Puv * (Huv_val + Fuv_alpha_val);
                    sum_of_alpha_portion += result;
                }
            }

            double sum_of_beta_portion{0.0};
            for(int i = 0; i < num_basis_funcs; i++){
                for(int j = 0; j < num_basis_funcs; j++){
                    double Huv_val = Huv(i, j);  // Function call
                    double Fuv_beta_val = Fuv(i, j, false);  // Function call
                    double Puv = P_beta(i, j); // Value extraction from class attribute
                    double result = Puv * (Huv_val + Fuv_beta_val);
                    sum_of_beta_portion += result;
                }
            }

            double one_half_first_term = sum_of_alpha_portion * 0.5;
            double one_half_second_term = sum_of_beta_portion * 0.5;

            return one_half_first_term + one_half_second_term;


        }

        double get_nuclear_repulsion_energy(){

            double res{0.0};

            for(int atom_A = 0; atom_A < atoms_of_molecule.size(); atom_A++){
                for(int atom_B = 0; atom_B < atom_A; atom_B++){
                    EmpiricalFactors emp_factors;
                    Atom A = atoms_of_molecule.at(atom_A);
                    double zetaA = emp_factors.zeta_starA.at(A.Z);

                    Atom B = atoms_of_molecule.at(atom_B);
                    double zetaB = emp_factors.zeta_starA.at(B.Z);
                    arma::vec centerA = {A.x, A.y, A.z};
                    arma::vec centerB = {B.x, B.y, B.z};
                    double R_AB = get_distance_between_basis_centers(centerA, centerB);

                    double numerator = zetaA * zetaB;
                    double factor = (1 / R_AB);
                    double result = numerator * factor;
                    res += result;


                }
            }
            return res * 27.211324570273;
        }

        std::pair<double, double> SCF_algorithm(){

            // Even though it's called in the constructor let's make sure P_alpha, P_beta, and P_total begin as zero
            P_alpha = initialize_density_matrices();
            P_beta = initialize_density_matrices();
            construct_P_total();


            // Main Loop
            int max_iter = 50;
            double threshold = 1e-6;


            // We want to display the output in a similar manner to the solution so we can easily compare and debug
            // cout << "Num Alpha Electrons: " << num_alpha_e << endl;
            // cout << "Num Beta Electrons: " << num_beta_e << endl;
            // gamma_mat.print("Gamma Matrix");
            // overlap_matrix.print("Overlap Matrix");
            arma::mat H_core = build_H_core();
            // H_core.print("H_core");

            arma::mat Fa;
            arma::mat Fb;

            for(int iter = 0; iter < max_iter; iter++){

                // Build Fa and Fb
                Fa = build_fock_matrix(true);
                Fb = build_fock_matrix(false);


                // Get the molecular coefficients
                arma::mat coeff_matrix_alpha = get_molecular_coefficients(true);
                arma::mat coeff_matrix_beta = get_molecular_coefficients(false);
                arma::mat P_alpha_new = build_density_matrix(coeff_matrix_alpha, num_alpha_e);
                arma::mat P_beta_new = build_density_matrix(coeff_matrix_beta, num_beta_e);

                arma::mat P_total_new = P_alpha_new + P_beta_new;

                // Get the largest difference between the element of the new matrix and the old
                double delta_a = arma::abs(P_alpha_new - P_alpha).max();  // All other differences will be less than this.
                double delta_b = arma::abs(P_beta_new - P_beta).max();
                double delta = std::max(delta_a, delta_b);      // Take the largest one. If this one has converged so has the other.


                // cout << "Iteration Number: " << iter << endl << endl;
                // Fa.print("Fa ");
                // Fb.print("Fb ");
                // cout << "After solving Eigen Equation: " << iter << endl;
                // cout << "P = " << num_alpha_e << " Q = " << num_beta_e << endl;
                // coeff_matrix_alpha.print("Ca ");
                // coeff_matrix_beta.print("Cb ");
                
                // P_alpha_new.print("P_alpha_new");
                // P_beta_new.print("P_beta_new");
                // P_total_new.print("P_total_new");


                if (delta < threshold){
                    cout << " SCF converged after " << iter << " iterations." << endl;
                    break;
                }

                P_alpha = P_alpha_new;
                P_beta = P_beta_new;
                construct_P_total();

            }

            construct_P_total();
            Fock_alpha = Fa;
            Fock_beta = Fb;
            arma::mat F_total = Fa + Fb;

            /* Printing Important Final Values*/

            // Getting The Final Eigenvectors
            auto final_eigenvals_and_vecs_alpha = get_eigenvectors_and_eigenvalues(true);
            auto final_eigenvals_and_vecs_beta = get_eigenvectors_and_eigenvalues(false);

            
            // Extracting The Eigenvectors and Values given alpha num electrons
            arma::mat final_eigenvecs_alpha = final_eigenvals_and_vecs_alpha.first;
            arma::vec final_eigenvals_alpha = final_eigenvals_and_vecs_alpha.second;


            // Extractin the Eigenvectors and Values given beta num electrons
            arma::mat final_eigenvecs_beta = final_eigenvals_and_vecs_beta.first;
            arma::vec final_eigenvals_beta = final_eigenvals_and_vecs_beta.second;

            // Printing The Converged Eigenvectors For Alpha and Beta Number of Electrons
            cout << endl;
            // final_eigenvecs_alpha.print("Final Eigenvectors For Alpha Number of Electrons: ");
            cout << endl;
            // final_eigenvecs_beta.print("Final Eigenvectors For Beta Number of Electrons: ");

            // Printing The Converged Eigenvalus For Alpha and Beta Number of Electrons
            // final_eigenvals_alpha.print("Final Ea: ");
            cout << endl;
            // final_eigenvals_beta.print("Final Eb: ");
            cout << endl;


            double Electronic_energy = get_electronic_energy();
            // cout << "Electronic Energy: " << Electronic_energy << endl;

            double Nuclear_repulsion_energy = get_nuclear_repulsion_energy();
            // cout << "Nuclear Repulsion Energy: " << Nuclear_repulsion_energy << endl;

            return std::make_pair(Electronic_energy, Nuclear_repulsion_energy);
        }

    void SCF_debug(){


        P_alpha = initialize_density_matrices();
        P_beta = initialize_density_matrices();
        construct_P_total();


        arma::mat Fa = build_fock_matrix(true);
        arma::mat Fb = build_fock_matrix(false);

        Fa.print("Initial Fock A");
        Fb.print("Initial Fock B");

        arma::mat Ca = get_molecular_coefficients(true);
        arma::mat Cb = get_molecular_coefficients(false);

        Ca.print("Initial Ca");
        Cb.print("Initial Cb");

        arma::mat P_alpha_old = P_alpha;
        arma::mat P_beta_old = P_beta;

        P_alpha_old.print("P_alpha_old");
        P_beta_old.print("P_beta_old");

        P_alpha = build_density_matrix(Ca, num_alpha_e);
        P_beta = build_density_matrix(Cb, num_beta_e);

        P_alpha.print("P_alpha");
        P_beta.print("P_beta");
        construct_P_total();

        cout << "==================== End of Round 1 ==========================" << endl;


        arma::mat Fa2 = build_fock_matrix(true);
        arma::mat Fb2 = build_fock_matrix(false);

        Fa2.print("Initial Fock A round 2");
        Fb2.print("Initial Fock B round 2");

        arma::mat Ca2 = get_molecular_coefficients(true);
        arma::mat Cb2 = get_molecular_coefficients(false);

        Ca2.print("Initial Ca round 2");
        Cb2.print("Initial Cb round 2");

        P_alpha_old = P_alpha;
        P_beta_old = P_beta;

        P_alpha_old.print("P_alpha_old rd 2");
        P_beta_old.print("P_beta_old rd 2");

        P_alpha = build_density_matrix(Ca2, num_alpha_e);
        P_beta = build_density_matrix(Cb2, num_beta_e);

        P_alpha.print("P_alpha rd 2");
        P_beta.print("P_beta rd 2");
        construct_P_total();
    }

    
    // Helper function to complete extra credit
    void update_geometry(const arma::mat& new_coordinates) {
        
        // The updated coordinate system
        molecular_coordinates = new_coordinates;

        // For every atom in the system update its position
        for (size_t i = 0; i < atoms_of_molecule.size(); ++i) {
            atoms_of_molecule.at(i).x = new_coordinates(i, 0);
            atoms_of_molecule.at(i).y = new_coordinates(i, 1);
            atoms_of_molecule.at(i).z = new_coordinates(i, 2);
        }

        // Clear the old basis functions. They will need to be rebuilt
        basis_functions.clear();

        // Rebuild everything based on new geometry
        build_basis_functions(); 
        add_dprime_to_primitive();
        
        // Recalculate everything
        overlap_matrix = final_overlap();
        gamma_mat = gamma_matrix();
        
        // Reset density
        P_alpha = initialize_density_matrices();
        P_beta = initialize_density_matrices();
        construct_P_total();
    }

    void print_summary(){
        // This will print a summary of what we have so far
        molecular_coordinates.print("These are the coordinates");

        // Print Information on The Basis Functions
        cout << "\nThe List of Basis Functions:\n" << endl;
        for(int i =0; i < basis_functions.size(); i++){
            BasisFunction basis = basis_functions.at(i);
            cout << "Basis #" << i + 1 << ": " << endl;
            cout << "The Center: \n" << basis.center << endl;
            cout << "The Quantum numbers: \n" << basis.quantum_numbers << endl;

            Primitive prime1 = basis.primitives.at(0);
            Primitive prime2 = basis.primitives.at(1);
            Primitive prime3 = basis.primitives.at(2);

            arma::vec p1 = {prime1.coefficient, prime1.exponent, prime1.normalization};
            arma::vec p2 = {prime2.coefficient, prime2.exponent, prime2.normalization};
            arma::vec p3 = {prime3.coefficient, prime3.exponent, prime3.normalization};

            cout <<"The three primatives: \n" << endl;

            cout << "Primitive 1: \n";
            cout << "Coefficient: " << p1.at(0) << endl;
            cout << "Exponent: " << p1.at(1) << endl;
            cout << "Normalization Factor: " << p1.at(2) << endl << endl;

            cout << "Primitive 2: \n";
            cout << "Coefficient: " << p2.at(0) << endl;
            cout << "Exponent: " << p2.at(1) << endl;
            cout << "Normalization Factor: " << p2.at(2) << endl << endl;

            cout << "Primitive 3: \n";
            cout << "Coefficient: " << p3.at(0) << endl;
            cout << "Exponent: " << p3.at(1) << endl;
            cout << "Normalization Factor: " << p3.at(2) << endl << endl;

        }

        overlap_matrix.print("This is the final overlap matrix.");
    }

};
