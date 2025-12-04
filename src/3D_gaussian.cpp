#include <fstream> 
#include <functional>
#include <cmath> 
#include <cassert> 
#include <string> 
#include <iostream> 
#include <numbers>
#include <stdexcept>
#include <vector>
#include<armadillo>


#include "3D_gaussian.hpp"


// Three dimensional Gaussian Class   
Gaussian_3D::Gaussian_3D(double X_coord, double Y_coord, double Z_coord, double exponent_, int momentum_){

            this-> X_A = X_coord;
            this-> Y_A = Y_coord;
            this-> Z_A = Z_coord;
            this-> centroid = {X_coord, Y_coord, Z_coord};
            this-> exponent = exponent_;
            this-> momentum = momentum_;
            L0 = {{0,0,0}};
            L1 = {{1,0,0}, {0,1,0}, {0,0,1}};
            L2 = {{2,0,0}, {1,1,0}, {1,0,1}, {0,2,0}, {0,1,1}, {0,0,2}};


        }

    // Acessors
    double Gaussian_3D::get_X_coord(){
        return this-> X_A;
    }

    double Gaussian_3D::get_Y_coord(){
        return this-> Y_A;

    }

    double Gaussian_3D::get_Z_coord(){
        return this-> Z_A;
    }

    arma::vec Gaussian_3D::get_centroid(){
        return this-> centroid;
    }

    double Gaussian_3D::get_exponent(){
        return this -> exponent;
    }

    int Gaussian_3D::get_momentum(){
        return this -> momentum;
    }

    std::vector<arma::vec> Gaussian_3D::get_momentum_vector(int L){
        // Check that valid momentum is passed in
        if(L > 2){
             
            throw std::invalid_argument("We don't provide support for that momentum");
        }


        // Control - flow to deliver correct triplet
        if(L == 0){return L0;}
        else if(L == 1){return L1;}
        else {return L2;}

    }

    std::vector<arma::vec> Gaussian_3D::get_momentum_vec_for_this_G(){
        if(this -> momentum == 0){return L0;}
        else if(this -> momentum == 1){return L1;}
        else {return L2;}
    }

    // Mutators
    void Gaussian_3D::set_X_coord(double X_coord){
        this-> X_A = X_coord;
    }

    void Gaussian_3D::set_Y_coord(double Y_coord){
        this-> Y_A = Y_coord;
    }

    void Gaussian_3D::set_Z_coord(double Z_coord){
        this-> Z_A = Z_coord;
    }



// Function to evaluate the factorial of a number
int compute_factorial(int N){
    
    if(N < 0){
        throw std::invalid_argument("Please pass in a positive integer.");
    }

    if (N == 0){
        int one = 1;
        return one;
    }

    int total = N;
    for(int i = 1; i < N; i++){
        total *= (N - i);
    }

    return total;
}

// Function to evaluate n choose k (here we follow the convention used in instructions and name this m_choose_n)
int m_choose_n(int M, int N){
    int m_factorial = compute_factorial(M);
    int n_factorial = compute_factorial(N);
    int m_minus_n_factorial = compute_factorial(M - N);

    return m_factorial / (n_factorial * m_minus_n_factorial);
}

// Function to compute the product center of two Gaussian ()
arma::vec get_product_center(Gaussian_3D G_A, Gaussian_3D G_B){

    // Initialize vector
    arma::vec center_product{0,0,0};

    // Get the centroid from the gaussians
    arma::vec centroid_A = G_A.get_centroid();
    arma::vec centroid_B = G_B.get_centroid();

    // Get the exponents from the gaussians
    double alpha = G_A.get_exponent();
    double beta = G_B.get_exponent();

    // Extract coordinates from the first centroid
    double X_1 = centroid_A.at(0);
    double Y_1 = centroid_A.at(1);
    double Z_1 = centroid_A.at(2);

    // Extract coordinates from the second centroid
    double X_2 = centroid_B.at(0);
    double Y_2 = centroid_B.at(1);
    double Z_2 = centroid_B.at(2);

    // Add exponents
    double alpha_plus_beta = alpha + beta;

    // Get numerators
    double numerate_X = (alpha * X_1) + (beta * X_2);
    double numerate_Y = (alpha * Y_1) + (beta * Y_2);
    double numerate_Z = (alpha * Z_1) + (beta * Z_2);

    // Get results for the three coordinates
    double res_X = numerate_X / alpha_plus_beta;
    double res_Y = numerate_Y / alpha_plus_beta;
    double res_Z = numerate_Z / alpha_plus_beta;

    // Store result
    center_product = {res_X, res_Y, res_Z};

    // cout << "Testing, is alpha and beta what you expect: " << alpha_plus_beta << endl;


    return center_product;
}

double get_leading_term(Gaussian_3D Gauss_1, Gaussian_3D Gauss_2, int coord){
    
    double result {0};

    // Get exponents
    double alpha = Gauss_1.get_exponent();
    double beta = Gauss_2.get_exponent();

    // Alpha plus beta and alpha times beta
    double alpha_plus_beta = alpha + beta;
    double alpha_times_beta = alpha * beta;

    // Difference of the centers squared
    arma::vec G1_center_coord = Gauss_1.get_centroid();
    arma::vec G2_center_coord = Gauss_2.get_centroid();

    // Get squared difference of proper coordinate
    double diff_of_coord = G1_center_coord.at(coord) - G2_center_coord.at(coord);
    double sq_diff_of_coord = diff_of_coord * diff_of_coord;
   
    // The square root portion of equation 2.9
    constexpr double pi = 3.14159265358979323846;   // Approximation of pi
    double root_of_pi_over_alpha_plus_beta = std::sqrt(pi / alpha_plus_beta);

    // The numerator of the inside of the exp term
    double AB_times_sq_diff_of_coord = alpha_times_beta * sq_diff_of_coord;

    // The full inside of the exp term
    double first_portion_inside = - AB_times_sq_diff_of_coord / alpha_plus_beta;

    // The full exp term
    double first_portion = std::exp(first_portion_inside);

    // The full leading term, what we want to compute
    result = first_portion * root_of_pi_over_alpha_plus_beta;

    return result;
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

// // First version: Rewrote this as SAB_1D below, that version is the one the code relies on
// double the_main_event(Gaussian_3D Gauss_1, Gaussian_3D Gauss_2, int coord){

//     // Check that the coordinate is reasonable
//     if(coord > 2){
//         throw std::invalid_argument("Coordinate number must be between zero and two.");
//     }


//     // Get the product center at the proper coordinate
//     arma::vec product_center = get_product_center(Gauss_1, Gauss_2);
//     double Xp = product_center.at(coord);

//     // Get the proper coordinate of the centroid of Gauss 1
//     double XA{0};
//     if(coord == 0){XA = Gauss_1.get_X_coord();}
//     else if(coord == 1){XA = Gauss_1.get_Y_coord();}
//     else {XA = Gauss_1.get_Z_coord();}

//     // Get the proper coordinate of the centroid of Gauss 2
//     double XB{0};
//     if(coord == 0){XB = Gauss_2.get_X_coord();}
//     else if(coord == 1){XB = Gauss_2.get_Y_coord();}
//     else {XB = Gauss_2.get_Z_coord();}

//     // Get alpha and beta
//     double alpha = Gauss_1.get_exponent();
//     double beta = Gauss_2.get_exponent();

//     // Get momentum value
//     double total_momentum_G1 = Gauss_1.get_momentum();
//     double total_momentum_G2 = Gauss_2.get_momentum();

//     // Get lA & lB
//     std::vector<arma::vec> G1_triplet_vec =  Gauss_1.get_momentum_vector(total_momentum_G1);
//     std::vector<arma::vec> G2_triplet_vec = Gauss_2.get_momentum_vector(total_momentum_G2);

   
//     // Get Xp - XA and Xp - Xb
//     double Xp_minus_XA = Xp - XA;
//     double Xp_minus_XB = Xp - XB;

//     // Get alpha plus beta & two times alpha plus beta
//     double alpha_plus_beta = alpha + beta;
//     double two_times_alpha_plus_beta = 2 * alpha_plus_beta;

//     double true_total {0};
//     for(auto lA_triplet : G1_triplet_vec){
//         for(auto lB_triplet : G2_triplet_vec){

//             if (lA_triplet.at((coord + 1) % 3) != 0 || lA_triplet.at((coord + 2) % 3) != 0)
//             continue;

//             // keep only axis-aligned component for B on this coord
//             if (lB_triplet.at((coord + 1) % 3) != 0 || lB_triplet.at((coord + 2) % 3) != 0)
//                 continue;

//             int lA = lA_triplet.at(coord);
//             int lB = lB_triplet.at(coord);

//             for(int i = 0; i <= lA; i++){
                
//                 for(int j = 0; j <= lB; j++){

//                     if( ((i + j) % 2 == 1) && (XA == XB)){
//                         continue;
//                     }

//                     if ((i + j) % 2 == 1 && std::abs(Xp - XA) < 1e-12 && std::abs(Xp - XB) < 1e-12) {
//                         continue; // odd polynomial → integral must vanish
//                     }

//                     // Compute the double factorial component
//                     int i_plus_j_minus_one = i + j - 1;
//                     double double_fac = double_factorial(i_plus_j_minus_one);

//                     int lA_minus_i = lA - i;
//                     int lB_minus_j = lB - j;
                    

//                     double Xp_minus_XA_raised = std::pow(Xp_minus_XA, lA_minus_i);
//                     double Xp_minus_XB_raised = std::pow(Xp_minus_XB, lB_minus_j);

//                     double i_plus_j_divided_by_two = (i + j) / 2.0;

//                     double numerator = double_fac * Xp_minus_XA_raised * Xp_minus_XB_raised;
//                     double denominator = std::pow(two_times_alpha_plus_beta, i_plus_j_divided_by_two);

//                     double prelim_result = numerator / denominator;

//                     // get lA choose i and lB choose j
//                     double lA_choose_i = m_choose_n(lA, i);
//                     double lB_choose_j = m_choose_n(lB, j);

//                     double result = lA_choose_i * lB_choose_j * prelim_result;

//                     true_total += result;

//                 }

                
//             }

//         }

//     }

//     double leading_term = get_leading_term(Gauss_1, Gauss_2, coord);

//     return leading_term * true_total;
// }

double SAB_1D(Gaussian_3D G1, Gaussian_3D G2, int lA, int lB, int coord){

    // Ensure the user has passed in appropriate values
    if(coord > 2 || coord <0){
        throw std::invalid_argument("Coordinate number must be between zero and two.");
    }

    if(lA > 2 || lA <0){
        throw std::invalid_argument("Momentum must be positive and we currently only support values between zero and two.");
    }

    if(lB > 2 || lB < 0){
        throw std::invalid_argument("Momentum must be positive and we currently only support values between zero and two.");
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


// Function to construct the final overlap matrix
arma::mat build_overlap_matrix(Gaussian_3D G1, Gaussian_3D G2){
    // This function will take all of the building blocks and construct the final overlap matrix S


    // To compute the overlap of the shell we need to extract the shells
    std::vector<arma::vec> G1_shells = G1.get_momentum_vec_for_this_G();
    std::vector<arma::vec> G2_shells = G2.get_momentum_vec_for_this_G();

    // This matrix is where we will store the result
    arma::mat overlap_matrix(G1_shells.size(), G2_shells.size());

    // We will now loop through the shells: Note right now the are of the form std::vector<arma::vec>

    // The outer loop is the lA momentum shell
    for(int i = 0; i < G1_shells.size(); i++){
        
        // The inner loop is the lB momentum shell
        for(int j =0; j < G2_shells.size(); j++){

            // We need to extract the arma::vecs first
            arma::vec lA = G1_shells.at(i); // At i since the outer shell is lA
            arma::vec lB = G2_shells.at(j); // At j since the inner shell is lB

            // Now we need the correct components for each part of the total SB
            int lAx = lA.at(0); int lAy = lA.at(1); int lAz = lA.at(2); // the x, y, & z components of lA
            int lBx = lB.at(0); int lBy = lB.at(1); int lBz = lB.at(2); // the x, y, & z components of lB

            // Now we can pass these in to our function
            double SAB_X = SAB_1D(G1, G2, lAx, lBx, 0); // Zero for the x-coordinate
            double SAB_Y = SAB_1D(G1, G2, lAy, lBy, 1); // One for the y-coordinate
            double SAB_Z = SAB_1D(G1, G2, lAz, lBz, 2); // Two for the z-coordinate

            // Multiply them together to get the result
            double SAB_total = SAB_X * SAB_Y * SAB_Z;

            // Add them to the correct location in the matrix
            overlap_matrix(i, j) = SAB_total;
            
        }
    }

    // Here we transpose the matrix to match the expected output
    overlap_matrix = overlap_matrix.t();

    return overlap_matrix;

}
