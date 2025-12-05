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
#include "new_molecule.hpp"
#include "CNDO.hpp"
#include "normal_modes.hpp"


HessianBuilder::HessianBuilder(CNDO& cndo_object)
    : cndo(cndo_object), 
      molecule(cndo_object.get_molecule())   
    {
        atomic_masses = {

            {1, 1.0079},
            {6, 12.011},
            {7, 14.007},
            {8, 15.999},
            {9, 18.998}
        };
    }



// Convert 3 x N gradient matrix into a 3N-dimensional vector
arma::vec HessianBuilder::flatten_gradient(const arma::mat& gradient){

    
    int num_atoms = molecule.atoms_of_molecule.size();
    arma::vec flattened_vec(3*num_atoms, arma::fill::zeros);

    for(int A= 0; A < num_atoms; A++){
        for(int k = 0; k < 3; k++){
            int idx = 3 * A + k;
            flattened_vec.at(idx) = gradient(k, A);

        }
    }

    return flattened_vec;

}

arma::mat HessianBuilder::compute_hessian(double h){

    int num_atoms = molecule.atoms_of_molecule.size();
    int N_times_3 = 3 * num_atoms;

    arma::mat result(N_times_3, N_times_3 , arma::fill::zeros);

    arma::mat original_coords = molecule.molecular_coordinates;


    for(int atom_idx = 0; atom_idx < num_atoms; atom_idx++ ){
        for(int coord_idx = 0; coord_idx < 3; coord_idx++){


            // Initialize matrices
            arma::mat coords_plus_h = original_coords;
            arma::mat coords_minus_h = original_coords;

            // Add h at given coordinates and update geometry
            coords_plus_h(atom_idx, coord_idx) += h;
            molecule.update_geometry(coords_plus_h);
            molecule.SCF_algorithm();

            // Create cndo object with perturb coordinate and get flatten vector
            CNDO cndo_plus(molecule);
            
            arma::vec gradient_plus = flatten_gradient(cndo_plus.compute_gradient());

            // Restore original geometry
            molecule.update_geometry(original_coords);

            // Repeat but this time subtract h
            coords_minus_h(atom_idx, coord_idx) -= h;
            molecule.update_geometry(coords_minus_h);
            molecule.SCF_algorithm();
            CNDO cndo_minus(molecule);
            
            arma::vec gradient_minus = flatten_gradient(cndo_minus.compute_gradient());

            // Get flattened gradient idx
            int idx = atom_idx * 3 + coord_idx;

            // Use armadillo to vectorize the central difference portion
            arma::vec hessian_column = (gradient_plus - gradient_minus) / (2 * h);
            

            // Store vector as column in the hessian matrix
            result.col(idx) = hessian_column;

            // Restore geometry
            molecule.update_geometry(original_coords);

        }

    }

    return result;

}

