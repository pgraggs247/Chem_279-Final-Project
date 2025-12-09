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
            arma::vec hessian_column = -(gradient_plus - gradient_minus) / (2 * h);
            

            // Store vector as column in the hessian matrix
            result.col(idx) = hessian_column;

            // Restore geometry
            molecule.update_geometry(original_coords);

        }

    }

    result = 0.5 * (result + result.t());

    return result;

}


arma::mat HessianBuilder::mass_weight(const arma::mat& hessian){

    int N = hessian.n_rows;
    int num_atoms = molecule.atoms_of_molecule.size();
    arma::vec mass_vec(N, arma::fill::zeros);
    arma::mat mass_weighted_hessian(N, N, arma::fill::zeros);

    const double amu_to_au = 1822.888486;

    for(int atom = 0; atom < num_atoms; atom++){
        int Z = molecule.atoms_of_molecule.at(atom).Z;

        if (atomic_masses.find(Z) == atomic_masses.end()){
            throw std::runtime_error("Code doesn't support one of the atoms in given molecule.");
        }

        double mass = atomic_masses.at(Z);
        double mass_au = mass * amu_to_au;

        mass_vec(3*atom + 0) = mass;
        mass_vec(3*atom + 1) = mass;
        mass_vec(3*atom + 2) = mass;


    }

    arma::vec inv_sqrt_mass = 1.0 / arma::sqrt(mass_vec);

    arma::mat D = arma::diagmat(inv_sqrt_mass);

    mass_weighted_hessian = D * hessian * D;

    return mass_weighted_hessian;

}


arma::mat HessianBuilder::compute_vibrations(double h){
    int num_atoms = molecule.atoms_of_molecule.size();
    int N3 = 3 * num_atoms;
    arma::mat hessian(N3, N3, arma::fill::zeros);
    arma::mat mass_weighted(N3, N3, arma::fill::zeros);

    hessian = compute_hessian(h);
    mass_weighted = mass_weight(hessian);

    arma::mat eigvecs;
    arma::vec eigvals;

    arma::eig_sym(eigvals, eigvecs, mass_weighted);

    arma::vec mass_vec(N3, arma::fill::zeros);

    for(int atom = 0; atom < num_atoms; atom++){
        int Z = molecule.atoms_of_molecule.at(atom).Z;
        double mass = atomic_masses.at(Z);

        mass_vec(3*atom + 0) = mass;
        mass_vec(3*atom + 1) = mass;
        mass_vec(3*atom + 2) = mass;
    }

    arma::vec sqrt_mass = arma::sqrt(mass_vec);
    arma::mat D_inv = arma::diagmat(sqrt_mass);

    arma::mat cart_modes(N3, N3, arma::fill::zeros);

    for(int mode = 0; mode < N3; mode++){
        arma::vec mw_eigvec = eigvecs.col(mode);
        arma::vec cart_vec = D_inv * mw_eigvec;
        cart_modes.col(mode) = cart_vec;
    }

    return cart_modes;


}


arma::vec HessianBuilder::compute_frequencies(const arma::vec& eigvals){
    arma::vec freqs(eigvals.n_elem, arma::fill::zeros);

    const double au_to_wavenumbers = 5140.48;
    // const double au_to_wavenumbers = 219474.63;


    for(arma::uword i = 0; i < eigvals.n_elem; i++){
        double lambda = eigvals(i);

        if(std::abs(lambda) < 1e-5){
            freqs(i) = 0.0;
            continue;
        }

        if (lambda > 0.0){
            freqs(i) = std::sqrt(lambda) * au_to_wavenumbers;
        } else {
            freqs(i) = -std::sqrt(std::abs(lambda)) * au_to_wavenumbers;
        }


    }

    return freqs;
}

// void HessianBuilder::write_mode_xyz(const arma::mat& cart_modes,
//                                     int mode_index,
//                                     int n_frames,
//                                     double amplitude,
//                                     const std::string& base_filename)
// {
//     const double bohr_to_ang = 0.529177;
//     int num_atoms = molecule.atoms_of_molecule.size();
//     int N3 = 3 * num_atoms;

//     // Extract the mode vector (length 3N)
//     arma::vec mode_vec = cart_modes.col(mode_index);

//     // Convert current equilibrium geometry (bohr) to an arma::mat
//     arma::mat geom = molecule.molecular_coordinates;  // (N x 3)

//     for (int frame = 0; frame < n_frames; frame++)
//     {
//         double phase = (2.0 * M_PI * frame) / n_frames;
//         double displacement_scale = amplitude * std::sin(phase);

//         // Output filename: example: mode_5_frame_012.xyz
//         std::ostringstream fname;
//         fname << base_filename << "_mode_" << mode_index
//               << "_frame_" << std::setw(3) << std::setfill('0') << frame
//               << ".xyz";

//         std::ofstream out(fname.str());
//         if (!out.is_open()) {
//             throw std::runtime_error("ERROR: Could not open XYZ output file.");
//         }

//         // --- XYZ HEADER ---
//         out << num_atoms << "\n";
//         out << "Mode " << mode_index
//             << ", Frame " << frame
//             << ", Amplitude = " << amplitude
//             << "\n";

//         // --- PRINT ATOMS ---
//         for (int a = 0; a < num_atoms; a++)
//         {
//             double dx = displacement_scale * mode_vec(3*a + 0);
//             double dy = displacement_scale * mode_vec(3*a + 1);
//             double dz = displacement_scale * mode_vec(3*a + 2);

//             double x = (geom(a,0) + dx) * bohr_to_ang;
//             double y = (geom(a,1) + dy) * bohr_to_ang;
//             double z = (geom(a,2) + dz) * bohr_to_ang;

//             int Z = molecule.atoms_of_molecule[a].Z;

//             // Avogadro expects element symbols.
//             std::string symbol = symbol_from_Z(Z);

//             out << symbol << "  "
//                 << std::fixed << std::setprecision(6)
//                 << x << "  " << y << "  " << z << "\n";
//         }

//         out.close();
//     }

//     std::cout << "Wrote animations for mode " << mode_index
//               << " to files starting with: " << base_filename << "\n";
// }

void HessianBuilder::write_mode_xyz(const arma::mat& cart_modes,
                                    int mode_index,
                                    int n_frames,
                                    double amplitude,
                                    const std::string& base_filename)
{
    const double bohr_to_ang = 0.529177;
    int num_atoms = molecule.atoms_of_molecule.size();
    int N3 = 3 * num_atoms;

    arma::vec mode_vec = cart_modes.col(mode_index);
    arma::mat geom = molecule.molecular_coordinates;  // N x 3

    // Create ONE trajectory file:
    std::ostringstream full_name;
    full_name << base_filename << "_mode_" << mode_index << ".xyz";
    std::ofstream out(full_name.str());

    if (!out.is_open()) {
        throw std::runtime_error("ERROR: Could not open XYZ output file.");
    }

    for (int frame = 0; frame < n_frames; frame++)
    {
        double phase = (2.0 * M_PI * frame) / n_frames;
        double displacement_scale = amplitude * std::sin(phase);

        // --- XYZ HEADER FOR THIS FRAME ---
        out << num_atoms << "\n";
        out << "Mode " << mode_index
            << ", Frame " << frame
            << ", Amplitude = " << amplitude
            << "\n";

        // --- PRINT ATOMS FOR THIS FRAME ---
        for (int a = 0; a < num_atoms; a++)
        {
            double dx = displacement_scale * mode_vec(3*a + 0);
            double dy = displacement_scale * mode_vec(3*a + 1);
            double dz = displacement_scale * mode_vec(3*a + 2);

            double x = (geom(a,0) + dx) * bohr_to_ang;
            double y = (geom(a,1) + dy) * bohr_to_ang;
            double z = (geom(a,2) + dz) * bohr_to_ang;

            int Z = molecule.atoms_of_molecule[a].Z;
            std::string symbol = symbol_from_Z(Z);

            out << symbol << "  "
                << std::fixed << std::setprecision(6)
                << x << "  " << y << "  " << z << "\n";
        }
    }

    out.close();

    std::cout << "Wrote multi-frame XYZ animation to: "
              << full_name.str() << "\n";
}
