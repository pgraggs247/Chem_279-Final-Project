#include <iostream>
#include <armadillo>
#include "new_molecule.hpp"
#include "CNDO.hpp"
#include "normal_modes.hpp"


using std::cout;
using std::endl;


int main(){
    // NewMolecule molecule("atoms/H2.xyz", 1, 1);
    // // molecule.SCF_algorithm();

    // CNDO cndo(molecule);
    // cndo.geometry_optimization();

    // HessianBuilder hes(cndo);
    // arma::mat H = hes.compute_hessian(0.0001);


    // H.print("The Hessian");
    // // arma::mat grad = cndo.compute_gradient();
    // // std::cout << grad << std::endl;

    // // arma::mat asym = H - H.t();
    // // std::cout << "Max |H - H^T| = " << arma::abs(asym).max() << std::endl;

    // // arma::vec eigvals;
    // // arma::mat eigvecs;
    // // arma::eig_sym(eigvals, eigvecs, asym);

    // // eigvals.head(6).print("Smallest Eigenvalues: ");


    // arma::mat mass_weighted_hessian = hes.mass_weight(H);
    // mass_weighted_hessian.print("Mass Weighted Hessian.");

    // arma::mat vibrations = hes.compute_vibrations(0.0001);
    // vibrations.print("Vibrations: ");


    NewMolecule molecule("atoms/water.xyz", 4, 4);
    
    CNDO cndo(molecule);

    cout << "\n=== GEOMETRY OPTIMIZATION ===\n";
    // cndo.geometry_optimization();

    HessianBuilder hes(cndo);

    cout << "\n=== COMPUTING HESSIAN ===\n";
    arma::mat H = hes.compute_hessian(1e-4);

    // Check symmetry
    arma::mat asym = H - H.t();
    double max_asym = arma::abs(asym).max();
    cout << "Max |H - H^T| = " << max_asym << endl;

    if (max_asym > 1e-6)
        cout << "Warning: Hessian is not symmetric!\n";

    cout << "\nHessian (atomic units):\n";
    H.print();

    cout << "\n=== MASS WEIGHTED HESSIAN ===\n";
    arma::mat MW = hes.mass_weight(H);
    MW.print();

    // Diagonalize mass-weighted Hessian
    arma::vec eigvals;
    arma::mat eigvecs;
    arma::eig_sym(eigvals, eigvecs, MW);

    cout << "\nEigenvalues of mass-weighted Hessian:\n";
    eigvals.t().print();

    // Count near-zero eigenvalues
    int zero_modes = 0;
    for (int i = 0; i < eigvals.n_elem; i++)
        if (std::abs(eigvals(i)) < 1e-6)
            zero_modes++;

    cout << "Number of ~zero eigenvalues = " << zero_modes << " (expected = 5)\n";

    arma::vec freqs = hes.compute_frequencies(eigvals);
    cout << "\nVibrational frequencies (cm^-1):\n";
    freqs.t().print("These Are The Frequencies.");

    cout << "\n=== CLASSIFICATION ===\n";
    for (int i = 0; i < freqs.n_elem; i++) {
        double f = freqs(i);
        if (std::abs(f) < 5.0)
            cout << "Mode " << i << ": Translational/Rotational (freq ≈ " << f << ")\n";
        else if (f > 0)
            cout << "Mode " << i << ": Real vibration  (freq = " << f << ")\n";
        else
            cout << "Mode " << i << ": Imaginary vibration (freq = " << f << ")\n";
    }

cout << "\n=== CARTESIAN VIBRATIONAL MODES ===\n";
arma::mat cart_modes = hes.compute_vibrations(1e-4);
cart_modes.print("Cartesian Normal Modes:");

// ------------------------------------------------------
// Ensure output directory exists
// ------------------------------------------------------
// std::filesystem::create_directories("mode_5");

// ------------------------------------------------------
// Write animation for mode 5 into ONE multi-frame XYZ file
// ------------------------------------------------------
// std::string output_base = "mode_5/H2O";   
// int mode_to_write = 8;       // choose vibration index
// int n_frames = 60;           // number of animation frames
// double amplitude = 0.15;     // amplitude in bohr

// hes.write_mode_xyz(
//     cart_modes,
//     mode_to_write,
//     n_frames,
//     amplitude,
//     output_base
// );

// cout << "Animation written to: " << output_base 
//      << "_mode_" << mode_to_write << ".xyz\n";

return 0;
}
