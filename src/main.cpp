#include <iostream>
#include <armadillo>
#include "new_molecule.hpp"
#include "CNDO.hpp"
#include "normal_modes.hpp"


using std::cout;
using std::endl;


int main(){
    NewMolecule molecule("atoms/H2.xyz", 1, 1);
    // molecule.SCF_algorithm();

    CNDO cndo(molecule);
    cndo.geometry_optimization();

    HessianBuilder hes(cndo);
    arma::mat H = hes.compute_hessian(0.0001);


    H.print("The Hessian");
    // arma::mat grad = cndo.compute_gradient();
    // std::cout << grad << std::endl;

    arma::mat asym = H - H.t();
    std::cout << "Max |H - H^T| = " << arma::abs(asym).max() << std::endl;
    return 0;
}