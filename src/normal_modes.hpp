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
#include "new_molecule.hpp"
#include "CNDO.hpp"


class HessianBuilder{

private:
    CNDO& cndo;
    NewMolecule& molecule;

    std::unordered_map<int, double> atomic_masses;

    std::string symbol_from_Z(int Z) const {
    switch (Z) {
        case 1: return "H";
        case 6: return "C";
        case 7: return "N";
        case 8: return "O";
        case 9: return "F";
        default: return "X";
    }
}


public:
    HessianBuilder(CNDO& C);

    arma::vec flatten_gradient(const arma::mat& gradient);

    arma::mat compute_hessian(double h);

    arma::mat mass_weight(const arma::mat& H);

    arma::mat compute_vibrations(double h);

    arma::vec compute_frequencies(const arma::vec& eigvals);

    void write_mode_xyz(const arma::mat& cart_modes,
                    int mode_index,
                    int n_frames,
                    double amplitude,
                    const std::string& base_filename);



};