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

public:
    HessianBuilder(CNDO& C);

    arma::vec flatten_gradient(const arma::mat& gradient);

    arma::mat compute_hessian(double h);

    arma::mat mass_weight(const arma::mat& H);

    void compute_vibrations(double h);

};