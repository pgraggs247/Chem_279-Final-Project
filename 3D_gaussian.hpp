#pragma once

#include <vector>
#include<armadillo>


class Gaussian_3D{

    private:
        double X_A;
        double Y_A;
        double Z_A;
        arma::vec centroid;
        double exponent;
        int momentum;
        std::vector<arma::vec> L0;
        std::vector<arma::vec> L1;
        std::vector<arma::vec> L2;

    public:
        Gaussian_3D(double X_coord, double Y_coord, double Z_coord, double exponent_, int momentum_);

    // Acessors
    double get_X_coord();

    double get_Y_coord();

    double get_Z_coord();

    arma::vec get_centroid();

    double get_exponent();

    int get_momentum();

    std::vector<arma::vec> get_momentum_vector(int L);

    std::vector<arma::vec> get_momentum_vec_for_this_G();

    // Mutators
    void set_X_coord(double X_coord);

    void set_Y_coord(double Y_coord);

    void set_Z_coord(double Z_coord);

};


int compute_factorial(int N);

int m_choose_n(int M, int N);

arma::vec get_product_center(Gaussian_3D G_A, Gaussian_3D G_B);

double get_leading_term(Gaussian_3D Gauss_1, Gaussian_3D Gauss_2, int coord);

double double_factorial(int N);

double the_main_event(Gaussian_3D Gauss_1, Gaussian_3D Gauss_2, int coord);

double SAB_1D(Gaussian_3D G1, Gaussian_3D G2, int lA, int lB, int coord);

arma::mat build_overlap_matrix(Gaussian_3D G1, Gaussian_3D G2);
