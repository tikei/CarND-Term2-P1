#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  // check validity of inputs
  if(estimations.size() == 0 || estimations.size() != ground_truth.size()){
    std::cout << "Invalid estimation or ground truth data!" << std::endl;
    return rmse;
  }
  // accumulate squared residuals
  for(unsigned int i=0; i < estimations.size(); ++i){
    VectorXd residual = estimations[i] - ground_truth[i];

    // square the residuals, component-wise (Hadamard) product
    residual = residual.array() * residual.array();

    rmse += residual;
  }

  // calculate mean
  rmse = rmse / estimations.size();

  // square root
  rmse = rmse .array().sqrt();

  return rmse;

}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {

  MatrixXd Hj(3,4);
  //recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  //pre-compute a set of terms to avoid repeated calculation
  float c1 = px*px+py*py;
  float c2 = sqrt(c1);
  float c3 = (c1*c2);

  //check division by zero
  double epsilon = 0.0001;
  if (std::fabs(px) < epsilon && std::fabs(py) < epsilon)
  {
      px = epsilon;
      py = epsilon;
  }

  // if(fabs(c1) < 0.0001){
  //     std::cout << "CalculateJacobian () - Error - Division by Zero" << std::endl;
  //     Hj = MatrixXd::Identity(3, 4);
  //     return Hj;
  // }

  //compute the Jacobian matrix
  Hj << (px/c2), (py/c2), 0, 0,
       -(py/c1), (px/c1), 0, 0,
        py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

  return Hj;
}

VectorXd Tools::CalculateHofX(const VectorXd& x_state){
  VectorXd hx(3);
  //recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  //pre-compute terms to avoid repeated calculation
  float c1 = px*px+py*py;
  float ro = sqrt(c1);

  float ro_dot = (px*vx + py*vy) / ro;
  float theta = std::atan2(py, px);

  hx << ro, theta, ro_dot;

  return hx;
}
