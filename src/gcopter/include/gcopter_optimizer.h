/*
 * GcopterWayptS3 — MINCO with hard intermediate waypoint constraints
 *                  and optional SFC polytope position constraints.
 * Adapted from SUPER (github.com/hku-mars/SUPER).
 *
 * Points layout: [free0, wpt0, free1, wpt1, ..., free_{N-1}, wpt_{N-1}, free_N]
 *   - even indices = free points  (optimization variables xi)
 *   - odd indices  = waypoints    (fixed, never updated by the optimizer)
 *
 * When hPolys is provided, SFC position constraints are enforced on all pieces
 * during numerical integration.
 */

#ifndef GCOPTER_OPTIMIZER_H
#define GCOPTER_OPTIMIZER_H

#include <Eigen/Eigen>
#include <vector>

#include "flatness.hpp"
#include "lbfgs.hpp"
#include "minco.hpp"
#include "trajectory.hpp"

namespace gcopter {

class GcopterWayptS3 {
 public:
  GcopterWayptS3() = default;

  int iter_num{0};
  double initSpeed{0.1};

  // magnitudeBounds = [v_max, omg_max, theta_max, thrust_min, thrust_max]^T  (5D)
  // penaltyWeights  = [weight_vel, weight_omg, weight_theta, weight_thrust]^T (4D)
  // physicalParams  = [mass, grav, horiz_drag, vert_drag, parasitic_drag, speed_eps]^T (6D)
  // hPolys / weightPos: optional SFC polytope constraints (pass empty to skip)
  bool setup(double timeWeight, const Eigen::Matrix3d &initialPVA, const Eigen::Matrix3d &terminalPVA,
             const Eigen::Matrix3Xd &_waypoints, double smoothingFactor, int integralResolution,
             const Eigen::VectorXd &magnitudeBounds, const Eigen::VectorXd &penaltyWeights,
             const Eigen::VectorXd &physicalParams,
             const std::vector<Eigen::MatrixX4d> &hPolys = {}, double weightPos = 0.0);

  double optimize(Trajectory<5> &traj, double relCostTol);

 private:
  minco::MINCO_S3NU minco;

  double rho{0.0};
  Eigen::Matrix3d headPVA = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d tailPVA = Eigen::Matrix3d::Zero();

  int pieceN{0};
  int temporalDim{0};
  int spatialDim{0};

  double smoothEps{0.0};
  int integralRes{0};
  Eigen::VectorXd magnitudeBd;   // 5D
  Eigen::VectorXd penaltyWt;     // 4D
  Eigen::VectorXd physicalPm;    // 6D

  flatness::FlatnessMap flatMap;

  lbfgs::lbfgs_parameter_t lbfgs_params;

  // points[3, 2*N+1] = [free0, wpt0, free1, wpt1, ..., free_N]
  Eigen::Matrix3Xd points;
  // waypoints[3, N] = input waypoints (for read-only reference)
  Eigen::Matrix3Xd waypoints;
  // free_points[3, N+1] = the subset of points that are optimization variables
  Eigen::Matrix3Xd free_points;
  Eigen::VectorXd times;

  // Gradients
  Eigen::Matrix3Xd gradByPoints;
  Eigen::VectorXd gradByTimes;
  Eigen::MatrixX3d partialGradByCoeffs;
  Eigen::VectorXd partialGradByTimes;

  // SFC polytope constraints
  bool hasSfc_{false};
  double weightPos_{0.0};
  std::vector<Eigen::MatrixX4d> hPolys_;
  std::vector<int> hIdx_;  // piece index → polytope index

  // time mapping (tau ↔ T)
  template <typename EIGENVEC>
  static void forwardT(const Eigen::VectorXd &tau, EIGENVEC &T);

  template <typename EIGENVEC>
  static void backwardT(const Eigen::VectorXd &T, EIGENVEC &tau);

  template <typename EIGENVEC>
  static void backwardGradT(const Eigen::VectorXd &tau, const Eigen::VectorXd &gradT, EIGENVEC &gradTau);

  // penalty
  static void attachPenaltyFunctional(const Eigen::VectorXd &T, const Eigen::MatrixX3d &coeffs, double smoothFactor,
                                      int integralResolution, const Eigen::VectorXd &magnitudeBounds,
                                      const Eigen::VectorXd &penaltyWeights,
                                      flatness::FlatnessMap &flatMap,
                                      bool hasSfc, double weightPos,
                                      const std::vector<Eigen::MatrixX4d> &hPolys, const std::vector<int> &hIdx,
                                      double &cost, Eigen::VectorXd &gradT, Eigen::MatrixX3d &gradC);

  static double costFunctional(void *ptr, const Eigen::VectorXd &x, Eigen::VectorXd &g);

  void setInitial();
};

}  // namespace gcopter

#endif  // GCOPTER_OPTIMIZER_H
