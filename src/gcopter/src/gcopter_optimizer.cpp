#include "gcopter_optimizer.h"

#include <ros/console.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

namespace gcopter {

// ============================================================
//  time mapping
// ============================================================

template <typename EIGENVEC>
void GcopterWayptS3::forwardT(const Eigen::VectorXd &tau, EIGENVEC &T) {
  const int n = tau.size();
  T.resize(n);
  for (int i = 0; i < n; i++) {
    if (tau(i) > 0.0) {
      T(i) = (0.5 * tau(i) + 1.0) * tau(i) + 1.0;
    } else {
      T(i) = 1.0 / ((0.5 * tau(i) - 1.0) * tau(i) + 1.0);
    }
  }
}

template <typename EIGENVEC>
void GcopterWayptS3::backwardT(const Eigen::VectorXd &T, EIGENVEC &tau) {
  const int n = T.size();
  tau.resize(n);
  for (int i = 0; i < n; i++) {
    if (T(i) > 1.0) {
      tau(i) = std::sqrt(2.0 * T(i) - 1.0) - 1.0;
    } else {
      tau(i) = 1.0 - std::sqrt(2.0 / T(i) - 1.0);
    }
  }
}

template <typename EIGENVEC>
void GcopterWayptS3::backwardGradT(const Eigen::VectorXd &tau, const Eigen::VectorXd &gradT, EIGENVEC &gradTau) {
  const int n = tau.size();
  gradTau.resize(n);
  for (int i = 0; i < n; i++) {
    if (tau(i) > 0) {
      gradTau(i) = gradT(i) * (tau(i) + 1.0);
    } else {
      double den = (0.5 * tau(i) - 1.0) * tau(i) + 1.0;
      gradTau(i) = gradT(i) * (1.0 - tau(i)) / (den * den);
    }
  }
}

template void GcopterWayptS3::forwardT<Eigen::VectorXd>(const Eigen::VectorXd &, Eigen::VectorXd &);
template void GcopterWayptS3::forwardT<Eigen::Map<Eigen::VectorXd>>(const Eigen::VectorXd &,
                                                                    Eigen::Map<Eigen::VectorXd> &);
template void GcopterWayptS3::backwardT<Eigen::VectorXd>(const Eigen::VectorXd &, Eigen::VectorXd &);
template void GcopterWayptS3::backwardT<Eigen::Map<Eigen::VectorXd>>(const Eigen::VectorXd &,
                                                                     Eigen::Map<Eigen::VectorXd> &);
template void GcopterWayptS3::backwardGradT<Eigen::VectorXd>(const Eigen::VectorXd &, const Eigen::VectorXd &,
                                                             Eigen::VectorXd &);
template void GcopterWayptS3::backwardGradT<Eigen::Map<Eigen::VectorXd>>(const Eigen::VectorXd &,
                                                                         const Eigen::VectorXd &,
                                                                         Eigen::Map<Eigen::VectorXd> &);

// ============================================================
//  smoothed L1
// ============================================================

static inline bool smoothedL1(double x, double mu, double &f, double &df) {
  if (x < 0.0) {
    return false;
  } else if (x > mu) {
    f = x - 0.5 * mu;
    df = 1.0;
    return true;
  } else {
    double xdmu = x / mu;
    double sqrxdmu = xdmu * xdmu;
    double mumxd2 = mu - 0.5 * x;
    f = mumxd2 * sqrxdmu * xdmu;
    df = sqrxdmu * ((-0.5) * xdmu + 3.0 * mumxd2 / mu);
    return true;
  }
}

// ============================================================
//  setup
// ============================================================

bool GcopterWayptS3::setup(double timeWeight, const Eigen::Matrix3d &initialPVA, const Eigen::Matrix3d &terminalPVA,
                           const Eigen::Matrix3Xd &_waypoints, double smoothingFactor, int integralResolution,
                           const Eigen::VectorXd &magnitudeBounds, const Eigen::VectorXd &penaltyWeights,
                           const Eigen::VectorXd &physicalParams, const std::vector<Eigen::MatrixX4d> &hPolys,
                           double weightPos) {
  rho = timeWeight;
  headPVA = initialPVA;
  tailPVA = terminalPVA;
  waypoints = _waypoints;

  smoothEps = smoothingFactor;
  integralRes = integralResolution;
  magnitudeBd = magnitudeBounds;
  penaltyWt = penaltyWeights;
  physicalPm = physicalParams;

  flatMap.reset(physicalParams(0), physicalParams(1), physicalParams(2), physicalParams(3), physicalParams(4),
                physicalParams(5));

  // --- SFC ---
  hasSfc_ = !hPolys.empty() && weightPos > 0.0;
  if (hasSfc_) {
    hPolys_ = hPolys;
    weightPos_ = weightPos;
  }

  const int N = static_cast<int>(waypoints.cols());
  const bool hasWaypoints = (N > 0);

  if (hasWaypoints) {
    pieceN = (N + 1) * 2;
  } else if (hasSfc_) {
    pieceN = static_cast<int>(hPolys_.size());
  } else {
    pieceN = 2;  // degenerate: start → end directly
  }
  temporalDim = pieceN;

  if (hasWaypoints) {
    spatialDim = (N + 1) * 3;
    minco.setConditions(headPVA, tailPVA, pieceN);
    points.resize(3, 2 * N + 1);
    free_points.resize(3, N + 1);
    gradByPoints.resize(3, N + 1);
  } else {
    // No waypoints: all interior junction points are free
    spatialDim = 3 * (pieceN - 1);
    minco.setConditions(headPVA, tailPVA, pieceN);
    points.resize(3, pieceN - 1);
    free_points.resize(3, pieceN - 1);
    gradByPoints.resize(3, pieceN - 1);
  }
  times.resize(pieceN);
  gradByTimes.resize(pieceN);
  partialGradByCoeffs.resize(6 * pieceN, 3);
  partialGradByTimes.resize(pieceN);

  // --- hIdx: piece → polytope mapping ---
  if (hasSfc_) {
    const int P = static_cast<int>(hPolys_.size());
    hIdx_.resize(pieceN);
    if (hasWaypoints) {
      // Waypoint mode: 2 pieces share 1 polytope (P = N+1, pieceN = 2(N+1))
      for (int i = 0; i < pieceN; i++) {
        hIdx_[i] = std::min(i / 2, P - 1);
      }
    } else {
      // Discrete mode: 1 piece : 1 polytope (P == pieceN)
      for (int i = 0; i < pieceN; i++) {
        hIdx_[i] = i;
      }
    }
  }

  return true;
}

// ============================================================
//  setInitial
// ============================================================

void GcopterWayptS3::setInitial() {
  const double minPieceTime = 0.05;

  if (waypoints.cols() == 0) {
    const int NP = static_cast<int>(points.cols());  // pieceN - 1
    const double totalDist = (tailPVA.col(0) - headPVA.col(0)).norm();

    // Distribute free points along straight line from head to tail
    for (int i = 0; i < NP; i++) {
      double alpha = static_cast<double>(i + 1) / (NP + 1);
      free_points.col(i) = headPVA.col(0) + alpha * (tailPVA.col(0) - headPVA.col(0));
      for (int j = 0; j < 3; j++) {
        free_points(j, i) += 0.001 * (static_cast<double>(rand()) / RAND_MAX - 0.5);
      }
      points.col(i) = free_points.col(i);
    }
    for (int i = 0; i < pieceN; i++) {
      times(i) = std::max(totalDist / (NP + 1) / initSpeed, minPieceTime);
    }
  } else {
    const int N = static_cast<int>(waypoints.cols());
    free_points.resize(3, N + 1);

    free_points.col(0) = (waypoints.col(0) + headPVA.col(0)) / 2.0;

    for (int i = 0; i < N - 1; i++) {
      free_points.col(i + 1) = (waypoints.col(i) + waypoints.col(i + 1)) / 2.0;
    }

    free_points.rightCols(1) = (waypoints.rightCols(1) + tailPVA.col(0)) / 2.0;

    for (int i = 0; i < free_points.cols(); i++) {
      for (int j = 0; j < 3; j++) {
        free_points(j, i) += 0.001 * (static_cast<double>(rand()) / RAND_MAX - 0.5);
      }
    }

    for (int i = 0; i < N; i++) {
      points.col(2 * i) = free_points.col(i);
      points.col(2 * i + 1) = waypoints.col(i);
    }
    points.rightCols(1) = free_points.rightCols(1);

    Eigen::Vector3d lastP = headPVA.col(0);
    for (int i = 0; i < N; i++) {
      Eigen::Vector3d curP = points.col(2 * i);
      times(2 * i) = std::max((curP - lastP).norm() / initSpeed, minPieceTime);
      lastP = curP;
      curP = points.col(2 * i + 1);
      times(2 * i + 1) = std::max((curP - lastP).norm() / initSpeed, minPieceTime);
      lastP = curP;
    }
    times(2 * N) = std::max((points.rightCols(1) - lastP).norm() / initSpeed, minPieceTime);
    times(2 * N + 1) = std::max((tailPVA.col(0) - points.rightCols(1)).norm() / initSpeed, minPieceTime);
  }
}

// ============================================================
//  attachPenaltyFunctional  (vel / omg / theta / thrust + SFC pos)
//  VLA-style: uses FlatnessMap to enforce physical constraints.
// ============================================================

void GcopterWayptS3::attachPenaltyFunctional(const Eigen::VectorXd &T, const Eigen::MatrixX3d &coeffs,
                                             double smoothFactor, int integralResolution,
                                             const Eigen::VectorXd &magnitudeBounds,
                                             const Eigen::VectorXd &penaltyWeights, flatness::FlatnessMap &flatMap,
                                             bool hasSfc, double weightPos,
                                             const std::vector<Eigen::MatrixX4d> &hPolys, const std::vector<int> &hIdx,
                                             double &cost, Eigen::VectorXd &gradT, Eigen::MatrixX3d &gradC) {
  const double vmax = magnitudeBounds(0);
  const double omgMax = magnitudeBounds(1);
  const double thetaMax = magnitudeBounds(2);
  const double thrustMin = magnitudeBounds(3);
  const double thrustMax = magnitudeBounds(4);

  const double vmaxSqr = vmax * vmax;
  const double omgSqrMax = omgMax * omgMax;
  const double thrustMean = 0.5 * (thrustMin + thrustMax);
  const double thrustRadi = 0.5 * fabs(thrustMax - thrustMin);
  const double thrustSqrRadi = thrustRadi * thrustRadi;

  const double weightVel = penaltyWeights(0);
  const double weightOmg = penaltyWeights(1);
  const double weightTheta = penaltyWeights(2);
  const double weightThrust = penaltyWeights(3);

  Eigen::Vector3d pos, vel, acc, jer, sna;
  Eigen::Vector3d totalGradPos, totalGradVel, totalGradAcc, totalGradJer;
  double totalGradPsi, totalGradPsiD;
  double thr, cos_theta;
  Eigen::Vector4d quat;
  Eigen::Vector3d omg;
  double gradThr;
  Eigen::Vector4d gradQuat;
  Eigen::Vector3d gradPos, gradVel, gradAcc, gradJer, gradOmg;

  double step, alpha;
  double s1, s2, s3, s4, s5;
  Eigen::Matrix<double, 6, 1> beta0, beta1, beta2, beta3, beta4;
  double violaPos, violaPosPena, violaPosPenaD;
  double violaVel, violaVelPena, violaVelPenaD;
  double violaOmg, violaOmgPena, violaOmgPenaD;
  double violaTheta, violaThetaPena, violaThetaPenaD;
  double violaThrust, violaThrustPena, violaThrustPenaD;
  Eigen::Vector3d outerNormal;
  double node, pena;

  const int pieceNum = static_cast<int>(T.size());
  const double integralFrac = 1.0 / integralResolution;

  for (int i = 0; i < pieceNum; i++) {
    const Eigen::Matrix<double, 6, 3> &c = coeffs.block<6, 3>(i * 6, 0);
    step = T(i) * integralFrac;
    for (int j = 0; j <= integralResolution; j++) {
      s1 = j * step;
      s2 = s1 * s1;
      s3 = s2 * s1;
      s4 = s2 * s2;
      s5 = s4 * s1;
      beta0(0) = 1.0, beta0(1) = s1, beta0(2) = s2, beta0(3) = s3, beta0(4) = s4, beta0(5) = s5;
      beta1(0) = 0.0, beta1(1) = 1.0, beta1(2) = 2.0 * s1, beta1(3) = 3.0 * s2, beta1(4) = 4.0 * s3, beta1(5) = 5.0 * s4;
      beta2(0) = 0.0, beta2(1) = 0.0, beta2(2) = 2.0, beta2(3) = 6.0 * s1, beta2(4) = 12.0 * s2, beta2(5) = 20.0 * s3;
      beta3(0) = 0.0, beta3(1) = 0.0, beta3(2) = 0.0, beta3(3) = 6.0, beta3(4) = 24.0 * s1, beta3(5) = 60.0 * s2;
      beta4(0) = 0.0, beta4(1) = 0.0, beta4(2) = 0.0, beta4(3) = 0.0, beta4(4) = 24.0, beta4(5) = 120.0 * s1;
      pos = c.transpose() * beta0;
      vel = c.transpose() * beta1;
      acc = c.transpose() * beta2;
      jer = c.transpose() * beta3;
      sna = c.transpose() * beta4;

      // Differential flatness: (vel, acc, jer) → (thrust, quat, omg)
      flatMap.forward(vel, acc, jer, 0.0, 0.0, thr, quat, omg);

      // Compute violations
      violaVel = vel.squaredNorm() - vmaxSqr;
      violaOmg = omg.squaredNorm() - omgSqrMax;
      cos_theta = 1.0 - 2.0 * (quat(1) * quat(1) + quat(2) * quat(2));
      violaTheta = acos(cos_theta) - thetaMax;
      violaThrust = (thr - thrustMean) * (thr - thrustMean) - thrustSqrRadi;

      // Accumulate penalty and gradients
      gradThr = 0.0;
      gradQuat.setZero();
      gradPos.setZero(), gradVel.setZero(), gradAcc.setZero();
      gradJer.setZero(), gradOmg.setZero();
      pena = 0.0;

      // ---- SFC polytope position constraint ----
      if (hasSfc) {
        int L = hIdx[i];
        int K = hPolys[L].rows();
        for (int k = 0; k < K; k++) {
          outerNormal = hPolys[L].block<1, 3>(k, 0);
          violaPos = outerNormal.dot(pos) + hPolys[L](k, 3);
          if (smoothedL1(violaPos, smoothFactor, violaPosPena, violaPosPenaD)) {
            gradPos += weightPos * violaPosPenaD * outerNormal;
            pena += weightPos * violaPosPena;
          }
        }
      }

      // ---- Velocity ----
      if (weightVel > 0.0 && smoothedL1(violaVel, smoothFactor, violaVelPena, violaVelPenaD)) {
        gradVel += weightVel * violaVelPenaD * 2.0 * vel;
        pena += weightVel * violaVelPena;
      }

      // ---- Body rate (omg) ----
      if (weightOmg > 0.0 && smoothedL1(violaOmg, smoothFactor, violaOmgPena, violaOmgPenaD)) {
        gradOmg += weightOmg * violaOmgPenaD * 2.0 * omg;
        pena += weightOmg * violaOmgPena;
      }

      // ---- Tilt angle (theta) ----
      if (weightTheta > 0.0 && smoothedL1(violaTheta, smoothFactor, violaThetaPena, violaThetaPenaD)) {
        gradQuat += weightTheta * violaThetaPenaD / sqrt(1.0 - cos_theta * cos_theta) * 4.0 *
                    Eigen::Vector4d(0.0, quat(1), quat(2), 0.0);
        pena += weightTheta * violaThetaPena;
      }

      // ---- Thrust ----
      if (weightThrust > 0.0 && smoothedL1(violaThrust, smoothFactor, violaThrustPena, violaThrustPenaD)) {
        gradThr += weightThrust * violaThrustPenaD * 2.0 * (thr - thrustMean);
        pena += weightThrust * violaThrustPena;
      }

      // Propogate flatness gradients back to derivative chain
      flatMap.backward(gradPos, gradVel, gradThr, gradQuat, gradOmg, totalGradPos, totalGradVel, totalGradAcc,
                       totalGradJer, totalGradPsi, totalGradPsiD);
      totalGradAcc += gradAcc;
      totalGradJer += gradJer;

      node = (j == 0 || j == integralResolution) ? 0.5 : 1.0;
      alpha = j * integralFrac;
      gradC.block<6, 3>(i * 6, 0) +=
          (beta0 * totalGradPos.transpose() + beta1 * totalGradVel.transpose() + beta2 * totalGradAcc.transpose() +
           beta3 * totalGradJer.transpose()) *
          node * step;
      gradT(i) += (totalGradPos.dot(vel) + totalGradVel.dot(acc) + totalGradAcc.dot(jer) + totalGradJer.dot(sna)) *
                      alpha * node * step +
                  node * integralFrac * pena;
      cost += node * step * pena;
    }
  }
}

// ============================================================
//  costFunctional  (L-BFGS callback)
// ============================================================

double GcopterWayptS3::costFunctional(void *ptr, const Eigen::VectorXd &x, Eigen::VectorXd &g) {
  GcopterWayptS3 &obj = *reinterpret_cast<GcopterWayptS3 *>(ptr);
  const int dimTau = obj.temporalDim;
  const int dimXi = obj.spatialDim;
  const double weightT = obj.rho;

  Eigen::Map<const Eigen::VectorXd> tau(x.data(), dimTau);
  Eigen::Map<const Eigen::VectorXd> xi(x.data() + dimTau, dimXi);
  Eigen::Map<Eigen::VectorXd> gradTau(g.data(), dimTau);
  Eigen::Map<Eigen::VectorXd> gradXi(g.data() + dimTau, dimXi);

  obj.iter_num++;

  forwardT(tau, obj.times);

  const bool hasWpts = (obj.waypoints.cols() > 0);
  for (int i = 0; i < obj.free_points.cols(); i++) {
    obj.free_points.col(i) = xi.segment<3>(3 * i);
    if (hasWpts) {
      obj.points.col(i * 2) = obj.free_points.col(i);  // even indices only
    } else {
      obj.points.col(i) = obj.free_points.col(i);  // 1:1
    }
  }

  double cost = 0.0;
  obj.minco.setParameters(obj.points, obj.times);
  obj.minco.getEnergy(cost);
  const double energyCost = cost;  // snapshot before penalties
  obj.partialGradByCoeffs.setZero();
  obj.partialGradByTimes.setZero();
  obj.minco.getEnergyPartialGradByCoeffs(obj.partialGradByCoeffs);
  obj.minco.getEnergyPartialGradByTimes(obj.partialGradByTimes);

  attachPenaltyFunctional(obj.times, obj.minco.getCoeffs(), obj.smoothEps, obj.integralRes, obj.magnitudeBd,
                          obj.penaltyWt, obj.flatMap, obj.hasSfc_, obj.weightPos_, obj.hPolys_, obj.hIdx_, cost,
                          obj.partialGradByTimes, obj.partialGradByCoeffs);
  const double penaltyCost = cost - energyCost;

  obj.minco.propogateGrad(obj.partialGradByCoeffs, obj.partialGradByTimes, obj.gradByPoints, obj.gradByTimes);

  const double timeCost = weightT * obj.times.sum();
  cost += timeCost;
  obj.gradByTimes.array() += weightT;

  backwardGradT(tau, obj.gradByTimes, gradTau);

  if (hasWpts) {
    for (int i = 0, idx = 0; i < obj.gradByPoints.cols(); i++) {
      if (i % 2 == 0) {
        gradXi.segment<3>(3 * idx) = obj.gradByPoints.col(i);
        idx++;
      }
    }
  } else {
    for (int i = 0; i < obj.gradByPoints.cols(); i++) {
      gradXi.segment<3>(3 * i) = obj.gradByPoints.col(i);
    }
  }

  // --- diagnostic: first call ---
  if (obj.iter_num == 1) {
    double gTau_norm = gradTau.norm();
    double gXi_norm = gradXi.norm();
    ROS_INFO("  [Opt] iter=%d  cost=%.2f (energy=%.2f, penalty=%.2f, time=%.2f)  |gTau|=%.1f  |gXi|=%.1f",
             obj.iter_num, cost, energyCost, penaltyCost, timeCost, gTau_norm, gXi_norm);

    // --- SFC junction diagnostic ---
    if (obj.hasSfc_) {
      const int P = static_cast<int>(obj.hPolys_.size());
      int n_violated = 0;
      double max_viol = 0.0;
      double sum_viol = 0.0;
      int worst_junc = -1, worst_poly = -1;

      // Build junction list: [head, points(all cols), tail]
      std::vector<Eigen::Vector3d> juncs;
      juncs.push_back(obj.headPVA.col(0));
      for (int c = 0; c < obj.points.cols(); c++) {
        juncs.push_back(obj.points.col(c));
      }
      juncs.push_back(obj.tailPVA.col(0));

      for (int k = 0; k < obj.pieceN; k++) {
        int L = obj.hIdx_[k];
        const Eigen::MatrixX4d &H = obj.hPolys_[L];
        const Eigen::Vector3d &pt = juncs[k];  // start junction of piece k
        double viol_max_k = 0.0;
        for (int r = 0; r < H.rows(); r++) {
          double d = H.row(r).head<3>().dot(pt) + H(r, 3);
          if (d > viol_max_k) viol_max_k = d;
        }
        if (viol_max_k > 0.0) {
          n_violated++;
          sum_viol += viol_max_k;
          if (viol_max_k > max_viol) {
            max_viol = viol_max_k;
            worst_junc = k;
            worst_poly = L;
          }
        }
      }

      if (n_violated > 0) {
        ROS_WARN("  [SFC diag] %d/%d junctions violate their polytope | sum=%.3f max=%.3f (junc=%d poly=%d)",
                 n_violated, obj.pieceN, sum_viol, max_viol, worst_junc, worst_poly);
        const Eigen::Vector3d &wpt = juncs[worst_junc];
        const Eigen::MatrixX4d &Hw = obj.hPolys_[worst_poly];
        ROS_WARN("  [SFC diag] worst junc[%d] pos=(%.2f,%.2f,%.2f) poly[%d] faces=%d", worst_junc, wpt(0), wpt(1),
                 wpt(2), worst_poly, static_cast<int>(Hw.rows()));
        for (int r = 0; r < Hw.rows(); r++) {
          double d = Hw.row(r).head<3>().dot(wpt) + Hw(r, 3);
          if (d > 0.0) {
            ROS_WARN("  [SFC diag]   face[%d]: n=(%.3f,%.3f,%.3f) b=%.3f  viol=%.4f", r, Hw(r, 0), Hw(r, 1), Hw(r, 2),
                     Hw(r, 3), d);
          }
        }
      } else {
        ROS_INFO("  [SFC diag] all %d junctions inside their polytopes ✓", obj.pieceN);
      }
    }
  }

  return cost;
}

// ============================================================
//  optimize
// ============================================================

double GcopterWayptS3::optimize(Trajectory<5> &traj, double relCostTol) {
  Eigen::VectorXd x(temporalDim + spatialDim);
  Eigen::Map<Eigen::VectorXd> tau_map(x.data(), temporalDim);
  Eigen::Map<Eigen::VectorXd> xi_map(x.data() + temporalDim, spatialDim);

  setInitial();
  for (int i = 0; i < free_points.cols(); i++) {
    xi_map.segment<3>(3 * i) = free_points.col(i);
  }
  backwardT(times, tau_map);

  iter_num = 0;
  double minCost;

  lbfgs_params.mem_size = 256;
  lbfgs_params.past = 3;
  lbfgs_params.min_step = 1.0e-12;
  lbfgs_params.g_epsilon = 1.0e-8;
  lbfgs_params.delta = relCostTol;

  int ret = lbfgs::lbfgs_optimize(x, minCost, &GcopterWayptS3::costFunctional, nullptr, nullptr, this, lbfgs_params);

  if (ret >= 0) {
    forwardT(tau_map, times);
    minco.setConditions(headPVA, tailPVA, temporalDim);
    minco.setParameters(points, times);
    minco.getTrajectory(traj);
  } else {
    traj.clear();
    minCost = INFINITY;
    std::cout << " -- [WayptS3] optimization failed: " << lbfgs::lbfgs_strerror(ret) << std::endl;
  }

  return minCost;
}

}  // namespace gcopter
