/*
    MIT License

    Copyright (c) 2021 Zhepei Wang (wangzhepei@live.com)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

#ifndef SFC_GEN_HPP
#define SFC_GEN_HPP

#include <ompl/base/DiscreteMotionValidator.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/planners/rrt/InformedRRTstar.h>
#include <ompl/util/Console.h>

#include <Eigen/Eigen>
#include <deque>
#include <memory>

#include "firi.hpp"
#include "geo_utils.hpp"

namespace sfc_gen {

template <typename Map>
inline double planPath(const Eigen::Vector3d &s, const Eigen::Vector3d &g, const Eigen::Vector3d &lb,
                       const Eigen::Vector3d &hb, const Map *mapPtr, const double &timeout,
                       std::vector<Eigen::Vector3d> &p) {
  auto space(std::make_shared<ompl::base::RealVectorStateSpace>(3));

  ompl::base::RealVectorBounds bounds(3);
  bounds.setLow(0, 0.0);
  bounds.setHigh(0, hb(0) - lb(0));
  bounds.setLow(1, 0.0);
  bounds.setHigh(1, hb(1) - lb(1));
  bounds.setLow(2, 0.0);
  bounds.setHigh(2, hb(2) - lb(2));
  space->setBounds(bounds);

  auto si(std::make_shared<ompl::base::SpaceInformation>(space));

  si->setStateValidityChecker([&](const ompl::base::State *state) {
    const auto *pos = state->as<ompl::base::RealVectorStateSpace::StateType>();
    const Eigen::Vector3d position(lb(0) + (*pos)[0], lb(1) + (*pos)[1], lb(2) + (*pos)[2]);
    return mapPtr->query(position) == 0;
  });
  si->setup();

  ompl::msg::setLogLevel(ompl::msg::LOG_NONE);

  ompl::base::ScopedState<> start(space), goal(space);
  start[0] = s(0) - lb(0);
  start[1] = s(1) - lb(1);
  start[2] = s(2) - lb(2);
  goal[0] = g(0) - lb(0);
  goal[1] = g(1) - lb(1);
  goal[2] = g(2) - lb(2);

  auto pdef(std::make_shared<ompl::base::ProblemDefinition>(si));
  pdef->setStartAndGoalStates(start, goal);
  pdef->setOptimizationObjective(std::make_shared<ompl::base::PathLengthOptimizationObjective>(si));
  auto planner(std::make_shared<ompl::geometric::InformedRRTstar>(si));
  planner->setProblemDefinition(pdef);
  planner->setup();

  ompl::base::PlannerStatus solved;
  solved = planner->ompl::base::Planner::solve(timeout);

  double cost = INFINITY;
  if (solved) {
    p.clear();
    const ompl::geometric::PathGeometric path_ =
        ompl::geometric::PathGeometric(dynamic_cast<const ompl::geometric::PathGeometric &>(*pdef->getSolutionPath()));
    for (size_t i = 0; i < path_.getStateCount(); i++) {
      const auto state = path_.getState(i)->as<ompl::base::RealVectorStateSpace::StateType>()->values;
      p.emplace_back(lb(0) + state[0], lb(1) + state[1], lb(2) + state[2]);
    }
    cost = pdef->getSolutionPath()->cost(pdef->getOptimizationObjective()).value();
  }

  return cost;
}

// A* grid search — deterministic, shortest path, zero external dependencies.
// Works with any map type that has query(pos) → uint8_t (0 = free).
template <typename Map>
inline double planPathAStar(const Eigen::Vector3d &s, const Eigen::Vector3d &g, const Eigen::Vector3d &lb,
                             const Eigen::Vector3d &hb, const Map *mapPtr, double /*unused*/,
                             std::vector<Eigen::Vector3d> &p) {
  p.clear();
  const double res = 0.1;  // grid resolution

  const int nx = static_cast<int>(std::ceil((hb(0) - lb(0)) / res));
  const int ny = static_cast<int>(std::ceil((hb(1) - lb(1)) / res));
  const int nz = static_cast<int>(std::ceil((hb(2) - lb(2)) / res));
  const int stride_y = nx;
  const int stride_z = nx * ny;

  auto posToId = [&](const Eigen::Vector3d &pos) -> int {
    int x = static_cast<int>((pos(0) - lb(0)) / res);
    int y = static_cast<int>((pos(1) - lb(1)) / res);
    int z = static_cast<int>((pos(2) - lb(2)) / res);
    x = std::max(0, std::min(x, nx - 1));
    y = std::max(0, std::min(y, ny - 1));
    z = std::max(0, std::min(z, nz - 1));
    return x + y * stride_y + z * stride_z;
  };

  auto idToPos = [&](int id) -> Eigen::Vector3d {
    int z = id / stride_z;
    int r = id % stride_z;
    int y = r / stride_y;
    int x = r % stride_y;
    return Eigen::Vector3d(lb(0) + (x + 0.5) * res, lb(1) + (y + 0.5) * res, lb(2) + (z + 0.5) * res);
  };

  auto heuristic = [&](int a, int b) -> double {
    return (idToPos(a) - idToPos(b)).norm();
  };

  const int N = nx * ny * nz;
  const double INF = 1e20;

  // Per-node data: gScore, parent, closed flag
  std::vector<double> gScore(N, INF);
  std::vector<int> parent(N, -1);
  std::vector<bool> closed(N, false);

  int startId = posToId(s);
  int goalId = posToId(g);

  gScore[startId] = 0.0;

  using PQItem = std::pair<double, int>;  // (fScore, id)
  std::priority_queue<PQItem, std::vector<PQItem>, std::greater<PQItem>> open;
  open.push({heuristic(startId, goalId), startId});

  // 6-connected neighbors
  const int neighbors[6] = {1, -1, stride_y, -stride_y, stride_z, -stride_z};

  bool found = false;
  while (!open.empty()) {
    auto [f, cur] = open.top();
    open.pop();

    if (closed[cur]) continue;
    closed[cur] = true;

    if (cur == goalId) {
      found = true;
      break;
    }

    Eigen::Vector3d curPos = idToPos(cur);
    for (int k = 0; k < 6; k++) {
      int nb = cur + neighbors[k];
      if (nb < 0 || nb >= N) continue;

      // Check axis bounds
      int dz = nb / stride_z;
      int dr = nb % stride_z;
      int dy = dr / stride_y;
      int dx = dr % stride_y;
      if (dx < 0 || dx >= nx || dy < 0 || dy >= ny || dz < 0 || dz >= nz) continue;

      if (closed[nb]) continue;

      Eigen::Vector3d nbPos = idToPos(nb);
      if (mapPtr->query(nbPos) != 0) continue;  // occupied

      double edgeCost = (nbPos - curPos).norm();
      double newG = gScore[cur] + edgeCost;
      if (newG < gScore[nb]) {
        gScore[nb] = newG;
        parent[nb] = cur;
        open.push({newG + heuristic(nb, goalId), nb});
      }
    }
  }

  if (!found) return INFINITY;

  // Reconstruct path
  std::vector<int> idPath;
  for (int at = goalId; at != -1; at = parent[at]) {
    idPath.push_back(at);
  }
  std::reverse(idPath.begin(), idPath.end());

  for (int id : idPath) {
    p.push_back(idToPos(id));
  }
  return 0.0;
}

inline void convexCover(const std::vector<Eigen::Vector3d> &path, const std::vector<Eigen::Vector3d> &points,
                        const Eigen::Vector3d &lowCorner, const Eigen::Vector3d &highCorner, const double &progress,
                        const double &range, std::vector<Eigen::MatrixX4d> &hpolys, const double eps = 1.0e-6) {
  hpolys.clear();
  const int n = path.size();
  Eigen::Matrix<double, 6, 4> bd = Eigen::Matrix<double, 6, 4>::Zero();
  bd(0, 0) = 1.0;
  bd(1, 0) = -1.0;
  bd(2, 1) = 1.0;
  bd(3, 1) = -1.0;
  bd(4, 2) = 1.0;
  bd(5, 2) = -1.0;

  Eigen::MatrixX4d hp, gap;
  Eigen::Vector3d a, b = path[0];
  std::vector<Eigen::Vector3d> valid_pc;
  std::vector<Eigen::Vector3d> bs;
  valid_pc.reserve(points.size());
  for (int i = 1; i < n;) {
    a = b;
    if ((a - path[i]).norm() > progress) {
      b = (path[i] - a).normalized() * progress + a;
    } else {
      b = path[i];
      i++;
    }
    bs.emplace_back(b);

    bd(0, 3) = -std::min(std::max(a(0), b(0)) + range, highCorner(0));
    bd(1, 3) = +std::max(std::min(a(0), b(0)) - range, lowCorner(0));
    bd(2, 3) = -std::min(std::max(a(1), b(1)) + range, highCorner(1));
    bd(3, 3) = +std::max(std::min(a(1), b(1)) - range, lowCorner(1));
    bd(4, 3) = -std::min(std::max(a(2), b(2)) + range, highCorner(2));
    bd(5, 3) = +std::max(std::min(a(2), b(2)) - range, lowCorner(2));

    valid_pc.clear();
    for (const Eigen::Vector3d &p : points) {
      if ((bd.leftCols<3>() * p + bd.rightCols<1>()).maxCoeff() < 0.0) {
        valid_pc.emplace_back(p);
      }
    }
    Eigen::Map<const Eigen::Matrix<double, 3, -1, Eigen::ColMajor>> pc(valid_pc[0].data(), 3, valid_pc.size());

    firi::firi(bd, pc, a, b, hp);

    if (hpolys.size() != 0) {
      const Eigen::Vector4d ah(a(0), a(1), a(2), 1.0);
      if (3 <= ((hp * ah).array() > -eps).cast<int>().sum() + ((hpolys.back() * ah).array() > -eps).cast<int>().sum()) {
        firi::firi(bd, pc, a, a, gap, 1);
        hpolys.emplace_back(gap);
      }
    }

    hpolys.emplace_back(hp);
  }
}

inline void shortCut(std::vector<Eigen::MatrixX4d> &hpolys) {
  std::vector<Eigen::MatrixX4d> htemp = hpolys;
  if (htemp.size() == 1) {
    Eigen::MatrixX4d headPoly = htemp.front();
    htemp.insert(htemp.begin(), headPoly);
  }
  hpolys.clear();

  int M = htemp.size();
  Eigen::MatrixX4d hPoly;
  bool overlap;
  std::deque<int> idices;
  idices.push_front(M - 1);
  for (int i = M - 1; i >= 0; i--) {
    for (int j = 0; j < i; j++) {
      if (j < i - 1) {
        overlap = geo_utils::overlap(htemp[i], htemp[j], 0.01);
      } else {
        overlap = true;
      }
      if (overlap) {
        idices.push_front(j);
        i = j + 1;
        break;
      }
    }
  }
  for (const auto &ele : idices) {
    hpolys.push_back(htemp[ele]);
  }
}

}  // namespace sfc_gen

#endif
