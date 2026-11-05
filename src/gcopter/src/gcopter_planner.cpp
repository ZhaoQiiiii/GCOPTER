#include "gcopter_planner.h"

namespace gcopter {

bool GcopterPlanner::loadOccupancyGrid(const std::string &occ_file) {
  std::ifstream f(occ_file, std::ios::binary);
  if (!f.is_open()) {
    ROS_ERROR("Failed to open %s", occ_file.c_str());
    return false;
  }

  int32_t dims_raw[3];
  double origin_raw[3];
  double grid_resolution;
  f.read(reinterpret_cast<char *>(dims_raw), 12);
  f.read(reinterpret_cast<char *>(origin_raw), 24);
  f.read(reinterpret_cast<char *>(&grid_resolution), 8);

  Eigen::Vector3i dims(dims_raw[0], dims_raw[1], dims_raw[2]);
  Eigen::Vector3d grid_origin(origin_raw[0], origin_raw[1], origin_raw[2]);

  if (f.fail() || dims(0) <= 0 || dims(1) <= 0 || dims(2) <= 0) {
    ROS_ERROR("Invalid .occ header in %s", occ_file.c_str());
    return false;
  }

  int total_voxels = dims(0) * dims(1) * dims(2);
  std::vector<uint8_t> occ_data(total_voxels);
  f.read(reinterpret_cast<char *>(occ_data.data()), total_voxels);

  if (f.gcount() < total_voxels) {
    ROS_ERROR("Grid file too short: expected %d bytes, got %ld", total_voxels, f.gcount());
    return false;
  }

  ROS_INFO("Loaded occupancy grid: %dx%dx%d, res=%.2fm, origin=(%.2f,%.2f,%.2f)", dims(0), dims(1), dims(2),
           grid_resolution, grid_origin(0), grid_origin(1), grid_origin(2));

  constexpr int margin_voxels = 2;
  Eigen::Vector3d map_origin = grid_origin - Eigen::Vector3d::Constant(margin_voxels * grid_resolution);
  Eigen::Vector3i map_size = dims + Eigen::Vector3i::Constant(2 * margin_voxels);

  voxel_map_ = voxel_map::VoxelMap(map_size, map_origin, grid_resolution);

  int ny_nz = dims(1) * dims(2);
  int nz = dims(2);
  int occupied_count = 0;
  std::vector<float> vis_data;
  constexpr double vis_max_z = 2.0;

  for (int x = 0; x < dims(0); x++) {
    for (int y = 0; y < dims(1); y++) {
      for (int z = 0; z < dims(2); z++) {
        if (occ_data[x * ny_nz + y * nz + z]) {
          Eigen::Vector3d pos = grid_origin + Eigen::Vector3d(x + 0.5, y + 0.5, z + 0.5) * grid_resolution;
          voxel_map_.setOccupied(pos);
          occupied_count++;

          if (pos(2) < vis_max_z) {
            vis_data.push_back(static_cast<float>(pos(0)));
            vis_data.push_back(static_cast<float>(pos(1)));
            vis_data.push_back(static_cast<float>(pos(2)));
          }
        }
      }
    }
  }

  voxel_map_.dilate(std::ceil(config_.inflateRadius / grid_resolution));

  map_initialized_ = true;
  ROS_INFO("VoxelMap ready: %dx%dx%d, origin=(%.2f,%.2f,%.2f), %d occupied voxels", map_size(0), map_size(1),
           map_size(2), map_origin(0), map_origin(1), map_origin(2), occupied_count);

  std::vector<float> vis_data_inf;
  {
    const Eigen::Vector3i ms = voxel_map_.getSize();
    const Eigen::Vector3d morigin = voxel_map_.getOrigin();
    const double mscale = voxel_map_.getScale();
    const std::vector<uint8_t> &vox = voxel_map_.getVoxels();
    const int sy = ms(0);
    const int sz = ms(0) * ms(1);
    for (int z = 0; z < ms(2); z++) {
      const double wz = morigin(2) + (z + 0.5) * mscale;
      if (wz >= vis_max_z) continue;
      for (int y = 0; y < ms(1); y++) {
        for (int x = 0; x < ms(0); x++) {
          if (vox[x + y * sy + z * sz] != voxel_map::Unoccupied) {
            vis_data_inf.push_back(static_cast<float>(morigin(0) + (x + 0.5) * mscale));
            vis_data_inf.push_back(static_cast<float>(morigin(1) + (y + 0.5) * mscale));
            vis_data_inf.push_back(static_cast<float>(wz));
          }
        }
      }
    }
  }

  publishVoxelVis(grid_vis_pub_, vis_data, "/gcopter/voxel_map_vis", vis_max_z);
  publishVoxelVis(grid_inflate_vis_pub_, vis_data_inf, "/gcopter/voxel_map_inflated_vis", vis_max_z);

  return true;
}

void GcopterPlanner::publishVoxelVis(ros::Publisher &pub, const std::vector<float> &vis_data, const char *topic_name,
                                     double vis_max_z) {
  sensor_msgs::PointCloud2 vis_msg;
  vis_msg.header.frame_id = "world";
  vis_msg.header.stamp = ros::Time::now();
  vis_msg.height = 1;
  vis_msg.is_bigendian = false;
  vis_msg.is_dense = true;
  vis_msg.point_step = 12;
  vis_msg.width = vis_data.size() / 3;
  vis_msg.row_step = vis_msg.point_step * vis_msg.width;

  vis_msg.fields.resize(3);
  vis_msg.fields[0].name = "x";
  vis_msg.fields[0].offset = 0;
  vis_msg.fields[0].datatype = sensor_msgs::PointField::FLOAT32;
  vis_msg.fields[0].count = 1;
  vis_msg.fields[1].name = "y";
  vis_msg.fields[1].offset = 4;
  vis_msg.fields[1].datatype = sensor_msgs::PointField::FLOAT32;
  vis_msg.fields[1].count = 1;
  vis_msg.fields[2].name = "z";
  vis_msg.fields[2].offset = 8;
  vis_msg.fields[2].datatype = sensor_msgs::PointField::FLOAT32;
  vis_msg.fields[2].count = 1;

  vis_msg.data.resize(vis_data.size() * sizeof(float));
  std::memcpy(vis_msg.data.data(), vis_data.data(), vis_msg.data.size());
  pub.publish(vis_msg);
  ROS_INFO("Published %zu points to %s (z < %.1fm)", vis_data.size() / 3, topic_name, vis_max_z);
}

// ============================================================
//  Shared SFC builder: subsample route → convexCover → shortCut
// ============================================================
namespace {
// target_polys < 0 → treat `route` as the exact junction sequence (no subsampling),
// yielding route.size()-1 polytopes aligned 1:1 with trajectory pieces.
void buildSFC(const std::vector<Eigen::Vector3d> &route, int target_polys, const voxel_map::VoxelMap &voxel_map,
              bool use_shortcut, std::vector<Eigen::MatrixX4d> &hPolys) {
  std::vector<Eigen::Vector3d> surfPts;
  voxel_map.getSurf(surfPts);

  double route_len = 0.0;
  for (size_t i = 1; i < route.size(); i++) {
    route_len += (route[i] - route[i - 1]).norm();
  }

  std::vector<Eigen::Vector3d> route_sparse;
  if (target_polys < 0) {
    route_sparse = route;  // exact junction sequence
  } else {
    int sample_n = target_polys + 1;
    if (static_cast<int>(route.size()) <= sample_n) {
      route_sparse = route;
    } else {
      route_sparse.reserve(sample_n);
      for (int k = 0; k < sample_n; k++) {
        int idx = k * (static_cast<int>(route.size()) - 1) / (sample_n - 1);
        route_sparse.push_back(route[idx]);
      }
    }
  }

  // progress = largest gap in route_sparse → convexCover never splits a segment,
  // so it emits exactly route_sparse.size()-1 polytopes (1:1 with trajectory pieces).
  double max_gap = 0.0;
  for (size_t i = 1; i < route_sparse.size(); i++) {
    max_gap = std::max(max_gap, (route_sparse[i] - route_sparse[i - 1]).norm());
  }
  double progress = max_gap * 1.01 + 1.0e-6;
  double range = std::max(1.5, std::min(progress * 0.8, 3.0));

  ROS_INFO("  SFC: route_len=%.1fm, target=%d, %zu→%zu pts, progress=%.2f, range=%.2f", route_len, target_polys,
           route.size(), route_sparse.size(), progress, range);

  sfc_gen::convexCover(route_sparse, surfPts, voxel_map.getOrigin(), voxel_map.getCorner(), progress, range, hPolys);
  if (use_shortcut) {
    sfc_gen::shortCut(hPolys);
  }
}
}  // namespace

bool GcopterPlanner::planOnce(const Eigen::Vector3d &start, const Eigen::Vector3d &goal) {
  // --- A* path search ---
  std::vector<Eigen::Vector3d> route;
  sfc_gen::planPathAStar<voxel_map::VoxelMap>(start, goal, voxel_map_.getOrigin(), voxel_map_.getCorner(), &voxel_map_,
                                              0.0, route);

  if (route.size() < 2) {
    ROS_WARN("A* failed to find a path");
    return false;
  }

  // --- SFC ---
  double route_len = 0.0;
  for (size_t i = 1; i < route.size(); i++) route_len += (route[i] - route[i - 1]).norm();
  int target_polys = std::max(4, static_cast<int>(route_len / 1.0));
  std::vector<Eigen::MatrixX4d> hPolys;
  buildSFC(route, target_polys, voxel_map_, false, hPolys);  // no shortCut → more pieces = more DOFs

  if (hPolys.empty()) {
    ROS_WARN("SFC generation failed");
    return false;
  }

  // --- Setup (no waypoints, SFC active) ---
  Eigen::Matrix3d iniState, finState;
  iniState << start, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();
  finState << goal, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();

  Eigen::VectorXd magnitudeBounds(5);
  magnitudeBounds(0) = config_.maxVelMag;      // v_max
  magnitudeBounds(1) = config_.maxBdrMag;      // omg_max
  magnitudeBounds(2) = config_.maxTiltAngle;   // theta_max
  magnitudeBounds(3) = config_.minThrust;      // thrust_min
  magnitudeBounds(4) = config_.maxThrust;      // thrust_max

  Eigen::VectorXd penaltyWeights(4);
  penaltyWeights(0) = config_.chiVec[1];        // vel
  penaltyWeights(1) = config_.chiVec[2];        // omg
  penaltyWeights(2) = config_.chiVec[3] * 0.1;  // theta (scaled)
  penaltyWeights(3) = config_.chiVec[3];        // thrust

  Eigen::VectorXd physicalParams(6);
  physicalParams(0) = config_.vehicleMass;
  physicalParams(1) = config_.gravAcc;
  physicalParams(2) = config_.horizDrag;
  physicalParams(3) = config_.vertDrag;
  physicalParams(4) = config_.parasDrag;
  physicalParams(5) = config_.speedEps;

  const double weightPos = config_.chiVec[0];

  traj_.clear();
  yaw_traj_.clear();

  Eigen::Matrix3Xd emptyWpt;
  if (!waypt_optimizer_.setup(config_.weightT, iniState, finState, emptyWpt, config_.smoothingEps,
                              config_.integralIntervs, magnitudeBounds, penaltyWeights, physicalParams,
                              hPolys, weightPos)) {
    ROS_WARN("planOnce: setup failed");
    return false;
  }

  waypt_optimizer_.initSpeed = config_.initSpeed;
  if (std::isinf(waypt_optimizer_.optimize(traj_, config_.relCostTol))) {
    ROS_WARN("planOnce: optimization failed");
    traj_.clear();
    return false;
  }

  // --- Yaw: LOOK_TARGET → linear interpolation to target ---
  buildYaw(traj_, YawMode::LOOK_TARGET);

  ROS_INFO("planOnce: %d pieces, %.2fs", traj_.getPieceNum(), traj_.getTotalDuration());
  return traj_.getPieceNum() > 0;
}

void GcopterPlanner::buildYaw(const Trajectory<5> &pos_traj, YawMode yaw_mode) {
  yaw_traj_.clear();

  double dur = pos_traj.getTotalDuration();
  if (dur < 1e-6) return;

  double yaw_start = current_yaw_;
  double yaw_end = has_target_yaw_ ? target_yaw_ : current_yaw_;
  double dyaw = yaw_end - yaw_start;
  while (dyaw > M_PI) dyaw -= 2.0 * M_PI;
  while (dyaw < -M_PI) dyaw += 2.0 * M_PI;

  switch (yaw_mode) {
    case YawMode::LOOK_TARGET: {
      // ==========================================================
      // 离散任务: yaw 线性插值到目标朝向
      // ==========================================================
      minco::MINCO_S3NU yaw_minco;
      Eigen::Matrix3d yawH = Eigen::Matrix3d::Zero(), yawT = Eigen::Matrix3d::Zero();
      yawH(0, 0) = yaw_start;
      yawT(0, 0) = yaw_end;
      yaw_minco.setConditions(yawH, yawT, 2);

      Eigen::Matrix3Xd yaw_pts(3, 1);
      yaw_pts.setZero();
      yaw_pts(0, 0) = yaw_start + dyaw * 0.5;
      Eigen::VectorXd yaw_times(2);
      yaw_times(0) = dur * 0.5;
      yaw_times(1) = dur * 0.5;
      yaw_minco.setParameters(yaw_pts, yaw_times);
      yaw_minco.getTrajectory(yaw_traj_);
      break;
    }
    case YawMode::LOOK_FORWARD: {
      // ==========================================================
      // 连续任务: yaw 跟随速度方向 (SUPER-style)
      // ==========================================================
      const double yaw_dot_max = 1.0;
      const double interp_dt = M_PI / yaw_dot_max;

      int n_samples = dur < interp_dt ? 0 : static_cast<int>(dur / interp_dt);
      if (n_samples > 20) n_samples = 20;

      std::vector<double> yaw_wpts;
      double last_yaw = yaw_start;
      for (int i = 0; i < n_samples; i++) {
        double t = dur * static_cast<double>(i + 1) / (n_samples + 1);
        Eigen::Vector3d vel = pos_traj.getVel(t);
        double cur_yaw;
        if (vel.head<2>().norm() > 0.1) {
          cur_yaw = std::atan2(vel(1), vel(0));
          double d = cur_yaw - last_yaw;
          while (d > M_PI) {
            cur_yaw -= 2.0 * M_PI;
            d -= 2.0 * M_PI;
          }
          while (d < -M_PI) {
            cur_yaw += 2.0 * M_PI;
            d += 2.0 * M_PI;
          }
        } else {
          cur_yaw = last_yaw;
        }
        yaw_wpts.push_back(cur_yaw);
        last_yaw = cur_yaw;
      }
      {
        double d = yaw_end - last_yaw;
        while (d > M_PI) {
          yaw_end -= 2.0 * M_PI;
          d -= 2.0 * M_PI;
        }
        while (d < -M_PI) {
          yaw_end += 2.0 * M_PI;
          d += 2.0 * M_PI;
        }
      }

      int yaw_pieceN = n_samples + 1;
      minco::MINCO_S3NU yaw_minco;
      Eigen::Matrix3d yawH = Eigen::Matrix3d::Zero(), yawT = Eigen::Matrix3d::Zero();
      yawH(0, 0) = yaw_start;
      yawT(0, 0) = yaw_end;
      yaw_minco.setConditions(yawH, yawT, yaw_pieceN);

      Eigen::Matrix3Xd yaw_pts(3, n_samples > 0 ? n_samples : 1);
      yaw_pts.setZero();
      if (n_samples > 0) {
        for (int i = 0; i < n_samples; i++) yaw_pts(0, i) = yaw_wpts[i];
      } else {
        yaw_pts(0, 0) = yaw_start + dyaw * 0.5;
      }
      Eigen::VectorXd yaw_times(yaw_pieceN);
      for (int i = 0; i < yaw_pieceN; i++) yaw_times(i) = dur / yaw_pieceN;
      yaw_minco.setParameters(yaw_pts, yaw_times);
      yaw_minco.getTrajectory(yaw_traj_);
      break;
    }
    case YawMode::FIX_YAW: {
      // ==========================================================
      // Pattern 模式: yaw 固定为起点朝向
      // ==========================================================
      minco::MINCO_S3NU yaw_minco;
      Eigen::Matrix3d yawH = Eigen::Matrix3d::Zero(), yawT = Eigen::Matrix3d::Zero();
      yawH(0, 0) = yaw_start;
      yawT(0, 0) = yaw_start;
      yaw_minco.setConditions(yawH, yawT, 1);

      Eigen::Matrix3Xd yaw_pts(3, 1);
      yaw_pts.setZero();
      yaw_pts(0, 0) = yaw_start;
      Eigen::VectorXd yaw_times(1);
      yaw_times(0) = dur;
      yaw_minco.setParameters(yaw_pts, yaw_times);
      yaw_minco.getTrajectory(yaw_traj_);
      break;
    }
  }
}

bool GcopterPlanner::planOnceWaypt(const Eigen::Vector3d &start, const std::vector<Eigen::Vector3d> &waypoints) {
  const int M = static_cast<int>(waypoints.size());
  if (M < 1) {
    ROS_WARN("planOnceWaypt: empty waypoints");
    return false;
  }

  // Last waypoint is the terminal state, not an intermediate one.
  // Keeping it in wptMat creates two zero-length pieces at the tail, whose
  // 0.05s minimum duration forces huge acc/jerk violations.
  const int N = M - 1;  // intermediate waypoints
  Eigen::Matrix3Xd wptMat(3, N);
  for (int i = 0; i < N; i++) wptMat.col(i) = waypoints[i];

  // Boundary conditions: start and final waypoint with vel=acc=0
  Eigen::Matrix3d iniState, finState;
  iniState << start, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();
  finState << waypoints.back(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();

  // Kinematic bounds (same as GCOPTER config)
  Eigen::VectorXd magnitudeBounds(5);
  magnitudeBounds(0) = config_.maxVelMag;      // v_max
  magnitudeBounds(1) = config_.maxBdrMag;      // omg_max
  magnitudeBounds(2) = config_.maxTiltAngle;   // theta_max
  magnitudeBounds(3) = config_.minThrust;      // thrust_min
  magnitudeBounds(4) = config_.maxThrust;      // thrust_max

  // Penalty weights (vel / omg / theta / thrust)
  Eigen::VectorXd penaltyWeights(4);
  penaltyWeights(0) = config_.chiVec[1];        // vel
  penaltyWeights(1) = config_.chiVec[2];        // omg
  penaltyWeights(2) = config_.chiVec[3] * 0.1;  // theta (scaled)
  penaltyWeights(3) = config_.chiVec[3];        // thrust

  // Physical parameters for flatness map
  Eigen::VectorXd physicalParams(6);
  physicalParams(0) = config_.vehicleMass;
  physicalParams(1) = config_.gravAcc;
  physicalParams(2) = config_.horizDrag;
  physicalParams(3) = config_.vertDrag;
  physicalParams(4) = config_.parasDrag;
  physicalParams(5) = config_.speedEps;

  // --- Reachability check: A* per segment ---
  int astar_ok = 0, astar_fail = 0;
  for (int i = 0; i < M; i++) {
    const Eigen::Vector3d &seg_start = (i == 0) ? start : waypoints[i - 1];
    const Eigen::Vector3d &seg_end = waypoints[i];
    std::vector<Eigen::Vector3d> seg_route;
    sfc_gen::planPathAStar<voxel_map::VoxelMap>(seg_start, seg_end, voxel_map_.getOrigin(), voxel_map_.getCorner(),
                                                &voxel_map_, 0.0, seg_route);
    if (seg_route.size() >= 2) {
      astar_ok++;
    } else {
      astar_fail++;
      ROS_WARN("  A* seg %d/%d: dist=%.2fm unreachable", i + 1, M, (seg_end - seg_start).norm());
    }
  }
  ROS_INFO("  A* reachability: %d/%d OK, %d unreachable", astar_ok, M, astar_fail);

  // --- SFC: seeds at waypoints only (not free mid-points) ---
  // MINCO pieces: 2 per waypoint segment.  hIdx maps piece i → polytope (i+1)/2,
  // so consecutive piece-pairs share the same polytope that covers the segment.
  std::vector<Eigen::Vector3d> sfcRoute;
  sfcRoute.push_back(start);
  for (int i = 0; i < N; i++) {
    sfcRoute.push_back(waypoints[i]);
  }
  sfcRoute.push_back(waypoints.back());  // terminal = wp24

  std::vector<Eigen::MatrixX4d> hPolys;
  buildSFC(sfcRoute, -1, voxel_map_, false, hPolys);

  if (hPolys.empty()) {
    ROS_WARN("planOnceWaypt: SFC generation failed");
    return false;
  }

  double weightPos = config_.chiVec[0];
  ROS_INFO("  Optimizer: weightPos=%.0f, wpts=%d (N=%d intermediate), pieces=%d, hPolys=%zu", weightPos, M, N,
           (N + 1) * 2, hPolys.size());

  traj_.clear();
  yaw_traj_.clear();

  if (!waypt_optimizer_.setup(config_.weightT, iniState, finState, wptMat, config_.smoothingEps,
                              config_.integralIntervs, magnitudeBounds, penaltyWeights, physicalParams,
                              hPolys, weightPos)) {
    ROS_WARN("planOnceWaypt: setup failed");
    return false;
  }

  waypt_optimizer_.initSpeed = config_.initSpeed;
  if (std::isinf(waypt_optimizer_.optimize(traj_, config_.relCostTol))) {
    ROS_WARN("planOnceWaypt: optimization failed");
    traj_.clear();
    return false;
  }

  // --- Yaw: configurable (LOOK_FORWARD or FIX_YAW) ---
  buildYaw(traj_, config_.yawMode);

  ROS_INFO("planOnceWaypt: %d waypoints, %d pieces, %.2fs, %d iters", M, traj_.getPieceNum(), traj_.getTotalDuration(),
           waypt_optimizer_.iter_num);
  return traj_.getPieceNum() > 0;
}

}  // namespace gcopter
