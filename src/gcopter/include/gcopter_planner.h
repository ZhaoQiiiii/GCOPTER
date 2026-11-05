#ifndef GCOPTER_PLANNER_H
#define GCOPTER_PLANNER_H

#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <perception_utils/perception_utils.h>
#include <quadrotor_msgs/PositionCommand.h>
#include <ros/console.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float64.h>
#include <tf/tf.h>
#include <visualization_msgs/Marker.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "firi.hpp"
#include "flatness.hpp"
#include "gcopter.hpp"
#include "gcopter_optimizer.h"
#include "geo_utils.hpp"
#include "sfc_gen.hpp"
#include "trajectory.hpp"
#include "voxel_map.hpp"

namespace gcopter {

enum class YawMode { LOOK_TARGET, LOOK_FORWARD, FIX_YAW };

struct Config {
  std::string goalTopic;
  std::string odomTopic;
  double inflateRadius;
  double timeoutRRT;
  double maxVelMag;
  double maxBdrMag;
  double maxTiltAngle;
  double minThrust;
  double maxThrust;
  double vehicleMass;
  double gravAcc;
  double horizDrag;
  double vertDrag;
  double parasDrag;
  double speedEps;
  double initSpeed;
  double weightT;
  double yawAlignWeight;
  std::vector<double> chiVec;
  double smoothingEps;
  int integralIntervs;
  double relCostTol;
  double initX;
  double initY;
  double initZ;
  double initYaw;
  double kx0, kx1, kx2;
  double kv0, kv1, kv2;

  explicit Config(const ros::NodeHandle &nh_priv) {
    nh_priv.param<std::string>("GoalTopic", goalTopic, "/uav_actions");
    nh_priv.param<std::string>("OdomTopic", odomTopic, "/drone_0_visual_slam/odom");
    nh_priv.param("InflateRadius", inflateRadius, 0.3);
    nh_priv.param("TimeoutRRT", timeoutRRT, 0.02);
    nh_priv.param("MaxVelMag", maxVelMag, 1.0);
    nh_priv.param("MaxBdrMag", maxBdrMag, 2.1);
    nh_priv.param("MaxTiltAngle", maxTiltAngle, 1.05);
    nh_priv.param("MinThrust", minThrust, 2.0);
    nh_priv.param("MaxThrust", maxThrust, 12.0);
    nh_priv.param("VehicleMass", vehicleMass, 0.98);
    nh_priv.param("GravAcc", gravAcc, 9.8);
    nh_priv.param("HorizDrag", horizDrag, 0.0);
    nh_priv.param("VertDrag", vertDrag, 0.0);
    nh_priv.param("ParasDrag", parasDrag, 0.0);
    nh_priv.param("SpeedEps", speedEps, 0.0001);
    nh_priv.param("InitSpeed", initSpeed, 0.1);
    nh_priv.param("WeightT", weightT, 20.0);
    nh_priv.param("YawAlignWeight", yawAlignWeight, 10.0);
    nh_priv.param("SmoothingEps", smoothingEps, 1.0e-2);
    nh_priv.param("IntegralIntervs", integralIntervs, 16);
    nh_priv.param("RelCostTol", relCostTol, 1.0e-5);
    nh_priv.param("InitX", initX, 0.0);
    nh_priv.param("InitY", initY, 0.0);
    nh_priv.param("InitZ", initZ, 1.5);
    nh_priv.param("InitYaw", initYaw, 0.0);
    nh_priv.param("Kx0", kx0, 5.0);
    nh_priv.param("Kx1", kx1, 5.0);
    nh_priv.param("Kx2", kx2, 5.0);
    nh_priv.param("Kv0", kv0, 3.0);
    nh_priv.param("Kv1", kv1, 3.0);
    nh_priv.param("Kv2", kv2, 3.0);
    nh_priv.getParam("ChiVec", chiVec);
    if (chiVec.size() < 4) {
      chiVec = {1.0e+4, 1.0e+4, 1.0e+4, 1.0e+5};
    }

    std::string yawModeStr;
    nh_priv.param<std::string>("YawMode", yawModeStr, std::string("LOOK_FORWARD"));
    if (yawModeStr == "LOOK_TARGET")
      yawMode = YawMode::LOOK_TARGET;
    else if (yawModeStr == "FIX_YAW")
      yawMode = YawMode::FIX_YAW;
    else
      yawMode = YawMode::LOOK_FORWARD;
  }

  YawMode yawMode;
};

class GcopterPlanner {
 public:
  enum State { INIT = 0, WAIT_GOAL, PLAN, EXECUTE };

  explicit GcopterPlanner(const Config &cfg);
  void init(ros::NodeHandle &nh);

 private:
  // ---- Grid I/O ----
  bool loadOccupancyGrid(const std::string &occ_file);
  void publishVoxelVis(ros::Publisher &pub, const std::vector<float> &vis_data, const char *topic_name,
                       double vis_max_z);

  // ---- ROS Callbacks ----
  void goalCallback(const geometry_msgs::PoseStamped::ConstPtr &msg);
  void waypointListCallback(const nav_msgs::Path::ConstPtr &msg);
  void odomCallback(const nav_msgs::Odometry::ConstPtr &msg);
  void yawCmdCallback(const std_msgs::Float64::ConstPtr &msg);
  static double normalizeAngle(double angle);

  // ---- Main FSM Loop (100 Hz) ----
  void execTimerCallback(const ros::TimerEvent &event);

  // ---- Planning ----
  bool planOnce(const Eigen::Vector3d &start, const Eigen::Vector3d &goal);
  bool planOnceWaypt(const Eigen::Vector3d &start, const std::vector<Eigen::Vector3d> &waypoints);
  void buildYaw(const Trajectory<5> &pos_traj, YawMode yaw_mode);

  // ---- Command Publishing ----
  void publishTrajectoryCommand(double t);
  void publishHoverCommand();
  void publishInitialHoverCommand();
  void publishHoverAt(const Eigen::Vector3d &pos);
  void publishGoalAbandoned();

  // ---- Visualization ----
  void publishGoalMarker();
  void publishWaypointMarkers();
  void clearGoalMarker();
  void clearWaypointMarkers();
  void publishTrajectoryMarker();
  void clearTrajectoryMarker();
  void publishFOVMarker();

  // ---- Member variables ----
  Config config_;
  ros::NodeHandle nh_;

  // Subscribers
  ros::Subscriber goal_sub_;
  ros::Subscriber waypoint_list_sub_;
  ros::Subscriber odom_sub_;
  ros::Subscriber yaw_cmd_sub_;

  // Publishers
  ros::Publisher pos_cmd_pub_;
  ros::Publisher traj_done_pub_;
  ros::Publisher goal_abandoned_pub_;
  ros::Publisher traj_marker_pub_;
  ros::Publisher fov_marker_pub_;
  ros::Publisher goal_marker_pub_;
  ros::Publisher grid_vis_pub_;
  ros::Publisher grid_inflate_vis_pub_;

  // Timer
  ros::Timer exec_timer_;

  // State
  State state_;
  bool map_initialized_;
  bool has_odom_;
  bool has_executed_traj_{false};

  // Data
  voxel_map::VoxelMap voxel_map_;
  nav_msgs::Odometry latest_odom_;
  Eigen::Vector3d latest_goal_{Eigen::Vector3d::Zero()};
  std::vector<Eigen::Vector3d> waypoints_;
  GcopterWayptS3 waypt_optimizer_;

  // Trajectory
  Trajectory<5> traj_;
  Trajectory<5> yaw_traj_;
  flatness::FlatnessMap flatmap_;
  std::unique_ptr<PerceptionUtils> percep_utils_;
  double exec_t_start_;
  int traj_id_{0};

  // Yaw / position cache
  double last_yaw_;
  double last_yaw_dot_{0.0};
  double current_yaw_{0.0};
  double target_yaw_{0.0};
  bool has_yaw_cmd_{false};
  bool has_target_yaw_{false};
  ros::Time yaw_cmd_stamp_{0.0};
  Eigen::Vector3d last_pos_{Eigen::Vector3d::Zero()};
};

}  // namespace gcopter

#endif  // GCOPTER_PLANNER_H
