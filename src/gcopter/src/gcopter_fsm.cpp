#include "gcopter_planner.h"

namespace gcopter {

GcopterPlanner::GcopterPlanner(const Config &cfg)
    : config_(cfg),
      state_(INIT),
      map_initialized_(false),
      has_odom_(false),
      exec_t_start_(0.0),
      last_yaw_(cfg.initYaw) {}

void GcopterPlanner::init(ros::NodeHandle &nh) {
  nh_ = nh;

  // Publishers used by loadOccupancyGrid must be created first
  grid_vis_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/gcopter/voxel_map_vis", 1, true);
  grid_inflate_vis_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/gcopter/voxel_map_inflated_vis", 1, true);

  // Inflated Map
  std::string occ_file;
  nh_.param<std::string>("occ_file", occ_file, "");
  if (!occ_file.empty()) {
    if (loadOccupancyGrid(occ_file)) {
      ROS_INFO("Occupancy grid loaded from %s", occ_file.c_str());
    } else {
      ROS_ERROR("Failed to load occupancy grid from %s", occ_file.c_str());
    }
  } else {
    ROS_WARN("occ_file not set. Occupancy grid will not be loaded.");
  }

  // ROS utils
  goal_sub_ =
      nh_.subscribe(config_.goalTopic, 1, &GcopterPlanner::goalCallback, this, ros::TransportHints().tcpNoDelay());
  waypoint_list_sub_ = nh_.subscribe("/waypoint_list", 1, &GcopterPlanner::waypointListCallback, this,
                                     ros::TransportHints().tcpNoDelay());
  odom_sub_ =
      nh_.subscribe(config_.odomTopic, 1, &GcopterPlanner::odomCallback, this, ros::TransportHints().tcpNoDelay());
  yaw_cmd_sub_ =
      nh_.subscribe("/planning/yaw_cmd", 1, &GcopterPlanner::yawCmdCallback, this, ros::TransportHints().tcpNoDelay());

  pos_cmd_pub_ = nh_.advertise<quadrotor_msgs::PositionCommand>("/planning/pos_cmd", 50);
  traj_done_pub_ = nh_.advertise<std_msgs::Bool>("/traj_done", 10);
  goal_abandoned_pub_ = nh_.advertise<std_msgs::Bool>("/goal_abandoned", 10);

  traj_marker_pub_ = nh_.advertise<visualization_msgs::Marker>("/gcopter/trajectory", 10);
  fov_marker_pub_ = nh_.advertise<visualization_msgs::Marker>("/gcopter/planning/position_cmd_vis", 10);
  goal_marker_pub_ = nh_.advertise<visualization_msgs::Marker>("/gcopter/goal_point", 100);

  exec_timer_ = nh_.createTimer(ros::Duration(0.01), &GcopterPlanner::execTimerCallback, this);

  // Flatness
  Eigen::VectorXd pm(6);
  pm(0) = config_.vehicleMass;
  pm(1) = config_.gravAcc;
  pm(2) = config_.horizDrag;
  pm(3) = config_.vertDrag;
  pm(4) = config_.parasDrag;
  pm(5) = config_.speedEps;
  flatmap_.reset(pm(0), pm(1), pm(2), pm(3), pm(4), pm(5));

  ros::NodeHandle nh_priv("~");
  percep_utils_.reset(new PerceptionUtils(nh_priv));

  ROS_INFO("GCOPTER initialized. Loading occupancy grid...");
}

//
// Callbacks
//

void GcopterPlanner::execTimerCallback(const ros::TimerEvent & /*event*/) {
  switch (state_) {
    case INIT:
      if (map_initialized_ && has_odom_) {
        ROS_INFO("Map and odom ready. Switching to WAIT_GOAL.");
        state_ = WAIT_GOAL;
      }
      break;

    case WAIT_GOAL:
      if (!has_executed_traj_) {
        publishInitialHoverCommand();
      } else {
        publishHoverCommand();
      }
      publishFOVMarker();
      break;

    case PLAN: {
      Eigen::Vector3d start(latest_odom_.pose.pose.position.x, latest_odom_.pose.pose.position.y,
                            latest_odom_.pose.pose.position.z);

      if (voxel_map_.query(start) != voxel_map::Unoccupied) {
        ROS_ERROR("Start position (%.2f,%.2f,%.2f) is occupied!", start(0), start(1), start(2));
        publishGoalAbandoned();
        clearWaypointMarkers();
        waypoints_.clear();
        state_ = WAIT_GOAL;
        break;
      }

      bool plan_ok = false;
      if (!waypoints_.empty()) {
        ROS_INFO("Planning waypt: start=(%.2f,%.2f,%.2f) with %zu waypoints", start(0), start(1), start(2),
                 waypoints_.size());
        plan_ok = planOnceWaypt(start, waypoints_);
      } else {
        ROS_INFO("Planning: start=(%.2f,%.2f,%.2f) -> goal=(%.2f,%.2f,%.2f)", start(0), start(1), start(2),
                 latest_goal_(0), latest_goal_(1), latest_goal_(2));
        plan_ok = planOnce(start, latest_goal_);
      }

      if (plan_ok) {
        exec_t_start_ = latest_odom_.header.stamp.toSec();
        state_ = EXECUTE;
        has_executed_traj_ = true;
        publishTrajectoryMarker();
        ROS_INFO("Plan SUCCESS. Trajectory duration=%.2fs, %d pieces.", traj_.getTotalDuration(), traj_.getPieceNum());
      } else {
        ROS_ERROR("Planning FAILED.");
        publishGoalAbandoned();
        clearWaypointMarkers();
        waypoints_.clear();
        state_ = WAIT_GOAL;
      }
      break;
    }

    case EXECUTE: {
      double elapsed = latest_odom_.header.stamp.toSec() - exec_t_start_;
      if (elapsed < 0.0) elapsed = 0.0;

      if (elapsed >= traj_.getTotalDuration()) {
        publishHoverAt(traj_.getPos(traj_.getTotalDuration()));  // hover at trajectory endpoint
        clearTrajectoryMarker();
        clearWaypointMarkers();

        std_msgs::Bool done;
        done.data = true;
        traj_done_pub_.publish(done);
        ROS_INFO("Trajectory done (%.2fs). Back to WAIT_GOAL.", elapsed);

        state_ = WAIT_GOAL;
      } else {
        publishTrajectoryCommand(elapsed);
        publishFOVMarker();
      }
      break;
    }
  }
}

void GcopterPlanner::goalCallback(const geometry_msgs::PoseStamped::ConstPtr &msg) {
  if (!map_initialized_) {
    ROS_WARN_THROTTLE(5.0, "Goal received but voxel map not yet initialized");
    return;
  }
  if (!has_odom_) {
    ROS_WARN_THROTTLE(5.0, "Goal received but odometry not yet available");
    return;
  }
  if (state_ == EXECUTE || state_ == PLAN) {
    ROS_WARN_THROTTLE(1.0, "Goal received while busy (state=%d), ignoring", state_);
    return;
  }

  latest_goal_ = Eigen::Vector3d(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);

  {
    tf::Quaternion q(msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z,
                     msg->pose.orientation.w);
    double roll, pitch;
    tf::Matrix3x3(q).getRPY(roll, pitch, target_yaw_);
    has_target_yaw_ = true;
  }

  if (voxel_map_.query(latest_goal_) != voxel_map::Unoccupied) {
    ROS_WARN("Goal (%.2f,%.2f,%.2f) is in occupied space, abandoning", latest_goal_(0), latest_goal_(1),
             latest_goal_(2));
    publishGoalAbandoned();
    return;
  }

  has_yaw_cmd_ = false;

  ROS_INFO("Goal received: (%.2f, %.2f, %.2f), triggering PLAN", latest_goal_(0), latest_goal_(1), latest_goal_(2));
  publishGoalMarker();
  state_ = PLAN;
}

void GcopterPlanner::waypointListCallback(const nav_msgs::Path::ConstPtr &msg) {
  if (!map_initialized_) {
    ROS_WARN_THROTTLE(5.0, "Waypoint list received but voxel map not yet initialized");
    return;
  }
  if (!has_odom_) {
    ROS_WARN_THROTTLE(5.0, "Waypoint list received but odometry not yet available");
    return;
  }
  if (state_ == EXECUTE || state_ == PLAN) {
    ROS_WARN_THROTTLE(1.0, "Waypoint list received while busy (state=%d), ignoring", state_);
    return;
  }

  const int N = msg->poses.size();
  if (N < 1) {
    ROS_WARN("Waypoint list is empty, ignoring");
    return;
  }

  waypoints_.clear();
  waypoints_.reserve(N);
  for (int i = 0; i < N; i++) {
    const auto &p = msg->poses[i].pose.position;
    Eigen::Vector3d wp(p.x, p.y, p.z);

    if (voxel_map_.query(wp) != voxel_map::Unoccupied) {
      ROS_WARN("Waypoint %d (%.2f,%.2f,%.2f) is in occupied space, abandoning entire list", i, wp(0), wp(1), wp(2));
      publishGoalAbandoned();
      waypoints_.clear();
      return;
    }
    waypoints_.push_back(wp);
  }

  // Extract yaw from the first waypoint's orientation
  {
    const auto &q = msg->poses[0].pose.orientation;
    tf::Quaternion tf_q(q.x, q.y, q.z, q.w);
    double roll, pitch;
    tf::Matrix3x3(tf_q).getRPY(roll, pitch, target_yaw_);
    has_target_yaw_ = true;
  }

  has_yaw_cmd_ = false;

  publishWaypointMarkers();

  ROS_INFO("Waypoint list received: %d waypoints, first=(%.2f,%.2f,%.2f), triggering PLAN", N, waypoints_[0](0),
           waypoints_[0](1), waypoints_[0](2));
  state_ = PLAN;
}

void GcopterPlanner::odomCallback(const nav_msgs::Odometry::ConstPtr &msg) {
  latest_odom_ = *msg;
  has_odom_ = true;

  tf::Quaternion q_odom(msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z,
                        msg->pose.pose.orientation.w);
  double roll, pitch;
  tf::Matrix3x3(q_odom).getRPY(roll, pitch, current_yaw_);
}

void GcopterPlanner::yawCmdCallback(const std_msgs::Float64::ConstPtr &msg) {
  target_yaw_ = msg->data;
  has_yaw_cmd_ = true;
  yaw_cmd_stamp_ = ros::Time::now();
  ROS_INFO("Yaw command received: %.2f rad (%.1f deg)", target_yaw_, target_yaw_ * 180.0 / M_PI);
}

double GcopterPlanner::normalizeAngle(double angle) {
  while (angle > M_PI) angle -= 2.0 * M_PI;
  while (angle < -M_PI) angle += 2.0 * M_PI;
  return angle;
}

}  // namespace gcopter
