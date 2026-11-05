#include "gcopter_planner.h"

namespace gcopter {

void GcopterPlanner::publishTrajectoryCommand(double t) {
  Eigen::Vector3d pos = traj_.getPos(t);
  Eigen::Vector3d vel = traj_.getVel(t);
  Eigen::Vector3d acc = traj_.getAcc(t);
  Eigen::Vector3d jer = traj_.getJer(t);

  double yaw, yaw_dot;
  if (yaw_traj_.getPieceNum() > 0 && t <= yaw_traj_.getTotalDuration()) {
    yaw = yaw_traj_.getPos(t)(0);
    yaw_dot = yaw_traj_.getVel(t)(0);
  } else {
    yaw = (vel.norm() > 0.1) ? std::atan2(vel(1), vel(0)) : last_yaw_;
    yaw_dot = 0.0;
  }

  quadrotor_msgs::PositionCommand cmd;
  cmd.header.stamp = ros::Time::now();
  cmd.header.frame_id = "world";
  cmd.trajectory_flag = quadrotor_msgs::PositionCommand::TRAJECTORY_STATUS_READY;
  cmd.trajectory_id = traj_id_;

  cmd.position.x = pos(0);
  cmd.position.y = pos(1);
  cmd.position.z = pos(2);
  cmd.velocity.x = vel(0);
  cmd.velocity.y = vel(1);
  cmd.velocity.z = vel(2);
  cmd.acceleration.x = acc(0);
  cmd.acceleration.y = acc(1);
  cmd.acceleration.z = acc(2);
  cmd.jerk.x = jer(0);
  cmd.jerk.y = jer(1);
  cmd.jerk.z = jer(2);
  cmd.yaw = yaw;
  cmd.yaw_dot = yaw_dot;

  cmd.kx[0] = config_.kx0;
  cmd.kx[1] = config_.kx1;
  cmd.kx[2] = config_.kx2;
  cmd.kv[0] = config_.kv0;
  cmd.kv[1] = config_.kv1;
  cmd.kv[2] = config_.kv2;

  pos_cmd_pub_.publish(cmd);

  last_yaw_ = yaw;
  last_pos_ = pos;
}

void GcopterPlanner::publishHoverCommand() {
  if (!has_odom_) return;
  Eigen::Vector3d pos(latest_odom_.pose.pose.position.x, latest_odom_.pose.pose.position.y,
                      latest_odom_.pose.pose.position.z);
  publishHoverAt(pos);
}

void GcopterPlanner::publishInitialHoverCommand() {
  if (!has_odom_) return;
  Eigen::Vector3d pos(config_.initX, config_.initY, config_.initZ);
  publishHoverAt(pos);
}

void GcopterPlanner::publishHoverAt(const Eigen::Vector3d &pos) {
  double cmd_yaw = last_yaw_;
  double cmd_yaw_dot = 0.0;
  if (has_yaw_cmd_) {
    const double YAW_DOT_MAX = 0.6 * M_PI;
    const double YAW_DOT_DOT_MAX = 2.0 * M_PI;

    double cur_yaw = last_yaw_;
    double d_yaw = normalizeAngle(target_yaw_ - cur_yaw);

    double dt = 0.01;
    double d_yaw_max;
    if (std::abs(last_yaw_dot_ + dt * std::copysign(YAW_DOT_DOT_MAX, d_yaw)) <= YAW_DOT_MAX) {
      d_yaw_max = last_yaw_dot_ * dt + 0.5 * std::copysign(YAW_DOT_DOT_MAX, d_yaw) * dt * dt;
    } else {
      double t1 = (std::copysign(YAW_DOT_MAX, d_yaw) - last_yaw_dot_) / std::copysign(YAW_DOT_DOT_MAX, d_yaw);
      d_yaw_max = last_yaw_dot_ * t1 + 0.5 * std::copysign(YAW_DOT_DOT_MAX, d_yaw) * t1 * t1 +
                  std::copysign(YAW_DOT_MAX, d_yaw) * (dt - t1);
    }

    if (std::abs(d_yaw) > std::abs(d_yaw_max)) d_yaw = d_yaw_max;

    cmd_yaw = cur_yaw + d_yaw;
    cmd_yaw_dot = d_yaw / dt;

    last_yaw_dot_ = cmd_yaw_dot;
  }

  quadrotor_msgs::PositionCommand cmd;
  cmd.header.stamp = ros::Time::now();
  cmd.header.frame_id = "world";
  cmd.trajectory_flag = quadrotor_msgs::PositionCommand::TRAJECTORY_STATUS_READY;
  cmd.trajectory_id = traj_id_;

  cmd.position.x = pos(0);
  cmd.position.y = pos(1);
  cmd.position.z = pos(2);
  cmd.velocity.x = 0.0;
  cmd.velocity.y = 0.0;
  cmd.velocity.z = 0.0;
  cmd.acceleration.x = 0.0;
  cmd.acceleration.y = 0.0;
  cmd.acceleration.z = 0.0;
  cmd.jerk.x = 0.0;
  cmd.jerk.y = 0.0;
  cmd.jerk.z = 0.0;
  cmd.yaw = cmd_yaw;
  cmd.yaw_dot = cmd_yaw_dot;

  cmd.kx[0] = config_.kx0;
  cmd.kx[1] = config_.kx1;
  cmd.kx[2] = config_.kx2;
  cmd.kv[0] = config_.kv0;
  cmd.kv[1] = config_.kv1;
  cmd.kv[2] = config_.kv2;

  pos_cmd_pub_.publish(cmd);
  last_yaw_ = cmd_yaw;
  last_pos_ = pos;
}

void GcopterPlanner::publishGoalAbandoned() {
  std_msgs::Bool msg;
  msg.data = true;
  goal_abandoned_pub_.publish(msg);
}

void GcopterPlanner::publishGoalMarker() {
  visualization_msgs::Marker mk;
  mk.header.frame_id = "world";
  mk.header.stamp = ros::Time::now();
  mk.ns = "gcopter_goal";
  mk.id = 0;
  mk.type = visualization_msgs::Marker::SPHERE;
  mk.action = visualization_msgs::Marker::ADD;
  mk.pose.position.x = latest_goal_(0);
  mk.pose.position.y = latest_goal_(1);
  mk.pose.position.z = latest_goal_(2);
  mk.pose.orientation.w = 1.0;
  mk.scale.x = 0.2;
  mk.scale.y = 0.2;
  mk.scale.z = 0.2;
  mk.color.r = 125.0 / 255.0;
  mk.color.g = 125.0 / 255.0;
  mk.color.b = 125.0 / 255.0;
  mk.color.a = 1.0;
  goal_marker_pub_.publish(mk);
}

void GcopterPlanner::clearGoalMarker() {
  visualization_msgs::Marker mk;
  mk.header.frame_id = "world";
  mk.header.stamp = ros::Time::now();
  mk.ns = "gcopter_goal";
  mk.id = 0;
  mk.action = visualization_msgs::Marker::DELETE;
  goal_marker_pub_.publish(mk);
}

void GcopterPlanner::publishWaypointMarkers() {
  // Delete all existing goal markers
  visualization_msgs::Marker del_all;
  del_all.header.frame_id = "world";
  del_all.header.stamp = ros::Time::now();
  del_all.ns = "gcopter_wpts";
  del_all.action = visualization_msgs::Marker::DELETEALL;
  goal_marker_pub_.publish(del_all);

  for (size_t i = 0; i < waypoints_.size(); i++) {
    visualization_msgs::Marker mk;
    mk.header.frame_id = "world";
    mk.header.stamp = ros::Time::now();
    mk.ns = "gcopter_wpts";
    mk.id = static_cast<int>(i);
    mk.type = visualization_msgs::Marker::SPHERE;
    mk.action = visualization_msgs::Marker::ADD;
    mk.pose.position.x = waypoints_[i](0);
    mk.pose.position.y = waypoints_[i](1);
    mk.pose.position.z = waypoints_[i](2);
    mk.pose.orientation.w = 1.0;
    mk.scale.x = 0.15;
    mk.scale.y = 0.15;
    mk.scale.z = 0.15;
    mk.color.r = 0.0;
    mk.color.g = 0.8;
    mk.color.b = 0.0;
    mk.color.a = 0.9;
    goal_marker_pub_.publish(mk);
  }
}

void GcopterPlanner::clearWaypointMarkers() {
  visualization_msgs::Marker del_all;
  del_all.header.frame_id = "world";
  del_all.header.stamp = ros::Time::now();
  del_all.ns = "gcopter_wpts";
  del_all.action = visualization_msgs::Marker::DELETEALL;
  goal_marker_pub_.publish(del_all);
}

void GcopterPlanner::publishTrajectoryMarker() {
  if (traj_.getPieceNum() == 0) return;

  visualization_msgs::Marker mk;
  mk.header.frame_id = "world";
  mk.header.stamp = ros::Time::now();
  mk.ns = "gcopter_traj";
  mk.id = 0;
  mk.type = visualization_msgs::Marker::LINE_STRIP;
  mk.action = visualization_msgs::Marker::ADD;
  mk.pose.orientation.w = 1.0;
  mk.scale.x = 0.05;
  mk.color.r = 1.0;
  mk.color.g = 0.0;
  mk.color.b = 0.0;
  mk.color.a = 1.0;

  const double dt = 0.05;
  const double dur = traj_.getTotalDuration();
  for (double t = 0.0; t <= dur; t += dt) {
    Eigen::Vector3d pt = traj_.getPos(t);
    geometry_msgs::Point p;
    p.x = pt(0);
    p.y = pt(1);
    p.z = pt(2);
    mk.points.push_back(p);
  }
  traj_marker_pub_.publish(mk);
}

void GcopterPlanner::clearTrajectoryMarker() {
  visualization_msgs::Marker mk;
  mk.header.frame_id = "world";
  mk.header.stamp = ros::Time::now();
  mk.ns = "gcopter_traj";
  mk.id = 0;
  mk.action = visualization_msgs::Marker::DELETE;
  traj_marker_pub_.publish(mk);
}

void GcopterPlanner::publishFOVMarker() {
  if (!has_odom_) return;

  Eigen::Vector3d pos(latest_odom_.pose.pose.position.x, latest_odom_.pose.pose.position.y,
                      latest_odom_.pose.pose.position.z);
  tf::Quaternion q(latest_odom_.pose.pose.orientation.x, latest_odom_.pose.pose.orientation.y,
                   latest_odom_.pose.pose.orientation.z, latest_odom_.pose.pose.orientation.w);
  double roll, pitch, yaw;
  tf::Matrix3x3(q).getRPY(roll, pitch, yaw);

  percep_utils_->setPose(pos, yaw);
  std::vector<Eigen::Vector3d> l1, l2;
  percep_utils_->getFOV(l1, l2);

  visualization_msgs::Marker mk;
  mk.header.frame_id = "world";
  mk.header.stamp = ros::Time::now();
  mk.ns = "gcopter_fov";
  mk.id = 0;
  mk.type = visualization_msgs::Marker::LINE_LIST;
  mk.action = visualization_msgs::Marker::ADD;
  mk.pose.orientation.w = 1.0;
  mk.scale.x = 0.04;
  mk.color.r = 0.0;
  mk.color.g = 0.0;
  mk.color.b = 0.0;
  mk.color.a = 0.5;

  for (size_t i = 0; i < l1.size(); ++i) {
    geometry_msgs::Point p1, p2;
    p1.x = l1[i](0);
    p1.y = l1[i](1);
    p1.z = l1[i](2);
    p2.x = l2[i](0);
    p2.y = l2[i](1);
    p2.z = l2[i](2);
    mk.points.push_back(p1);
    mk.points.push_back(p2);
  }

  fov_marker_pub_.publish(mk);
}

}  // namespace gcopter
