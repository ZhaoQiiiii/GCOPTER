#include "geometry_msgs/PoseStamped.h"
#include "nav_msgs/Path.h"
#include "quadrotor_msgs/PositionCommand.h"
#include "ros/ros.h"

using namespace std;

nav_msgs::Path drone_path;
ros::Publisher drone_path_pub;

void cmd_callback(const quadrotor_msgs::PositionCommand& cmd) {
  drone_path.header.frame_id = "world";
  drone_path.header.stamp = ros::Time::now();
  geometry_msgs::PoseStamped pose;
  pose.header.frame_id = "world";
  pose.header.stamp = ros::Time::now();
  pose.pose.position.x = cmd.position.x;
  pose.pose.position.y = cmd.position.y;
  pose.pose.position.z = cmd.position.z;
  pose.pose.orientation.w = 1.0;
  drone_path.poses.push_back(pose);
  drone_path_pub.publish(drone_path);
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "visualize_history_path");
  ROS_INFO("visualize_history_path node init ok");

  ros::NodeHandle n("~");
  std::string cmd_topic;
  n.param<std::string>("cmd_topic", cmd_topic, "/position_cmd");
  ros::Subscriber cmd_sub = n.subscribe(cmd_topic, 100, cmd_callback);

  drone_path_pub = n.advertise<nav_msgs::Path>("/drone_path", 10, true);

  ROS_INFO("Subscribing to cmd topic: %s", cmd_topic.c_str());

  ros::spin();

  return 0;
}
