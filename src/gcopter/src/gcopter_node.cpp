#include "gcopter_planner.h"

using namespace gcopter;

int main(int argc, char **argv) {
  ros::init(argc, argv, "gcopter_node");
  ros::NodeHandle nh("~");
  Config config(nh);

  GcopterPlanner planner(config);
  planner.init(nh);

  ros::spin();
  return 0;
}
