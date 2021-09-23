#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>

#include "ground_segmentation.h"

int main(int argc, char** argv) {
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::io::loadPLYFile(argv[1], cloud);
    
    GroundSegmentationParams params;

    params.visualize = true;    // # visualize segmentation result - USE ONLY FOR DEBUGGING
    params.n_bins = 120;        // # number of radial bins
    params.n_segments = 360;    // # number of radial segments.

    // Off-the-shelf params for Kitti
    // params.max_dist_to_line = 0.05; // # maximum vertical distance of point to line to be considered ground.
    // params.max_slope = 0.3;         // # maximum slope of a ground line.
    // params.long_threshold = 1.0;    // # distance between points after which they are considered far from each other.
    // params.max_long_height = 0.1;   // # maximum height change to previous point in long line.
    // params.max_start_height = 0.2;  // # maximum difference to estimated ground height to start a new line.
    // params.sensor_height = 1.8;     // # sensor height above ground.
    // params.line_search_angle = 0.1; // # how far to search in angular direction to find a line [rad].

    // Attempt to tune it for Nuscenes
    params.max_dist_to_line = 0.1; // # maximum vertical distance of point to line to be considered ground.
    params.max_slope = 0.4;         // # maximum slope of a ground line.
    params.long_threshold = 2.0;    // # distance between points after which they are considered far from each other.
    params.max_long_height = 0.2;   // # maximum height change to previous point in long line.
    params.max_start_height = 0.4;  // # maximum difference to estimated ground height to start a new line.
    params.sensor_height = 1.84;     // # sensor height above ground.
    params.line_search_angle = 0.2; // # how far to search in angular direction to find a line [rad].

    params.n_threads = 4;           // # number of threads to use.

    double r_min = 0.8;         // # minimum point distance.
    double r_max = 70.4;          // # maximum point distance.
    double max_fit_error = 0.1; // # maximum error of a point during line fit.
    
    params.r_min_square = r_min*r_min;
    params.r_max_square = r_max*r_max;
    params.max_error_square = max_fit_error * max_fit_error;
    
    GroundSegmentation segmenter(params);
    std::vector<int> labels;

    segmenter.segment(cloud, &labels);

    // Visualize.
    PointCloud::Ptr obstacle_cloud(new PointCloud());
    // Get cloud of ground points.
    PointCloud::Ptr ground_cloud(new PointCloud());
    for (size_t i = 0; i < cloud.size(); ++i) {
        if (labels.at(i) == 1) ground_cloud->push_back(cloud[i]);
        else obstacle_cloud->push_back(cloud[i]);
    }

    pcl::io::savePLYFile("obstacle.ply", *obstacle_cloud);
    pcl::io::savePLYFile("ground.ply", *ground_cloud);
}
