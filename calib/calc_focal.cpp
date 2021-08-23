#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>
#include <opencv2/imgcodecs.hpp>
#include <vector>

using namespace std;

int main(int argc, char *argv[])
{

  // input
  int width  = 1280;
  int height = 720;
  cv::Mat camera_matrix = (cv::Mat_<double>(3,3) <<
			   1020.05370, 0.00000, 721.80286,
			   0.00000, 1027.46396, 516.04788,
			   0.0, 0.0, 1.0);

  // double aperture_width  = 3.6736; // [mm] 入力すべき単位不明
  // double aperture_height = 2.7384; // [mm]

  // double aperture_width  = 0.0036736; // [um] 入力すべき単位不明
  // double aperture_height = 0.0027384; // [um]

  double aperture_width  = 0.0;
  double aperture_height = 0.0;

  cout << "\n----- INPUT -----\n";
  cout << "width: " << width << "  height: " << height << endl;
  cout << "camera_matrix:\n" << camera_matrix << endl;


  // output
  double fovx, fovy, focal_length, aspect_ratio;
  cv::Point2d principal_point;


  cv::calibrationMatrixValues(camera_matrix, cv::Size(width, height), aperture_width, aperture_height, fovx, fovy, focal_length, principal_point, aspect_ratio);
  cout << "\n----- OUTPUT -----\n";
  cout << "fovx: " << fovx << endl;
  cout << "fovy: " << fovy << endl;
  cout << "focal length: " << focal_length << endl;
  cout << "principal point: " << principal_point << endl;
  cout << "aspect ratio: " << aspect_ratio << endl;

  
  return 0;
}
