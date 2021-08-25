#include "vo_features.h"

using namespace cv;
using namespace std;

#define MAX_FRAME 150
#define MIN_NUM_FEAT 10


int main( int argc, char** argv ){
  Mat img_1, img_2;
  Mat R_f, t_f; //the final rotation and tranlation vectors containing the 

  ofstream myfile;
  myfile.open("results1_1.txt");

  double scale = 1.00;
  char filename1[200];
  char filename2[200];

  sprintf(filename1, "/home/users/imamura/WORK/Git/mono-vo/dataset/contest/run1/view_%08d.png", 0);
  sprintf(filename2, "/home/users/imamura/WORK/Git/mono-vo/dataset/contest/run1/view_%08d.png", 1);

  char text[100];
  int fontFace = FONT_HERSHEY_PLAIN;
  double fontScale = 1;
  int thickness = 1;  
  cv::Point textOrg(10, 50);

  //read the first two frames from the dataset
  Mat img_1_c = imread(filename1);
  Mat img_2_c = imread(filename2);

  if ( !img_1_c.data || !img_2_c.data ) { 
    std::cout<< " --(!) Error reading images " << std::endl; return -1;
  }

  int miss_cnt=0;

  
  // we work with grayscale images
  cvtColor(img_1_c, img_1, COLOR_BGR2GRAY);
  cvtColor(img_2_c, img_2, COLOR_BGR2GRAY);

  // feature detection, tracking
  vector<Point2f> points1, points2;  //vectors to store the coordinates of the feature points
  
  // vector<KeyPoint> の参照渡しはうまくいかない??????????
  vector<KeyPoint> key1, key2;
  vector<DMatch> match;
  tie(key1, key2, match) = feature_matching(img_1, img_2);

  tie(points1, points2) = sort_feature(key1, key2, match);
  // KeyPoint::convert(key1, points1, vector<int>());
  // KeyPoint::convert(key2, points2, vector<int>());

  cout << "-------------------------------\n";
  cout << points1.size() << ", " << points2.size() << endl;
  
  // KITTI
  // double focal = 718.8560;
  // cv::Point2d pp(607.1928, 185.2157);

  // pcam
  double focal = 1020.05;
  cv::Point2d pp(721.803, 512.326);

  
  ////recovering the pose and the essential matrix
  // R : 回転行列 3x3
  // t : 並進行列 1x3
  Mat E, R, t, mask;
  E = findEssentialMat(points2, points1, focal, pp, RANSAC, 0.999, 1.0, mask);
  recoverPose(E, points2, points1, R, t, focal, pp, mask);

  Mat prevImage = img_2;
  Mat currImage;
  vector<Point2f> prevFeatures = points2;
  vector<Point2f> currFeatures;

  char filename[100];

  R_f = R.clone();
  t_f = t.clone();

  clock_t begin = clock();

  namedWindow( "Road facing camera", WINDOW_AUTOSIZE );// Create a window for display.
  namedWindow( "Trajectory", WINDOW_AUTOSIZE );// Create a window for display.

  //Mat traj = Mat::zeros(600, 600, CV_8UC3);
  Mat traj = imread("../map.png"); // 350x490
  
  for(int numFrame=2; numFrame < MAX_FRAME; numFrame++)	{
    sprintf(filename, "/home/users/imamura/WORK/Git/mono-vo/dataset/contest/run1/view_%08d.png", numFrame);
    cout << "\nframe: " << numFrame << endl;
    Mat currImage_c = imread(filename);
    cvtColor(currImage_c, currImage, COLOR_BGR2GRAY);
    tie(key1, key2, match) = feature_matching(prevImage, currImage);
    tie(points1, points2) = sort_feature(key1, key2, match);
    // KeyPoint::convert(key1, prevFeatures, vector<int>());
    // KeyPoint::convert(key2, currFeatures, vector<int>());

    
    cout << "points1 size: " << points1.size() << ", points2 size: " << points2.size() << ", match size: "<< match.size() << endl;
    for(int i=0; i<points1.size(); i++){
      cout << points1.at(i).x << ", " << points1.at(i).y << endl;
    }
    cout << "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n";
    
    Mat dst_img;
    drawMatches(prevImage, key1, currImage, key2, match, dst_img);
    imshow("match", dst_img);
    
    if(match.size() <= 5){
      cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
      cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
      cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
      cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
      cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
    }
    E = findEssentialMat(currFeatures, prevFeatures, focal, pp, RANSAC, 0.999, 1.0, mask);
    recoverPose(E, currFeatures, prevFeatures, R, t, focal, pp, mask);
    
    Mat prevPts(2,prevFeatures.size(), CV_64F), currPts(2,currFeatures.size(), CV_64F);

    Mat currImage_draw = currImage.clone();
    Mat prevImage_draw = prevImage.clone();
    
    for(int i=0;i<prevFeatures.size();i++){  //this (x,y) combination makes sense as observed from the source code of triangulatePoints on GitHub
      prevPts.at<double>(0,i) = prevFeatures.at(i).x;
      prevPts.at<double>(1,i) = prevFeatures.at(i).y;
      
      currPts.at<double>(0,i) = currFeatures.at(i).x;
      currPts.at<double>(1,i) = currFeatures.at(i).y;
    }
    
    scale = 10.0;
    if((scale>0.1)&&(t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1))){
      t_f = t_f + scale*(R_f*t);
      R_f = R*R_f;
    }else{
      miss_cnt++;
      cout << "scale below 0.1, or incorrect translation. numFrame: " << numFrame << endl;
      // cout << t << endl;
      // cout << R << endl;
    }
    
    // lines for printing results
    myfile << t_f.at<double>(0) << " " << t_f.at<double>(1) << " " << t_f.at<double>(2) << endl;

    // a redetection is triggered in case the number of feautres being trakced go below a particular threshold
    if (prevFeatures.size() < MIN_NUM_FEAT){
      tie(key1, key2, match) = feature_matching(prevImage, currImage);
      tie(points1, points2) = sort_feature(key1, key2, match);
      // KeyPoint::convert(key1, prevFeatures, vector<int>());
      // KeyPoint::convert(key2, currFeatures, vector<int>());
    }

    prevImage = currImage.clone();
    prevFeatures = currFeatures;

    int x = int(t_f.at<double>(0)) + 18;
    int y = int(t_f.at<double>(2)) + 462;
    circle(traj, Point(x, y) ,1, CV_RGB(255,0,0), 2);


    //imshow( "Road facing camera", currImage_c );
    imshow( "Trajectory", traj );

    waitKey(20);

  }

  clock_t end = clock();
  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  cout << "Total time taken: " << elapsed_secs << "s" << endl;
  cout << "miss count: " << miss_cnt << endl;
  
  //cout << R_f << endl;
  //cout << t_f << endl;

  return 0;
}
