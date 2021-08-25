/*

  The MIT License

  Copyright (c) 2015 Avi Singh

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.

*/

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"


#include <iostream>
#include <ctype.h>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>

using namespace cv;
using namespace std;

void featureTracking(Mat img_1, Mat img_2, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status){ 

  //this function automatically gets rid of points for which tracking fails

  vector<float> err;					
  Size winSize=Size(21,21);
  TermCriteria termcrit=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);

  calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

  //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
  int indexCorrection = 0;
  for( int i=0; i<status.size(); i++){
    Point2f pt = points2.at(i- indexCorrection);
    if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0)){
      if((pt.x<0)||(pt.y<0)){
  	status.at(i) = 0;
      }
      points1.erase (points1.begin() + (i - indexCorrection));
      points2.erase (points2.begin() + (i - indexCorrection));
      indexCorrection++;
    }
  }
  
}


void featureDetection(Mat img_1, vector<Point2f>& points1){   //uses FAST as of now, modify parameters as necessary
  vector<KeyPoint> keypoints_1;

  // FAST
  int fast_threshold = 20;
  bool nonmaxSuppression = true;
  FAST(img_1, keypoints_1, fast_threshold, nonmaxSuppression);
  KeyPoint::convert(keypoints_1, points1, vector<int>());


  // draw
  /* Mat dst = Mat::zeros(img_1.rows, img_1.cols, CV_8UC3); */
  /* drawKeypoints(img_1, keypoints_1, dst); */
  /* imshow("FAST", dst); */

  // 
}



vector<KeyPoint> feature_detection(Mat img){
  Mat dst_img;
  vector<KeyPoint> keypoint;
  
  // A-KAZE
  Ptr<cv::AKAZE> akaze = AKAZE::create(AKAZE::DESCRIPTOR_MLDB, 0, 3, 0.001f);
  akaze->detect(img, keypoint);

  // ORB
  /* Ptr<cv::ORB> orb = ORB::create(500, 1.2f, 2); */
  /* orb->detect(img, keypoint); */
  
  
  drawKeypoints(img, keypoint, dst_img, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

  /* imshow("feature_img", dst_img); */
  /* waitKey(); */

  return keypoint;
}


  
tuple<vector<KeyPoint>, vector<KeyPoint>, vector<DMatch>> feature_matching(Mat img_1, Mat img_2){
  Mat dst_img;
  vector<KeyPoint> key1, key2;
  Mat des1, des2;
  
  //// キーポイントの検出 & 特徴量記述の計算
  // A-KAZE
  /* Ptr<AKAZE> akaze = AKAZE::create(AKAZE::DESCRIPTOR_MLDB, 0, 3, 0.001f); */
  /* akaze->detect(img_1, key1); */
  /* akaze->detect(img_2, key2); */
  /* akaze->compute(img_1, key1, des1); */
  /* akaze->compute(img_2, key2, des2); */
  

  // ORB
  Ptr<ORB> orb = ORB::create(30);
  orb->detect(img_1, key1);
  orb->detect(img_2, key2);
  orb->compute(img_1, key1, des1);
  orb->compute(img_2, key2, des2);


  // version問題?
  /* Ptr<FeatureDetector> detector = FeatureDetector::create("ORB"); */
  /* Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create("ORB"); */
  /* Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce"); */
  /* detector->detect(img_1, key1); */
  /* detector->detect(img_2, key2); */
  /* descriptorExtractor->compute(img_1, key1, des1); */
  /* descriptorExtractor->compute(img_2, key2, des2); */

  
  // 特徴量マッチングアルゴリズムの選択  
  Ptr<DescriptorMatcher> hamming = DescriptorMatcher::create("BruteForce-Hamming");

  // 特徴量マッチング
  vector<DMatch> match;
  vector<DMatch> match12, match21;
  hamming->match(des1, des2, match12);
  hamming->match(des2, des1, match21);

  for(size_t i=0; i<match12.size(); i++){
    DMatch m12 = match12[i];
    DMatch m21 = match21[m12.trainIdx];

    if(m21.trainIdx == m12.queryIdx)
      match.push_back(m12);
  }
  cout << "++++++++++++++++++++++++++++++++++\n";
  cout << "m12 <--> m21 match: " << match.size() << endl;


  // 特徴量距離が小さい順にソート
  for(int i=0; i<match.size()-1; i++){
    double min = match[i].distance;
    int n = i;
    for(int j=i+1; j<match.size(); j++){
      if(match[j].distance < min){
	n = j;
	min = match[j].distance;
      }
    }
    swap(match[i], match[n]);
  }
  // 上位n点を残して, 残りを削除
  int n = 20;
  if(match.size() > n)
    match.erase(match.begin()+n, match.end());


  
  // draw
  // drawMatches(img_1, key1, img_2, key2, match, dst_img);

  /* imshow("match feature", dst_img); */
  /* waitKey(); */


  return forward_as_tuple(key1, key2, match);
}



tuple<vector<Point2f>, vector<Point2f>> sort_feature(vector<KeyPoint> key1, vector<KeyPoint> key2, vector<DMatch> match){
  vector<Point2f> points1, points2;

  for(size_t i=0; i<match.size(); i++){
    Point2f p;

    p.x = key1[match[i].queryIdx].pt.x;
    p.y = key1[match[i].queryIdx].pt.y;
    points1.push_back(p);

    p.x = key2[match[i].trainIdx].pt.x;
    p.y = key2[match[i].trainIdx].pt.y;
    points2.push_back(p);
  }
  
  return forward_as_tuple(points1, points2);
}


