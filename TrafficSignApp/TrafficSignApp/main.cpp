#include <stdio.h>
#include <iostream>
#include <vector>
#include <time.h>

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/calib3d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

Mat surf_detection(Mat img_1, Mat img_2, Mat imgObject); /** @function main */

Mat objectDescriptor(Mat imgObject);
void saveToFile();

std::vector<KeyPoint> keyPointsObject;

int largestArea = 0;
int largestContourIndex = 0;
Rect boundRect;

RNG rng(12345);
vector<vector<Point> > contours;
vector<Vec4i> hierarchy;
Mat imgHSV;
Mat imgThresholded;

int main(int argc, char** argv)
{
	VideoCapture cap("C:\\sceneOld.mp4");

	//Save to file

	if (!cap.isOpened())  // if not success, exit program
	{
		cout << "Cannot open the web cam" << endl;
		return -1;
	}

	int iLowH = 65;
	int iHighH = 180;

	int iLowS = 150;
	int iHighS = 255;

	int iLowV = 60;
	int iHighV = 255;

	int iLastX = -1;
	int iLastY = -1;

	//Capture a temporary image from the camera
	Mat imgTmp;
	cap.read(imgTmp);

	Mat imgObject = imread("C:\\objectStop.png", CV_LOAD_IMAGE_GRAYSCALE); // read template image
	Mat objectDescript = objectDescriptor(imgObject);

	clock_t init, final;

	int index = 0;
	string filePathOrg = "D:\\Original\\Original_.jpg";
	string filePathHSV = "D:\\HSV\\HSV_.jpg";
	string filePathCnt = "D:\\Cnt\\Cnt_.jpg";
	string filePathFtr = "D:\\Ftr\\Ftr_.jpg";

	int length;

	while (true)
	{
		init = clock();
		Mat imgOriginal;

		bool bSuccess = cap.read(imgOriginal); // read a new frame from video

		if (!bSuccess) //if not success, break loop
		{
			cout << "Cannot read a frame from video stream" << endl;
			break;
		}

		cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

		inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image
			
			////morphological opening (remove small objects from the foreground)
		erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

		//erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		//dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

		Mat dst(imgOriginal.rows, imgOriginal.cols, CV_32FC1, Scalar::all(0));

		findContours(imgThresholded, contours, hierarchy, CV_RETR_EXTERNAL, CHAIN_APPROX_TC89_KCOS);

		for (int i = 0; i < contours.size(); i++)
		{
			double item = contourArea(contours[i], false);

			if (8000 < item && item < 50000)
			{
				boundRect = boundingRect(contours[i]);

				Scalar color(255, 255, 255);
				drawContours(dst, contours, i, color, CV_FILLED, 8, hierarchy);
				rectangle(imgOriginal, boundRect, Scalar(0, 255, 0), 1, 8, 0);

				//imshow("Largest Contour", imgOriginal(boundRect));

				Mat gray = imgOriginal(boundRect);
				cvtColor(gray, gray, COLOR_BGR2GRAY);

				Mat out = surf_detection(objectDescript, gray, imgObject);

				imshow("Feature Detection", out);

				//cout << item << endl;
			}
		}

		imshow("Original Contour Selected", imgOriginal); //show the original+contour Selected image

		if (waitKey(1) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}
		index++;
		final = clock() - init;

		cout << "Iteration: " << (double)final / ((double)CLOCKS_PER_SEC) << endl;
	}// End while()
	return 0;
}

Mat objectDescriptor(Mat imgObject)
{
	Ptr <SURF> detector = SURF::create(400);
	Ptr <SURF> extractor = SURF::create();

	detector->detect(imgObject, keyPointsObject);

	Mat descriptorObject;
	extractor->compute(imgObject, keyPointsObject, descriptorObject);

	return descriptorObject;
}

Mat surf_detection(Mat descriptors_object, Mat img_scene, Mat img_object)
{
	Mat img_matches;
	//-- Step 1: Detect the keypoints using SURF Detector
	Ptr <SURF> detector = SURF::create(400);

	std::vector<KeyPoint> keypoints_scene;

	detector->detect(img_scene, keypoints_scene);

	//-- Step 2: Calculate descriptors (feature vectors)
	Ptr <SURF> extractor = SURF::create();

	Mat descriptors_scene;

	extractor->compute(img_scene, keypoints_scene, descriptors_scene);

	std::vector<DMatch> matches;
	//-- Step 3: Matching descriptor vectors using Brute-Force matcher
	// matching descriptors
	BFMatcher matcher = BFMatcher(NORM_L2, false);

	matcher.match(descriptors_object, descriptors_scene, matches);

	//cout << "Matches size: " << matches.size() << endl;

	double max_dist = 0; double min_dist = 100;

	////-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < descriptors_object.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	//printf("-- Max dist : %f \n", max_dist);
	//printf("-- Min dist : %f \n", min_dist);

	////-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
	std::vector< DMatch > good_matches;

	for (int i = 0; i < descriptors_object.rows; i++)
	{
		if (matches[i].distance < 3 * min_dist)
		{
			good_matches.push_back(matches[i]);
		}
	}

	drawMatches(img_object, keyPointsObject, img_scene, keypoints_scene,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//-- Localize the object
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	for (int i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		obj.push_back(keyPointsObject[good_matches[i].queryIdx].pt);
		scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
	}

	Mat H = findHomography(obj, scene, CV_RANSAC);

	//-- Get the corners from the image_1 ( the object to be "detected" )
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0, 0);
	obj_corners[1] = cvPoint(img_object.cols, 0);
	obj_corners[2] = cvPoint(img_object.cols, img_object.rows);
	obj_corners[3] = cvPoint(0, img_object.rows);
	std::vector<Point2f> scene_corners(4);


	//perspectiveTransform(obj_corners, scene_corners, H);

	////-- Draw lines between the corners (the mapped object in the scene - image_2 )
	//line(img_matches, scene_corners[0] + Point2f(img_object.cols, 0), scene_corners[1] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
	//line(img_matches, scene_corners[1] + Point2f(img_object.cols, 0), scene_corners[2] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
	//line(img_matches, scene_corners[2] + Point2f(img_object.cols, 0), scene_corners[3] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
	//line(img_matches, scene_corners[3] + Point2f(img_object.cols, 0), scene_corners[0] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);

	//-- Show detected matches
	//imshow("Good Matches & Object detection", img_matches);
	//waitKey(500);

	//return img_matches;
	return img_matches;
}

void saveToFile()
{
	//filePathFtr.insert(11, to_string(index));
	//imwrite(filePathFtr, out);
	//length = to_string(index).size();
	//filePathFtr.erase(11, length);
}