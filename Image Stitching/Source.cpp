#include "opencv2\opencv.hpp"
#include "opencv2/nonfree/nonfree.hpp"  
#include "opencv2/legacy/legacy.hpp"
#include "time.h"
#include "math.h"

using namespace cv;
using namespace std;

// Bug: match, coordinate after homogrphy  matrix,  homogrphy  matrix
int main()
{

	Mat image01 = imread("87_2.jpg");
	Mat image02 = imread("87.jpg");
	Mat image01p = Mat(image01.rows, image01.cols, CV_8UC3);

	Mat finalDstImg = Mat(image01.rows * 2, image01.cols * 2, CV_8UC3);

	for (int x = 0; x < image01.rows; x++)
	{
		for (int y = 0; y < image01.cols; y++)
		{
			image01p.at<Vec3b>(x, y) = image01.at<Vec3b>(x, y);
		}
	}

	Mat dstImage01, dstImage02;

	SiftFeatureDetector siftDetector(5000);
	vector<KeyPoint> keyPoint1, keyPoint2;
	siftDetector.detect(image01, keyPoint1);
	siftDetector.detect(image02, keyPoint2);
	
	drawKeypoints(image01, keyPoint1, dstImage01, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(image02, keyPoint2, dstImage02, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

	imshow("KeyPoints of image1", dstImage01);
	imshow("KeyPoints of image2", dstImage02);

	int iterationNum = 200;
	int randomNum = 4;

	vector<KeyPoint>finalKeyPoints;
	vector<KeyPoint>finalKeyPoints2;
	vector<KeyPoint>finalfinalKeyPoints;

	srand((unsigned)time(NULL));
	
	SiftDescriptorExtractor siftDescriptor;
	Mat imageDesc1, imageDesc2;

	siftDescriptor.compute(image01, keyPoint1, imageDesc1);
	siftDescriptor.compute(image02, keyPoint2, imageDesc2);

	BruteForceMatcher<L2<float>> matcher;
	vector<DMatch> matchePoints;
	matcher.match(imageDesc1, imageDesc2, matchePoints, Mat());
	Mat imageOutput;


	float max_dist = 0; float min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < matchePoints.size(); i++)
	{
		float dist = matchePoints[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	std::vector< DMatch > good_matches;

	for (int i = 0; i < matchePoints.size(); i++)
	{
		if (2 * min_dist < 0.02)
		{
			min_dist = 0.01;
		}

		if (matchePoints[i].distance <= 2 * min_dist)
		{
			good_matches.push_back(matchePoints[i]);
		}
	}

	for (int i = 0; i < good_matches.size(); i++)
	{
		finalKeyPoints.push_back(keyPoint1[good_matches[i].queryIdx]);
		finalKeyPoints2.push_back(keyPoint2[good_matches[i].trainIdx]);
	}

	cout << good_matches.size() << endl;

	drawMatches(image01, keyPoint1, image02, keyPoint2, good_matches, imageOutput);

	imshow("Mathch Points", imageOutput);

	float score = 0;
	float min = 1000000;

	Mat Hp;

	for (int i = 0; i < iterationNum; i++)
	{

		vector<int> random;
		vector<KeyPoint> tempPoints;

		// find 4 random samples
		for (int j = 0; j < randomNum; j++)
		{

			int currentRandom = rand() % finalKeyPoints.size();
			bool repeat = false;

			for (int k = 0; k < random.size(); k++)
			{
				if (currentRandom == random[k])
				{
					repeat = true;
				}
			}

			if (!repeat)
			{
				tempPoints.push_back(finalKeyPoints[currentRandom]);
				random.push_back(currentRandom);
			}
			else
			{
				j--;
			}
		}


		for (int j = 0; j < 4; j++)
		{
			
			finalfinalKeyPoints = tempPoints;
		}

		siftDescriptor.compute(image01, finalfinalKeyPoints, imageDesc1);
		siftDescriptor.compute(image02, keyPoint2, imageDesc2);
		matcher.match(imageDesc1, imageDesc2, matchePoints, Mat());


		float A[9][9];

		for (int i = 0; i < 4; i++)
		{
			A[2 * i][0] = -finalfinalKeyPoints[i].pt.x;

			A[2 * i][1] = -finalfinalKeyPoints[i].pt.y;
			A[2 * i][2] = -1;
			A[2 * i][3] = 0;
			A[2 * i][4] = 0;
			A[2 * i][5] = 0;
			A[2 * i][6] = finalfinalKeyPoints[i].pt.x*keyPoint2[matchePoints[i].trainIdx].pt.x;
			A[2 * i][7] = finalfinalKeyPoints[i].pt.y*keyPoint2[matchePoints[i].trainIdx].pt.x;
			A[2 * i][8] = keyPoint2[matchePoints[i].trainIdx].pt.x;

			A[2 * i + 1][0] = 0;
			A[2 * i + 1][1] = 0;
			A[2 * i + 1][2] = 0;
			A[2 * i + 1][3] = -finalfinalKeyPoints[i].pt.x;
			A[2 * i + 1][4] = -finalfinalKeyPoints[i].pt.y;
			A[2 * i + 1][5] = -1;
			A[2 * i + 1][6] = finalfinalKeyPoints[i].pt.x*keyPoint2[matchePoints[i].trainIdx].pt.y;
			A[2 * i + 1][7] = finalfinalKeyPoints[i].pt.y*keyPoint2[matchePoints[i].trainIdx].pt.y;
			A[2 * i + 1][8] = keyPoint2[matchePoints[i].trainIdx].pt.y;
		}

		for (int i = 0; i < 8; i++)
		{
			A[8][i] = 0;
		}

		A[8][8] = 1;

		float b[9][1];

		for (int i = 0; i < 8; i++)
		{
			b[i][0] = 0;
		}
		b[8][0] = 1;

		Mat Ap = Mat(9, 9, CV_32FC1, A);
		Mat B = Mat(9, 1, CV_32FC1, b);

		Mat x = Ap.inv() * B;

		//cout << "x=" << endl << " " << x << endl;

		float H[3][3] =
		{ { x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2, 0) },
		{ x.at<float>(3, 0), x.at<float>(4, 0), x.at<float>(5, 0) },
		{ x.at<float>(6, 0), x.at<float>(7, 0), x.at<float>(8, 0) } };


		Mat Hpp = Mat(3, 3, CV_32FC1, H);

		//cout << "H=" << endl << " " << Hp << endl;

		for (int j = 0; j < good_matches.size(); j++)
		{
			float originFeaturePos[3][1] =
			{ { keyPoint1[good_matches[j].queryIdx].pt.x },
			{ keyPoint1[good_matches[j].queryIdx].pt.y },
			{ 1 } };

			Mat originFeaturePosp = Mat(3, 1, CV_32FC1, originFeaturePos);
			Mat result;
			result = Hpp*originFeaturePosp;

			score += sqrt(pow(((result.at<float>(0, 0) / result.at<float>(2, 0)) - keyPoint2[good_matches[j].trainIdx].pt.x), 2) + pow(((result.at<float>(1, 0) / result.at<float>(2, 0)) - keyPoint2[good_matches[j].trainIdx].pt.y), 2));
		}

		if (score < min)
		{
			min = score;
			Hp = Hpp;
		}

		cout << i << ":" << score << endl;
		score = 0;
	}

	cout << "min" << ":" << min << endl;

	int cons = 200;

	for (int x = 0; x < image01.rows; x++)
	{
		for (int y = 0; y < image01.cols; y++)
		{
			float originPos[3][1] =
			{ { y },
			{ x },
			{ 1 } };

			Mat originPosp = Mat(3, 1, CV_32FC1, originPos);
			Mat result;
			result = Hp*originPosp;

			float xp = (result.at<float>(0, 0) / result.at<float>(2, 0)) + cons * 2, yp = (result.at<float>(1, 0) / result.at<float>(2, 0)) + cons;

			if (xp >= 0 && xp < finalDstImg.cols && yp >= 0 && yp < finalDstImg.rows)
			{
				finalDstImg.at<Vec3b>(yp, xp) = image01p.at<Vec3b>(x, y);
			}
		}
	}

	vector<float> featureAfterH_x, featureAfterH_y;
	float matchFeature_x = keyPoint2[matchePoints[0].trainIdx].pt.y, matchFeature_y = keyPoint2[matchePoints[0].trainIdx].pt.x; // origin
	float test_x = keyPoint2[matchePoints[2].trainIdx].pt.y, test_y = keyPoint2[matchePoints[2].trainIdx].pt.x;
	
	cout << "match coordinate:" << matchFeature_x << "," << matchFeature_y << endl;

	for (int i = 0; i < finalfinalKeyPoints.size(); i++)
	{
		float originPos[3][1] =
		{ { finalfinalKeyPoints[i].pt.x },
		{ finalfinalKeyPoints[i].pt.y },
		{ 1 } };

		Mat originPosp = Mat(3, 1, CV_32FC1, originPos);
		Mat result;
		result = Hp*originPosp;

		float xp = (result.at<float>(0, 0) / result.at<float>(2, 0)) + cons * 2, yp = (result.at<float>(1, 0) / result.at<float>(2, 0)) + cons;
		featureAfterH_x.push_back(yp); // origin
		featureAfterH_y.push_back(xp); // origin

		cout << "Feature  after transformation:" << featureAfterH_x[i] << "," << featureAfterH_y[i] << endl;
	}

	for (int i = 0; i < 1; i++)
	{

		for (int x = 1; x < finalDstImg.rows - 1; x++)
		{
			for (int y = 1; y < finalDstImg.cols - 1; y++)
			{
				if (finalDstImg.at<Vec3b>(x, y) == finalDstImg.at<Vec3b>(0, 0))
				{
					
					if (finalDstImg.at<Vec3b>(x, y + 1) != finalDstImg.at<Vec3b>(0, 0))
					{
						finalDstImg.at<Vec3b>(x, y) = finalDstImg.at<Vec3b>(x, y + 1);
					}
					
					else if (finalDstImg.at<Vec3b>(x + 1, y) != finalDstImg.at<Vec3b>(0, 0))
					{
						finalDstImg.at<Vec3b>(x, y) = finalDstImg.at<Vec3b>(x + 1, y);
					}
					
					else if (finalDstImg.at<Vec3b>(x, y - 1) != finalDstImg.at<Vec3b>(0, 0))
					{
						finalDstImg.at<Vec3b>(x, y) = finalDstImg.at<Vec3b>(x, y - 1);
					}
					else if (finalDstImg.at<Vec3b>(x - 1, y) != finalDstImg.at<Vec3b>(0, 0))
					{
						finalDstImg.at<Vec3b>(x, y) = finalDstImg.at<Vec3b>(x - 1, y);
					}
					
				}
			}
		}
	}
	
	imshow("Output", finalDstImg);

	Mat mergeImg = Mat(image01.rows * 2, 2000, CV_8UC3);
	
	float offset_x = featureAfterH_x[0] - matchFeature_x, offset_y = featureAfterH_y[0] - matchFeature_y;
	float test_off_x = featureAfterH_x[2] - test_x, test_off_y = featureAfterH_y[2] - test_y;

	cout << "offset:" << offset_x << "," << offset_y << endl;
	cout << "testoffset:" << test_off_x << "," << test_off_y << endl;

	for (int x = 0; x < finalDstImg.rows; x++)
	{
		for (int y = 0; y < finalDstImg.cols; y++)
		{
			mergeImg.at<Vec3b>(x, y) = finalDstImg.at<Vec3b>(x, y);

		}
	}

	for (int x = 0; x < image02.rows; x++)
	{
		for (int y = 0; y < image02.cols; y++)
		{
			mergeImg.at<Vec3b>(x + offset_x, y + offset_y) = image02.at<Vec3b>(x, y);

		}
	}
	
	imshow("Merge", mergeImg);
	
	waitKey(0);
	return 0;
}