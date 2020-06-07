#include "opencv2\opencv.hpp"
#include "opencv2/nonfree/nonfree.hpp"  
#include "opencv2/legacy/legacy.hpp"
#include "time.h"
#include "math.h"

using namespace cv;
using namespace std;

Mat homomat(vector<KeyPoint> &_4matchKeyPoints01, vector<KeyPoint> &_4matchKeyPoints02)
{
	Mat A = Mat(8, 9, CV_32F);
	
	for (int i = 0; i < 4; i++)
	{
		A.at<float>(i * 2, 0) = _4matchKeyPoints01[i].pt.x;
		A.at<float>(i * 2, 1) = _4matchKeyPoints01[i].pt.y;
		A.at<float>(i * 2, 2) = 1.0;
		A.at<float>(i * 2, 3) = 0.0;
		A.at<float>(i * 2, 4) = 0.0;
		A.at<float>(i * 2, 5) = 0.0;
		A.at<float>(i * 2, 6) = -_4matchKeyPoints01[i].pt.x * _4matchKeyPoints02[i].pt.x;
		A.at<float>(i * 2, 7) = -_4matchKeyPoints01[i].pt.y * _4matchKeyPoints02[i].pt.x;
		A.at<float>(i * 2, 8) = -_4matchKeyPoints02[i].pt.x;

		A.at<float>(i * 2 + 1, 0) = 0.0;
		A.at<float>(i * 2 + 1, 1) = 0.0;
		A.at<float>(i * 2 + 1, 2) = 0.0;
		A.at<float>(i * 2 + 1, 3) = _4matchKeyPoints01[i].pt.x;
		A.at<float>(i * 2 + 1, 4) = _4matchKeyPoints01[i].pt.y;
		A.at<float>(i * 2 + 1, 5) = 1.0;
		A.at<float>(i * 2 + 1, 6) = -_4matchKeyPoints01[i].pt.x * _4matchKeyPoints02[i].pt.y;
		A.at<float>(i * 2 + 1, 7) = -_4matchKeyPoints01[i].pt.y * _4matchKeyPoints02[i].pt.y;
		A.at<float>(i * 2 + 1, 8) = -_4matchKeyPoints02[i].pt.y;
	}
	
	Mat h = Mat(3, 3, CV_32F);

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			h.at<float>(i, j) = SVD(A).vt.at<float>(7, i * 3 + j);
		}
	}
	
	return h;
}

float computeScore(Mat &H, vector<KeyPoint> &matchKeyPoints01, vector<KeyPoint> &matchKeyPoints02)
{
	float score = 0.0;

	for (int i = 0; i < matchKeyPoints01.size(); i++)
	{

		Mat sourcePoint = Mat(3, 1, CV_32F), targetPoint = Mat(3, 1, CV_32F);
		sourcePoint.at<float>(0, 0) = matchKeyPoints01[i].pt.x;
		sourcePoint.at<float>(1, 0) = matchKeyPoints01[i].pt.y;
		sourcePoint.at<float>(2, 0) = 1.0;

		targetPoint = H * sourcePoint;

		float targetPoint_x = targetPoint.at<float>(0, 0) / targetPoint.at<float>(2, 0);
		float targetPoint_y = targetPoint.at<float>(1, 0) / targetPoint.at<float>(2, 0);

		score += sqrt(pow(matchKeyPoints02[i].pt.x - targetPoint_x, 2) + pow(matchKeyPoints02[i].pt.y - targetPoint_y, 2));
	}

	return score;
}

Mat featheringBlending(Mat &image01, Mat &image02)
{
	Mat blend = Mat(image01.rows, image01.cols, CV_8UC3);

	for (float i = 0; i < image01.rows; i++)
	{
		for (float j = 0; j < image01.cols; j++)
		{
			float w = j / (image01.cols - 1);
			Vec3f a = image01.at<Vec3b>(i, j);
			Vec3f b = image02.at<Vec3b>(i, j);

			if (a == Vec3f(0, 0, 0))
			{
				w = 1.0;
			}
			else if (b == Vec3f(0, 0, 0))
			{
				w = 0.0;
			}

			blend.at<Vec3b>(i, j) = (1.0 - w) * a + w * b;
		}
	}

	return blend;
}

Mat pyraidBlending(Mat &image01, Mat &image02)
{

	vector<Mat> GP01, GP02, LP01, LP02, LS, LSP;

	// build gaussian pyramid
	buildPyramid(image01, GP01, 7);
	buildPyramid(image02, GP02, 7);

	LP01.push_back(GP01[GP01.size() - 2]);
	LP02.push_back(GP02[GP02.size() - 2]);
	
	// build laplacian pyramid
	for (int i = GP01.size() - 2; i >= 1; i--)
	{
		Mat tmp01, tmp02, lap01, lap02;

		pyrUp(GP01[i], tmp01);
		resize(tmp01, tmp01, GP01[i - 1].size());
		subtract(GP01[i - 1], tmp01, lap01);

		pyrUp(GP02[i], tmp02);
		resize(tmp02, tmp02, GP02[i - 1].size());
		subtract(GP02[i - 1], tmp02, lap02);

		LP01.push_back(lap01);
		LP02.push_back(lap02);
	}

	// combine laplacian pyramid
	for (int i = 0; i < LP01.size(); i++)
	{
		Mat H;
		Mat leftHalf = LP01[i](Rect(0, 0, LP01[i].cols / 2, LP01[i].rows));
		Mat rightHalf = LP02[i](Rect(LP02[i].cols / 2, 0, LP02[i].cols / 2, LP02[i].rows));
		hconcat(leftHalf, rightHalf, H);

		LS.push_back(H);
	}

	LSP.push_back(LS[0]);

	// reconstruct from combined laplacian pyramid
	for (int i = 0; i < LS.size() - 1; i++)
	{

		Mat tmp, tmpLSP;
		pyrUp(LSP[i], tmp);
		resize(tmp, tmp, LS[i + 1].size());
		
		tmpLSP = LS[i + 1] + tmp;

		LSP.push_back(tmpLSP);
		
	}

	return LSP.back();
}


void warp(Mat image01, Mat image02, Mat &H, string name)
{
	
	float min_x = INT_MAX, min_y = INT_MAX;
	float max_x = INT_MIN, max_y = INT_MIN;

	// detect image boundary
	for (float y = 0; y < image01.rows; y++)
	{
		for (float x = 0; x < image01.cols; x++)
		{
			Mat sourcePoint = Mat(3, 1, CV_32F), targetPoint = Mat(3, 1, CV_32F);
			sourcePoint.at<float>(0, 0) = x;
			sourcePoint.at<float>(1, 0) = y;
			sourcePoint.at<float>(2, 0) = 1.0;

			targetPoint = H * sourcePoint;
			float targetPoint_x = targetPoint.at<float>(0, 0) / targetPoint.at<float>(2, 0);
			float targetPoint_y = targetPoint.at<float>(1, 0) / targetPoint.at<float>(2, 0);

			if (targetPoint_x < min_x)
			{
				min_x = targetPoint_x;
			}

			if (targetPoint_y < min_y)
			{
				min_y = targetPoint_y;
			}

			if (targetPoint_x > max_x)
			{
				max_x = targetPoint_x;
			}

			if (targetPoint_y > max_y)
			{
				max_y = targetPoint_y;
			}
		}
	}

	Mat warpImg = Mat(max_y - min_y + 1, max_x - min_x + 1, CV_8UC3, Scalar(0, 0, 0));
	float offsetX, offsetY;
	int cornerX, cornerY, baseCornerX, baseCornerY;
	int type;

	if (min_x <= 0 && min_y <= 0)
	{
		offsetX = -min_x;
		offsetY = -min_y;

		cornerX = image02.cols;
		cornerY = image02.rows;

		baseCornerX = warpImg.cols;
		baseCornerY = warpImg.rows;

		type = 0;
	}
	else if (max_x > 0 && min_y <= 0)
	{
		offsetX = warpImg.cols - max_x;
		offsetY = -min_y;

		cornerX = 0;
		cornerY = image02.rows;

		baseCornerX = 0;
		baseCornerY = warpImg.rows;

		type = 1;
	}
	else if (min_x <= 0 && max_y > 0)
	{
		offsetX = -min_x;
		offsetY = warpImg.rows - max_y;

		cornerX = image02.cols;
		cornerY = 0;

		baseCornerX = warpImg.cols;
		baseCornerY = 0;

		type = 2;
	}
	else if (max_x > 0 && max_y > 0)
	{
		offsetX = warpImg.cols - max_x;
		offsetY = warpImg.rows - max_y;

		cornerX = 0;
		cornerY = 0;

		baseCornerX = 0;
		baseCornerY = 0;

		type = 3;
	}

	cout << "type: " << type << endl;

	// transform image
	for (float y = 0; y < image01.rows; y++)
	{
		for (float x = 0; x < image01.cols; x++)
		{

			Mat sourcePoint = Mat(3, 1, CV_32F), targetPoint = Mat(3, 1, CV_32F);
			// pt ®y¼Ð¬Û¤Ï
			sourcePoint.at<float>(0, 0) = x;
			sourcePoint.at<float>(1, 0) = y;
			sourcePoint.at<float>(2, 0) = 1.0;

			targetPoint =  H * sourcePoint;

			float targetPoint_x = targetPoint.at<float>(0, 0) / targetPoint.at<float>(2, 0);
			float targetPoint_y = targetPoint.at<float>(1, 0) / targetPoint.at<float>(2, 0);

			targetPoint_x += offsetX;
			targetPoint_y += offsetY;

			if (targetPoint_x >= 0 && targetPoint_y >= 0 && targetPoint_x < warpImg.cols && targetPoint_y < warpImg.rows)
				warpImg.at<Vec3b>(targetPoint_y, targetPoint_x) = image01.at<Vec3b>(y, x);
			else
				cout << targetPoint_x << "," << targetPoint_y << endl;
		}
	}

	
	int splattingSize = 3;

	// interpolation
	for (int i = 0; i < warpImg.rows - splattingSize; i++)
	{
		for (int j = 0; j < warpImg.cols - splattingSize; j++)
		{
			if (warpImg.at<Vec3b>(i, j) == Vec3b(0, 0, 0))
			{
				for (int ki = 0; ki < splattingSize; ki++)
				{
					bool isBreak = false;
					for (int kj = 0; kj < splattingSize; kj++)
					{
						
						if (warpImg.at<Vec3b>(i + ki, j + kj) != Vec3b(0, 0, 0))
						{
							warpImg.at<Vec3b>(i, j) = warpImg.at<Vec3b>(i + ki, j + kj);
							isBreak = true;
							break;
						}
					}

					if (isBreak)
						break;
				}
			}
		}
	}
	
	cornerX += offsetX;
	cornerY += offsetY;

	int combineImgRows;
	int combineImgCols;

	if (cornerY > baseCornerY || cornerY < 0)
	{
		combineImgRows = warpImg.rows + abs(cornerY - baseCornerY);
	}
	else
	{
		combineImgRows = warpImg.rows;
	}

	if (cornerX > baseCornerX || cornerX < 0)
	{
		combineImgCols = warpImg.cols + abs(cornerX - baseCornerX);
	}
	else
	{
		combineImgCols = warpImg.cols;
	}


	Mat combineImg = Mat(combineImgRows, combineImgCols, CV_8UC3, Scalar(0, 0, 0));

	int blendingLeftTopCornerI = -1, blendingLeftTopCornerJ = -1;
	int blendingRightBottomCornerI = -1, blendingRightBottomCornerJ = -1;

	int warpLTCX = -1, warpLTCY = -1;
	int warpRBCX = -1, warpRBCY = -1;

	int _02LTCX = -1, _02LTCY = -1;
	int _02RBCX = -1, _02RBCY = -1;

	// blend image
	for (int i = 0; i < combineImg.rows; i++)
	{
		for (int j = 0; j < combineImg.cols; j++)
		{

			if (type == 0)
			{
				bool overlap1 = false, overlap2 = false;

				if (i < warpImg.rows && j < warpImg.cols)
				{
					combineImg.at<Vec3b>(i, j) = warpImg.at<Vec3b>(i, j);
					overlap1 = true;
				}

				if (i >= offsetY && j >= offsetX)
				{
					if (i < offsetY + image02.rows && j < offsetX + image02.cols)
					{
						combineImg.at<Vec3b>(i, j) = image02.at<Vec3b>(i - offsetY, j - offsetX);
						overlap2 = true;
					}
				}

				if (overlap1 && overlap2 && blendingLeftTopCornerI == -1)
				{
					blendingLeftTopCornerI = i;
					blendingLeftTopCornerJ = j;

					warpLTCX = j;
					warpLTCY = i;

					_02LTCX = j - offsetX;
					_02LTCY = i - offsetY;
				}

				if (overlap1 && overlap2)
				{
					blendingRightBottomCornerI = i;
					blendingRightBottomCornerJ = j;

					warpRBCX = j;
					warpRBCY = i;

					_02RBCX = j - offsetX;
					_02RBCY = i - offsetY;
				}
			}
			else if (type == 1)
			{
				bool overlap1 = false, overlap2 = false;

				if (i >= 0 && j >= abs(cornerX - baseCornerX))
				{
					if (i < warpImg.rows && j < abs(cornerX - baseCornerX) + warpImg.cols)
					{
						combineImg.at<Vec3b>(i, j) = warpImg.at<Vec3b>(i, j - abs(cornerX - baseCornerX));
						overlap1 = true;
					}
				}

				if (i >= offsetY && j >= 0)
				{
					if (i < offsetY + image02.rows && j < image02.cols)
					{
						combineImg.at<Vec3b>(i, j) = image02.at<Vec3b>(i - offsetY, j);
						overlap2 = true;
					}
				}

				if (overlap1 && overlap2 && blendingLeftTopCornerI == -1)
				{
					blendingLeftTopCornerI = i;
					blendingLeftTopCornerJ = j;

					warpLTCX = j - abs(cornerX - baseCornerX);
					warpLTCY = i;

					_02LTCX = j;
					_02LTCY = i - offsetY;
				}

				if (overlap1 && overlap2)
				{
					blendingRightBottomCornerI = i;
					blendingRightBottomCornerJ = j;

					warpRBCX = j - abs(cornerX - baseCornerX);
					warpRBCY = i;

					_02RBCX = j;
					_02RBCY = i - offsetY;
				}
			}
			else if (type == 2)
			{
				bool overlap1 = false, overlap2 = false;

				if (i >= abs(cornerY - baseCornerY) && j >= 0)
				{
					if (i < abs(cornerY - baseCornerY) + warpImg.rows && j < warpImg.cols)
					{
						combineImg.at<Vec3b>(i, j) = warpImg.at<Vec3b>(i - abs(cornerY - baseCornerY), j);
						overlap1 = true;
					}
				}

				if (i >= 0 && j >= offsetX)
				{
					if (i < image02.rows && j < offsetX + image02.cols)
					{
						combineImg.at<Vec3b>(i, j) = image02.at<Vec3b>(i, j - offsetX);
						overlap2 = true;
					}
				}

				if (overlap1 && overlap2 && blendingLeftTopCornerI == -1)
				{
					blendingLeftTopCornerI = i;
					blendingLeftTopCornerJ = j;

					warpLTCX = j;
					warpLTCY = i - abs(cornerY - baseCornerY);

					_02LTCX = j - offsetX;
					_02LTCY = i;
				}

				if (overlap1 && overlap2)
				{
					blendingRightBottomCornerI = i;
					blendingRightBottomCornerJ = j;

					warpRBCX = j;
					warpRBCY = i - abs(cornerY - baseCornerY);

					_02RBCX = j - offsetX;
					_02RBCY = i;
				}
			}
			else if (type == 3)
			{
				bool overlap1 = false, overlap2 = false;

				if (i >= abs(cornerY - baseCornerY) && j >= abs(cornerX - baseCornerX))
				{
					if (i < abs(cornerY - baseCornerY) + warpImg.rows && j < abs(cornerX - baseCornerX) + warpImg.cols)
					{
						combineImg.at<Vec3b>(i, j) = warpImg.at<Vec3b>(i - abs(cornerY - baseCornerY), j - abs(cornerX - baseCornerX));
						overlap1 = true;
					}
				}

				if (i >= 0 && j >= 0)
				{
					if (i < image02.rows && j < image02.cols)
					{
						combineImg.at<Vec3b>(i, j) = image02.at<Vec3b>(i, j);
						overlap2 = true;
					}
				}

				if (overlap1 && overlap2 && blendingLeftTopCornerI == -1)
				{
					blendingLeftTopCornerI = i;
					blendingLeftTopCornerJ = j;

					warpLTCX = j - abs(cornerX - baseCornerX);
					warpLTCY = i - abs(cornerY - baseCornerY);

					_02LTCX = j;
					_02LTCY = i;
				}

				if (overlap1 && overlap2)
				{
					blendingRightBottomCornerI = i;
					blendingRightBottomCornerJ = j;

					warpRBCX = j - abs(cornerX - baseCornerX);
					warpRBCY = i - abs(cornerY - baseCornerY);

					_02RBCX = j;
					_02RBCY = i;
				}
			}
		}
	}

	Mat overlap01 = warpImg(Rect(warpLTCX, warpLTCY, warpRBCX - warpLTCX + 1, warpRBCY - warpLTCY + 1));
	Mat overlap02 = image02(Rect(_02LTCX, _02LTCY, _02RBCX - _02LTCX + 1, _02RBCY - _02LTCY + 1));
	Mat blend;
	
	if (type == 0 || type == 2)
		blend = featheringBlending(overlap01, overlap02);
	else
		blend = featheringBlending(overlap02, overlap01);


	for (int i = blendingLeftTopCornerI; i <= blendingRightBottomCornerI; i++)
	{
		for (int j = blendingLeftTopCornerJ; j <= blendingRightBottomCornerJ; j++)
		{
			combineImg.at<Vec3b>(i, j) = blend.at<Vec3b>(i - blendingLeftTopCornerI, j - blendingLeftTopCornerJ);
		}
	}

	//Mat result;
	//warpPerspective(image01, result, H, Size(image01.cols, image01.rows));

	//imshow(name + "WP", result);
	imshow(name, warpImg);
	imshow(name + "combine", combineImg);
	imwrite("combine.jpg", combineImg);
}


int main()
{

	Mat image01 = imread("l1+l2.jpg");
	Mat image02 = imread("l3+l4.jpg");

	// dectect SIFT
	Mat image4keyPoints01, image4keyPoints02;

	SiftFeatureDetector siftDetector(1000);
	vector<KeyPoint> keyPoints01, keyPoints02;
	siftDetector.detect(image01, keyPoints01);
	siftDetector.detect(image02, keyPoints02);
	
	drawKeypoints(image01, keyPoints01, image4keyPoints01, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(image02, keyPoints02, image4keyPoints02, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

	imshow("KeyPoints of image1", image4keyPoints01);
	imshow("KeyPoints of image2", image4keyPoints02);

	int iterationNum = 20000;
	int randomNum = 4;

	vector<KeyPoint>matchKeyPoints01;
	vector<KeyPoint>matchKeyPoints02;
	vector<KeyPoint>finalKeyPoints;

	srand((unsigned)time(NULL));
	
	// describe SIFT
	SiftDescriptorExtractor siftDescriptor;
	Mat imageDesc01, imageDesc02;

	siftDescriptor.compute(image01, keyPoints01, imageDesc01);
	siftDescriptor.compute(image02, keyPoints02, imageDesc02);

	// SIFT matching
	BruteForceMatcher<L2<float>> matcher;
	BFMatcher matcher02(NORM_L2);

	vector<DMatch> goodMatchPoints;
	vector<vector<DMatch>> knnMatchPoints;
	matcher02.knnMatch(imageDesc01, imageDesc02, knnMatchPoints, 3);
	Mat image4matchPoint;
	vector<Point2f> srcPt, dstPt;

	for (int i = 0; i < knnMatchPoints.size(); i++)
	{
		float ratioDistance = knnMatchPoints[i][0].distance / knnMatchPoints[i][1].distance;
		if (ratioDistance < 0.6)
		{
			goodMatchPoints.push_back(knnMatchPoints[i][0]);
			matchKeyPoints01.push_back(keyPoints01[goodMatchPoints.back().queryIdx]);
			matchKeyPoints02.push_back(keyPoints02[goodMatchPoints.back().trainIdx]);

			srcPt.push_back(keyPoints01[goodMatchPoints.back().queryIdx].pt);
			dstPt.push_back(keyPoints02[goodMatchPoints.back().trainIdx].pt);
		}
	}

	drawKeypoints(image01, matchKeyPoints01, image4keyPoints01, Scalar(255, 0, 0), DrawMatchesFlags::DEFAULT);
	drawKeypoints(image02, matchKeyPoints02, image4keyPoints02, Scalar(255, 0, 0), DrawMatchesFlags::DEFAULT);

	imshow("KeyPoints of image1", image4keyPoints01);
	imshow("KeyPoints of image2", image4keyPoints02);

	drawMatches(image01, keyPoints01, image02, keyPoints02, goodMatchPoints, image4matchPoint);
	imshow("matchPoints", image4matchPoint);

	float minScore = INT_MAX;
	Mat bestH;

	// RANSAC
	for (int i = 0; i < iterationNum; i++)
	{
		vector<KeyPoint> _4matchKeyPoints01, _4matchKeyPoints02;
		set<int> tmpMatchPointsIdx;
		
		while (_4matchKeyPoints01.size() < randomNum)
		{
			int currentRandom = rand() % matchKeyPoints01.size();

			if (!tmpMatchPointsIdx.count(currentRandom))
			{
				_4matchKeyPoints01.push_back(matchKeyPoints01[currentRandom]);
				_4matchKeyPoints02.push_back(matchKeyPoints02[currentRandom]);
				
				tmpMatchPointsIdx.insert(currentRandom);
			}
		}

		Mat H = homomat(_4matchKeyPoints01, _4matchKeyPoints02);
		
		H = H / H.at<float>(2, 2);

		float score = computeScore(H, matchKeyPoints01, matchKeyPoints02);
		//cout << score << endl;

		if (score < minScore)
		{
			cout << i << ":" << score << endl;
			minScore = score;
			bestH = H;
		}
		
	}
	Mat H_API = findHomography(srcPt, dstPt);
	Mat H_API5;
	H_API.convertTo(H_API5, CV_32FC1);
	float correctScore = computeScore(H_API5, matchKeyPoints01, matchKeyPoints02);

	cout << "myMinScore: " << minScore << endl;
	cout << "API minScore: " << correctScore << endl;

	cout << "good match size: " << matchKeyPoints02.size() << endl;

	cout << "myH: " << bestH << endl;
	cout << "APIH: " << H_API5 << endl;

	warp(image01, image02, bestH, "mine");
	//warp(image01, image02, H_API5, "API");


	waitKey(0);
}