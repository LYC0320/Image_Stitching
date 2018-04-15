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

	Mat image01 = imread("1.jpg");
	Mat image02 = imread("2.jpg");
	//Mat image01p = image01;
	Mat image01p = Mat(image01.rows, image01.cols, CV_8UC3);

	Mat finalDstImg = Mat(image01.rows * 2, image01.cols * 2, CV_8UC3);

	
	for (int x = 0; x < image01.rows; x++)
	{
		for (int y = 0; y < image01.cols; y++)
		{
			image01p.at<Vec3b>(x, y) = image01.at<Vec3b>(x, y);
		}
	}
	

	//imshow("KeyPoints of image122", image01p);

	Mat dstImage01, dstImage02;
	
	//Mat image1, image2;
	//GaussianBlur(image01, image1, Size(3, 3), 0.5);
	//GaussianBlur(image02, image2, Size(3, 3), 0.5);
	
	SiftFeatureDetector siftDetector(5000);
	vector<KeyPoint> keyPoint1, keyPoint2;
	siftDetector.detect(image01, keyPoint1);
	siftDetector.detect(image02, keyPoint2);
	
	drawKeypoints(image01, keyPoint1, dstImage01, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(image02, keyPoint2, dstImage02, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	//namedWindow("KeyPoints of image1", 0);
	//namedWindow("KeyPoints of image2", 0);

	imshow("KeyPoints of image1", dstImage01);
	imshow("KeyPoints of image2", dstImage02);

	/*
	for (int i = 0; i < keyPoint1.size(); i++)
	{
		cout << keyPoint1[i].angle << ":" << keyPoint1[i].pt << endl;
	}

	cout << "--------------" << endl;

	for (int i = 0; i < keyPoint2.size(); i++)
	{
		cout << keyPoint2[i].angle << ":" << keyPoint2[i].pt << endl;
		
	}
	*/
	

	int iteraationNum = 5000;
	int randomNum = 4;
	float threshold = 20;
	float finalA, finalB, FinalC;
	float max = 0; 
	vector<KeyPoint>finalKeyPoints;
	vector<DMatch>finalKeyPoints2;
	vector<KeyPoint>finalfinalKeyPoints;

	srand((unsigned)time(NULL));

	for (int i = 0; i < iteraationNum; i++)
	{
		vector<KeyPoint> tempPoints;
		vector<int> random;
		float xMean = 0, yMean = 0;
		float matrixA[4][2];
		float a = 0, b = 0, c = 0, d = 0;
		float trA, detA;
		float L1, L2p;


		// find 8 random samples
		for (int j = 0; j < randomNum; j++)
		{


			int currentRandom = rand() % keyPoint1.size();
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
				tempPoints.push_back(keyPoint1[currentRandom]);
				random.push_back(currentRandom);
			}
			else
			{
				j--;
			}
		}


		for (int k = 0; k < tempPoints.size(); k++)
		{
			//cout <<"Sample:"<< tempPoints[k].angle << ":" << tempPoints[k].pt << endl;
			xMean += tempPoints[k].pt.x;
			yMean += tempPoints[k].pt.y;
		}

		xMean = xMean / randomNum;
		yMean = yMean / randomNum;

		//cout << "------------" << endl;
		//cout << "Mean:" << xMean << "," << yMean << endl;
		//cout << "------------" << endl;
		
		for (int j = 0; j < randomNum; j++)
		{
			matrixA[j][0] = tempPoints[j].pt.x - xMean;
			matrixA[j][1] = tempPoints[j].pt.y - yMean;

			//cout <<"Matrix:" <<matrixA[j][0] << "," << matrixA[j][1] << endl;
		}

		//cout << "------------" << endl;


		// AT*A
		for (int j = 0; j < randomNum; j++)
		{
			a += matrixA[j][0] * matrixA[j][0];
			b += matrixA[j][0] * matrixA[j][1];
			c += matrixA[j][1] * matrixA[j][0];
			d += matrixA[j][1] * matrixA[j][1];
		}

		//cout <<"AT*A:"<< a << "," << b << endl;
		//cout << "AT*A:" << c << "," << d << endl;

		//cout << "------------" << endl;

		trA = a + d;
		detA = a*d - b*c;

		// x^2-trA*x+detA = 0
		L1 = (trA + sqrt(trA*trA - 4 * detA))*0.5;
		L2p = (trA - sqrt(trA*trA - 4 * detA))*0.5;

		//cout <<"®ö¥´:"<< L1 << "," << L2p << endl;

		//cout << "------------" << endl;

		//cout <<"Kernel" <<a - L2p << "," << b << "," << endl;
		//cout << "Kernel" << c << "," << d - L2p << endl;

		//cout << "------------" << endl;

		// use the smallest eigenvalue to find eigenvector
		a = -b / (a - L2p);
		b = 1;


		// normalize (a,b)
		b = b / sqrt(a*a + b);
		a = a/sqrt(a*a + 1);
		c = -a*xMean - b*yMean;

		//cout<<"Line:" << a << "," << b << "," << c << endl;

		

		//cout << image01.rows <<","<< image01.cols;

		int score = 0;
		
		for (int j = 0; j < keyPoint1.size(); j++)
		{
			if (fabsf(a*keyPoint1[j].pt.x + b*keyPoint1[j].pt.y + c) < threshold)
			{
				score++;
			}
		}

		if (score > max)
		{
			max = score;
			finalA = a;
			finalB = b;
			FinalC = c;
			//finalKeyPoints = tempPoints;
		}
		
	}

	for (int j = 0; j < keyPoint1.size(); j++)
	{
		if (fabsf(finalA*keyPoint1[j].pt.x + finalB*keyPoint1[j].pt.y + FinalC) < threshold)
		{
			finalKeyPoints.push_back(keyPoint1[j]);
		}
	}
	

	/*
	cout << "Score:" << max << endl;

	//cout << "Final Line1:" << finalA << "*x+" << finalB << "*y+" << FinalC << "=0" << endl;

	for (int k = 0; k < finalKeyPoints.size(); k++)
	{
		cout << "FinalKeyPoints1:" << finalKeyPoints[k].angle << ":" << finalKeyPoints[k].pt << endl;
		//xMean += tempPoints[k].pt.x;
		//yMean += tempPoints[k].pt.y;
	}
	max = 0;
	*/

	

	/*
	for (int i = 0; i < iteraationNum; i++)
	{
		vector<KeyPoint> tempPoints;
		vector<int> random;
		float xMean = 0, yMean = 0;
		float matrixA[4][2];
		float a = 0, b = 0, c = 0, d = 0;
		float trA, detA;
		float L1, L2p;

		for (int j = 0; j < randomNum; j++)
		{


			int currentRandom = rand() % keyPoint2.size();
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
				tempPoints.push_back(keyPoint2[currentRandom]);
				random.push_back(currentRandom);
			}
			else
			{
				j--;
			}
		}


		for (int k = 0; k < tempPoints.size(); k++)
		{
			//cout <<"Sample:"<< tempPoints[k].angle << ":" << tempPoints[k].pt << endl;
			xMean += tempPoints[k].pt.x;
			yMean += tempPoints[k].pt.y;
		}

		xMean = xMean / randomNum;
		yMean = yMean / randomNum;

		//cout << "------------" << endl;
		//cout << "Mean:" << xMean << "," << yMean << endl;
		//cout << "------------" << endl;

		for (int j = 0; j < randomNum; j++)
		{
			matrixA[j][0] = tempPoints[j].pt.x - xMean;
			matrixA[j][1] = tempPoints[j].pt.y - yMean;

			//cout <<"Matrix:" <<matrixA[j][0] << "," << matrixA[j][1] << endl;
		}

		//cout << "------------" << endl;

		for (int j = 0; j < randomNum; j++)
		{
			a += matrixA[j][0] * matrixA[j][0];
			b += matrixA[j][0] * matrixA[j][1];
			c += matrixA[j][1] * matrixA[j][0];
			d += matrixA[j][1] * matrixA[j][1];
		}

		//cout <<"AT*A:"<< a << "," << b << endl;
		//cout << "AT*A:" << c << "," << d << endl;

		//cout << "------------" << endl;

		trA = a + d;
		detA = a*d - b*c;
		L1 = (trA + sqrt(trA*trA - 4 * detA))*0.5;
		L2p = (trA - sqrt(trA*trA - 4 * detA))*0.5;

		//cout <<"®ö¥´:"<< L1 << "," << L2p << endl;

		//cout << "------------" << endl;

		//cout <<"Kernel" <<a - L2p << "," << b << "," << endl;
		//cout << "Kernel" << c << "," << d - L2p << endl;

		//cout << "------------" << endl;

		a = -b / (a - L2p);
		b = 1;

		b = b / sqrt(a*a + b);
		a = a / sqrt(a*a + 1);
		c = -a*xMean - b*yMean;

		//cout<<"Line:" << a << "," << b << "," << c << endl;



		//cout << image01.rows <<","<< image01.cols;

		int score = 0;

		for (int j = 0; j < keyPoint2.size(); j++)
		{
			if (fabsf(a*keyPoint2[j].pt.x + b*keyPoint2[j].pt.y + c) < threshold)
			{
				score++;
			}
		}

		if (score > max)
		{
			max = score;
			finalA = a;
			finalB = b;
			FinalC = c;
			finalKeyPoints2 = tempPoints;
		}

	}

	cout << "Score:" << max << endl;

	//cout << "Final Line2:" << finalA << "*x+" << finalB << "*y+" << FinalC << "=0" << endl;

	for (int k = 0; k < finalKeyPoints2.size(); k++)
	{
		cout << "FinalKeyPoints2:" << finalKeyPoints2[k].angle << ":" << finalKeyPoints2[k].pt << endl;
		//xMean += tempPoints[k].pt.x;
		//yMean += tempPoints[k].pt.y;
	}
	*/
	
	
	SiftDescriptorExtractor siftDescriptor;
	Mat imageDesc1, imageDesc2;
	siftDescriptor.compute(image01, finalKeyPoints, imageDesc1);
	siftDescriptor.compute(image02, keyPoint2, imageDesc2);

	BruteForceMatcher<L2<float>> matcher;
	vector<DMatch> matchePoints;
	matcher.match(imageDesc1, imageDesc2, matchePoints, Mat());
	Mat imageOutput;
	//drawMatches(image01, finalKeyPoints, image02, keyPoint2, matchePoints, imageOutput);

	for (int i = 0; i < matchePoints.size(); i++)
	{

		if (finalKeyPoints2.size() < 4)
		{
			finalKeyPoints2.push_back(matchePoints[i]);
		}
		else
		{
			for (int j = 0; j < 4; j++)
			{
				if (matchePoints[i].distance < finalKeyPoints2[j].distance)
				{
					if (fabs((float)finalKeyPoints[matchePoints[i].queryIdx].pt.x - finalKeyPoints[finalKeyPoints2[0].queryIdx].pt.x)>0.5 && fabs((float)finalKeyPoints[matchePoints[i].queryIdx].pt.x - finalKeyPoints[finalKeyPoints2[1].queryIdx].pt.x)>0.5 && fabs((float)finalKeyPoints[matchePoints[i].queryIdx].pt.x - finalKeyPoints[finalKeyPoints2[2].queryIdx].pt.x) > 0.5&&fabs((float)finalKeyPoints[matchePoints[i].queryIdx].pt.x - finalKeyPoints[finalKeyPoints2[3].queryIdx].pt.x)>0.5)
					{
						finalKeyPoints2[j] = matchePoints[i];
						break;
					}

					//cout << (float)finalKeyPoints[matchePoints[i].queryIdx].pt.x<<endl;
					//cout << (float)finalKeyPoints[finalKeyPoints2[j].queryIdx].pt.x << endl;
					//cout << finalKeyPoints[matchePoints[i].queryIdx].pt.x - finalKeyPoints[finalKeyPoints2[j].queryIdx].pt.x << endl;

					

					//break;
				}
			}
		}
	}

	

	for (int j = 0; j < 4; j++)
	{
		finalfinalKeyPoints.push_back(finalKeyPoints[finalKeyPoints2[j].queryIdx]);

		cout << finalKeyPoints2[j].distance << endl;
		cout << finalKeyPoints2[j].queryIdx << endl;
		cout << finalKeyPoints[finalKeyPoints2[j].queryIdx].pt << endl;
		cout << "Size" << finalKeyPoints.size()<<endl;
		
	}

	siftDescriptor.compute(image01, finalfinalKeyPoints, imageDesc1);
	siftDescriptor.compute(image02, keyPoint2, imageDesc2);
	matcher.match(imageDesc1, imageDesc2, matchePoints, Mat());


	drawMatches(image01, finalfinalKeyPoints, image02, keyPoint2, matchePoints, imageOutput);

	imshow("Mathch Points", imageOutput);
	
	//cout << matchePoints[0].trainIdx;

	//namedWindow("Mathch Points", 0);
	
	

	float A[9][9];
	//float AT[9][8];

	//cout << keyPoint2[matchePoints[0].trainIdx].pt.x << endl;
	
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
	cout << "x=" << endl << " " << x << endl;
	//cout << x.at<float>(1, 0) << endl;

	float H[3][3] = 
	{ { x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2, 0) },
	{ x.at<float>(3, 0), x.at<float>(4, 0), x.at<float>(5, 0) },
	{ x.at<float>(6, 0), x.at<float>(7, 0), x.at<float>(8, 0) } };

	Mat Hp = Mat(3, 3, CV_32FC1, H);

	cout << "H=" << endl << " " << Hp << endl;

	cout << image01.rows << "," << image01.cols << endl;

	int cons = 350;

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


	//Mat outputImg = Mat(image01.rows, image01.cols, CV_8UC3);

	//warpPerspective(image01p, outputImg, Hp, Size(image01p.cols * 2, image01p.rows * 2), INTER_NEAREST, BORDER_CONSTANT);
	
	
	//imshow("123", outputImg);


	
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

	Mat mergeImg = Mat(image01.rows * 2, 2000, CV_8UC3);

	for (int i = 0; i < featureAfterH_x.size(); i++)
	{
		//mergeImg.at<Vec3b>(featureAfterH_x[i], featureAfterH_y[i])[0] = 255;
		//mergeImg.at<Vec3b>(featureAfterH_x[i], featureAfterH_y[i])[1] = 0;
		//mergeImg.at<Vec3b>(featureAfterH_x[i], featureAfterH_y[i])[2] = 0;
	}
	
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
	

	imshow("Output", finalDstImg);
	imshow("Merge", mergeImg);
	
	/*
	for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < 9; j++)
		{
			AT[j][i] = A[i][j];
		}
	}
	*/
	
	
	//Mat AM = Mat(8, 9, CV_32FC1, A);
	//Mat ATM = Mat(9, 8, CV_32FC1, AT);
	//Mat C;
	//C = ATM * AM;
	//cout << "A =" << endl << " " << AM << endl << endl;
	//cout << "AT =" << endl << " " << ATM << endl << endl;
	//cout << "C =" << endl << " " << C<< endl << endl;


	
	waitKey(0);
	return 0;
}