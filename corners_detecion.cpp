/************************************************************
* corners_detection.cpp										*
* version 1.0												*
* MrZigZag @ Peking University								*
* 2015-04-24												*
*************************************************************
*															*
* 1.DETECT FEATURE POINTS OF A FRONTAL FACE!				*
* 2.Function getCenterMap and findIrisCenter are from		*
*	Erroll Wood's EyeTab system.							*
*															*
* 3.initializeModule: CASCADE_FILENAME_FACE represents the	*
* path of file "haarcascade_frontalface_alt.xml", and		*
* CASCADE_FILENAME_EYEPAIR represents the path of file		*
* "haarcascade_mcs_eyepair_big.xml". YOU MUST RUN THIS		*
* FUNCTION BEFORE USING OTHER PARTS IN THIS MODULE!			*
*															*
* 4.getFeaturePoints: find ten feature points in a face. 	*
* They are eye corners(4), iris centers(2), nostril,		*
* centers(2), mouth corners(2)								*
************************************************************/
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <vector>

using namespace std;
using namespace cv;

#define SMALL_FRAME_WIDTH 512

CascadeClassifier face_cascade;
CascadeClassifier eyepair_cascade;
const float EYE_PART_RATIOS[5] = {0.0, 0.40, 0.20, 0.40, 0.0};
const Size MIN_FACE_SIZE = Size(50, 50);
const Size MIN_EYE_PAIR_SIZE = Size(67, 17);
const int fastSize_width = 40;
const int DARKNESS_WEIGHT_SCALE = 100;

/*********************************************************
* You must run initializeModule before using the other	 *
* functions.											 *
*********************************************************/

void initializeModule(String CASCADE_FILENAME_FACE, String CASCADE_FILENAME_EYEPAIR)
{
	face_cascade.load(CASCADE_FILENAME_FACE);
	eyepair_cascade.load(CASCADE_FILENAME_EYEPAIR);
}

Mat getCenterMap(Mat& eye_grey) 
{

	// Calculate image gradients
	Mat grad_x, grad_y;
	Sobel(eye_grey, grad_x, CV_32F, 1, 0, 5);
	Sobel(eye_grey, grad_y, CV_32F, 0, 1, 5);

	// Get magnitudes of gradients, and calculate thresh
	Mat mags;
	Scalar mean, stddev;
	magnitude(grad_x, grad_y, mags);
	meanStdDev(mags, mean, stddev);
	int mag_thresh = stddev.val[0] / 2 + mean.val[0];

	// Threshold out gradients with mags which are too low
	grad_x.setTo(0, mags < mag_thresh);
	grad_y.setTo(0, mags < mag_thresh);

	// Normalize gradients
	grad_x = grad_x / (mags + 1); // (+1 is hack to guard against div by 0)
	grad_y = grad_y / (mags + 1);

	// Initialize 1d vectors of x and y indicies of Mat
	vector<int> x_inds_vec, y_inds_vec;
	for (int i = 0; i < eye_grey.size().width; i++)
		x_inds_vec.push_back(i);
	for (int i = 0; i < eye_grey.size().height; i++)
		y_inds_vec.push_back(i);

	// Repeat vectors to form indices Mats
	Mat x_inds(x_inds_vec), y_inds(y_inds_vec);
	x_inds = repeat(x_inds.t(), eye_grey.size().height, 1);
	y_inds = repeat(y_inds, 1, eye_grey.size().width);
	x_inds.convertTo(x_inds, CV_32F);	// Has to be float for arith. with dx, dy
	y_inds.convertTo(y_inds, CV_32F);

	// Set-up Mats for main loop
	Mat ones = Mat::ones(x_inds.rows, x_inds.cols, CV_32F);	// for re-use with creating normalized disp. vecs
	Mat darkness_weights = (255 - eye_grey) / DARKNESS_WEIGHT_SCALE;
	Mat accumulator = Mat::zeros(eye_grey.size(), CV_32F);
	Mat diffs, dx, dy;

	// Loop over all pixels, testing each as a possible center
	for (int y = 0; y < eye_grey.rows; ++y) {

		// Get pointers for each row
		float* grd_x_p = grad_x.ptr<float>(y);
		float* grd_y_p = grad_y.ptr<float>(y);
		uchar* d_w_p = darkness_weights.ptr<uchar>(y);

		for (int x = 0; x < eye_grey.cols; ++x) {

			// Deref and increment pointers
			float grad_x_val = *grd_x_p++;
			float grad_y_val = *grd_y_p++;

			// Skip if no gradient
			if (grad_x_val == 0 && grad_y_val == 0)
				continue;

			dx = ones * x - x_inds;
			dy = ones * y - y_inds;

			magnitude(dx, dy, mags);
			dx = dx / mags;
			dy = dy / mags;

			diffs = (dx * grad_x_val + dy * grad_y_val) * *d_w_p++;
			diffs.setTo(0, diffs < 0);

			accumulator = accumulator + diffs;
		}
	}

	// Normalize and convert accumulator
	accumulator = accumulator / eye_grey.total();
	normalize(accumulator, accumulator, 0, 255, NORM_MINMAX);
	accumulator.convertTo(accumulator, CV_8U);

	return accumulator;
}

Point findIrisCenter(Mat& eye_bgr)
{

	// Convert BGR coarse ROI to gray
	Mat eye_grey, eye_grey_small;
	cvtColor(eye_bgr, eye_grey, CV_BGR2GRAY);

	// Resize the image to a constant fast size
	// TODO: prevent upscaling
	float scale = fastSize_width / (float)eye_grey.size().width;
	resize(eye_grey, eye_grey_small, Size(0, 0), scale, scale);
	GaussianBlur(eye_grey_small, eye_grey_small, Size(5, 5), 0);

	// Create centermap
	Mat centermap = getCenterMap(eye_grey_small);

	// Find position of max value in small-size centermap
	Point maxLoc;
	minMaxLoc(centermap, NULL, NULL, NULL, &maxLoc);

	// Return re-scaled center to full size
	return maxLoc * (1 / scale);
}


/**********************************************************
* Find feature points of a face in an image. The points in*
* the vector are 4 eye corners, 2 iris centers, 2 nostril *
* centers, 2 mouth coners. The function will return 0 if  *
* finishing successfully, else, -1.						  *
**********************************************************/
int getFeaturePoints(Mat& frame_bgr, vector<Point>& feature_points)
{
	if (frame_bgr.empty()) return -1;
	Mat frame_gray, frame_thresh;
	cvtColor(frame_bgr, frame_gray, CV_BGR2GRAY);

	Mat frame_small;
	float scale = SMALL_FRAME_WIDTH / (float)frame_bgr.size().width;
	resize(frame_gray, frame_small, Size(0, 0), scale, scale);

	vector<Rect> detected_faces;
	face_cascade.detectMultiScale(frame_small, detected_faces, 1.2, 3, 0, MIN_FACE_SIZE);
	if (detected_faces.size() == 0) return -1;
	Rect face_roi = Rect(detected_faces[0].x / scale, detected_faces[0].y / scale, detected_faces[0].width / scale, detected_faces[0].height / scale);

	vector<Rect> detected_eyepairs;
	eyepair_cascade.detectMultiScale(frame_gray(face_roi), detected_eyepairs, 1.2, 3, 0, MIN_EYE_PAIR_SIZE);

	if (detected_eyepairs.size() == 0) return -1;
	Rect binocular_roi = Rect(face_roi.x + detected_eyepairs[0].x, face_roi.y + detected_eyepairs[0].y, detected_eyepairs[0].width, detected_eyepairs[0].height);

	Rect eye_roi[2];
    Mat eye_roi_mat[2], nose_roi_mat, mouth_roi_mat;
	eye_roi[0] = Rect(binocular_roi.x + binocular_roi.width * EYE_PART_RATIOS[0], binocular_roi.y, binocular_roi.width * EYE_PART_RATIOS[1], binocular_roi.height);
	eye_roi[1] = Rect(binocular_roi.x + binocular_roi.width*(EYE_PART_RATIOS[0] + EYE_PART_RATIOS[1] + EYE_PART_RATIOS[2]), binocular_roi.y,
		binocular_roi.width * EYE_PART_RATIOS[3], binocular_roi.height);
	Rect nose_roi(eye_roi[0].x + eye_roi[0].width * 0.5, eye_roi[0].y + eye_roi[0].height, eye_roi[1].x - eye_roi[0].x, (face_roi.y + face_roi.height - eye_roi[0].y - eye_roi[0].height) * 0.5);
	Rect mouth_roi(nose_roi.x - face_roi.width / 20, nose_roi.y + nose_roi.height, nose_roi.width + face_roi.width / 10, nose_roi.height);

    eye_roi_mat[0] = Mat(frame_bgr, eye_roi[0]);
    eye_roi_mat[1] = Mat(frame_bgr, eye_roi[1]);

	// find iris centers
	Point iris_centers[2] = { Point(0, 0), Point(0, 0) };
    
	iris_centers[0] = findIrisCenter(eye_roi_mat[0]);
	iris_centers[1] = findIrisCenter(eye_roi_mat[1]);

	if (iris_centers[0] == Point(0, 0) || iris_centers[1] == Point(0, 0)) return -1;
	iris_centers[0] += eye_roi[0].tl();
	iris_centers[1] += eye_roi[1].tl();

	cvtColor(frame_bgr, frame_thresh, CV_BGR2GRAY);
	equalizeHist(frame_thresh(binocular_roi), frame_thresh(binocular_roi));
    //equalizeHist(frame_thresh(face_roi), frame_thresh(face_roi));
	threshold(frame_thresh(eye_roi[0]), frame_thresh(eye_roi[0]), 35, 100, CV_THRESH_BINARY);
	threshold(frame_thresh(eye_roi[1]), frame_thresh(eye_roi[1]), 35, 100, CV_THRESH_BINARY);
	dilate(frame_thresh(eye_roi[0]), frame_thresh(eye_roi[0]), Mat());
	dilate(frame_thresh(eye_roi[1]), frame_thresh(eye_roi[1]), Mat());

    imshow("0", frame_thresh(eye_roi[0]));
    imshow("1", frame_thresh(eye_roi[1]));
	// eye corners detection
	Rect eye_corners_roi[2];
	eye_corners_roi[0] = Rect(eye_roi[0].x, iris_centers[0].y - eye_roi[0].height / 4, eye_roi[0].width, eye_roi[0].height / 2);
	eye_corners_roi[1] = Rect(eye_roi[1].x, iris_centers[1].y - eye_roi[1].height / 4, eye_roi[1].width, eye_roi[1].height / 2);
	Point eye_corners[4];
	{
		vector<Point> detected_eye_corners[2];
		int max_corners = 10;
		double quality_level = 0.1;
		double min_distance = 10;
		int block_size = 8;
		bool use_harris_detector = false;
		double k = 0.4;
		goodFeaturesToTrack(frame_thresh(eye_corners_roi[0]),
			detected_eye_corners[0],
			max_corners,
			quality_level,
			min_distance,
			Mat(),
			block_size,
			use_harris_detector,
			k);
		goodFeaturesToTrack(frame_thresh(eye_corners_roi[1]),
			detected_eye_corners[1],
			max_corners,
			quality_level,
			min_distance,
			Mat(),
			block_size,
			use_harris_detector,
			k);
		if (detected_eye_corners[0].size() == 0 || detected_eye_corners[1].size() == 0) return -1;
		for (int i = 0; i < detected_eye_corners[0].size(); i++)
		{
			detected_eye_corners[0][i].x += eye_corners_roi[0].x;
			detected_eye_corners[0][i].y += eye_corners_roi[0].y;
		}
		for (int i = 0; i < detected_eye_corners[1].size(); i++)
		{
			detected_eye_corners[1][i].x += eye_corners_roi[1].x;
			detected_eye_corners[1][i].y += eye_corners_roi[1].y;
		}

		// filtering corners
		int corner_value_left = -10000, corner_value_right = -10000;
		int corner_select_left = 0, corner_select_right = 0;
		for (int i = 0; i < detected_eye_corners[0].size(); i++)
		{
			if (detected_eye_corners[0][i].y - detected_eye_corners[0][i].x > corner_value_left)
			{
				corner_value_left = detected_eye_corners[0][i].y - detected_eye_corners[0][i].x;
				corner_select_left = i;
			}

			if (detected_eye_corners[0][i].y + detected_eye_corners[0][i].x > corner_value_right)
			{
				corner_value_right = detected_eye_corners[0][i].y + detected_eye_corners[0][i].x;
				corner_select_right = i;
			}
		}
		eye_corners[0] = detected_eye_corners[0][corner_select_left];
		eye_corners[1] = detected_eye_corners[0][corner_select_right];

		corner_value_left = -10000;
		corner_value_right = -10000;
		for (int i = 0; i < detected_eye_corners[1].size(); i++)
		{
			if (detected_eye_corners[1][i].y - detected_eye_corners[1][i].x > corner_value_left)
			{
				corner_value_left = detected_eye_corners[1][i].y - detected_eye_corners[1][i].x;
				corner_select_left = i;
			}

			if (detected_eye_corners[1][i].y + detected_eye_corners[1][i].x > corner_value_right)
			{
				corner_value_right = detected_eye_corners[1][i].y + detected_eye_corners[1][i].x;
				corner_select_right = i;
			}
		}
		eye_corners[2] = detected_eye_corners[1][corner_select_left];
		eye_corners[3] = detected_eye_corners[1][corner_select_right];
	}
    /*
	equalizeHist(frame_thresh(nose_roi), frame_thresh(nose_roi));
	equalizeHist(frame_thresh(mouth_roi), frame_thresh(mouth_roi));
	threshold(frame_thresh(nose_roi), frame_thresh(nose_roi), 15, 100, CV_THRESH_BINARY_INV);
	threshold(frame_thresh(mouth_roi), frame_thresh(mouth_roi), 20, 100, CV_THRESH_BINARY);
	erode(frame_thresh(nose_roi), frame_thresh(nose_roi), Mat());
	dilate(frame_thresh(mouth_roi), frame_thresh(mouth_roi), Mat());

	// nostril detection
	vector<vector<Point> > nose_contours;
	findContours(frame_thresh(nose_roi), nose_contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	Point nostril_centers[2];
	{
		vector<Point> nostril_contours[2];
		int max_num[2] = { 0, 0 };
		vector <vector<Point> > ::const_iterator it_contours = nose_contours.begin();
		for (; it_contours != nose_contours.end(); it_contours++)
		{
			if (it_contours->size() > max_num[0])
			{
				max_num[0] = it_contours->size();
				nostril_contours[0] = *it_contours;
				if (max_num[0] > max_num[1])
				{
					max_num[0] = max_num[1];
					nostril_contours[0] = nostril_contours[1];
					max_num[1] = it_contours->size();
					nostril_contours[1] = *it_contours;
				}
			}
		}
		if (max_num[0] == 0 || max_num[1] == 0) return -1;

		int total_x, total_y;
		for (int i = 0; i < 2; i++)
		{
			total_x = 0;
			total_y = 0;
			for (vector<Point>::const_iterator it_points = nostril_contours[i].begin(); it_points != nostril_contours[i].end(); it_points++)
			{
				total_x += it_points->x;
				total_y += it_points->y;
			}
			nostril_centers[i] = Point(total_x / max_num[i] + nose_roi.x, total_y / max_num[i] + nose_roi.y);
		}

		if (nostril_centers[0].x > nostril_centers[1].x)
		{
			Point tmp = nostril_centers[0];
			nostril_centers[0] = nostril_centers[1];
			nostril_centers[1] = tmp;
		}
	}

	// mouth corners detection
	Point mouth_corners[2];
	{
		vector<Point> mouth_corners_detected;
		int max_corners = 10;
		double quality_level = 0.1;
		double min_distance = 10;
		int block_size = 8;
		bool use_harris_detector = false;
		double k = 0.4;
		goodFeaturesToTrack(frame_thresh(mouth_roi),
			mouth_corners_detected,
			max_corners,
			quality_level,
			min_distance,
			Mat(),
			block_size,
			use_harris_detector,
			k);
		Point mouth_corners_tmp[2];
		int min_x = 10000;
		int max_x = 0;
		for (vector<Point>::const_iterator it_mouth = mouth_corners_detected.begin(); it_mouth != mouth_corners_detected.end(); it_mouth++)
		{
			if (it_mouth->x < min_x)
			{
				min_x = it_mouth->x;
				mouth_corners_tmp[0] = *it_mouth;
			}
			else if (it_mouth->x > max_x)
			{
				max_x = it_mouth->x;
				mouth_corners_tmp[1] = *it_mouth;
			}
		}
		if (min_x == 10000 || max_x == 0) return -1;
		mouth_corners[0] = Point(mouth_corners_tmp[0].x + mouth_roi.x, mouth_corners_tmp[0].y + mouth_roi.y);
		mouth_corners[1] = Point(mouth_corners_tmp[1].x + mouth_roi.x, mouth_corners_tmp[1].y + mouth_roi.y);
	}
*/
	for (int i = 0; i < 4; i++)
		feature_points.push_back(eye_corners[i]);
	feature_points.push_back(iris_centers[0]);
	feature_points.push_back(iris_centers[1]);
    feature_points.push_back(Point(binocular_roi.width, binocular_roi.height));
	//feature_points.push_back(nostril_centers[0]);
	//feature_points.push_back(nostril_centers[1]);
	//feature_points.push_back(mouth_corners[0]);
	//feature_points.push_back(mouth_corners[1]);

	if (feature_points.size() < 6) return -1;
	return 0;
}



