#include <opencv2/opencv.hpp>
using namespace cv;

void initializeModule(String CASCADE_FILENAME_FACE, String CASCADE_FILENAME_EYEPAIR);
int getFeaturePoints(Mat& frame_bgr, vector<Point>& feature_points, int TRACKING = 0, int type = 0, int kernel = 0);
Mat getCenterMap(Mat& eye_grey);
Point findIrisCenter(Mat& eye_bgr);
int getFaceRegions(Mat frame_bgr, vector<Rect>& regions);
Point2f calcMovement(Mat frame);
int blinkDetection(Mat eye_bgr);