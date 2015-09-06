#include <opencv2/core/core.hpp>
using namespace cv;

void initializeModule(String CASCADE_FILENAME_FACE, String CASCADE_FILENAME_EYEPAIR);
int getFeaturePoints(Mat& frame_bgr, vector<Point>& feature_points);
Mat getCenterMap(Mat& eye_grey);
Point findIrisCenter(Mat& eye_bgr);
