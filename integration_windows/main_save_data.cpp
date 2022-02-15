
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <type_traits>
#include <map>
#include <string>
#include <math.h>

#include <opencv2/opencv.hpp>

#include "FacialExpressionAnalysis_Interface.h"
#include "Macros.h"

using namespace std;
using namespace cv;

#if (defined(_WIN32) || defined(_WIN64))
#else // Linux
#define sprintf_s sprintf
#endif

/////////////////////////////////
// command line arguments:
// none: -
// first: open video/image file
// for testing, image file = "day01_0087.jpg"



map<string, Mat> getCameraParams(string);
Mat getUndistortedImage(string, Mat, Mat);
Mat loadFaceModel(string);
Mat getLandmarks(Mat, string, string);
vector<Mat> getHeadPose(Mat, Mat, Mat, Mat);
vector<Mat> normalize(Mat, Mat, Mat, Mat, Mat);
void displayPatch(Mat);
void savePatch(Mat, string);
void saveLandmarks(Mat, string);
void saveHeadpose(vector<Mat>, Mat, string);
void saveFEAypr(double[], string);

int main(int argc, char **argv)
{
	string imgFile = argv[1];
    string person = argv[2];
    string index = argv[3];
    string directory = "../fyp_normalisation/integration_test_data/" + person + "/";
    string camParamPath = directory + "calib.txt";
    string normalizedImageOutputPath = directory + index + "/norm.png";
    string headposeOutputPath = directory + index + "/headpose.txt";
    string facialLandmarksOutputPath = directory + index + "/facialLandmarks.txt";
    string FEAyprOutputPath = directory + index + "/ypr.txt";

	auto cameraParams = getCameraParams(camParamPath);
    auto img = getUndistortedImage(imgFile, cameraParams["camera_matrix"], cameraParams["distortion"]);
    //imshow("Undistorted image", img);
    //waitKey(0);
    auto faceModel = loadFaceModel("./face_model.txt");
    // cout << "face model = " << endl << " "  << faceModel << endl << endl;
    auto landmarks = getLandmarks(img, facialLandmarksOutputPath, FEAyprOutputPath);
    // cout << "landmarks = " << endl << " "  << landmarks << endl << endl;
    auto headPose = getHeadPose(faceModel, landmarks, cameraParams["camera_matrix"], cameraParams["distortion"]);
    auto res = normalize(img, cameraParams["camera_parameter"], headPose[0], headPose[1], faceModel);
    
	Mat eyePatch = res[0];
    Mat normHeadPose = res[1];
    // displayPatch(eyePatch);
    savePatch(eyePatch, normalizedImageOutputPath);
    saveHeadpose(headPose, normHeadPose, headposeOutputPath);
	
}

map<string, Mat> getCameraParams(string camParamPath) {
    cout << "getting cameraParams" << endl;
    // hard code values -> todo: implement camera calibration
    //mat cameramat = (mat_<double>(3,3) << 994.73532636, 0, 624.66344095, 0, 998.16646784, 364.08742557, 0, 0, 1);
    //mat distortion = (mat_<double>(1,5) << -0.16321888, 0.66783406, -0.00121854, -0.00303158, -1.02159927);
    //mat cameraparam = (mat_<double>(1,4) << 994.73532636, 998.16646784, 624.66344095, 364.08742557);

   // read 3*6 face model matrix fro txt file
    Mat cameraMat = Mat::zeros(3, 3, CV_64FC1);
    Mat distortion = Mat::zeros(1, 5, CV_64FC1);

    string line;
    fstream ifs;
    ifs.open(camParamPath);

    for (int i = 0; i < cameraMat.rows; i++) {
        for (int j = 0; j < cameraMat.cols; j++) {
            getline(ifs, line);
            cameraMat.at<double>(i, j) = stof(line);
        }
    }

    getline(ifs, line);

    for (int i = 0; i < distortion.rows; i++) {
        for (int j = 0; j < distortion.cols; j++) {
            getline(ifs, line);
            distortion.at<double>(i, j) = stof(line);
        }
    }

    ifs.close();

    double fx = cameraMat.at<double>(0, 0);
    double cx = cameraMat.at<double>(0, 2);
    double fy = cameraMat.at<double>(1, 1);
    double cy = cameraMat.at<double>(1, 2);

    Mat cameraParam = (Mat_<double>(1, 4) << fx, fy, cx, cy);

    map<string, Mat> cameraParams {{"camera_matrix", cameraMat}, {"distortion", distortion}, {"camera_parameter", cameraParam}};
    return cameraParams;
}

Mat getUndistortedImage(string imgFile, Mat cameraMatrix, Mat cameraDistortion) {
    cout << "undistoring image" << endl;
	Mat img = cv::imread(imgFile, IMREAD_COLOR);
	if (img.empty())
	{
		std::cout << "Unable to read image";
		exit(1);
	}

    Mat undistorted;
    undistort(img, undistorted, cameraMatrix, cameraDistortion);
    return img;
}

Mat loadFaceModel(string file) {
    cout << "loading face model" << endl;

    // read 3*6 face model matrix fro txt file
    Mat face = Mat::zeros(3,6, CV_64FC1);
    string line;
    fstream ifs;
    ifs.open(file);
    
    for (int i = 0; i < face.rows; i++) {
        for (int j = 0; j < face.cols; j++) {
            getline(ifs, line);
            face.at<double>(i,j) = stof(line);
        }
    }
    
    ifs.close();
    return face;
}

// Use FEA to get 6 facial landmarks (4 eye corners and 2 mouth corners)
Mat getLandmarks(Mat img, string facialLandmarksOutputPath, string FEAyprOutputPath) {
    // return (Mat_<int>(6,2) << 548, 409, 605, 405, 698, 398, 757, 391, 606, 567, 724, 559);
    cout << "getting landmarks" << endl;


	FacialExpressionAnalysis_Interface fea( 1, 50, 0.5 );
	
	if (fea.init_status != FEA::OK)
	{
		std::cout << "FEA initialization failed";
		exit(1);  // check log file for details
	}
	
	cv::Rect bbox(0, 0, 0, 0);
	FEA::RESULT result;
	double avi[3] = { 0, 0, 0 };
	double ypr[3] = { 0, 0, 0 };
	int eyeBlink = 0;
	double eyeOpenness = 0;
	float score = 0;
	cv::Mat facepoints;
	
	result = fea.detect_face(img, bbox);
	if (result == FEA::OK)
	{
		result = fea.detect_facepoints(img, bbox, facepoints, score);
		if (result == FEA::OK)
		{
			result = fea.calc_expression(facepoints, score, avi, ypr, eyeBlink, eyeOpenness);
		}
	}

	Mat landmarks = Mat::zeros(6, 2, CV_32F);
	int indices[6] = {19, 22, 25, 28, 31, 37}; // indeices of four eye corners and two mouth corners

	for (int i = 0; i < 6; i++) {
		int index = indices[i];
		landmarks.at<float>(i, 0) = facepoints.at<float>(0, index);
		landmarks.at<float>(i, 1) = facepoints.at<float>(1, index);
	}

    saveLandmarks(landmarks, facialLandmarksOutputPath);
    saveFEAypr(ypr, FEAyprOutputPath);

	return landmarks;
}

void saveLandmarks(Mat landmarks, string facialLandmarksOutputPath) {
    ofstream file;
    file.open(facialLandmarksOutputPath);
    for (int i = 0; i < landmarks.rows; i++) {
        for (int j = 0; j < landmarks.cols; j++) {
            file << landmarks.at<float>(i, j) << "\t";
        }
        file << "\n";
    }

    file.close();
}

void saveFEAypr(double ypr[], string FEAyprOutputPath) {
    ofstream file;
    file.open(FEAyprOutputPath);
    for (int i = 0; i < 3; i++) {
        file << ypr[i] << "\n";
    }
    file.close();
    
}

vector<Mat> getHeadPose(Mat faceModel, Mat landmarks, Mat cameraMatrix, Mat cameraDistortion) {
    int numPts = faceModel.cols;
    Mat facePts = faceModel.t();
    facePts = facePts.reshape(0, numPts);
    
    // Mat landmarksFloat;
    // landmarks.convertTo(landmarksFloat, CV_32FC1);

    // Mat rvec, tvec;
    // solvePnP(facePts, landmarksFloat, cameraMatrix, cameraDistortion, rvec, tvec, false, SOLVEPNP_EPNP);
    // solvePnP(facePts, landmarksFloat, cameraMatrix, cameraDistortion, rvec, tvec, true); // further optimize
    // cout << "rvec = " << endl << " "  << rvec << endl << endl;
    // cout << "tvec = " << endl << " "  << tvec << endl << endl;

	Mat rvec, tvec;
    solvePnP(facePts, landmarks, cameraMatrix, cameraDistortion, rvec, tvec, false, SOLVEPNP_EPNP);
    solvePnP(facePts, landmarks, cameraMatrix, cameraDistortion, rvec, tvec, true); // further optimize

    return vector<Mat> {rvec, tvec};
}

vector<Mat> normalize(Mat img, Mat cameraParams, Mat rvec, Mat tvec, Mat faceModel) {
    // todo: check non-null for rvec & tvec

    // Calculate rotation matrix and euler angles
    rvec = rvec.reshape(0, 3);
    tvec = tvec.reshape(0, 3);
    Mat rotateMat;
    Rodrigues(rvec, rotateMat);

    // Reconstruct frame
    Mat fullFrame;
    cvtColor(img, fullFrame, COLOR_BGR2RGB);

    // Form camera matrix
    Mat cameraMat = (Mat_<double>(3,3) << cameraParams.at<double>(0), 0, cameraParams.at<double>(2), 
                                            0, cameraParams.at<double>(1), cameraParams.at<double>(3), 
                                            0,0,1);

    // get normalized camera parameters -> todo: put into constants
    int focalLength = 1300;
    int distance = 600;
    int width = 256;
    int height = 64;

    Mat normCameraMat = (Mat_<double>(3,3) << focalLength, 0, 0.5 * width, 
                                            0, focalLength, 0.5 * height, 
                                            0,0,1);
    
    Mat fc = rotateMat * faceModel; // 3D positions of facial landmarks before translation
    Mat re = 0.5*(fc(Range::all(), Range(0,1)).clone() + fc(Range::all(), Range(1,2)).clone()) + tvec; // center of left eye
    Mat le = 0.5*(fc(Range::all(), Range(2,3)).clone() + fc(Range::all(), Range(3,4)).clone()) + tvec; // center of right eye
    Mat gazeOrigin = (re + le) / 2; // center of two eyes

    
    // Code below is an adaptation of code by Xucong Zhang
    // https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/gaze-based-human-computer-interaction/revisiting-data-normalization-for-appearance-based-gaze-estimation/
    
    
    double zScale = distance / norm(gazeOrigin, NORM_L2, noArray());
    Mat S = Mat::eye(3, 3, CV_64F);
    S.at<double>(2, 2) = zScale; // scale matrix S

    Mat hRx = rotateMat(Range::all(), Range(0,1)).clone();
    Mat forward = (gazeOrigin / norm(gazeOrigin, NORM_L2, noArray()));
    Mat down = forward.cross(hRx);
    down /= norm(down, NORM_L2, noArray());
    Mat right = down.cross(forward);
    right /= norm(right, NORM_L2, noArray());
    Mat R;
    hconcat(right, down, R);
    hconcat(R, forward, R);
    R = R.t(); // rotation matrix R

    Mat W = (normCameraMat * S) * (R * cameraMat.inv());// transformation matrix
    Mat patch; 
    warpPerspective(fullFrame, patch, W, Size(width, height)); // image normalization

    // Normalise head pose
    Mat headMat = R * rotateMat;
    Mat normHeadPose = (Mat_<double>(1,2) << asin(headMat.at<double>(1, 2)), atan2(headMat.at<double>(0, 2), headMat.at<double>(2, 2)));

    return vector<Mat> {patch, normHeadPose};
}

void displayPatch(Mat patchBGR) {
    Mat display;
    cvtColor(patchBGR, display, COLOR_BGR2RGB);
    imshow("patch", display);
    waitKey(0);
}

void savePatch(Mat patchBGR, string normalizedImageOutputPath) {
    Mat display;
    cvtColor(patchBGR, display, COLOR_BGR2RGB);
    imwrite(normalizedImageOutputPath, display);
}

void saveHeadpose(vector<Mat> before, Mat after, string headposeOutputPath) {
    ofstream file;
    file.open(headposeOutputPath);
    Mat rvec = before[0];
    Mat tvec = before[1];
    file << "rvec:" << "\n";
    for (int i = 0; i < rvec.rows; i++) {
        for (int j = 0; j < rvec.cols; j++) {
            file << rvec.at<double>(i, j) << "\t";
        }
        file << "\n";
    }

    file << "tvec:" << "\n";
    for (int i = 0; i < tvec.rows; i++) {
        for (int j = 0; j < tvec.cols; j++) {
            file << tvec.at<double>(i, j) << "\t";
        }
        file << "\n";
    }

    file << "normalised:" << "\n";
    for (int i = 0; i < after.rows; i++) {
        for (int j = 0; j < after.cols; j++) {
            file << after.at<double>(i, j) << "\t";
        }
        file << "\n";
    }

    file.close();

}