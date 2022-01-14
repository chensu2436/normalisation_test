#include <vector>
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <math.h>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

map<string, Mat> getCameraParams();
Mat getUndistortedImage(Mat, Mat);
Mat loadFaceModel(string);
Mat getLandmarks();
vector<Mat> getHeadPose(Mat, Mat, Mat, Mat);
vector<Mat> normalize(Mat, Mat, Mat, Mat, Mat);
void displayPatch(Mat);

int main() {
    auto cameraParams = getCameraParams();
    auto img = getUndistortedImage(cameraParams["camera_matrix"], cameraParams["distortion"]);
    // imshow("Undistorted image", img);
    // waitKey(0);
    auto faceModel = loadFaceModel("./face_model.txt");
    // cout << "face model mat = " << endl << " "  << faceModel << endl << endl;
    auto landmarks = getLandmarks();
    // cout << "face point mat = " << endl << " "  << facePts << endl << endl;
    auto headPose = getHeadPose(faceModel, landmarks, cameraParams["camera_matrix"], cameraParams["distortion"]);
    auto res = normalize(img, cameraParams["camera_parameter"], headPose[0], headPose[1], faceModel);
    Mat eyePatch = res[0];
    Mat normHeadPose = res[1];
    displayPatch(eyePatch);
}

map<string, Mat> getCameraParams() {
    // hard code values
    Mat cameraMat = (Mat_<double>(3,3) << 994.73532636, 0, 624.66344095, 0, 998.16646784, 364.08742557, 0, 0, 1);
    Mat distortion = (Mat_<double>(1,5) << -0.16321888, 0.66783406, -0.00121854, -0.00303158, -1.02159927);
    Mat cameraParam = (Mat_<double>(1,4) << 994.73532636, 998.16646784, 624.66344095, 364.08742557);

    map<string, Mat> cameraParams {{"camera_matrix", cameraMat}, {"distortion", distortion}, {"camera_parameter", cameraParam}};
    return cameraParams;
}

Mat getUndistortedImage(Mat cameraMatrix, Mat cameraDistortion) {
    //todo: change to argv[1]
    string imgPath = "../data/example/day01_0087.jpg";
    Mat img = imread(imgPath, IMREAD_COLOR);
    // cout << "img mat = " << endl << " "  << img << endl << endl;
    Mat undistorted;
    undistort(img, undistorted, cameraMatrix, cameraDistortion);
    return img;
}

Mat loadFaceModel(string file) {
    // read 3*6 face model matrix fro txt file
    Mat face = Mat::zeros(3,6, CV_64FC1);
    string line;
    fstream ifs;
    ifs.open(file);
    
    for (int i = 0; i < face.rows; i++) {
        for (int j = 0; j < face.cols; j++) {
            getline(ifs, line);
            // cout.precision(dbl::max_digits10);
            face.at<double>(i,j) = stof(line);
        }
    }
    
    ifs.close();
    return face;
}

Mat getLandmarks(){
    // hard code values -> to be integrated with SDK on windows
    return (Mat_<int>(6,2) << 548, 409, 605, 405, 698, 398, 757, 391, 606, 567, 724, 559);
}

vector<Mat> getHeadPose(Mat faceModel, Mat landmarks, Mat cameraMatrix, Mat cameraDistortion) {
    int numPts = faceModel.cols;
    Mat facePts = faceModel.t();
    facePts = facePts.reshape(0, numPts);
    
    Mat landmarksFloat;
    landmarks.convertTo(landmarksFloat, CV_32FC1);

    Mat rvec, tvec;
    solvePnP(facePts, landmarksFloat, cameraMatrix, cameraDistortion, rvec, tvec, false, SOLVEPNP_EPNP);
    solvePnP(facePts, landmarksFloat, cameraMatrix, cameraDistortion, rvec, tvec, true); // further optimize
    // cout << "rvec = " << endl << " "  << rvec << endl << endl;
    // cout << "tvec = " << endl << " "  << tvec << endl << endl;

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