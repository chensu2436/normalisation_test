#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <type_traits>

#include <opencv2/opencv.hpp>

#include "FacialExpressionAnalysis_Interface.h"
#include "Macros.h"

using namespace std;

#if (defined(_WIN32) || defined(_WIN64))
#else // Linux
#define sprintf_s sprintf
#endif

//*********************
// select mode/options
//*********************
#define SINGLE_IMAGE
//#define VIDEO // or live camera (without command line arguments)
//#define MULTI_IMAGES  // Windows-only
#define INTEL
//#define CATEGORIZE // post-processing of results
//#include <windows.h>

// undefine to analyze only largest face
//#define ALL_FACES

/////////////////////////////////
// command line arguments:
// none: open camera
// first: open video/image file
//
int main(int argc, char **argv)
{
	FacialExpressionAnalysis_Interface fea( 1, 50, 0.5 );
	
	if (fea.init_status != FEA::OK)
	{
		std::cout << "FEA initialization failed";
		return 1;  // check log file for details
	}
	
	char str[200];
	cv::Mat img; 
	std::string expr_name, expr_intensity;
	cv::Rect bbox(0, 0, 0, 0);
	FEA::RESULT result;

#ifdef VIDEO

	cv::VideoCapture cap;
	bool capRresult = false;
	if (argc > 1) 
		capRresult = cap.open(argv[1]); // open video file
	else
		capRresult = cap.open(0); // open camera

	if (!capRresult)
	{
		printf("Unable to open video file/camera.\n");
	}

	// auto-rotation of frames enabled in OpenCV versions 4.5 and above or 3.4.12 and above 
	int rotation = 0;
	// manual frame rotation input only required for OpenCV versions 4.4 and below or 3.4.11 and below
	if (argc > 2)
	{
		rotation = atoi(argv[2]);
	}
	// only for OpenCV versions 4.5 and above or 3.4.12 and above 
	rotation = cap.get(cv::CAP_PROP_ORIENTATION_META);

	if (cap.isOpened())
	{
		printf("Video resolution: %.0fx%.0f, frame rate: %.1f fps, frame rotation: %d deg\n", cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT), cap.get(cv::CAP_PROP_FPS), rotation);
		cv::Mat frame;
		int frameNum = -1;
		double msecPerFrame = 0;

		std::chrono::high_resolution_clock clock;

		cv::Rect bbox(0, 0, 0, 0);
		while (cap.read(frame) )
		{
			double msec = cap.get(cv::CAP_PROP_POS_MSEC);
			int camFrameNum = (int) cap.get(cv::CAP_PROP_POS_FRAMES);
			if (camFrameNum > 0)
			{
				frameNum = camFrameNum;
			}
			else
			{
				frameNum++;
			}
			//if (frameNum < 2500) continue; ///**********************************//////////
			if (msec < 0)
			{
				msec = 0;
			}
			/* // manual frame rotation only required for OpenCV versions 4.4 and below or 3.4.11 and below
			cv::Mat temp;
			switch (rotation)
			{
			case 90: // rotate clockwise
				cv::transpose(frame, temp);
				cv::flip(temp, frame, 1);
				break;
			case 180:
				cv::flip(frame, temp, 0);
				cv::flip(temp, frame, 1);
				break;
			case 270: // rotate counter-clockwise
				cv::transpose(frame, temp);
				cv::flip(temp, frame, 0);
				break;
			default:;
			}
			*/
#ifdef ALL_FACES
			std::vector<cv::Rect> faces;

			std::vector<double> arousal;
			std::vector<double> valence;
			std::vector<double> intensity;
			std::vector<double> yaw;
			std::vector<double> pitch;
			std::vector<double> roll;
			std::vector<float> confidence;
#ifdef INTEL
			std::vector<cv::Mat> facepoints;
			std::vector<int> eyeBlink;
			std::vector<double> eyeOpenness;

			std::chrono::high_resolution_clock::time_point startTime = clock.now(); ///// timer /////

			faces.clear();
			bool track = false; // detection only
			result = fea.detect_faces(frame, faces);
			if (result == FEA::OK)
			{
				result = fea.detect_facepoints(frame, faces, facepoints, confidence);
				if (result == FEA::OK)
				{
					result = fea.calc_expression(facepoints, confidence, arousal, valence, intensity, yaw, pitch, roll, eyeBlink, eyeOpenness);
				}
			}
			std::chrono::duration<double, std::milli> fp_ms = clock.now() - startTime; ///// timer /////
			msecPerFrame = fp_ms.count();

			sprintf_s(str, "%5d,%.3f, %3.0f, %d", frameNum, msec / 1000, msecPerFrame, track);
			int faceIndex = 0;
			for (auto face = faces.begin(); face != faces.end(); face++, faceIndex++) // all faces
			{
				printf("%s, %zu, %d, %d,%d,%d,%d, %3.0f, ", str, faces.size(), faceIndex, face->x, face->y, face->width, face->height, confidence[faceIndex] * 100);
				printf("%.2f,%.2f,%.2f, %.1f,%.1f,%.1f\n", arousal[faceIndex], valence[faceIndex], intensity[faceIndex], yaw[faceIndex], pitch[faceIndex], roll[faceIndex]);
			}
			if (faces.size() == 0)
			{
				printf("%s, %zu, %d, %d,%d,%d,%d, %3.0f, ", str, faces.size(), 0, 0, 0, 0, 0, 0.);
				printf("%.2f,%.2f,%.2f, %.1f,%.1f,%.1f\n", 0., 0., 0., 0., 0., 0.);
			}
			/*
			cv::Mat imgout = frame;
			cv::rectangle(imgout, bbox, cv::Scalar(0, 0, 255)); ///////////////////
			std::vector<int> compression_params;
			compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
			compression_params.push_back(85);

			char fn[100];
			sprintf_s(fn, "f%04d.jpg", frameNum);
			std::string fnStr = fn;
			cv::imwrite(fnStr, imgout, compression_params);
			*/
#else
			std::chrono::high_resolution_clock::time_point startTime = clock.now(); ///// timer /////

			faces.clear();
			bool track = false; // detection only, no tracking (usually slower)
			//bool track = (frameNum % 15 > 0); // re-detect faces every N frames
			result = fea.calc_expression(frame, faces, track, arousal, valence, intensity, yaw, pitch, roll, confidence);
			//result = fea.calc_expression(frame, cv::Rect(), true, avi, ypr, expr_word, score, str);

			std::chrono::duration<double, std::milli> fp_ms = clock.now() - startTime; ///// timer /////
			msecPerFrame = fp_ms.count();

			sprintf_s(str, "%5d,%.3f, %3.0f, %d", frameNum, msec / 1000, msecPerFrame, track);
			int faceIndex = 0;
			//std::vector<cv::Rect>::iterator face;
			for (auto face = faces.begin(); face != faces.end(); face++, faceIndex++) // all faces
			{
				printf("%s, %zu, %d, %d,%d,%d,%d, %3.0f, ", str, faces.size(), faceIndex, face->x, face->y, face->width, face->height, confidence[faceIndex] * 100);
				printf("%.2f,%.2f,%.2f, %.1f,%.1f,%.1f\n", arousal[faceIndex], valence[faceIndex], intensity[faceIndex], yaw[faceIndex], pitch[faceIndex], roll[faceIndex]);
			}
			if (faces.size() == 0)
			{
				printf("%s, %zu, %d, %d,%d,%d,%d, %3.0f, ", str, faces.size(), 0, 0, 0, 0, 0, 0.);
				printf("%.2f,%.2f,%.2f, %.1f,%.1f,%.1f\n", 0., 0., 0., 0., 0., 0.);
			}
#endif
#else // largest face only
			double avi[3] = { 0, 0, 0 };
			double ypr[3] = { 0, 0, 0 };
			int eyeBlink = 0;
			double eyeOpenness = 0;
			float score = 0;
#ifdef INTEL
			std::chrono::high_resolution_clock::time_point startTime = clock.now(); ///// timer /////

			cv::Mat facepoints;
			bbox = cv::Rect(0, 0, 0, 0);
			bool track = false; // detection only, no tracking 
			result = fea.detect_face(frame, bbox);
			if (result == FEA::OK)
			{
				result = fea.detect_facepoints(frame, bbox, facepoints, score);
				if (result == FEA::OK)
				{
					result = fea.calc_expression(facepoints, score, avi, ypr, eyeBlink, eyeOpenness);
				}
			}
			std::chrono::duration<double, std::milli> fp_ms = clock.now() - startTime; ///// timer /////
			msecPerFrame = fp_ms.count();

			sprintf_s(str, "%5d,%.3f, %3.0f, %d", frameNum, msec / 1000, msecPerFrame, track);
			if (bbox.area() == 0 || score < 0.5)
			{
				printf("%s, %d,%d,%d,%d, %3.0f, ", str, 0, 0, 0, 0, 0.);
				printf("%.2f,%.2f,%.2f, %.1f,%.1f,%.1f, %d,%.2f\n", 0., 0., 0., 0., 0., 0., 0, 0.);
			}
			else
			{
				printf("%s, %d,%d,%d,%d, %3.0f, ", str, bbox.x, bbox.y, bbox.width, bbox.height, score * 100);
				printf("%.2f,%.2f,%.2f, %.1f,%.1f,%.1f, %d,%.2f\n", avi[0], avi[1], avi[2], ypr[0], ypr[1], ypr[2], eyeBlink, eyeOpenness);
			}
			/*
			cv::Mat imgout = frame;
			cv::rectangle(imgout, bbox, cv::Scalar(0, 0, 255)); ///////////////////
			std::vector<int> compression_params;
			compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
			compression_params.push_back(85);

			char fn[100];
			sprintf_s(fn, "f%04d.jpg", frameNum);
			std::string fnStr = fn;
			cv::imwrite(fnStr, imgout, compression_params);
			*/
#else
			std::chrono::high_resolution_clock::time_point startTime = clock.now(); ///// timer /////

			bbox = cv::Rect(0, 0, 0, 0);
			bool track = false; // detection only, no tracking (usually slower)
			//bool track = (frameNum % 15 > 0); // re-detect faces every N frames
			result = fea.calc_expression(frame, bbox, track, avi, ypr, eyeBlink, eyeOpenness, score);
			std::chrono::duration<double, std::milli> fp_ms = clock.now() - startTime; ///// timer /////
			msecPerFrame = fp_ms.count();

			sprintf_s(str, "%5d,%.3f, %3.0f, %d", frameNum, msec / 1000, msecPerFrame, track);
			if (bbox.area() == 0 || score < 0.5)
			{
				printf("%s, %d,%d,%d,%d, %3.0f, ", str, 0, 0, 0, 0, 0.);
				printf("%.2f,%.2f,%.2f, %.1f,%.1f,%.1f, %d,%.2f\n", 0., 0., 0., 0., 0., 0., 0, 0.);
			}
			else
			{
				printf("%s, %d,%d,%d,%d, %3.0f, ", str, bbox.x, bbox.y, bbox.width, bbox.height, score * 100);
				printf("%.2f,%.2f,%.2f, %.1f,%.1f,%.1f, %d,%.2f\n", avi[0], avi[1], avi[2], ypr[0], ypr[1], ypr[2], eyeBlink, eyeOpenness);
			}
			/*
			cv::Mat imgout = frame;
			cv::rectangle(imgout, bbox, cv::Scalar(0, 0, 255)); ///////////////////
			std::vector<int> compression_params;
			compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
			compression_params.push_back(85);

			char fn[100];
			sprintf_s(fn, "f%04d.jpg", frameNum);
			std::string fnStr = fn;
			cv::imwrite(fnStr, imgout, compression_params);
			*/
#endif
#endif
		}
	}
#endif

#ifdef SINGLE_IMAGE
#ifdef INTEL
	double avi[3] = { 0, 0, 0 };
	double ypr[3] = { 0, 0, 0 };
	int eyeBlink = 0;
	double eyeOpenness = 0;
	float score = 0;
	cv::Mat facepoints;
	
	img = cv::imread(argv[1]);
	if (img.empty())
	{
		std::cout << "Unable to read image";
		return 1;
	}
	
	result = fea.detect_face(img, bbox);
	if (result == FEA::OK)
	{
		result = fea.detect_facepoints(img, bbox, facepoints, score);
		if (result == FEA::OK)
		{
			result = fea.calc_expression(facepoints, score, avi, ypr, eyeBlink, eyeOpenness);
		}
	}
	/*
	if (bbox.area() == 0 || score < 0.5)
	{
		printf("%s, %d,%d,%d,%d, %3.0f, ", str, 0, 0, 0, 0, 0.);
		printf("%.2f,%.2f,%.2f, %.1f,%.1f,%.1f, %d,%.2f\n", 0., 0., 0., 0., 0., 0., 0, 0.);
	}
	else
	{
		printf("%s, %d,%d,%d,%d, %3.0f, ", str, bbox.x, bbox.y, bbox.width, bbox.height, score * 100);
		printf("%.2f,%.2f,%.2f, %.1f,%.1f,%.1f, %d,%.2f\n", avi[0], avi[1], avi[2], ypr[0], ypr[1], ypr[2], eyeBlink, eyeOpenness);
	}
	*/
	ofstream file;
	file.open("output.txt");
	file << facepoints.type() << "\n";
	for (int i = 0; i < facepoints.rows; i++) {
		for (int j = 0; j < facepoints.cols; j++) {
			file << facepoints.at<float>(i, j) << "\t";
		}
		file << "\n";
	}

	file << "bounding box\n";
	file << bbox.x << "\t";
	file << bbox.y << "\t";
	file << bbox.width << "\t";
	file << bbox.height << "\t";

	file.close();

	cv::rectangle(img, bbox, cv::Scalar(0, 0, 255)); ///////////////////
	cv::imshow("image", img);
	cv::waitKey(0);

#endif
#endif

#ifdef MULTI_IMAGES
	WIN32_FIND_DATAA data0, data1;
	char pattern0[255], folder1[255], pattern1[255];
	sprintf_s(pattern0, "%s\\\\*", argv[1]);  // command line argument: folder name
	HANDLE hFind = FindFirstFileA(pattern0, &data0);
	if (hFind != INVALID_HANDLE_VALUE)
	{
		do
		{
			if (strlen(data0.cFileName) < 5) continue; // exclude ".", ".."
			sprintf_s(folder1, "%s\\\\%s", argv[1], data0.cFileName);  // command line argument: folder name
			sprintf_s(pattern1, "%s\\\\*", folder1);  // command line argument: folder name
			HANDLE hFind = FindFirstFileA(pattern1, &data1);
			if (hFind != INVALID_HANDLE_VALUE)
			{
				do
				{
					if (strlen(data1.cFileName) < 5) continue; // exclude ".", ".."
					cv::Rect bbox(0, 0, 0, 0);
					char fn[500];
					sprintf_s(fn, "%s\\\\%s", folder1, data1.cFileName);
					img = cv::imread(fn);
					//printf("%s, %s, %d, %d\n", data0.cFileName, data1.cFileName, img.cols, img.rows);
					result = fea.calc_expression(img, bbox, false, avi, ypr, score);
					if (bbox.area() == 0)
					{
						for (int i = 0; i < 196; i++) printf("0,");
					}
					printf("%s, %s, %d,%d,%d,%d, %3.0f, ", data0.cFileName, data1.cFileName, bbox.x, bbox.y, bbox.width, bbox.height, score * 100);
					printf("%.2f,%.2f,%.2f, %.1f,%.1f,%.1f\n", avi[0], avi[1], avi[2], ypr[0], ypr[1], ypr[2]);
				} while (FindNextFileA(hFind, &data1));
				FindClose(hFind);
			}
		} while (FindNextFileA(hFind, &data0));
		FindClose(hFind);
	}
#endif

#ifdef CATEGORIZE // post-processing
	FILE* fin = fopen(argv[1],"r");
	FILE* fout = fopen(argv[2],"w");
	if (fin == NULL)
	{
		std::cout << "Unable to read file";
		return 1;
	}
	if (fout == NULL)
	{
		std::cout << "Unable to write file";
		return 1;
	}
	fgets(str, 200, fin);
	fputs(str, fout);
	int month, day, hour, min, sec, frame, faces, face, facex, facey, w, h;
	float arousal, valence, intensity, yaw, pitch, roll, blink, open;
	while (fgets(str,200,fin))
	{
		int cc = sscanf(str, "%d,%d, %d,%d,%d, %d,%d,%d, %d,%d,%d,%d, %f,%f,%f, %f,%f,%f, %f,%f\n",
			&month, &day, &hour, &min, &sec, &frame, &faces, &face, &facex, &facey, &w, &h, &arousal, &valence, &intensity, &yaw, &pitch, &roll, &blink, &open);
		avi[0] = arousal;
		avi[1] = valence;
		avi[2] = intensity;
		fea.get_expression_names(avi, expr_name, expr_intensity);
		fprintf(fout, "%d,%d, %d,%d,%d, %d,%d,%d, %d,%d,%d,%d, %.2f,%.2f,%.2f, %.2f,%.2f,%.2f, %.2f,%.2f, %s,%s\n",
			month, day, hour, min, sec, frame, faces, face, facex, facey, w, h, arousal, valence, intensity, yaw, pitch, roll, blink, open, expr_intensity.c_str(), expr_name.c_str());
	}
	fclose(fin);
	fclose(fout);

#endif
}
