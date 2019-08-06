#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>

std::string type2str(int type) {
  std::string r;
  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);
  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }
  r += "C";
  r += (chans+'0');
  return r;
}

int main(int argc, char** argv){
	
	cv::Mat image = cv::imread(argv[1]); // first part of assignment done
	cv::Mat imagef;
	image.convertTo(imagef,CV_64FC3);
	for(int r=0;r<imagef.rows;r++){
		double avgb,avgg,avgr;
		avgb = avgg =avgr = 0.0;
		for (int c = 0; c < imagef.cols; c++){
			avgb +=imagef.at<cv::Vec3d>(r,c)[0];
			avgg +=imagef.at<cv::Vec3d>(r,c)[1];
			avgr +=imagef.at<cv::Vec3d>(r,c)[2];
		}
		avgb = avgb/imagef.cols;
		avgg = avgg/imagef.cols;
		avgr = avgr/imagef.cols;
		for (int c = 0; c < imagef.cols; c++){
			imagef.at<cv::Vec3d>(r,c)[0] = avgb;
			imagef.at<cv::Vec3d>(r,c)[1] = avgg;
			imagef.at<cv::Vec3d>(r,c)[2] = avgr;
		}
	}	
	
	cv::Mat img;
	imagef.convertTo(img,CV_8UC3);

	// Check for failure
	if (image.empty()) {
		std::cout << "Could not open or find the image" << std::endl;
		std::cin.get(); //wait for any key press
		return -1;
	}

	cv::imwrite(argv[2],img);
	return(0);
}
