#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>


// imshow(winname, mat) -> None
// . The function may scale the image, depending on its depth:
// . - If the image is 8-bit unsigned, it is displayed as is.
// . - If the image is 16-bit unsigned or 32-bit integer, the pixels are divided by 256. 
//     That is, the value range [0,255\*256] is mapped to [0,255].
// . - If the image is 32-bit or 64-bit floating-point, the pixel values are multiplied by 255. That is, the
// .   value range [0,1] is mapped to [0,255].

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



cv::Mat gamma(cv::Mat image){
  cv::Mat ret(image.rows,image.cols,CV_32FC3);
  for(int r = 0 ; r<image.rows;r++){
    for (int c = 0; c < image.cols; c++){
      for(int ch = 0;ch<image.channels();ch++){
        if(image.at<cv::Vec3f>(r,c)[ch]<=0.0031308)
          ret.at<cv::Vec3f>(r,c)[ch] = 12.92*image.at<cv::Vec3f>(r,c)[ch];
        else
          ret.at<cv::Vec3f>(r,c)[ch] = 1.055*std::pow(image.at<cv::Vec3f>(r,c)[ch],1/2.2) - 0.055;
      }
    }
  }
  return(ret);
}


cv::Mat LLenhancement(const cv::Mat& Luminance){
    cv::Mat logL;
    cv::log(1 + Luminance, logL);
    logL = logL/cv::log(10);
    double minVal; 
    double maxVal; 
    cv::Point minLoc; 
    cv::Point maxLoc;

    cv::minMaxLoc( logL, &minVal, &maxVal, &minLoc, &maxLoc );
    logL = (logL - minVal)/(maxVal-minVal); //[0,b]
    cv::Mat M(logL.rows,logL.cols,CV_32FC1);
    // cv::exp(logL,M);
    for(int r = 0 ; r<M.rows;r++){
        for (int c = 0; c < M.cols; c++){
            M.at<float>(r,c) = std::pow(10.0,logL.at<float>(r,c));
        }
    }

    cv::minMaxLoc( M, &minVal, &maxVal, &minLoc, &maxLoc );
    
    M = ((M - minVal)/(maxVal-minVal));
    return(M);
}

cv::Mat HistEquivalization(const cv::Mat& img){
    cv::Mat M(img.rows,img.cols,CV_32FC1);
    int numbins = 256;

    std::vector<float> hist(numbins);
    for (int i = 0; i < numbins; i++) hist[i] = 0;
    
    for(int r = 0 ; r < img.rows;r++){
        for (int c = 0; c < img.cols; c++){
            if((int)img.at<float>(r,c)*numbins<numbins)
                hist[(int)img.at<float>(r,c)*numbins] += 1;
            else
                hist[numbins - 1] += 1;
        }
    }
    //cdf
    for (int i = 1; i < numbins; i++) hist[i] = hist[i] + hist[i-1];
    for (int i = 0; i < numbins; i++) hist[i] = hist[i]/(img.rows*img.cols);
    for(int r = 0 ; r < img.rows;r++){
        for (int c = 0; c < img.cols; c++){
            if((int)img.at<float>(r,c)*numbins<numbins)
                M.at<float>(r,c) = hist[(int)img.at<float>(r,c)*numbins];
            else
                M.at<float>(r,c) = hist[numbins - 1];
        }
    }

    return(M);
}

cv::Mat unsharpMasking(const cv::Mat& img){
    cv::Mat Blurred;
    cv::GaussianBlur(img,Blurred,cv::Size(0,0),0.3,0.3);
    cv::Mat details = img - Blurred;
    cv::Mat M = details*100 + Blurred;
    return(M);
}

int main(int argc, char** argv){
    
    cv::Mat image = cv::imread(argv[1],CV_LOAD_IMAGE_UNCHANGED);
    std::cout<<type2str(image.type())<<"\n";

    // image = image*1000;
    // cv::resize(image, image, cv::Size(image.cols/2, image.rows/2)); // to half size or even smaller
    // image = toneMapping(image);
    double r,b,g;
    r = 0.299;
    b = 0.144;
    g = 0.587;
    cv:: Mat Luminance(image.rows,image.cols,CV_32FC1);;
    for(int r = 0;r<image.rows;r++){
        for(int c = 0 ;c<image.cols; c++){
            Luminance.at<float>(r,c) = b*image.at<cv::Vec3f>(r,c)[0] + g*image.at<cv::Vec3f>(r,c)[1] + r*image.at<cv::Vec3f>(r,c)[2];
        }
    }
    
    cv::Mat M = LLenhancement(Luminance);


    // M = HistEquivalization(Luminance);//order is important because the range with change otherwise
    M = unsharpMasking(M);

    cv::Mat coloured(image.rows,image.cols,CV_32FC3);
    for(int r = 0 ; r<image.rows;r++){
        for (int c = 0; c < image.cols; c++){
            float Lin= Luminance.at<float>(r,c);
            int k = 250;
            coloured.at<cv::Vec3f>(r,c)[0] = k*(float)((image.at<cv::Vec3f>(r,c)[0]/Luminance.at<float>(r,c))*M.at<float>(r,c));
            coloured.at<cv::Vec3f>(r,c)[1] = k*(float)((image.at<cv::Vec3f>(r,c)[1]/Luminance.at<float>(r,c))*M.at<float>(r,c));
            coloured.at<cv::Vec3f>(r,c)[2] = k*(float)((image.at<cv::Vec3f>(r,c)[2]/Luminance.at<float>(r,c))*M.at<float>(r,c));
        }
    }

    coloured = gamma(coloured);
    // std::cout<<coloured;
    cv::namedWindow( "Display window",CV_WINDOW_FREERATIO);
    // cv::imshow( "Display window", M);
    // cv::imshow( "Display window", Luminance);
    // cv::imshow( "Display window", logL);
    cv::imshow( "Display window", coloured);
    // cv::imshow( "Display window", image);


    cv::waitKey(0); 
    cv::destroyAllWindows();	
    return(0);
}
