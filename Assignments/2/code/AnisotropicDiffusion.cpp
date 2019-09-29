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
void plotHistogram(cv::Mat img){
    std::vector<cv::Mat> bgr_planes;
    cv::split( img, bgr_planes );
    /// Establish the number of bins
    int histSize = 256;

    double minVal; 
    double maxVal;
    cv::Point minLoc; 
    cv::Point maxLoc;
    cv::minMaxLoc( bgr_planes[0], &minVal, &maxVal, &minLoc, &maxLoc );

    float range[] = { (float)minVal, (float)maxVal } ;
    // float range[] = { 0, 256 } ;
    const float* histRange = { range };
    bool uniform = true; bool accumulate = false;
    // cv::Mat b_hist, g_hist, r_hist;

    std::vector<cv::Mat> hist(img.channels());

    /// Compute the histograms:
    for (int i = 0; i < img.channels(); i++){
        cv::calcHist( &bgr_planes[i], 1, 0, cv::Mat(), hist[i], 1, &histSize, &histRange, uniform, accumulate );
    }
    
    
    // Draw the histograms for B, G and R
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );
    cv::Mat histImage( hist_h, hist_w, CV_8UC3, cv::Scalar( 0,0,0) );
    /// Normalize the result to [ 0, histImage.rows ]

    for (int i = 0; i < img.channels(); i++){
        cv::normalize(hist[i], hist[i], 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
    }
    
    /// Draw for each channel
    std::vector<int>  Bcolors = {255,0,0};
    std::vector<int>  Gcolors = {0,255,0};
    std::vector<int>  Rcolors = {0,0,255};

    for( int i = 1; i < histSize; i++ )
    {
        for (int j = 0; j < img.channels(); j++){
            cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(hist[j].at<float>(i-1)) ) ,
                    cv::Point( bin_w*(i), hist_h - cvRound(hist[j].at<float>(i))),
                    cv::Scalar( Bcolors[j], Gcolors[j], Rcolors[j]), 2, 8, 0  );
        }
    }
    
    /// Display
    // imwrite( "./DogeAndBurnImages/img.jpg",histImage );
    cv::namedWindow("calcHist Demo", CV_WINDOW_FREERATIO );
    cv::imshow("calcHist Demo", histImage );
    cv::waitKey(0);
    cv::destroyAllWindows();	
}

void clipping(cv::Mat& M){
  for(int i = 0 ;i<M.rows;i++){
    for(int j = 0;j<M.cols;j++ ){
      if(M.at<float>(i,j) < 0) M.at<float>(i,j) = 0;
      if(M.at<float>(i,j) > 1) M.at<float>(i,j) = 1;
    }
  }
}


cv::Mat addGaussianNoise(const cv::Mat& M){
    cv::Mat L = M.clone();
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.7,0.9);
    float randn = distribution(generator);
    for(int i = 0;i < L.rows; i++){
        for(int j = 0; j < L.cols; j++){
            randn = distribution(generator);
            L.at<float>(i,j) = L.at<float>(i,j) + fabs(randn)/7;
        }
    }
    return(L);
}


cv::Mat Lapn(const cv::Mat& M){
  cv::Mat Lap;
  int kernel_size = 3;
  cv::Mat ker(3,1,CV_32F);
  ker.at<float>(0) =  1;
  ker.at<float>(1) = -1;
  ker.at<float>(2) =  0;
  // cv::Mat ker(3,3,CV_32F);
  // ker.at<float>(0,0) = 0;ker.at<float>(0,0) = 1;ker.at<float>(0,0) = 0;
  // ker.at<float>(0,0) = 0;ker.at<float>(0,0) = -1;ker.at<float>(0,0) = 0;
  // ker.at<float>(0,0) = 0;ker.at<float>(0,0) = 0;ker.at<float>(0,0) = 0;

  cv::filter2D(M,Lap,CV_32F,ker);
  return(Lap);
}


cv::Mat Laps(const cv::Mat& M){
  cv::Mat Lap;
  int kernel_size = 3;
  cv::Mat ker(3,1,CV_32F);
  ker.at<float>(0) = 0;
  ker.at<float>(1) = -1;
  ker.at<float>(2) = 1;
  // cv::Mat ker(3,3,CV_32F);
  // ker.at<float>(0,0) = 0;ker.at<float>(0,0) = 0;ker.at<float>(0,0) = 0;
  // ker.at<float>(0,0) = 0;ker.at<float>(0,0) = -1;ker.at<float>(0,0) = 0;
  // ker.at<float>(0,0) = 0;ker.at<float>(0,0) = 1;ker.at<float>(0,0) = 0;

  cv::filter2D(M,Lap,CV_32F,ker);
  return(Lap);
}


cv::Mat Lape(const cv::Mat& M){
  cv::Mat Lap;
  int kernel_size = 3;
  cv::Mat ker(1,3,CV_32F);
  ker.at<float>(0) =  0;
  ker.at<float>(1) = -1;
  ker.at<float>(2) = 1;  

  // cv::Mat ker(3,3,CV_32F);
  // ker.at<float>(0,0) = 0;ker.at<float>(0,0) = 0;ker.at<float>(0,0) = 0;
  // ker.at<float>(0,0) = 0;ker.at<float>(0,0) = -1;ker.at<float>(0,0) = 1;
  // ker.at<float>(0,0) = 0;ker.at<float>(0,0) = 0;ker.at<float>(0,0) = 0;


  cv::filter2D(M,Lap,CV_32F,ker);
  return(Lap);
}


cv::Mat Lapw(const cv::Mat& M){
  cv::Mat Lap;
  int kernel_size = 3;
  cv::Mat ker(1,3,CV_32F);
  ker.at<float>(0) =  1;
  ker.at<float>(1) = -1;
  ker.at<float>(2) =  0;
  // cv::Mat ker(3,3,CV_32F);
  // ker.at<float>(0,0) = 0;ker.at<float>(0,0) = 0;ker.at<float>(0,0) = 0;
  // ker.at<float>(0,0) = 1;ker.at<float>(0,0) = -1;ker.at<float>(0,0) = 0;
  // ker.at<float>(0,0) = 0;ker.at<float>(0,0) = 0;ker.at<float>(0,0) = 0;

  cv::filter2D(M,Lap,CV_32F,ker);
  return(Lap);
}

cv::Mat g(const cv::Mat& Lp){
  cv::Mat V = cv::abs(Lp);
  double k = 0.08; // select in some way
  V = V/k;
  V = V.mul(V);
  cv::Mat res;
  cv::exp(-V,res);
  return(res);
}


cv::Mat AnisotropicDiffusion(const cv::Mat& I0){
    int t = 20;
    double lambda = 0.14;
    cv::Mat Ln,Ls,Le,Lw,I;
    cv::Mat Cn,Cs,Ce,Cw;
    I = I0.clone();
    for (int i = 0; i < t; i++){
        Ln = Lapn(I);
        Ls = Laps(I);
        Le = Lape(I);
        Lw = Lapw(I);
        Cn = g(Ln);
        Cs = g(Ls);
        Ce = g(Le);
        Cw = g(Lw);
        // std::cout<<lambda*(Cn.mul(Ln) + Cs.mul(Ls) + Ce.mul(Le) + Cw.mul(Lw));
        I =  I + lambda*(Cn.mul(Ln) + Cs.mul(Ls) + Ce.mul(Le) + Cw.mul(Lw));
    }
    return(I);
}

int main(int argc, char** argv){
    cv::Mat image = cv::imread(argv[1],CV_LOAD_IMAGE_GRAYSCALE);
    image.convertTo(image,CV_32F);
    image = image/255;
    std::cout<<type2str(image.type())<<"\n";
    std::cout<< image.size()<<"\n";

    cv::Mat noised = addGaussianNoise(image);

    // cv::Mat diffused = AnisotropicDiffusion(image);
    cv::Mat diffused = AnisotropicDiffusion(noised);
    // std::cout<<diffused;
    
    cv::namedWindow( "Original",CV_WINDOW_FREERATIO);
    cv::imshow( "Original",image);


    cv::namedWindow( "G_Noise",CV_WINDOW_FREERATIO);
    cv::imshow( "G_Noise",noised);
    

    cv::namedWindow( "Anisotropicdiff",CV_WINDOW_FREERATIO);
    cv::imshow( "Anisotropicdiff",diffused);

    cv::waitKey(0); 
    cv::destroyAllWindows();	
    return(0);
}
