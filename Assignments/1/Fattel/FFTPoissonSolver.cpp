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

cv::Mat Laplac(const cv::Mat& M){
  cv::Mat Lap;
  int kernel_size = 3;
  int scale = 1;
  int delta = 0;
  int ddepth = CV_64F;
  cv::Laplacian( M, Lap,ddepth, kernel_size,scale,delta,cv::BORDER_DEFAULT );
  for(int i = 0; i < Lap.rows; i++){
    Lap.at<double>(i,0) = 0;
    Lap.at<double>(i,Lap.cols - 1) = 0;
  }
  for(int i = 0; i < Lap.cols; i++){
    Lap.at<double>(0,i) = 0;
    Lap.at<double>( Lap.rows - 1 ,i) = 0;
  }
  return(Lap);
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
// void showdft(const cv::Mat& M){
//     cv::Mat SpArr[2];
//     cv::split(M,SpArr);
//     cv::Mat Mag;
//     cv::magnitude(SpArr[0],SpArr[1],Mag);
//     Mag += cv::Scalar::all(1);
//     cv::log(Mag,Mag);
//     cv::normalize(Mag,Mag,0,1,CV_MINMAX);
//     recenter(Mag);
//     cv::namedWindow( "DFT",CV_WINDOW_FREERATIO);
//     cv::imshow("DFT",Mag);
//     // cv::normalize(Mag,Mag,0,1,CV_MINMAX);
//     cv::waitKey(0); 
//     cv::destroyAllWindows();
// }

void dftImage(const cv::Mat& G,cv::Mat& complexI){
    int m = cv::getOptimalDFTSize( G.rows );
    int n = cv::getOptimalDFTSize( G.cols );
    // cv::normalize(G,G,0,1,CV_MINMAX);
    cv::Mat padded;
    cv::copyMakeBorder(G, padded, 0, m - G.rows, 0, n - G.cols,cv::BORDER_CONSTANT, cv::Scalar::all(0));
    // std::cout<<G;
    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::merge(planes, 2, complexI);         // Add to the expanded another plane with zero
    cv::dft(complexI, complexI,cv::DFT_COMPLEX_OUTPUT);
}



cv::Mat solvePoisson( const cv::Mat& F,const cv::Mat& boundaryImage){
  cv::Mat extendedF(2*(F.rows + 1),F.cols,CV_64F);
  for(int i = 0; i < F.cols; i++ ){
    extendedF.at<double>(0,i) = 0;
    extendedF.at<double>(F.rows + 1,i) = 0;
  }
  for(int i = 0; i < F.rows; i++){
    for(int j =0; j < F.cols; j++){
      extendedF.at<double>(i+1,j) = F.at<double>(i,j);
    }
  }
  for(int i = F.rows + 2; i < extendedF.rows; i++){
    for(int j = 0; j < F.cols; j++){
      extendedF.at<double>(i,j) = -F.at<double>(2*F.rows - i + 1 ,j);
    }
  }
  cv::Mat complexI;
  dftImage(F,complexI);
  // dftImage(extendedF,complexI);
  for(int r = 0;r<complexI.rows;r++){
      for(int c = 0 ;c<complexI.cols; c++){
          double M,N;
          M = complexI.rows; N = complexI.cols;
          double p = r - 1;
          double q = c - 1;
          // double k = -4*M_PI*M_PI*((p*p)/(Lx*Lx) + ((q*q)/(Ly*Ly) ));
          double k = -4*((sin(M_PI*p/(M)))*(sin(M_PI*p/(M))) +  (sin(M_PI*q/(N)))*(sin(M_PI*q/(N))))  ;
          if(abs(k)<0.001) continue;
          complexI.at<cv::Vec2f>(r,c)[0] = complexI.at<cv::Vec2f>(r,c)[0]/k;
          complexI.at<cv::Vec2f>(r,c)[1] = complexI.at<cv::Vec2f>(r,c)[1]/k;

      }
  }
  
  complexI.at<cv::Vec2f>(0,0)[0] = 0;
  complexI.at<cv::Vec2f>(0,0)[1] = 0;


  cv::Mat FinalImage;

  cv::dft(complexI,FinalImage,cv::DFT_INVERSE|cv::DFT_SCALE);
  
  cv::Mat components[2];
  cv::split(FinalImage,components);
  double min, max;
  cv::minMaxLoc(components[0], &min, &max);
  std::cout<<"Min "<<min<<" Max "<<max<<"\n";
  components[0] = (components[0] - min)/(max - min);
  
  clipping(components[0]);
  // cv::Mat dis;
  // cv::resize(components[0],dis,cv::Size(F.cols,(int)(1.3*F.rows)));
  return(components[0]);
}

cv::Mat gamma(cv::Mat image){
  cv::Mat ret(image.rows,image.cols,CV_64FC3);
  for(int r = 0 ; r<image.rows;r++){
    for (int c = 0; c < image.cols; c++){
      for(int ch = 0;ch<image.channels();ch++){
        if(image.at<cv::Vec3d>(r,c)[ch]<=0.0031308)
          ret.at<cv::Vec3d>(r,c)[ch] = 12.92*image.at<cv::Vec3d>(r,c)[ch];
        else
          ret.at<cv::Vec3d>(r,c)[ch] = 1.055*std::pow(image.at<cv::Vec3d>(r,c)[ch],1/2.2) - 0.055;
      }
    }
  }
  return(ret);
}

int main(int argc, char** argv){
    cv::Mat image = cv::imread(argv[1],CV_LOAD_IMAGE_UNCHANGED);
    image.convertTo(image,CV_64FC3);
    std::cout<<type2str(image.type())<<"\n";
    std::cout<< image.size()<<"\n";
    // image = image*1000;
    // cv::resize(image, image, cv::Size(image.cols/2, image.rows/2)); // to half size or even smaller
    // image = toneMapping(image);
    double r,b,g,logavg;
    r = 0.299;
    b = 0.144;
    g = 0.587;
    cv:: Mat Luminance(image.rows,image.cols,CV_64FC1);
    for(int r = 0;r<image.rows;r++){
        for(int c = 0 ;c<image.cols; c++){
            Luminance.at<double>(r,c) = b*image.at<cv::Vec3d>(r,c)[0] + g*image.at<cv::Vec3d>(r,c)[1] + r*image.at<cv::Vec3d>(r,c)[2];
            logavg += log(0.0001 + Luminance.at<double>(r,c));
            // logavg += Luminance.at<double>(r,c);
        }
    }
    logavg = exp(logavg/((image.cols)*(image.rows)));
    std::cout<<"logavg "<<logavg<<"\n";

    double a = 0.18;
    cv::Mat L(image.rows,image.cols,CV_64FC1);
    L = (a*(Luminance)/logavg);
    cv::Mat boundaryImage = L.clone();
    for(int i = L.rows/4 ;i < L.rows/2 ; i++ ){
      for(int j = L.cols/4; j< L.cols/2 ; j++ ){
        boundaryImage.at<double>(i,j) = 0;
      }
    }
    // cv::resize(L,L,cv::Size(64,64));
    cv::Mat Lap = Laplac(L);
    cv::Mat solved = solvePoisson(Lap,boundaryImage);
    // cv::Mat complexI;
    // // dftImage(L,complexI);
    // dftImage(Lap,complexI);
    // // dftImage(G,complexI);
    
    // // getlhsF(complexI);
    
    // for(int r = 0;r<complexI.rows;r++){
    //     for(int c = 0 ;c<complexI.cols; c++){
    //         double M,N;
    //         M = complexI.rows; N = complexI.cols;
    //         double p = r - 1;
    //         double q = c - 1;
    //         // double k = -4*M_PI*M_PI*((p*p)/(Lx*Lx) + ((q*q)/(Ly*Ly) ));
    //         double k = -4*((sin(M_PI*p/(M)))*(sin(M_PI*p/(M))) +  (sin(M_PI*q/(N)))*(sin(M_PI*q/(N))))  ;
    //         if(abs(k)<0.001) continue;
    //         complexI.at<cv::Vec2f>(r,c)[0] = complexI.at<cv::Vec2f>(r,c)[0]/k;
    //         complexI.at<cv::Vec2f>(r,c)[1] = complexI.at<cv::Vec2f>(r,c)[1]/k;

    //     }
    // }
    
    // cv::Mat FinalImage;
    
    // cv::dft(complexI,FinalImage,cv::DFT_INVERSE|cv::DFT_SCALE);
    // cv::Mat components[2];
    // cv::split(FinalImage,components);
    // // std::cout<<components[0];
    // double min, max;
    // cv::minMaxLoc(components[0], &min, &max);
    // std::cout<<"Min "<<min<<" Max "<<max<<"\n";
    cv::namedWindow( "Laplacian",CV_WINDOW_FREERATIO);
    cv::imshow( "Laplacian", Lap);

    cv::namedWindow( "Original",CV_WINDOW_FREERATIO);
    cv::imshow( "Original", L);

    cv::namedWindow( "Rapid Poisson Solver",CV_WINDOW_FREERATIO);
    cv::imshow( "Rapid Poisson Solver", solved);



    cv::waitKey(0); 
    cv::destroyAllWindows();	
    return(0);
}
