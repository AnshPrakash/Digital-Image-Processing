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

void recenter(cv::Mat& M){
  	M = M(cv::Rect(0, 0, M.cols & -2, M.rows & -2));
    int centerX = M.cols/2;
    int centerY = M.rows/2;
    cv::Mat q1(M,cv::Rect(0,0,centerX,centerY));
    cv::Mat q2(M,cv::Rect(centerX,0,centerX,centerY));
    cv::Mat q3(M,cv::Rect(0,centerY,centerX,centerY));
    cv::Mat q4(M,cv::Rect(centerX,centerY,centerX,centerY));
    cv::Mat swapMap;
    q1.copyTo(swapMap);
    q4.copyTo(q1);
    swapMap.copyTo(q4);
    q2.copyTo(swapMap);
    q3.copyTo(q2);
    swapMap.copyTo(q3);
}


void showdft(const cv::Mat& M){
    cv::Mat SpArr[2];
    cv::split(M,SpArr);
    cv::Mat Mag;
    cv::magnitude(SpArr[0],SpArr[1],Mag);
    Mag += cv::Scalar::all(1);
    cv::log(Mag,Mag);
    cv::normalize(Mag,Mag,0,1,CV_MINMAX);
    recenter(Mag);
    cv::namedWindow( "DFT",CV_WINDOW_FREERATIO);
    cv::imshow("DFT",Mag);
    // cv::normalize(Mag,Mag,0,1,CV_MINMAX);
    cv::waitKey(0); 
    cv::destroyAllWindows();
}


float gaussfunc(float sigma, double x , double x0 , double y ,double y0){
    double val = 0;
    return(exp(-((x-x0)*(x-x0)+(y-y0)*(y - y0))/(2*sigma*sigma) ));
}
bool comp(int a, int b) 
{ 
    return (a < b); 
} 
cv::Mat gaussKer(float sigma){
  int size = std::max((int)(sigma*4),3,comp);
  cv::Mat ker(size+2,size+2,CV_32F,cv::Scalar(0));
  float sum = 0;
  for (int i = 1; i <= size; i++){
      for (int j = 1; j <= size; j++){
          ker.at<float>(i,j) = gaussfunc(sigma,i,size/2,j,size/2);
          sum += ker.at<float>(i,j);
      }
  }
  ker = ker/sum;
  // std::cout<< ker<<"\n";
  return(ker);
}
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

void idftImage(const cv::Mat& G,cv::Mat& complexI){
    cv::Mat M[2];
    cv::split(G,M);
    int m = cv::getOptimalDFTSize( M[0].rows );
    int n = cv::getOptimalDFTSize( M[0].cols );
    // cv::normalize(G,G,0,1,CV_MINMAX);
    cv::Mat padded[2];
    cv::copyMakeBorder(M[0], padded[0], 0, m - M[0].rows, 0, n - M[0].cols,cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::copyMakeBorder(M[1], padded[1], 0, m - M[0].rows, 0, n - M[0].cols,cv::BORDER_CONSTANT, cv::Scalar::all(0));

    // std::cout<<G;
    // cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::merge(padded, 2, complexI);         // Add to the expanded another plane with zero
    cv::dft(complexI, complexI,cv::DFT_INVERSE|cv::DFT_SCALE|cv::DFT_REAL_OUTPUT);
}



cv::Mat addGaussianNoise(const cv::Mat& M){
    cv::Mat L = M.clone();
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.5,0.5);
    float randn = distribution(generator);
    for(int i = 0;i < L.rows; i++){
        for(int j = 0; j < L.cols; j++){
            randn = distribution(generator);
            L.at<float>(i,j) = L.at<float>(i,j) + fabs(randn)/10;
        }
    }
    return(L);
}

double PSNR(const cv::Mat& I1,const cv::Mat& I2,int type){
    cv::Mat I = I1.clone();
    cv::Mat K = I2.clone();
    I.convertTo(I,CV_32F);
    K.convertTo(K,CV_32F);
    double MSE = 0;
    double MAXi = 1.0;
    MAXi = (type == 1)? 255:1;
    // std::cout<<"Maxi "<<MAXi<<"\n";
    for(int i = 0; i < I.rows; i++){
        for(int j = 0; j < I.cols; j++ ){
            MSE += (I.at<float>(i,j) - K.at<float>(i,j))*(I.at<float>(i,j) - K.at<float>(i,j));
        }
    }
    MSE = MSE/(I.rows*I.cols);
    double PSNR = 10*log10((MAXi*MAXi)/MSE);
    // std::cout<<"s: "<<PSNR<<" "<<MSE<<" "<<(MAXi*MAXi)/MSE<<" \n____________\n";
    return(PSNR);
}

cv::Mat WienerFilter(cv::Mat& H,const cv::Mat& G,double r){
  //conjugate
  // std::cout<<type2str(G.type())<<" \n";
  for(int u = 0; u < H.rows; u++){
      for(int v = 0; v < H.cols; v++){
          H.at<cv::Vec2f>(u,v)[1] = -H.at<cv::Vec2f>(u,v)[1];
      }
  }
  cv::Mat F_p;
  cv::Mat SpArr[2];
  cv::split(H,SpArr);
  cv::Mat Mag;
  cv::magnitude(SpArr[0],SpArr[1],Mag);
  Mag = Mag.mul(Mag);
  std::vector<cv::Mat> vec;
  vec.push_back(Mag.clone());
  vec.push_back(Mag.clone());
  cv::merge(vec,Mag);
  cv::Mat buff = H/(Mag + r);
  // std::cout<<type2str(buff.type())<<" "<<type2str(G.type());
  cv::mulSpectrums(G,buff,F_p,false);
  // F_p =  (H/(Mag + r))*G; // H have two channels while Mag have one channel check for this
  cv::Mat complex;
  idftImage(F_p,complex);
  cv::split(complex,SpArr);
  // return(H);
  return(SpArr[0]);

}




cv::Mat Blur(cv::Mat& M,cv::Mat ker){
  cv::Mat fft_M,fft_ker;
  cv::Mat padded_ker(M.rows,M.cols,CV_32F,cv::Scalar(0));;// = pad_recenterKer(M.rows,M.cols,ker);;
  ker.copyTo(padded_ker(cv::Rect(0,0, ker.cols,ker.rows)));
  dftImage(M,fft_M);
  dftImage(padded_ker,fft_ker);
  cv::Mat fft_G;
  cv::mulSpectrums(fft_M,fft_ker,fft_G,false);
  cv::Mat G;
  idftImage(fft_G,G);
  return(G);
}

int main(int argc, char** argv){
    cv::Mat image = cv::imread(argv[1],CV_LOAD_IMAGE_GRAYSCALE);
    image.convertTo(image,CV_32F);
    image = image/255;
    std::cout<<type2str(image.type())<<"\n";
    // cv::resize(image,image,cv::Size(16,16));
    std::cout<< image.size()<<"\n";
    
    cv::Mat blurred = Blur(image,gaussKer(5));
    cv::Mat ker = gaussKer(5);
    cv::Mat padded_ker(image.rows,image.cols,CV_32F,cv::Scalar(0));;// = pad_recenterKer(M.rows,M.cols,ker);;
    ker.copyTo(padded_ker(cv::Rect(0,0, ker.cols,ker.rows)));    
    cv::Mat H;
    cv::Mat dft_blurr;
    dftImage(blurred,dft_blurr);
    dftImage(padded_ker,H);
    cv::Mat filtered = WienerFilter(H,dft_blurr,0.000);
    // cv::filter2D(image,blurred,CV_32F,gaussKer(2,3));

    cv::namedWindow( "Original",CV_WINDOW_FREERATIO);
    cv::imshow( "Original",image);

    cv::namedWindow( "Blurred",CV_WINDOW_FREERATIO);
    cv::imshow( "Blurred",blurred);


    cv::namedWindow( "filtered",CV_WINDOW_FREERATIO);
    cv::imshow( "filtered",filtered);

    cv::waitKey(0); 
    cv::destroyAllWindows();	
    return(0);
}
