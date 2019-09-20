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
    cv::dft(complexI, complexI,cv::DFT_INVERSE|cv::DFT_SCALE);
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

cv::Mat saltpepperNoise(const cv::Mat& M){
    cv::Mat L = M.clone();
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.5,0.2);
    float randn = distribution(generator);
    for(int i = 0;i < L.rows; i++){
        for(int j = 0; j < L.cols; j++){
            randn = distribution(generator);
            if(fabs(randn) > 0.9 ){
                randn = distribution(generator);
                L.at<float>(i,j) = (randn < 0.5) ? 0 :1;
            }
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

cv::Mat medianfilters(const cv::Mat& img){
    cv::Mat MinMed;
    cv::Mat Med;
    cv::Mat M = img*255;
    M.convertTo(M,CV_8U);
    double minn = -10000000;
    for(int k = 3 ;k < 10 ; k+=2){
        cv::medianBlur(M,Med,k);
        // std::cout<<type2str(Med.type())<<"\n";
        if(minn < PSNR(Med,M,1)){
            minn = PSNR(Med,M,1);
            MinMed = Med;
        }
    }
    MinMed.convertTo(MinMed,CV_32F);
    MinMed = MinMed/255;
    std::cout<<minn<<"\n";
    return(MinMed);
}

cv::Mat MeanKernel(int k ){
    cv::Mat ker(k,k,CV_32F,cv::Scalar::all(1));
    return(ker);
}

cv::Mat meanfilters(const cv::Mat& M){
    cv::Mat MinMean;
    cv::Mat MeanM;
    double minn = -1000000000;
    cv::Mat ker;
    for(int k = 3 ;k < 10 ; k++){
        ker = MeanKernel(k);
        cv::filter2D(M,MeanM,CV_32F,ker);
        MeanM = MeanM/(k*k);
        if(minn < PSNR(MeanM,M,0)){
            minn = PSNR(MeanM,M,0);
            MinMean = MeanM;
        }
    }
    std::cout<<minn<<"\n";
    return(MeanM);
}


int main(int argc, char** argv){
    cv::Mat image = cv::imread(argv[1],CV_LOAD_IMAGE_UNCHANGED);
    image.convertTo(image,CV_32F);
    image = image/255;
    std::cout<<type2str(image.type())<<"\n";
    std::cout<< image.size()<<"\n";
    cv::Mat G = addGaussianNoise(image);
    cv::Mat SP = saltpepperNoise(image);
    // std::cout<< PSNR(image,G,0);
    cv::Mat G_medf,G_Mean;
    // G_medf = G;
    G_medf = medianfilters(G);
    G_Mean = meanfilters(G);

    cv::Mat SP_medf,SP_Mean;
    SP_medf = medianfilters(SP);
    SP_Mean = meanfilters(SP);


    
    
    cv::namedWindow( "Original",CV_WINDOW_FREERATIO);
    cv::imshow( "Original",image);


    cv::namedWindow( "Gaussian",CV_WINDOW_FREERATIO);
    cv::imshow( "Gaussian",G);

    cv::namedWindow( "G_Medf",CV_WINDOW_FREERATIO);
    cv::imshow( "G_Medf",G_medf);
    
    cv::namedWindow( "G_Meanf",CV_WINDOW_FREERATIO);
    cv::imshow( "G_Meanf",G_Mean);
    

    cv::namedWindow( "Salt and Pepper",CV_WINDOW_FREERATIO);
    cv::imshow( "Salt and Pepper",SP);


    cv::namedWindow( "SP_Medf",CV_WINDOW_FREERATIO);
    cv::imshow( "SP_Medf",SP_medf);
    
    cv::namedWindow( "SP_Meanf",CV_WINDOW_FREERATIO);
    cv::imshow( "SP_Meanf",SP_Mean);
    

    cv::waitKey(0); 
    cv::destroyAllWindows();	
    return(0);
}
