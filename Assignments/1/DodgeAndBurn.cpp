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

// void clipping(cv::Mat &M ){
//     for(int r = 0 ; r<M.rows;r++){
//         for (int c = 0; c < M.cols; c++){
//             for(int ch = 0;ch< M.channels();ch++){
//                 if(M.at<cv::Vec3f>(r,c)[ch]>1) M.at<cv::Vec3f>(r,c)[ch] = 1;
//             }
//         }
//     }
// }

void clipping(cv::Mat &M ){
    for(int r = 0 ; r<M.rows;r++){
        for (int c = 0; c < M.cols; c++){
            if(M.at<float>(r,c)<0){
                M.at<float>(r,c) = 0;
            }
            if(M.at<float>(r,c)>1){
                M.at<float>(r,c) = 1;
            }
        }
    }
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
// static float  sigma_0, sigma_1;
// #define SIGMA_I(i)       (sigma_0 + ((float)i/8)*(sigma_1 - sigma_0))
// #define S_I(i)           (exp (SIGMA_I(i)))

int main(int argc, char** argv){
    // sigma_0 = log(2);
    // sigma_1 = log(43);
    cv::Mat image = cv::imread(argv[1],CV_LOAD_IMAGE_UNCHANGED);
    image.convertTo(image,CV_32FC3);
    // plotHistogram(image);
    // std::cout<<type2str(image.type())<<"\n";
    
    float r,b,g;
    r = 0.299;
    b = 0.144;
    g = 0.587;
    cv:: Mat Luminance(image.rows,image.cols,CV_32FC1);;
    float logavg = 0;
    for(int r = 0;r<image.rows;r++){
        for(int c = 0 ;c<image.cols; c++){
            Luminance.at<float>(r,c) = b*image.at<cv::Vec3f>(r,c)[0] + g*image.at<cv::Vec3f>(r,c)[1] + r*image.at<cv::Vec3f>(r,c)[2];
            logavg += log(0.0001 + Luminance.at<float>(r,c));
            // logavg += Luminance.at<float>(r,c);
        }
    }
    double minVal;
    double maxVal;
    cv::Point minLoc;
    cv::Point maxLoc;
    cv::minMaxLoc( Luminance, &minVal, &maxVal, &minLoc, &maxLoc );
    std::cout<<"MaxVal " << maxVal<<"\n";
    std::cout<<"MinVal " << minVal<<"\n";
    // std::cout<<(logavg)/((image.cols)*(image.rows))<<"\n";

    logavg = exp(logavg/((image.cols)*(image.rows)));
    std::cout<<"logavg "<<logavg<<"\n";

    float a = 0.72;
    cv::Mat L(image.rows,image.cols,CV_32FC1);
    L = (a*(Luminance)/logavg);
    float alpha = 1.6;
    float s = 0.35; // 1/2*root(2)

    std::vector<cv::Mat> Vi;
    // // Just 8 scales
    for (int i = 0; i < 8; i++){
        cv::Mat response;
        // cv::GaussianBlur(L,response,cv::Size(0,0),S_I(i),S_I(i));
        cv::GaussianBlur(L,response,cv::Size(0,0),s,s);
        s = alpha*s;
        Vi.push_back(response);
    }
    float thershold = 0.05;
    float phi = 8.0;
    s = 0.35;

    std::vector<cv::Mat> V(7);
    for(int i = 0 ; i<7 ; i++){
        V[i] = (Vi[i] - Vi[i+1])/(((std::pow(2,phi)*a) /(s*s) )+ Vi[i] );
        s = alpha*s;
    }
    // cv::Mat selectScale(image.rows,image.cols,CV_32S);
    // selectScale = cv::Scalar(-1);
    cv::Mat Ld(image.rows,image.cols,CV_32FC1);
    int arr[8];
    arr[0 ] = 0;arr[1] = 0;arr[2] = 0;arr[3] = 0;arr[4] = 0;arr[5] = 0;arr[6] = 0;arr[7] = 0;
    for(int r = 0;r<image.rows;r++){
        for(int c = 0 ;c<image.cols; c++){
            for(int sm = 0; sm<V.size(); sm++ ){
                if( fabs(V[sm].at<float>(r,c)) > thershold || sm == V.size()-1 ){
                    // std::cout<<fabs(V[sm].at<float>(r,c))<<" , "<<sm<<"\n";
                    arr[sm] += 1;
                    Ld.at<float>(r,c) = L.at<float>(r,c)/(1 + Vi[sm].at<float>(r,c));
                    break;
                }
            }
        }
    }
    // std::cout<<"_________\n";
    // for(int i =0;i<8;i++){
    //     std::cout<<arr[i]<<"\n";
    // }
    // std::cout<<"_________\n";


    clipping(Ld);
    cv::Mat coloured(image.rows,image.cols,CV_32FC3);
    for(int r = 0 ; r<coloured.rows;r++){
        for (int c = 0; c < coloured.cols; c++){
            float Lin= Luminance.at<float>(r,c);
            // float Lin = L.at<float>(r,c);
            int k = 50.0;
            coloured.at<cv::Vec3f>(r,c)[0] = k*(float)((image.at<cv::Vec3f>(r,c)[0]/Lin)*Ld.at<float>(r,c));
            coloured.at<cv::Vec3f>(r,c)[1] = k*(float)((image.at<cv::Vec3f>(r,c)[1]/Lin)*Ld.at<float>(r,c));
            coloured.at<cv::Vec3f>(r,c)[2] = k*(float)((image.at<cv::Vec3f>(r,c)[2]/Lin)*Ld.at<float>(r,c));
        }
    }

    if(image.rows>1000 || image.cols>1000) cv::resize(coloured,coloured,cv::Size(780,1000));
    if(image.rows>1000 || image.cols>1000) cv::resize(Ld,Ld,cv::Size(780,1000));

    coloured = gamma(coloured);
    // image = gamma(image);
    // plotHistogram(image);
    // plotHistogram(coloured);
    cv::minMaxLoc( Ld, &minVal, &maxVal, &minLoc, &maxLoc );
    
    std::cout<<"MaxVal Ld " << maxVal<<"\n";
    std::cout<<"MinVal Ld" << minVal<<"\n";
    
    // plotHistogram(Ld);
    imwrite( "./DogeAndBurnImages/img.jpg",coloured*255 );
    cv::namedWindow( "DogeAndBurn",CV_WINDOW_FREERATIO);
    // cv::imshow( "4", image);
    cv::imshow( "DogeAndBurn", coloured);
    // cv::imshow( "4", Ld);
    cv::waitKey(0);
    cv::destroyAllWindows();	
    return(0);
}
