#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>
#include "Laplace.h"



// imshow(winname, mat) -> None
// . The function may scale the image, depending on its depth:
// . - If the image is 8-bit unsigned, it is displayed as is.
// . - If the image is 16-bit unsigned or 32-bit integer, the pixels are divided by 256. 
//     That is, the value range [0,255\*256] is mapped to [0,255].
// . - If the image is 32-bit or 64-bit doubleing-point, the pixel values are multiplied by 255. That is, the
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

double neumann_error(const cv::Mat& F)
{
  size_t n1 = F.rows;
  size_t n2 = F.cols;
  assert(n1>1 && n2>1);

  // assuming 0 boundary conditions, otherwise would have to adjust
  // right hand side F as in poisolve()

  // calculate min L2 error ||Laplace U - F|| = |\hat F[0]| * ||EV[0]||
  // where \hat F, are the coordinates of F in EV-space,
  // and can simply be calculated as follows
  double sum=0.0;
  double fac=1.0;
  for(size_t i=0; i<n1; i++) {
    for(size_t j=0; j<n2; j++) {
       fac=1.0;
       if(j==0 || j==n2-1)
          fac*=0.5;
       if(i==0 || i==n1-1)
          fac*=0.5;
       sum += fac*F.at<double>(i,j);
    }
  }
  double F00 = sum/((n1-1)*(n2-1));      // \hat F [0][0] (EV space)
  double norm_ev = sqrt((double)(n1*n2));  // EV[0]=(1,...,1) --> norm=sqrt(n)
  double l2_error = F00*norm_ev;
  return l2_error;
}


void plotHistogram(const cv::Mat& img){
    std::vector<cv::Mat> bgr_planes;
    cv::split( img, bgr_planes );
    for(int i = 0 ; i < img.channels(); i++) bgr_planes[i].convertTo(bgr_planes[i],CV_32F);
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

// given a right hand side F, we can find a constant Neumann-boundary
// value which will have a solution U, so that Laplace U = F,
// returns the boundary value
double neumann_compat(const cv::Mat& F,
                      double a1, double a2, double h1, double h2){
  size_t n1=F.rows;
  size_t n2=F.cols;

  double l2_error = neumann_error(F);
  double norm_ev = sqrt((double)(n1*n2));  // EV[0]=(1,...,1) --> norm=sqrt(n)
  double F00 = l2_error/norm_ev;         // \hat F [0][0] (EV space)

  // with non-zero Neumann boundary condition, rhs F is modified,
  // as in poisolve(), so we can calculate the exact boundary
  // value to make the l2_error zero
  double bd = F00 / (2.0*a1/(h1*(n1-1)) + 2.0*a2/(h2*(n2-1)));
  return bd;
}

cv::Mat addNeumannBoundary(const cv::Mat& M){
  cv::Mat dst;
  int top, bottom, left, right;
  int borderType = cv::BORDER_CONSTANT;
  cv::Scalar value;
  value = cv::Scalar(0);
  top = bottom = left = right = 1;
  cv::copyMakeBorder( M, dst, top, bottom, left, right, borderType, value );
  return(dst);
}

cv::Mat jacobi( const cv::Mat& xk,const cv::Mat& b){
  //xk have boundary added
  int n = b.rows - 1;
  int m = b.cols - 1 ;
  assert(n>0 && m>0); 
  double dx = 1/(double)n;
  double dy = 1/(double)m;
  cv::Mat xkp1(xk.rows,xk.cols,CV_64F);
  for(int i = 1 ; i <= n; i++){
    for(int j = 1; j <= m; j++){
      xkp1.at<double>(i,j) = (b.at<double>(i - 1,j - 1) 
                            - (xk.at<double>(i+1,j) + xk.at<double>(i-1,j))/(dx*dx) 
                            - (xk.at<double>(i,j+1) + xk.at<double>(i,j-1))/(dy*dy) )
                            / ( - 2/(dx*dx) - 2/(dy*dy) );
    }
  }
  return(xkp1);
}



cv::Mat removeBorder(cv::Mat& M){
  cv::Mat BL(M.rows -2,M.cols - 2,CV_64F);
  for(int i = 0 ; i < BL.rows; i++){
    for(int j = 0; j < BL.cols; j++){
      BL.at<double>(i,j) = M.at<double>(i+1,j+1);
    }
  }
  return(BL);
}




cv::Mat multigrid(const cv::Mat& U,const cv::Mat& b ){
  cv::Mat x0 = U.clone();
  x0 = addNeumannBoundary(x0);
  for(int i=0;i<10;i++) {
    x0 = jacobi(x0,b);
  }
  cv::Mat xk;
  x0 = removeBorder(x0);
  cv::Laplacian( x0, xk, CV_64F, 3, 1,0,cv::BORDER_DEFAULT );
  cv::Mat r = xk - b;
  cv::Mat rc;
  cv::pyrDown(r, rc, cv::Size(r.cols/2, r.rows/2));
  cv::Mat ec(rc.rows,rc.cols,CV_64F,cv::Scalar(0));
  ec = addNeumannBoundary(ec);
  for(int i=0; i<10 ;i++) {
    ec = jacobi(ec,rc);
  }
  ec = removeBorder(ec);
  cv::Mat e;
  cv::pyrUp(ec,e, cv::Size(r.cols, r.rows));
  x0 = (x0 - e);
  return(x0);
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
    cv::resize(L,L,cv::Size(64,64));
    cv::Mat Lap;
    cv::GaussianBlur( L, Lap, cv::Size(3,3), 0.3, 0.3,cv::BORDER_DEFAULT);
    int kernel_size = 3;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_64F;
    cv::Laplacian( Lap, Lap, ddepth, kernel_size, scale,delta, cv::BORDER_DEFAULT );
    

    cv::Mat mul_sol(Lap.rows,Lap.cols,CV_64F,cv::Scalar(0));
    cv::Mat jac_sol(Lap.rows,Lap.cols,CV_64F,cv::Scalar(0));
    
    cv::Mat rhs = addNeumannBoundary(Lap);
    jac_sol = addNeumannBoundary(jac_sol);
    for(int i=0;i<1000;i++) {
      jac_sol = jacobi(jac_sol,rhs);
    }
    
    
    for(int i = 0;i<200;i++)  mul_sol = multigrid(mul_sol,Lap);
    std::cout<<mul_sol.size();
    
    // plotHistogram(sol);
    // std::cout<<jac_sol;
    

    cv::namedWindow( "Laplacian",CV_WINDOW_FREERATIO);
    cv::imshow( "Laplacian", Lap);

    cv::namedWindow( "Original",CV_WINDOW_FREERATIO);
    cv::imshow( "Original", L);

    cv::namedWindow( "Multigrid Solution",CV_WINDOW_FREERATIO);
    cv::imshow( "Multigrid Solution", 1000*mul_sol);

    cv::namedWindow( "Jacobi Solution",CV_WINDOW_FREERATIO);
    cv::imshow( "Jacobi Solution", 1000*jac_sol);

    
    // cv::namedWindow( "Error",CV_WINDOW_FREERATIO);
    // cv::imshow( "Error", 1*(L - jac_sol));





    cv::waitKey(0); 
    cv::destroyAllWindows();	
    return(0);
}
