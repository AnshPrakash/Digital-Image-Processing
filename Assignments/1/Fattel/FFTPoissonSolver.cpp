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

void getlhsF(cv::Mat& complexI){
    // recenter(complexI);
    for(int r = 0;r<complexI.rows;r++){
        for(int c = 0 ;c<complexI.cols; c++){
            double Lx,Ly;
            Lx = complexI.rows; Ly = complexI.cols;
            double p = r;// - Lx/2;
            double q = c;// - Ly/2;
            
            // double k = -4*M_PI*M_PI*((p*p)/(Lx*Lx) + ((q*q)/(Ly*Ly) ));
            double k = -4*(sin(M_PI*p/Lx)*sin(M_PI*p/Lx) + sin(M_PI*q/Ly)*sin(M_PI*q/Ly) )/2;
            if(abs(k)<0.1) continue;
            complexI.at<cv::Vec2f>(r,c)[0] = complexI.at<cv::Vec2f>(r,c)[0]/k;
            complexI.at<cv::Vec2f>(r,c)[1] = complexI.at<cv::Vec2f>(r,c)[1]/k;
        }
    }
    // recenter(complexI);
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



// cv::Mat solvePoisson( const cv::Mat& F,const std::vector<double>& bd1a, const std::vector<double>& bd1b,
//                       const std::vector<double>& bd2a, const std::vector<double>& bd2b,
//                       bool add_boundary_to_solution = false){

//   double h1=1.0, h2=1.0, a1=1.0, a2=1.0;
//   double bdvalue = neumann_compat(F,a1,a2,h1,h2);
//   // Poisson equation solver
//   trunc = pde::poisolve(U,F,a1,a2,h1,h2,bdvalue,bdtype,false);
//   size_t n1=F.rows;
//   size_t n2=F.cols;
//   assert(n1>0 && n2>0);

//   // adjust right hand side F with boundary condition (nothing to do for =0)
//   cv::Mat rhs = F.clone();
//   assert( bd1a.size() == bd1b.size() && bd1a.size() == n2 );
//   assert( bd2a.size() == bd2b.size() && bd2a.size() == n1 );
//   double c1,c2;
//   //Neuamann boundary
//   c1=2.0*a1/h1;
//   c2=2.0*a2/h2;
//   for(size_t i=0; i<n2; i++) {
//     rhs.at<0,i>    -= c1 * bd1a[i];
//     rhs.at<n1-1,i> -= c1 * bd1b[i];
//   }
//   for(size_t i=0; i<n1; i++) {
//     rhs.at<i,0>    -= c2 * bd2a[i];
//     rhs.at<i,n2-1> -= c2 * bd2b[i];
//   }
  
//   return(F);
// }
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
    // cv::resize(L,L,cv::Size(64,64));
    cv::Mat Lap = Laplac(L);
    cv::Mat complexI;
    // dftImage(L,complexI);
    dftImage(Lap,complexI);
    // dftImage(G,complexI);
    
    // getlhsF(complexI);
    
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
    
    cv::Mat FinalImage;
    
    cv::dft(complexI,FinalImage,cv::DFT_INVERSE|cv::DFT_SCALE);
    cv::Mat components[2];
    cv::split(FinalImage,components);
    // std::cout<<components[0];
    double min, max;
    cv::minMaxLoc(components[0], &min, &max);
    std::cout<<"Min "<<min<<" Max "<<max<<"\n";
    
    cv::namedWindow( "Laplacian",CV_WINDOW_FREERATIO);
    cv::imshow( "Laplacian", Lap);

    cv::namedWindow( "Original",CV_WINDOW_FREERATIO);
    cv::imshow( "Original", L);

    cv::namedWindow( "Rapid Poisson Solver",CV_WINDOW_FREERATIO);
    cv::imshow( "Rapid Poisson Solver", components[0]);



    cv::waitKey(0); 
    cv::destroyAllWindows();	
    return(0);
}
