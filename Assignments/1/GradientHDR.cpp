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


cv::Mat GradAttentuation(cv::Mat logI){
    const int level = 4;
    cv::Mat Blurred;
    cv::GaussianBlur(logI,Blurred,cv::Size(0,0),0.3,0.3);
    cv::Mat gradx,grady;
    cv::Mat scaling[level];
    cv::Mat phi[level];
    for(int l = 0 ; l<level ; l++){
        Sobel(Blurred, gradx, CV_32FC1, 1, 0, 3);
        Sobel(Blurred, grady, CV_32FC1, 0, 1, 3);
        cv::Mat gradi(gradx.rows,gradx.cols,CV_32FC1);
        cv::Mat scalingMat(gradx.rows,gradx.cols,CV_32FC1);
        for(int r = 0;r<gradi.rows;r++){
            for(int c = 0 ;c<gradi.cols; c++){
                gradi.at<float>(r,c) = std::pow(gradx.at<float>(r,c)*gradx.at<float>(r,c) + grady.at<float>(r,c)*grady.at<float>(r,c),0.5);
                scalingMat.at<float>(r,c) =  (gradi.at<float>(r,c)<0.001) ? 1:std::pow(gradi.at<float>(r,c)/0.1,-0.2);
                // G.at<float>(r,c) = scaling.at<float>(r,c)*grad.at<float>(r,c);
            }
        }    
        scaling[l] = scalingMat;
        cv::Mat out;
        cv::pyrDown(Blurred, out, cv::Size(Blurred.cols/2, Blurred.rows/2));
        Blurred = out;
        // std::cout<< Blurred.rows<<" "<<Blurred.cols<<"\n";
    }
    phi[level-1] = scaling[level-1];
    for(int l = level-2; l>=0; l--){
        cv::Mat Interpolated;
        cv::pyrUp(phi[l+1], Interpolated, cv::Size(scaling[l].cols, scaling[l].rows));
        phi[l] = scaling[l].mul(Interpolated);
    }
    // phi[0] = logI;
    return(phi[0]);
}

void recenter(cv::Mat& M){
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
    cv::Mat logL;
    cv::log(cv::Scalar::all(1)+ Luminance, logL);
    cv::Mat Blurred;
    cv::Mat gradx,grady;
    cv::GaussianBlur(logL,Blurred,cv::Size(0,0),0.3,0.3);
    Sobel(Blurred, gradx, CV_32FC1, 1, 0,3);
    Sobel(Blurred, grady, CV_32FC1, 0, 1,3);
    cv::Mat grad(gradx.rows,gradx.cols,CV_32FC1);
    cv::Mat G(grad.rows,grad.cols,CV_32FC1);
    cv::Mat phi = GradAttentuation(logL);
    for(int r = 0;r<grad.rows;r++){
        for(int c = 0 ;c<grad.cols; c++){
            grad.at<float>(r,c) = std::pow(gradx.at<float>(r,c)*gradx.at<float>(r,c) + grady.at<float>(r,c)*grady.at<float>(r,c),0.5);
            G.at<float>(r,c) = phi.at<float>(r,c)*grad.at<float>(r,c);
        }
    }
    cv::Mat divG,divGx,divGy;
    Sobel(G, divGx, CV_32FC1, 1, 0, 3);
    Sobel(G, divGy, CV_32FC1, 0, 1, 3);
    divG = divGx + divGy;
    cv::Mat complexI;
    dftImage(divG,complexI);
    

    for(int r = 0;r<complexI.rows;r++){
        for(int c = 0 ;c<complexI.cols; c++){
            double p,q;
            p = r - complexI.rows/2;
            q = c - complexI.cols/2;
            double Lx,Ly;
            Lx = complexI.rows; Ly = complexI.cols;
            double k = 4*M_PI*M_PI*((p*p)/(Ly*Ly) + ((q*q)/(Lx*Lx) ));
            if(abs(k)<0.0001) continue;
            complexI.at<cv::Vec2f>(r,c)[0] = complexI.at<cv::Vec2f>(r,c)[0]/k;
            complexI.at<cv::Vec2f>(r,c)[1] = complexI.at<cv::Vec2f>(r,c)[1]/k;
        }
    }
    cv::Mat FinalImage;
    cv::dft(complexI,FinalImage,cv::DFT_REAL_OUTPUT|cv::DFT_INVERSE|cv::DFT_SCALE);
    


    cv::namedWindow( "2",CV_WINDOW_FREERATIO);
    cv::imshow( "2", grad);
    // cv::namedWindow( "3",CV_WINDOW_FREERATIO);
    // cv::imshow( "3", scaling);
    cv::namedWindow( "3",CV_WINDOW_FREERATIO);
    cv::imshow( "3", divG);
    cv::namedWindow( "1",CV_WINDOW_FREERATIO);
    cv::imshow( "1", FinalImage);
    // cv::imshow( "Display window", grady);
    // cv::imshow( "Display window", image);


    cv::waitKey(0); 
    cv::destroyAllWindows();	
    return(0);
}
