/********************************************************************************************/
/*                              Vocus2 header file                                          */
/*   Based on paper : "Traditional Saliency Reloaded: A Good Old Model in New Shape",       */
/*    S. Frintrop, et.al, in Proceedings of the IEEE International Conference               */
/*           on Computer Vision and Pattern Recognition (CVPR), 2015.                       */
/*           Implemenation : Kirankumar V. Adam (kiranadam@gmail.com)                       */
/********************************************************************************************/


#ifndef vocus2_HPP_
#define vocus2_HPP_

#include "opencv2/opencv.hpp"
#include <iostream>
#include <cmath>

#include <string>
#include <fstream>

#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>

#include <boost/serialization/nvp.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/extended_type_info.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/assume_abstract.hpp>


using namespace cv;
using namespace std;


class vocus2
{

	public :

		int no_scales;  // The value to determine scale space per octave.
		double c_sigma; // Center sigma for the images. 
		double s_sigma; // Surround sigma for the images.

		Mat f_ground, b_ground;
		vocus2();
		vocus2(int no_scales, double c_sigma, double s_sigma);		

		// function for Saliency Map
		Mat saliency_map(Mat& img);	
		
		// function for Histogram of GrayScale Image
		Mat histCalc(Mat& img);
		
		// Plot the histogram
		void plotHist(Mat& hist); 

		//get most salient region
		vector<Mat> segment(Mat& src, Mat& salmap);
		
		//get image with circle
		vector<Mat> image_segment(Mat& src, Mat& salmap);
		

	private :

		vector<vector<Mat>> center_pyr_I;  // center Intensity channel pyramid  
		vector<vector<Mat>> center_pyr_RG; // center RG channel pyramid
		vector<vector<Mat>> center_pyr_BY; // center BY channel pyramid
		
		vector<vector<Mat>> surround_pyr_I;  // Surrounding Intensity channel pyramid  
		vector<vector<Mat>> surround_pyr_RG; // Surrounding RG channel pyramid
		vector<vector<Mat>> surround_pyr_BY; // Surrounding BY channel pyramid		

		vector<Mat> C_S_pyr_I;    // center surround Intensity pyramid 
		vector<Mat> C_S_pyr_RG;   // center surround RG pyramid
		vector<Mat> C_S_pyr_BY;   // center surround BY pyramid

		vector<Mat> S_C_pyr_I;    // surround center Intensity pyramid
		vector<Mat> S_C_pyr_RG;   // surround center RG pyramid
		vector<Mat> S_C_pyr_BY;   // surround center BY pyramid
		
		vector<Mat> feature_I;	  // Intensity Feature map vector
		vector<Mat> feature_RG;	  // RG Feature map vector	
		vector<Mat> feature_BY;   // BY Feature map vector
		
		vector<Mat> conspicuity;  // conspicuity map vector

		// Image processing function BRG image to Intensity, RG, BY channels
		vector<Mat> image_process(Mat& img);
		//vector<Mat> image_processing(Mat& img);

		// Map fusion function for maps
		Mat map_fusion(vector<Mat> maps);

		// function for building center image vector pyramid
		vector<vector<Mat>> center_pyramid(Mat& img, double c_sigma);

		// function for building center surround image vector twin pyramid
		void center_surround_pyr(Mat& img);
		
		// function for Center Surround Contrast Pyramid
		void contrast_pyr();

		// function for feature map
		void feature_map();

		// function for consicuity map
		void consicuity_map(); 

};


#endif
