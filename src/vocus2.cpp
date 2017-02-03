/********************************************************************************************/
/*                              Vocus2 C++ implematation file                               */
/*   Based on paper : "Traditional Saliency Reloaded: A Good Old Model in New Shape",       */
/*    S. Frintrop, et.al, in Proceedings of the IEEE International Conference               */
/*           on Computer Vision and Pattern Recognition (CVPR), 2015.                       */
/*           Implemenation : Kirankumar V. Adam (kiranadam@gmail.com)                       */
/********************************************************************************************/

#include "opencv2/opencv.hpp"
#include "vocus2.hpp"
#include <omp.h>
#include <algorithm>

// default constructor
vocus2 :: vocus2()
{
	this->no_scales = 5;
	this->c_sigma = 5;
	this->s_sigma = 15;
}

// Parameterized constructor
vocus2 :: vocus2(int no_scales, double c_sigma, double s_sigma)
{
	this->no_scales = no_scales;
	this->c_sigma = c_sigma;
	this->s_sigma = s_sigma;
}

// Image processing like BRG image to Intensity, RG, BY channels.
vector<Mat> vocus2 :: image_process(Mat& img)
{
	vector<Mat> img_channels;
	img_channels.resize(3);

	// Convert the image into floating values of 32 bit
	Mat conv_img;
	img.convertTo(conv_img, CV_32FC3);

	// Vector definiation for Image split channels	
	vector<Mat> split_img;
	
	// Split the BRG channels
	split(conv_img, split_img);

	#pragma omp parallel sections
	{	
		// Calculating intensity and color channels
		#pragma omp section
		img_channels[0] = (split_img[0] + split_img[1] + split_img[2])/(3*255.f);  // Intensity channel

		#pragma omp section		
		img_channels[1] = (split_img[1] - split_img[2] + 255.f) / (2*255.f);  // RG channel

		#pragma omp section		
		img_channels[2] = (split_img[0] - (split_img[1] + split_img[2])/2.f + 255.f) / (2*255.f); // BY channel
	}
	return img_channels;	
} 

// Map fusion function for maps
Mat vocus2 :: map_fusion(vector<Mat> map)
{
	Mat fused_map = Mat::zeros(map[0].rows, map[0].cols, CV_32F);  // defination for fused map. 
	int no_maps = map.size();  // get the size of vector for the maps.
	vector<Mat> temp ; 	// temp. vector matrix for the storage purpose.
	temp.resize(no_maps);   // defining the size of the maps.

	#pragma omp parallel for schedule(dynamic, 1)
	for(int i = 0; i<no_maps; i++)
	{
		if(fused_map.size() == map[i].size())
		{
			temp[i] = map[i];
		}
		else
		{
			resize(map[i],temp[i],fused_map.size(),0,0,INTER_CUBIC); // upsampling the image here 
		}
	}

	for(int i = 0; i<no_maps; i++)
	{
		add(fused_map,temp[i],fused_map,Mat(),CV_32F); // fusing the maps here.
	}

	fused_map /= (float)no_maps;

	return fused_map;
}


// function for building center image vector pyramid
vector<vector<Mat>> vocus2 :: center_pyramid(Mat& img, double c_sigma)
{
	vector<vector<Mat>> cent_pyr;

	// Check the maximum level for the octave to continue
	int octave_level = min((int) min(log2(img.rows),log2(img.cols)), no_scales);
	cent_pyr.resize(octave_level);

	// Image for the first iteration
	Mat temp = img.clone();
	
	GaussianBlur(temp, temp, Size(), 2.f*c_sigma, 2.f*c_sigma, BORDER_REPLICATE);
	resize(temp, temp, Size(), 0.5, 0.5, INTER_NEAREST);
	

	for(int i=0; i<octave_level; i++)	
	{
		cent_pyr[i].resize(no_scales);

		for(int j=0; j<no_scales; j++)
		{
			if(i==0)
			{	
				cent_pyr[i][j] = temp;
			}
			else
			{
				temp = cent_pyr[i-1][j].clone();
				double sigma = pow(2.0, (double) j / (double) no_scales) * c_sigma;
				GaussianBlur(temp,temp,Size(),sigma,sigma,BORDER_REPLICATE);
				resize(temp, cent_pyr[i][j], Size(temp.cols/2, temp.rows/2), 0, 0, INTER_NEAREST);				
			}
		}
	}
	
	return cent_pyr;
}

// function for building center surround image vector twin pyramid
void vocus2 :: center_surround_pyr(Mat& img)
{
	vector<vector<Mat>> sur_pyr;
	
	// Get the three image channels
	vector<Mat> img_channels = image_process(img);

	// Build pyramids for each channel
	#pragma omp parallel sections
	{
		#pragma omp section
		center_pyr_I = center_pyramid(img_channels[0],c_sigma); 

		#pragma omp section	
		center_pyr_RG = center_pyramid(img_channels[1],c_sigma);

		#pragma omp section	
		center_pyr_BY = center_pyramid(img_channels[2],c_sigma);
	}

	// Make memory for surrounding pyramid
	surround_pyr_I.resize(center_pyr_I.size());
	surround_pyr_RG.resize(center_pyr_I.size());
	surround_pyr_BY.resize(center_pyr_I.size());

	// Center Surround ratio for sigma as in paper
	double sigma_x = sqrt(s_sigma*s_sigma - c_sigma*c_sigma);

	// Build GaussianBlur Image Pyramid for each channel
	for(unsigned int i=0; i< center_pyr_I.size(); i++)
	{
		surround_pyr_I[i].resize(no_scales);
		surround_pyr_RG[i].resize(no_scales);
		surround_pyr_BY[i].resize(no_scales);

		#pragma omp parallel for
		for(int j=0; j<no_scales; j++)
		{
			double sigma = pow(2.0, (double) j / (double) no_scales) * sigma_x;

			GaussianBlur(center_pyr_I[i][j], surround_pyr_I[i][j], Size(), sigma, sigma, BORDER_REPLICATE);
			GaussianBlur(center_pyr_RG[i][j], surround_pyr_RG[i][j], Size(), sigma, sigma, BORDER_REPLICATE);
			GaussianBlur(center_pyr_BY[i][j], surround_pyr_BY[i][j], Size(), sigma, sigma, BORDER_REPLICATE);
		}
	}
}

// function for Center Surround Contrast pyramid
void vocus2 :: contrast_pyr()
{
	// Get size of vector pyramid
	int pyr_size = (int) surround_pyr_I.size() * no_scales;
	
	// Initialize the memory Center Surround and Surround Center vector 		
	C_S_pyr_I.resize(pyr_size);
	C_S_pyr_RG.resize(pyr_size);
	C_S_pyr_BY.resize(pyr_size);

	S_C_pyr_I.resize(pyr_size);
	S_C_pyr_RG.resize(pyr_size);
	S_C_pyr_BY.resize(pyr_size);

	// get the Contrast Pyramid
	
	for(unsigned int i=0; i<surround_pyr_I.size(); i++)
	{
		#pragma omp parallel for
		for(int j=0; j<no_scales; j++)
		{
			int index = i*no_scales +j;

			// Center Surround difference for each channel

			Mat c_s = center_pyr_I[i][j] - surround_pyr_I[i][j]; // for channel I
			threshold(c_s, C_S_pyr_I[index], 0, 1, THRESH_TOZERO);
			
			c_s = center_pyr_RG[i][j] - surround_pyr_RG[i][j]; // for channel RG
			threshold(c_s, C_S_pyr_RG[index], 0, 1, THRESH_TOZERO);
			
			c_s = center_pyr_BY[i][j] - surround_pyr_BY[i][j]; // for channel BY
			threshold(c_s, C_S_pyr_BY[index], 0, 1, THRESH_TOZERO);
			
			// Surround Center difference for each channel
			
			c_s = surround_pyr_I[i][j] - center_pyr_I[i][j]; // for channel I
			threshold(c_s, S_C_pyr_I[index], 0, 1, THRESH_TOZERO);			
			
			c_s = surround_pyr_RG[i][j] - center_pyr_RG[i][j]; // for channel RG
			threshold(c_s, S_C_pyr_RG[index], 0, 1, THRESH_TOZERO);

			c_s = surround_pyr_BY[i][j] - center_pyr_BY[i][j]; // for channel BY
			threshold(c_s, S_C_pyr_BY[index], 0, 1, THRESH_TOZERO);		
		}
	} 
}

// function for the feature map
void vocus2 :: feature_map()
{
	// Feature map vector for intensity
	feature_I.push_back(map_fusion(C_S_pyr_I));
	feature_I.push_back(map_fusion(S_C_pyr_I));

	// Feature map vector for RG
 	feature_RG.push_back(map_fusion(C_S_pyr_RG));
	feature_RG.push_back(map_fusion(S_C_pyr_RG));

	// Feature map vector for BY
 	feature_BY.push_back(map_fusion(C_S_pyr_BY));
	feature_BY.push_back(map_fusion(S_C_pyr_BY));
}

// function for consicuity map
void vocus2 :: consicuity_map()
{
	// Consicuity map for each channel
	conspicuity.push_back(map_fusion(feature_I)); // Intensity conspicuity map
	conspicuity.push_back(map_fusion(feature_RG)); // RG conspicuity map
	conspicuity.push_back(map_fusion(feature_BY)); // BY conspicuity map 
}

// Function to calculate saliency map
Mat vocus2 :: saliency_map(Mat& img)
{
	//cout <<"Channels:"<< img.channels()<<endl;
	CV_Assert(img.channels() == 3);
	//cout<<"Image size "<<img.rows<<"  "<<img.cols<<endl;
	
	// Get twin pyramids for I, RG, BY channels
	center_surround_pyr(img);

	// Get contrast pyramids based on above twin pyramids
	contrast_pyr();

	// Get feature maps based on above contrast twin pyramids
	feature_map();	

	// Get consicuity maps based on above twin feature map pyramids
	consicuity_map();

	// Get the Saliency map based on above consicuity map vector
	Mat sal_map = map_fusion(conspicuity);
		
	resize(sal_map, sal_map, Size(sal_map.cols*2, sal_map.rows*2), 0, 0, INTER_NEAREST);
	//cout<<"Saliency map size "<<sal_map.rows<<"  "<<sal_map.cols<<endl;	

	// Convert CV_32F to CV_8U
	float _max = *max_element(sal_map.begin<float>(),sal_map.end<float>());
	float _min = *min_element(sal_map.begin<float>(),sal_map.end<float>());
	
	Mat U_sal;
	sal_map.convertTo(U_sal,CV_8U,255.0/(_max-_min),0);

	return U_sal;
}	

// Function to calculate Histogramm
Mat vocus2 :: histCalc(Mat& img)
{
	CV_Assert(img.channels() == 1 && img.type() == CV_8U);
	Mat hist;
	int channels[] = {0};
	int bin = 256;
	float hrange[] = {0,255};
	const float* ranges[] = {hrange}; 
	calcHist(&img, 1, channels, Mat(), hist, 1, &bin, ranges, true, false);
	
	return hist;	
}

// Plot the histogram
void vocus2 :: plotHist(Mat& hist)
{
	int hist_w = 512; 
	int hist_h = 400;
	int binSize = 256;
    	int bin_w = cvRound( (double) hist_w/binSize );
 
    	Mat histImage(hist_h, hist_w, CV_8UC1, Scalar( 0,0,0));
    	normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
     
    	for( int i = 1; i < binSize; i++ )
    	{
      		line(histImage, Point(bin_w*(i-1),hist_h - cvRound(hist.at<float>(i-1))), Point(bin_w*(i),hist_h - cvRound(hist.at<float>(i))), Scalar(255,0,0), 2, 8, 0);
    	}
 
    	namedWindow( "Result", 1 );    
	imshow( "Result", histImage );
 	
    	waitKey(0);    

}

vector<Mat> vocus2 :: image_segment(Mat& src, Mat& salmap)
{
	double max_val;
	Point max_ptr;
	float threshold_wt = 0.6;
	minMaxLoc(salmap, nullptr, &max_val, nullptr, &max_ptr);

	vector<Point> msr; // Most significant region
	msr.push_back(max_ptr); // Push max ptr

	int pos = 0;
	float thresh = threshold_wt*max_val;


	Mat considered = Mat::zeros(salmap.size(), CV_8U);
	considered.at<uchar>(max_ptr) = 1;

	while(pos < (int)msr.size())
	{
		int r = msr[pos].y;
		int c = msr[pos].x;
		

		for(int dr = -1; dr <= 1; dr++)
		{
			for(int dc = -1; dc <= 1; dc++)
			{
				if(dc == 0 && dr == 0)
					continue;
				if(considered.ptr<uchar>(r+dr)[c+dc] != 0)
					continue;
				if(r+dr < 0 || r+dr >= salmap.rows)
					continue;
				if(c+dc < 0 || c+dc >= salmap.cols)
					continue;
				

				if((float)salmap.ptr<uchar>(r+dr)[c+dc] >= thresh)
				{
					msr.push_back(Point(c+dc, r+dr));
					considered.ptr<uchar>(r+dr)[c+dc] = 1;	
				}
			}
		}
		pos++;
	}

	// Get the bounding box for the Grab cut segmentation
	Rect rect = boundingRect(msr);
	
	vector<Mat> img_vec;

	Mat result; // segmentation result (mask value)
	Mat bg,fg; // the models

	// GrabCut segmentation
	grabCut(src, result, rect, bg, fg, 1, GC_INIT_WITH_RECT); 

	// Get the pixels marked as likely foreground
	compare(result,cv::GC_PR_FGD,result,cv::CMP_EQ);
	
	// Generate output image
	Mat foreground(src.size(),CV_8UC3,cv::Scalar(0,0,0));
	Mat background(src.size(),CV_8UC3,cv::Scalar(0,0,0));
	
	src.copyTo(foreground,result); 
	src.copyTo(background,~result);

	img_vec.push_back(foreground);
	img_vec.push_back(background);

	return img_vec;
}
