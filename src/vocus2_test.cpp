/********************************************************************************************/
/*                              Test main file                                              */
/*   Based on paper : "Traditional Saliency Reloaded: A Good Old Model in New Shape",       */
/*    S. Frintrop, et.al, in Proceedings of the IEEE International Conference               */
/*           on Computer Vision and Pattern Recognition (CVPR), 2015.                       */
/*           Implemenation : Kirankumar V. Adam (kiranadam@gmail.com)                       */
/********************************************************************************************/

#include "opencv2/opencv.hpp"
#include "vocus2.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <string>

using namespace std;
using namespace cv;

template<class T>
T base_name(T const & path, T const & delims = "/\\");

template<class T>
T remove_extension(T const & filename);

int main(int , char **)
{
	string path;
	cout<<"Enter Image path : "<<endl;
	cin>>path;	
	
	Mat img = imread(path,1);

	if(!img.data)
	{
		cout<<"Path not found"<<endl;
		return -1;
	}
	else
	{
		imshow("Image",img);
	}

	//original
	vocus2 voc(5,1,200);

	
	Mat sal_map = voc.saliency_map(img);
	namedWindow("Saliency Map",CV_WINDOW_KEEPRATIO);
	imshow("Saliency Map",sal_map);

	vector<Mat> msr = voc.image_segment(img,sal_map);


	string file = remove_extension(base_name(path));

	imwrite("../../vocus2/"+file+"_fg.jpg",msr[0]);  // saving foreground
	imwrite("../../vocus2/"+file+"_bg.jpg",msr[1]);  // saving background

	imshow("FG",msr[0]);
	imshow("BG",msr[1]);
	waitKey(-1);

	return  0;
}


template<class T>
T base_name(T const & path, T const & delims = "/\\")
{
  return path.substr(path.find_last_of(delims) + 1);
}

template<class T>
T remove_extension(T const & filename)
{
  typename T::size_type const p(filename.find_last_of('.'));
  return p > 0 && p != T::npos ? filename.substr(0, p) : filename;
}
