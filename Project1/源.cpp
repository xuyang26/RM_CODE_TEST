#include<opencv2\opencv.hpp>
#include<opencv2\imgproc\imgproc.hpp>
using namespace cv;
using namespace std;

/*因为关于关于边缘检测可以检测出边界的像素，但是他并没有将轮廓作为一个整体，
所以这里介绍的轮廓检测就是把这些边缘像素组装成轮廓，因此轮廓检测前提是二值图像*/

int main()
{
	Mat srcImage, srcImage_gray, srcImage_binary, cannyImage, srcImage_blur;
	srcImage = imread("rm.jpg");
	cout << srcImage.size() << endl;
	if (!srcImage.data)
	{
		printf("图片载入失败"); return false;
	}
	imshow("原图像", srcImage);
	cvtColor(srcImage, srcImage_gray, COLOR_RGB2GRAY);
	blur(srcImage_gray, srcImage_blur, Size(3, 3), Point(-1, -1), 4); //高斯滤波平滑需要输入图像为灰度图像

	//imshow("srcImage_blur", srcImage_blur);
	Canny(srcImage_blur,        //Canny边缘检测通常要先高斯滤波平滑
		cannyImage,
		50, 150,             //Canny 边缘检测使用双阈值，通常高阈值：低阈值为2:1或者3:1
		3);

	Mat dstImage = Mat::zeros(srcImage.size(), CV_8UC3);

	/***********定义轮廓和层次结构***********/
	vector<vector<Point>>contours;
	vector<Vec4i>hierarchy;

	/*********************查找轮廓************************************************/

	findContours(cannyImage,                //找轮廓图输入必须是二值图
		contours,                          //表示找到的轮廓，每个轮廓存储为一个点向量
		hierarchy,                        //输出的向量包含图像的拓扑信息，存储查找到的第i个轮廓的后[i][0]、前[i][1]、父[i][2]、子轮廓[i][3]
		RETR_CCOMP,                       //可选轮廓的检索模式（有提取最外层轮廓，有提取所以轮廓，有 网状结构
		CHAIN_APPROX_SIMPLE);        //可选轮廓的近似方式，压缩水平，垂直，对角线方向只保留点元素，比如矩形轮廓只用四个点来保存轮廓

	cout << "轮廓个数" << contours.size() << endl;
	/***********************遍历所有顶层轮廓，以随机颜色绘制每个组件颜色*********/
	vector<vector<Point>>hull(contours.size());
	for (size_t i = 0; i < contours.size(); i++)
	{
		//由于颜色是随机产生的，则可大概的根据颜色的种类判断轮廓的个数
		Scalar color(rand() % 255,      //rand()是产生随机数-32768--32768之间的随机数 ，因为要求的随机数不能大于255，所以有rand（）&255操作
			rand() % 255,          //rand()&255实际上与rand()%255 效果一样
			rand() % 255);         //rand()&255更能显示出水平，255的二进制为0000 0000 1111 1111，rand（）产生的随机数和255进行“位与                                                 //”运算
		drawContours(dstImage,
			contours,
			i,                    //轮廓索引
			color,                //轮廓颜色
			1,                   //画轮廓线粗，如果是-1表示填充
			8,                   //轮廓线型
			hierarchy,
			-1,                   //第五个参数表示绘制轮廓的最大层数，如果是0，只绘制contour；如果是1，追加绘制和contour同层的所有轮廓                                               //如果是2,追加绘制比contour
						   //低一层的轮廓，以此类推；如果值是负值，则函数并不绘制contour后的轮廓，但是将画出其子轮廓，一直到abs                                                //(max_level) - 1层。
			Point(0, 0));   //偏移量为（0,0）表示在dstImage图上原来一模一样的位置画轮廓。如果Point（100,100）则相对原图偏移了（100,100）
							  //rand()是产生随机数-32768--32768之间的随机数 ，因为要求的随机数不能大于255，所以有rand（）&255操作

	/*************************************查找凸包并绘制凸包******************/
	convexHull(contours[i], hull[i], false);
	drawContours(dstImage, hull, i, color, 3, 8, hierarchy,0,Point());//凸包属于一种特殊的轮廓因此画凸包还是使用drawContours函数
	}
	putText(dstImage, "RM", Point(50, 60), FONT_HERSHEY_COMPLEX, 2, Scalar(255, 255, 255), 4, 8);
	imshow("轮廓图", dstImage);

	waitKey(0);
	return 0;
}