#include<opencv2\opencv.hpp>
#include<opencv2\imgproc\imgproc.hpp>
using namespace cv;
using namespace std;

/*��Ϊ���ڹ��ڱ�Ե�����Լ����߽�����أ���������û�н�������Ϊһ�����壬
����������ܵ����������ǰ���Щ��Ե������װ������������������ǰ���Ƕ�ֵͼ��*/

int main()
{
	Mat srcImage, srcImage_gray, srcImage_binary, cannyImage, srcImage_blur;
	srcImage = imread("rm.jpg");
	cout << srcImage.size() << endl;
	if (!srcImage.data)
	{
		printf("ͼƬ����ʧ��"); return false;
	}
	imshow("ԭͼ��", srcImage);
	cvtColor(srcImage, srcImage_gray, COLOR_RGB2GRAY);
	blur(srcImage_gray, srcImage_blur, Size(3, 3), Point(-1, -1), 4); //��˹�˲�ƽ����Ҫ����ͼ��Ϊ�Ҷ�ͼ��

	//imshow("srcImage_blur", srcImage_blur);
	Canny(srcImage_blur,        //Canny��Ե���ͨ��Ҫ�ȸ�˹�˲�ƽ��
		cannyImage,
		50, 150,             //Canny ��Ե���ʹ��˫��ֵ��ͨ������ֵ������ֵΪ2:1����3:1
		3);

	Mat dstImage = Mat::zeros(srcImage.size(), CV_8UC3);

	/***********���������Ͳ�νṹ***********/
	vector<vector<Point>>contours;
	vector<Vec4i>hierarchy;

	/*********************��������************************************************/

	findContours(cannyImage,                //������ͼ��������Ƕ�ֵͼ
		contours,                          //��ʾ�ҵ���������ÿ�������洢Ϊһ��������
		hierarchy,                        //�������������ͼ���������Ϣ���洢���ҵ��ĵ�i�������ĺ�[i][0]��ǰ[i][1]����[i][2]��������[i][3]
		RETR_CCOMP,                       //��ѡ�����ļ���ģʽ������ȡ���������������ȡ������������ ��״�ṹ
		CHAIN_APPROX_SIMPLE);        //��ѡ�����Ľ��Ʒ�ʽ��ѹ��ˮƽ����ֱ���Խ��߷���ֻ������Ԫ�أ������������ֻ���ĸ�������������

	cout << "��������" << contours.size() << endl;
	/***********************�������ж����������������ɫ����ÿ�������ɫ*********/
	vector<vector<Point>>hull(contours.size());
	for (size_t i = 0; i < contours.size(); i++)
	{
		//������ɫ����������ģ���ɴ�ŵĸ�����ɫ�������ж������ĸ���
		Scalar color(rand() % 255,      //rand()�ǲ��������-32768--32768֮�������� ����ΪҪ�����������ܴ���255��������rand����&255����
			rand() % 255,          //rand()&255ʵ������rand()%255 Ч��һ��
			rand() % 255);         //rand()&255������ʾ��ˮƽ��255�Ķ�����Ϊ0000 0000 1111 1111��rand�����������������255���С�λ��                                                 //������
		drawContours(dstImage,
			contours,
			i,                    //��������
			color,                //������ɫ
			1,                   //�������ߴ֣������-1��ʾ���
			8,                   //��������
			hierarchy,
			-1,                   //�����������ʾ�����������������������0��ֻ����contour�������1��׷�ӻ��ƺ�contourͬ�����������                                               //�����2,׷�ӻ��Ʊ�contour
						   //��һ����������Դ����ƣ����ֵ�Ǹ�ֵ��������������contour������������ǽ���������������һֱ��abs                                                //(max_level) - 1�㡣
			Point(0, 0));   //ƫ����Ϊ��0,0����ʾ��dstImageͼ��ԭ��һģһ����λ�û����������Point��100,100�������ԭͼƫ���ˣ�100,100��
							  //rand()�ǲ��������-32768--32768֮�������� ����ΪҪ�����������ܴ���255��������rand����&255����

	/*************************************����͹��������͹��******************/
	convexHull(contours[i], hull[i], false);
	drawContours(dstImage, hull, i, color, 3, 8, hierarchy,0,Point());//͹������һ�������������˻�͹������ʹ��drawContours����
	}
	putText(dstImage, "RM", Point(50, 60), FONT_HERSHEY_COMPLEX, 2, Scalar(255, 255, 255), 4, 8);
	imshow("����ͼ", dstImage);

	waitKey(0);
	return 0;
}