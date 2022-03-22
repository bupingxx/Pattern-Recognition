#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

// ���������ع�һ��ͼ��
static Mat norm_0_255(InputArray _src) {
    Mat src = _src.getMat();
    Mat dst;
    switch(src.channels()) {
    case 1:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }
    return dst;
}

// ��ȡѵ����
static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

int main(int argc, const char *argv[]) {

    // ѵ������·��
    string fn_csv = "data.csv";

    // ѵ������ͼƬ�ͱ�ǩ
    vector<Mat> images;
    vector<int> labels;

    // ��ȡѵ����
    read_csv(fn_csv, images, labels);

    // ��õ�һ��ͼƬ�ĸ߶ȣ�֮������reshape
    int height = images[0].rows;

    // �����һ��ͼƬ��ѵ������ȥ������Ϊ����Ĳ���ͼƬ
    Mat testSample = images[images.size() - 1];
    int testLabel = labels[labels.size() - 1];
    images.pop_back();
    labels.pop_back();

    // ���ͼƬ�߶ȡ�����
    cout << "height:" << height << endl;
    cout << "images:" << images.size() << endl;

    // ����FisherFaceģ����������ʶ����ͼƬ�ͱ�ǩ����ѵ��
    // ����������FLD��ָ������һ��������������
    Ptr<FaceRecognizer> model = createFisherFaceRecognizer(50);
    model->train(images, labels);

    // Ԥ�����ͼ��ı�ǩ��������
    int predictedLabel = model->predict(testSample);
    string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
    cout << result_message << endl;

    // �������ֵ����������ѵ������������ֵ
    Mat eigenvalues = model->getMat("eigenvalues");
    Mat W = model->getMat("eigenvectors");
    Mat mean = model->getMat("mean");

    // ��ʾ��ֵ��
    imshow("mean", norm_0_255(mean.reshape(1, images[0].rows)));

    // ��ʾ������
    for (int i = 0; i < min(10, W.cols); i++) {
        string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
        cout << msg << endl;
        Mat ev = W.col(i).clone();
        // reshape��ԭʼ�ߴ粢��һ��
        Mat grayscale = norm_0_255(ev.reshape(1, height));
        // ��ʾͼ��Ӧ��Bone colormap�Ը��õظ�֪
        Mat cgrayscale;
        applyColorMap(grayscale, cgrayscale, COLORMAP_BONE);

        imshow(format("fisherface_%d", i), cgrayscale);
    }

    // ��ʾ�ع���ͼ��
    for(int num_component = min(W.cols, 10); num_component < min(W.cols, 300); num_component += 15) {
        // ��ģ���зָ���������
        Mat ev = W.col(num_component);
        Mat projection = subspaceProject(ev, mean, images[0].reshape(1,1));
        Mat reconstruction = subspaceReconstruct(ev, mean, projection);
        // ��һ�����
        reconstruction = norm_0_255(reconstruction.reshape(1, images[0].rows));

        imshow(format("fisherface_reconstruction_%d", num_component), reconstruction);
    }

    waitKey(0);
    return 0;
}
