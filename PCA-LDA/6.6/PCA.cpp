#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

// 创建并返回归一化图像
static Mat norm_0_255(InputArray _src) {
    Mat src = _src.getMat();
    Mat dst;
    switch (src.channels()) {
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

// 读取训练集
static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if (!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

int main(int argc, const char* argv[]) {
    // 训练集的路径
    string fn_csv = "data.csv";

    // 训练集的图片和标签
    vector<Mat> images;
    vector<int> labels;

    // 读取训练集
    read_csv(fn_csv, images, labels);

    // 获得第一张图片的高度，之后用于reshape
    int height = images[0].rows;

    // 将最后一张图片从训练集中去除，作为后面的测试图片
    Mat testSample = images[images.size() - 1];
    int testLabel = labels[labels.size() - 1];
    images.pop_back();
    labels.pop_back();

    // 输出图片高度、数量
    cout << "height:" << height << endl;
    cout << "images:" << images.size() << endl;

    // 创建特征脸模型用于人脸识别，用图片和标签进行训练
    // 进行完整的PCA，指定生成一定数量的特征脸
    Ptr<FaceRecognizer> model = createEigenFaceRecognizer(50);
    model->train(images, labels);

    // 预测测试图像的标签，输出结果
    int predictedLabel = model->predict(testSample);
    string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
    cout << result_message << endl;

    // 获得特征值、特征脸、训练集的样本均值
    Mat eigenvalues = model->getMat("eigenvalues");
    Mat W = model->getMat("eigenvectors");
    Mat mean = model->getMat("mean");

    // 显示均值脸
    imshow("mean", norm_0_255(mean.reshape(1, images[0].rows)));

    // 显示特征脸
    for (int i = 0; i < min(10, W.cols); i++) {
        string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
        cout << msg << endl;
        Mat ev = W.col(i).clone();
        // reshape至原始尺寸并归一化
        Mat grayscale = norm_0_255(ev.reshape(1, height));
        // 显示图像并应用Jet colormap以更好地感知
        Mat cgrayscale;
        applyColorMap(grayscale, cgrayscale, COLORMAP_JET);

        imshow(format("eigenface_%d", i), cgrayscale);
    }

    // 显示重构的图像
    for (int num_components = min(W.cols, 10); num_components < min(W.cols, 300); num_components += 15) {
        // 从模型中分割特征向量
        Mat evs = Mat(W, Range::all(), Range(0, num_components));
        Mat projection = subspaceProject(evs, mean, images[0].reshape(1, 1));
        Mat reconstruction = subspaceReconstruct(evs, mean, projection);
        // 归一化结果
        reconstruction = norm_0_255(reconstruction.reshape(1, images[0].rows));

        imshow(format("eigenface_reconstruction_%d", num_components), reconstruction);
    }

    waitKey(0);
    return 0;
}