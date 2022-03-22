#### 5.2 研究PCA的平均向量

​	使用题目中的代码：

```matlab
% scale [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
% set the random number seed to 0 for reproducibility
rng(0);
avg = [1 2 3 4 5 6 7 8 9 10];
scale = 1;

data = randn(5000, 10) + repmat(avg*scale, 5000, 1);

m = mean(data);
m1 = m / norm(m);

% do PCA, but without centering
[~, S, V] = svd(data);
S = diag(S);
e1 = V(:,1);  % 第一个特征向量，未减去平均向量

% do correct PCA with centering
newdata = data - repmat(m, 5000, 1);
[U, S, V] = svd(newdata);
S = diag(S);
new_e1 = V(:,1); % 第一个特征向量，减去平均向量
```

​	当scale变化时，如果没有减去平均向量，得到的第一个特征向量值如下：

![image-20220315003537398](C:\Users\82458\Desktop\github\Pattern-Recognition\PCA-LDA\src\image-1.png)

​	如果有减去平均向量，得到的第一个特征向量值如下：

![image-20220315003527882](C:\Users\82458\Desktop\github\Pattern-Recognition\PCA-LDA\src\image-2.png)

​	可以看出，如果对每个样本进行减去平均向量的转换，那么在scale变化时，第一个特征向量值是不变的；如果没有减去平均向量，在scale较大时会得到错误的结果，直到scale减小到0.001，才会得到正确的特征向量。

​	用以下代码计算特征向量与平均向量之间的相关系数：

```matlab
% correlation between first eigenvector(new , old) and mean
avg = avg - mean(avg);
avg = avg / norm(avg);
e1 = e1 - mean(e1);
e1 = e1 / norm(e1);
new_e1 = new_e1 - mean(new_e1);
new_e1 = new_e1 / norm(new_e1);
corr1 = avg * e1;
corr2 = e1.' * new_e1;
```

​	结果如下，其中`corr1`为未减去平均向量的第一特征向量与平均向量的相关系数，`corr2`为两个第一特征向量的相关系数：

![image-20220315004226640](C:\Users\82458\Desktop\github\Pattern-Recognition\PCA-LDA\src\image-3.png)

​	可以看出在scale较小时，e1与平均向量相关程度较高，随着scale增大，相关程度渐渐变小，所以在`scale=0.001`之后e1逐渐趋于e2，`corr2`的绝对值因此也接近于1。



#### 5.3 实现PCA和白化变换

##### （a）生成样本

​	生成2000个二维样本，使用scatter函数画出图像如下：

![image-20220315105828619](C:\Users\82458\Desktop\github\Pattern-Recognition\PCA-LDA\src\image-4.png)

##### （b）PCA变换

​	对这些样本信息做PCA变换，保留所有的2个维度，代码如下：

```matlab
% b小题 做PCA变换
c = cov(x); %协方差
m = mean(x);
data = x - repmat(m, 2000, 1); %去均值
[U, S, V] = svd(c);
pca_data = data * V;
figure(2);
scatter(pca_data(:,1), pca_data(:,2));    % 画图
xlim([-10,10]);
ylim([-10,10]);
```

​	画出PCA后的样本，如下所示：

![image-20220315105837589](C:\Users\82458\Desktop\github\Pattern-Recognition\PCA-LDA\src\image-5.png)



##### （c）白化变换

​	对这些样本进行白化变换，保留所有的2个维度，代码如下：

```matlab
% c小题 做白化变换
pca_whiten_data = pca_data * (S)^(-0.5);
figure(3);
scatter(pca_whiten_data(:,1), pca_whiten_data(:,2));    % 画图
xlim([-5,4]);
ylim([-4,4]);
```

​	画出白化后的样本，如下所示：

![image-20220315105901234](C:\Users\82458\Desktop\github\Pattern-Recognition\PCA-LDA\src\image-6.png)



##### （d）解释

​	对于满足高斯分布的样本`x~N(μ, Σ)`，在去均值并对协方差矩阵进行SVD分解后，变换后的样本数据`y`也满足高斯分布，`y~N(0,Λ)`，其中`Λ`表示由`Cov(x)`的特征值所构成的对角矩阵。也就是说，PCA是对`x`先执行平移，然后进行旋转，使得高斯分布的轴能够与坐标轴平行，`y`的不同分量之间是彼此独立的。





#### 6.6 PCA+FLD 人脸识别

注：以下使用编程语言为C++，使用的OpenCV版本为2.4.9

##### （c）对数据集ORL运行PCA与FLD

​	参考[OpenCV官方文档](https://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html)，先创建一个CSV文件表示数据集中图片的存储位置以及标签（用`;`隔开），其内容大致如下：

```
ORL/s1/1.bmp;0
ORL/s1/10.bmp;0
...
ORL/s12/6.bmp;3
ORL/s12/7.bmp;3
...
ORL/s21/6.bmp;13
ORL/s21/7.bmp;13
...
```

​	CSV文件为附录文件中的`data.csv`，可使用一个简单的python脚本来生成CSV，此python脚本为附录中的`create_csv.py`。

​	使用OpenCV中的[FaceRecognizer](https://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_api.html#FaceRecognizer : public Algorithm)类来测试数据集，PCA代码如下：

```c++
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
```

​	FLD的代码类似与PCA，主要区别是在创建`FisherFace`模型时，FLD调用的是`createFisherFaceRecognizer()`函数：

```c++
    // 创建FisherFace模型用于人脸识别，用图片和标签进行训练
    // 进行完整的FLD，指定生成一定数量的特征脸
    Ptr<FaceRecognizer> model = createFisherFaceRecognizer(50);
    model->train(images, labels);
```

​	当PCA与FLD**均使用50个特征脸**时，结果如下：

###### PCA

​	训练集的平均脸：

![image-20220322235631526](C:\Users\82458\Desktop\github\Pattern-Recognition\PCA-LDA\src\image-7.png)

___

​	前10个Eigenvalue（特征值），以及预测结果：

![image-20220323002759089](C:\Users\82458\Desktop\github\Pattern-Recognition\PCA-LDA\src\image-8.png)

___

​	前10个Eigenface（resize为原图尺寸，使用`Jet Colormap`突出灰度值在特征脸中的分布）：

![image-20220322235749971](C:\Users\82458\Desktop\github\Pattern-Recognition\PCA-LDA\src\image-9.png)

___

​	根据Eigenface重构的人脸图像：

![image-20220323000219259](C:\Users\82458\Desktop\github\Pattern-Recognition\PCA-LDA\src\image-10.png)

___

###### FLD

​	训练集的平均脸：

![image-20220322235631526](C:\Users\82458\Desktop\github\Pattern-Recognition\PCA-LDA\src\image-11.png)

___

​	前10个Eigenvalue（特征值），以及预测结果：

![image-20220323001125799](C:\Users\82458\Desktop\github\Pattern-Recognition\PCA-LDA\src\image-12.png)

___

​	前10个Fisherface（resize为原图尺寸，使用`Bone Colormap`突出灰度值在fisherface中的分布）：

![image-20220323001503483](C:\Users\82458\Desktop\github\Pattern-Recognition\PCA-LDA\src\image-13.png)

___

​	根据Eigenface重构的人脸图像：

![image-20220323001356552](C:\Users\82458\Desktop\github\Pattern-Recognition\PCA-LDA\src\image-14.png)

___

​	有关颜色图的说明可以参考[jet](https://ww2.mathworks.cn/help/matlab/ref/jet.html)与[bone](https://ww2.mathworks.cn/help/matlab/ref/bone.html)。

##### （d）重构人脸图像

​	在重建的时候，由于FisherFace只关注各类目标间的不同特征，所以难以重建出原图像。而在EigenFace中，当使用更多数量的eigenfaces时，可以得到更好的重构图像。用于测试的原图像为：

![image-20220323003349607](C:\Users\82458\Desktop\github\Pattern-Recognition\PCA-LDA\src\image-15.png)

​	使用50个特征脸时得到的结果为：

![image-20220323002530972](C:\Users\82458\AppData\Roaming\Typora\typora-user-images\image-20220323002530972.png)

​	使用100个特征脸时得到的结果为：

![image-20220323002933659](C:\Users\82458\Desktop\current\模式识别\作业\PCA\reconstruct100.png)

​	使用150个特征脸时得到的结果为：

![image-20220323003014831](C:\Users\82458\Desktop\current\模式识别\作业\PCA\reconstruct150.png)

​	使用200个特征脸时得到的结果为：

![image-20220323003036223](C:\Users\82458\Desktop\current\模式识别\作业\PCA\reconstruct200.png)

​	使用250个特征脸时得到的结果为：

![image-20220323003102352](C:\Users\82458\Desktop\current\模式识别\作业\PCA\reconstruct250.png)

​	使用300个特征脸时得到的结果为：

![image-20220323003118301](C:\Users\82458\Desktop\current\模式识别\作业\PCA\reconstruct300.png)

​	使用350个特征脸时得到的结果为：

![image-20220323003732156](C:\Users\82458\AppData\Roaming\Typora\typora-user-images\image-20220323003732156.png)

​	使用400个特征脸时得到的结果为：

![image-20220323003743329](C:\Users\82458\AppData\Roaming\Typora\typora-user-images\image-20220323003743329.png)

​	可以看到，当使用200个特征脸时，可以重构出原本人脸的大致轮廓；当使用300个特征脸时，可以重构出原本人脸的基本细节，但是仍有一点模糊；当使用400个特征脸时，重构图像已经与原始输入的图像难以区分了。