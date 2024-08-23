import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve,train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import seaborn as sns

#处理遇见中文或者是负号无法显示的情况
plt.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def process_data():
    # 添加列名
    header_row = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
                  'ca','thal', 'target']
    # 载入数据
    heart = pd.read_csv('heart.csv', names = header_row)
    # 心脏病（0=否，1=是）
    heart["target"] = np.where(heart["target"] != 0, 1, 0)

    #划分特征值和目标值
    x = heart[
        ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',
         'thal']]
    label = heart.target
    # #查看各列数据类型
    # heart.info()
    # # 描述统计相关信息
    # desc = heart.describe(include = 'all')
    # print(desc)
    #
    #
    # # 患病分布情况，统计有多少人患病
    # heart_counts = heart['target'].value_counts()
    # # 设置画布fig，图像对象axes
    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 创建1行2列的画布，并设置画布大小
    # # 绘制条形图
    # ax = heart_counts.plot(kind='bar', ax=axes[0])  # 绘制条形图,画在第0个子图上,并返回axes对象
    # ax.set_title('患病分布')  # 设置标题
    # ax.set_xlabel('1:患病,0:未患病')  # x轴名称
    # ax.set_ylabel('人数')  # y轴名称
    # # 绘制饼图，显示数值保留两位小数，显示%，数值对应的标签名
    # heart_counts.plot(kind='pie', autopct='%.2f%%', labels=['患病', '未患病'])
    #
    # # 有多少个特征就画多少个直方图，一共有14个特征
    # fig, axes = plt.subplots(2, 7, figsize=(40, 10))
    # # 用一个循环绘图
    # for i in range(14):
    #     plt.subplot(2, 7, i + 1)  # 设置当前画板为当前画图对象，x+1表示第几个画布
    #     sns.distplot(heart.iloc[:, i], kde=True)  # 绘制直方图，所有特征下面的值的直方图，并显示曲线。取第x列的所有行
    # # 自动调整布局，轻量化调整
    # plt.tight_layout()
    # plt.show()
    # #显示心脏病各字段的相关性
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(heart.corr(), cmap='Blues', annot=True)
    # plt.show()
    #
    # heart.hist(figsize=(20, 16))
    # plt.show()

    return x,label

def knn_heart_diease(x,y,x_train,y_train,x_test,y_test):
    knn_score = []
    transfer = StandardScaler()
    # 标准化实验集
    x_train = transfer.fit_transform(x_train)
    # 标准化测试集
    x_test = transfer.transform(x_test)
    krange = range(1, 10)
    for i in krange:
        #KNN算法预估器
        clf = neighbors.KNeighborsClassifier(n_neighbors=i, metric="euclidean")
        clf = clf.fit(x_train,y_train)
        knn_score.append(clf.score(x_test, y_test))
    bestK = krange[knn_score.index(max(knn_score))]
    print(bestK)
    print(knn_score)
    print(max(knn_score))
    #绘制学习曲线
    plt.plot([k for k in krange], knn_score, color='blue')
    for i in krange:
        plt.text(i, knn_score[i - 1], (i, knn_score[i - 1]))
    xticks = plt.xticks([i for i in krange])
    plt.xlabel('Number of Neighbors (n_neighbors)')
    plt.ylabel('Scores')
    plt.title('K Neighbors Classifier scores for different K values')
    plt.show()
    plt_show(clf,x,y,'KNN')
    return max(knn_score)


def dectree_heart_diease(x,y,x_train,y_train,x_test,y_test):
    # 进行处理（特征工程）one hot 编码
    dict = DictVectorizer(sparse=False)
    x_train = dict.fit_transform(x_train.to_dict(orient="records"))
    print(dict.get_feature_names())
    x_test = dict.transform(x_test.to_dict(orient="records"))
    print(x_train)

    dtc_scores = []
    for i in range(1, len(x.columns) + 1):
        dtc_classifier = DecisionTreeClassifier(max_features=i, random_state=0)
        dtc_classifier.fit(x_train, y_train)
        dtc_scores.append(dtc_classifier.score(x_test, y_test))

    print(dtc_scores)
    best_feature = dtc_scores.index(max(dtc_scores)) + 1
    print(best_feature)
    print(max(dtc_scores))

    #找到最优参数，生成最优决策树
    dec = DecisionTreeClassifier(max_features=best_feature)
    # 用决策树进行预测
    dec.fit(x_train, y_train)
    # 预测准确率
    print("预测的准确率：", dec.score(x_test, y_test))
    plt_show(dec, x, y,'决策树')

    # 用随机森林进行预测(超参数调优)
    rf = RandomForestClassifier()
    # 参数准备
    param = {"n_estimators": [120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30]}
    # 网格搜索与交叉验证
    rf = GridSearchCV(rf, param_grid=param, cv=2)
    rf.fit(x_train, y_train)
    print("随机森林调优结果如下：")
    print("准确率:", rf.score(x_test, y_test))
    print("选择的参数模型：", rf.best_params_)
    # 生成决策树
    tree_process(dec)


    return dec.score(x_test,y_test),rf.score(x_test,y_test)

def svm_heart_diease(x,y,x_train,y_train,x_test,y_test):
    svc_scores = []
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']  # 'precomputed'

    for i in range(len(kernels)):
        svc_classifier = SVC(kernel=kernels[i])
        svc_classifier.fit(x_train, y_train)
        svc_scores.append(svc_classifier.score(x_test, y_test))

    bestkernel = kernels[svc_scores.index(max(svc_scores))]
    print(bestkernel)
    print(svc_scores)
    print(max(svc_scores))
    colors = ['red','green','blue','yellow']
    plt.bar(kernels, svc_scores, color=colors)
    for i in range(len(kernels)):
        plt.text(i, svc_scores[i], svc_scores[i])
    plt.xlabel('Kernels')
    plt.ylabel('Scores')
    plt.title('Support Vector Classifier scores for different kernels')
    plt.show()

    plt_show(svc_classifier, x, y, 'SVM')
    return max(svc_scores)


def plt_show(clf,x,y,name):

    train_size, train_scores, test_scores = learning_curve(clf, x, y, cv=10, scoring='accuracy',  # 10折交叉验证
                                                           train_sizes=np.linspace(0.1, 1.0, 5))  # 5次的训练数量占比
    mean_train = np.mean(train_scores, 1)
    # 得到得分范围的上下界
    upper_train = np.clip(mean_train + np.std(train_scores, 1), 0, 1)
    lower_train = np.clip(mean_train - np.std(train_scores, 1), 0, 1)

    mean_test = np.mean(test_scores, 1)
    # 得到得分范围的上下界
    upper_test = np.clip(mean_test + np.std(test_scores, 1), 0, 1)
    lower_test = np.clip(mean_test - np.std(test_scores, 1), 0, 1)

    plt.figure('Fig1')
    plt.plot(train_size, mean_train, 'ro-', label='train score')
    plt.plot(train_size, mean_test, 'go-', label='test score')
    ##填充上下界的范围
    plt.fill_between(train_size, upper_train, lower_train, alpha=0.2,  # alpha：覆盖区域的透明度[0,1],其值越大，表示越不透明
                     color='r')
    plt.fill_between(train_size, upper_test, lower_test, alpha=0.2,  # alpha：覆盖区域的透明度[0,1],其值越大，表示越不透明
                     color='g')
    plt.grid()
    plt.xlabel('train size')
    plt.ylabel('score')
    plt.legend(loc='lower right')
    plt.title(name)
    plt.savefig('train number-size.png')
    plt.show()

def tree_process(dec):
    export_graphviz(dec, out_file='tree.dot',
                    feature_names=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
                                   'oldpeak','slope', 'ca', 'thal'], class_names=['0', '1', '2', '3', '4'], filled=True,
                    rounded=True,special_characters=True)
    import os

    os.environ["PATH"] += os.pathsep + 'C:/Graphviz/bin'
    from subprocess import call
    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])


    from IPython.display import Image
    Image(filename='tree.png')

    import codecs
    txt_dir = 'tree.dot'
    txt_dir_utf8 = 'tree_utf8.dot'

    with codecs.open(txt_dir, 'r', encoding='utf-8') as f, codecs.open(txt_dir_utf8, 'w', encoding='utf-8') as wf:
        for line in f:
            lines = line.strip().split('\t')
            print(lines)
            if 'label' in lines[0]:
                newline = lines[0].replace('\n', '').replace(' ', '')
            else:
                newline = lines[0].replace('\n', '').replace('helvetica', ' "Microsoft YaHei" ')
            wf.write(newline + '\t')

    import pydot
    (graph,) = pydot.graph_from_dot_file('tree.dot', encoding="utf-8")
    graph.write_png('tree.png')


if __name__ == "__main__":
    x,label = process_data()
    #数据集的划分
    x_train, x_test, y_train, y_test = train_test_split(x, label, test_size=0.3)
    print("使用KNN算法预测结果如下：\n")
    score1 = knn_heart_diease(x,label,x_train,y_train,x_test,y_test)
    print("使用决策树算法预测结果如下：\n")
    score2,score3 = dectree_heart_diease(x,label,x_train,y_train,x_test,y_test)
    print("使用SVM算法预测结果如下:\n")
    score4 = svm_heart_diease(x,label,x_train,y_train,x_test,y_test)


    model = ['KNN', '决策树', '随机森林', 'SVM']
    score = [score1, score2, score3, score4]

    plt.figure(figsize=(15, 10))
    sns.barplot(x=score, y=model)
    plt.show()


