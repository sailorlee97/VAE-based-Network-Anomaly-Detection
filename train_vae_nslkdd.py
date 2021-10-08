"""
@Time    : 2021/7/15 16:56
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : chrislistudy@163.com
-------------------------------------------------
@FileName: train_vae_nslkdd.py
@Software: PyCharm
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import classification_report, roc_curve, auc
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary

from load_data.process_nslkdd import main_process
from models.VAE import VAE
from utils.plot_culve import plot_ROC, plot_loss_new,plt_loss,plot_confusion_matrix,plot_box
from sklearn.metrics import confusion_matrix
# from torch.utils.tensorboard import SummaryWriter


num_epochs = 80


def calculate_losses(x, preds):
    losses = np.zeros(len(x))
    for i in range(len(x)):
        losses[i] = ((preds[i] - x[i]) ** 2).mean(axis=None)

    return losses


def get_threshold(losses):
    # length = len(losses)
    sum_list = sum(losses[-5:])
    avg = sum_list / 5
    return avg


def cal_roc(score, standard):
    # 计算出每个模型预测结果对应的fpr，tpr
    y = standard.cpu().detach().numpy().flatten()  # y表示真实标签
    predict = score.cpu().detach().numpy().flatten()  # predict表示预测出来的值
    fpr, tpr, threshold = roc_curve(y, predict)
    AUC = auc(fpr, tpr)
    AUC = ('%.5f' % AUC)  # 只比较auc前5位数字
    return fpr, tpr, AUC


def plot_roc(fpr_list, tpr_list, auc_list, model_name):
    # 在一张图中画出多个model的roc曲线
    fig = plt.figure()
    legend_list = []
    for i in range(len(model_name)):
        plt.plot(fpr_list[i], tpr_list[i])  # 先x后y
        legend_list.append(model_name[i] + '(auc:' + str(auc_list[i]) + ')')
    plt.legend(legend_list)
    plt.xlabel('False Positve Rate')
    plt.ylabel('True Postive Rate')
    plt.title('ROC curve for RNA-disease model')
    fig.savefig("ROC.png")
    plt.show()
    return


def figure_prob(list):
    max_loss = max(list)
    min_loss = min(list)
    # min, max = findMinAndMax(list)
    prob = []
    Denominator = max_loss - min_loss
    for item in list:
        prob.append((item - min_loss) / Denominator)
    prob_ = np.array(prob)
    prob = prob_.reshape((prob_.shape[0], 1))
    return prob
def plot_roc_auc(y_test, y_test_scores, name):
    '''

    :param y_test: train data
    :param y_test_scores: prob
    :param name: conbined name
    :return: figure
    '''

    fpr, tpr, threshod = roc_curve(y_test, y_test_scores)

    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, 'darkorange', label='ROC (area = {0:.4f})'.format(roc_auc), lw=2)
    print('auc:',roc_auc)
    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
    plt.title('%s ROC Curve' % name)
    plt.legend(loc="lower right")
    plt.show()

if __name__ == '__main__':
    # losses = [1,2,3,4,5,6,8,3,2,63,8,3]
    # avg = get_threshold(losses)

    Loss_list = []
    data, x_test, y_test = main_process(100)
    # data = (data.astype(np.float32) - 127.5) / 127.5
    # X_normal = data.values.reshape(data.shape[0], 44)  # 变成矩阵格式
    # print(X_normal.shape)
    X_normal = data
    # 切分数据集
    x_test = x_test
    x_test_tensor = torch.FloatTensor(x_test)

    model = VAE(data.shape[1], 8)
    X_normal = torch.FloatTensor(X_normal)
    X_normal_data = TensorDataset(X_normal)  # 对tensor进行打包
    train_loader = DataLoader(dataset=X_normal_data, batch_size=32,
                              shuffle=True)  # 数据集放入Data.DataLoader中，可以生成一个迭代器，从而我们可以方便的进行批处理

    optimizer = torch.optim.Adam(model.parameters(), 0.00001)
    loss_func = torch.nn.MSELoss(reduction='mean')
    summary(model, ((data.shape[1], data.shape[1])))

    for epoch in range(num_epochs):
        total_loss = 0.
        for step, (x,) in enumerate(train_loader):
            # print('x:',x.shape)
            x_recon, z,mu, logvar= model.forward(x)
            # loss = calculate_losses(x_recon,x)
            loss = loss_func(x_recon, x)
            optimizer.zero_grad()
            # 计算中间的叶子节点，计算图
            loss.backward()
            # 内容信息反馈
            optimizer.step()
            total_loss += loss.item() * len(x)
            # print('Epoch :', epoch, ';Batch', step, ';train_loss:%.4f' % loss.data)
            # writer.add_scalar('loss', loss, step + len(train_loader) * epoch)  # 可视化变量loss的值
        total_loss /= len(X_normal)
        print('Epoch {}/{} : loss: {:.4f}'.format(
            epoch + 1, num_epochs, loss.item()))
        Loss_list.append(loss.item())
    plt_loss(num_epochs, Loss_list)

    # writerCSV = pd.DataFrame(data=Loss_list)
    # array = np.array(Loss_list)
    # writerCSV.to_csv('./csv_data/losses.csv', encoding='utf-8')

    # 阈值
    threshold = get_threshold(Loss_list)
    print('threshold:', threshold)

    #验证集 训练集的分布
    #normal_predictions, z_NORMAL,mu_NORMAL, logvar_NORMAL = model.forward(X_normal)
    #normal_predictions = normal_predictions.detach().numpy()
    #normal_losses = calculate_losses(normal_predictions, data)
    #sns.distplot(normal_losses, kde=True)
    #plt.show()


    # 对结果进行预测
    testing_set_predictions,z,mu, logvar  = model.forward(x_test_tensor)
    # print(testing_set_predictions)
    testing_set_predictions = testing_set_predictions.detach().numpy()
    # print(testing_set_predictions)

    # y_test = np.ones(len(x_test))
    # print(y_test)
    losses = calculate_losses(testing_set_predictions, x_test)

    testing_set_prediction_losses= np.zeros(len(losses))
    testing_set_prediction_losses[np.where(losses > threshold)] = 1
    print("-------------------------------使用阈值--------------------------------")
    print(classification_report(y_test, testing_set_prediction_losses,digits=4))




    np.savetxt('./score.csv', losses, delimiter = ',')

    losses_list = losses.tolist()
    prob = figure_prob(losses_list)
    prob = np.array(prob)
    prob = prob.reshape(len(prob))
    print("prob:", prob)

    plot_roc_auc(y_test, prob, 'vae')
    testing_set_predictions = np.zeros(len(prob))
    testing_set_predictions[np.where(prob > 0.0125)] = 1

    prob_normal  = prob[:50000]
    prob_malware = prob[50000:]
    #plt.hist(losses)
    sns.distplot(prob_normal, kde=True)
    sns.distplot(prob_malware, kde=True)
    plt.show()
    # sns.kdeplot(data, shade=True)

    # ----------------------------评价总体体系标准---------------------------------
    print("-------------------------------mse归一化处理--------------------------------")
    print(classification_report(y_test, testing_set_predictions,digits=4))
    cm = confusion_matrix(y_test, testing_set_predictions)
    print("混淆矩阵：",cm)
    classes = ['normal','malware']
    plot_confusion_matrix(cm,'confusion_matrix.png',classes ,title='confusion matrix')
    plot_loss_new(50500,losses_list)

    losses_normal = losses[:50000]
    losses_malware = losses[50000:]

    plt.figure(figsize=(8,5))
    plt.title('Examples of boxplot', fontsize=20)
    labels = 'normal', 'malware'
    plt.boxplot([losses_normal, losses_malware], labels=labels)
    plt.show()

    plot_box(losses_normal, losses_malware,'loss')
    plot_box(prob_normal,prob_malware,'prob')
