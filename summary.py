import numpy as np
import os
from glob import glob
import argparse
from sklearn import metrics
from utils import *
import pandas as pd
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, classes, normalize=False, title='Transition matrix when preliminary results change', cmap=plt.cm.Blues):


    plt.rc('font',family='Times New Roman')
    plt.rc('font', weight='bold')
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=12,fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    plt.axis("equal")

    ax = plt.gca()
    left, right = plt.xlim()
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        plt.text(j, i, num,
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if num > thresh else "black")

    plt.ylabel('Preliminary results',fontsize=12,fontweight='bold')
    plt.xlabel('Final results with CRF optimization',fontsize=12,fontweight='bold')


    plt.tight_layout()
    plt.savefig('CRF_CHANGE.png', transparent=True, dpi=800)

    plt.show()

def pd_toExcel(data, fileName):  # pandas库储存数据到excel
    y_label = data.item()['label'].tolist()
    y_pre1 = data.item()['y_pre1'].tolist()
    y_pre2 = data.item()['y_pre2'].tolist()

    dfData = {  # 用字典设置DataFrame所需数据
        'y_label': y_label,
        'y_pre1': y_pre1,
        'y_pre2': y_pre2
    }
    df = pd.DataFrame(dfData)  # 创建DataFrame
    df.to_excel(fileName, index=False)  # 存表，去除原始索引列（0,1,2...）


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TSANet')
    parser.add_argument('--output', type=str, default="./output_20/")
    args = parser.parse_args()

    data = np.load(os.path.join(args.output, "output_all.npy"), allow_pickle=True)
    print(type(data))

    pd_toExcel(data, os.path.join(args.output, "output_all.xlsx"))
    y_label = data.item()['label']
    y_pre1 = data.item()['y_pre1']
    y_pre2 = data.item()['y_pre2']
    print(type(y_label))
    print(y_label)

    print(y_label.shape)
    crf = np.zeros((5, 5))
    crf_true = np.zeros((5, 5))
    crf_change = np.zeros((5, 5))
    for i in range(len(y_label)):
        if (y_pre2[i] == y_label[i]) & (y_pre1[i] != y_label[i]):
            print("corected!", i, " from y_pre1 ", y_pre1[i], " to y_pre2 ", y_pre2[i])
            crf_true[y_pre1[i]][y_pre2[i]] = crf_true[y_pre1[i]][y_pre2[i]] + 1
        crf[y_pre1[i]][y_pre2[i]] = crf[y_pre1[i]][y_pre2[i]]+1
        if y_pre1[i] != y_pre2[i]:
            crf_change[y_pre1[i]][y_pre2[i]] = crf_change[y_pre1[i]][y_pre2[i]] + 1
    print(crf.astype(int))
    crfpre1_sum = np.int64(np.sum(crf.astype(int), 1))
    print(crfpre1_sum)
    for i in range(5):
        crf[i] = crf[i] / crfpre1_sum[i]
    print(np.around(crf, 2))

    print(crf_true.astype(int))

    print(crf_change.astype(int))
    correct_rate = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            correct_rate[i][j] = crf_true[i][j]/crf_change[i][j]
    print(np.around(correct_rate, 2))

    label = ["W", "N1", "N2", "N3", "REM"]
    plot_confusion_matrix(crf_true, label)

    confusion_all1 = metrics.confusion_matrix(np.array(y_label), np.array(y_pre1))
    mf1_1 = metrics.f1_score(np.array(y_label), np.array(y_pre1), average='macro')
    ka_1 = metrics.cohen_kappa_score(np.array(y_label), np.array(y_pre1))
    ac_1 = metrics.accuracy_score(np.array(y_label), np.array(y_pre1))

    confusion_all2 = metrics.confusion_matrix(np.array(y_label), np.array(y_pre2))
    mf1_2 = metrics.f1_score(np.array(y_label), np.array(y_pre2), average='macro')
    ka_2 = metrics.cohen_kappa_score(np.array(y_label), np.array(y_pre2))
    ac_2 = metrics.accuracy_score(np.array(y_label), np.array(y_pre2))

    pr_sum = np.int64(np.sum(confusion_all2.astype(int), 0))
    print(pr_sum, pr_sum.sum())
    re_sum = np.int64(np.sum(confusion_all2.astype(int), 1))
    print(re_sum, re_sum.sum())

    pr = np.zeros((5))
    for i in range(5):
        pr[i] = confusion_all2[i][i] / pr_sum[i]

    re = np.zeros((5))
    for i in range(5):
        re[i] = confusion_all2[i][i] / re_sum[i]

    f1 = np.zeros((5))
    for i in range(5):
        f1[i] = 2 * pr[i] * re[i] / (pr[i] + re[i])

    pr_sum1 = np.int64(np.sum(confusion_all1.astype(int), 0))
    re_sum1 = np.int64(np.sum(confusion_all1.astype(int), 1))

    pr1 = np.zeros((5))
    for i in range(5):
        pr1[i] = confusion_all1[i][i] / pr_sum1[i]

    re1 = np.zeros((5))
    for i in range(5):
        re1[i] = confusion_all1[i][i] / re_sum1[i]

    f11 = np.zeros((5))
    for i in range(5):
        f11[i] = 2 * pr1[i] * re1[i] / (pr1[i] + re1[i])
    print("acc_preliminary: %.4f, acc_final: %.4f, kappa: %.4f, mf1: %.4f" %(ac_1, ac_2, ka_2, mf1_2))
    print("W：%.4f, N1：%.4f, N2：%.4f, N3：%.4f, REM：%.4f" % (f1[0], f1[1], f1[2], f1[3], f1[4]))
    print("W：%.4f, N1：%.4f, N2：%.4f, N3：%.4f, REM：%.4f" % (f11[0], f11[1], f11[2], f11[3], f11[4]))
    print(confusion_all2)


