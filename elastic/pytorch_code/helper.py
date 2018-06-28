
import matplotlib

matplotlib.use("PDF")
import matplotlib.pyplot as plt

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import scipy

import datetime
from io import StringIO
import shutil
import random
import sys
import time
import sys
import csv



def LOG(message, logFile):
    ts = datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    msg = "[%s] %s" % (ts, message)

    with open(logFile, "a") as fp:
        fp.write(msg + "\n")

    print(msg)

def log_summary(model, logFile):
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    model.summary()

    sys.stdout = old_stdout
    summary = mystdout.getvalue()

    LOG("Model summary:", logFile)

    for line in summary.split("\n"):
        LOG(line, logFile) 


def log_error(labels, predictions):
    total = 0
    # print(predictions[0])
    # print(labels[0])
    for i in range(len(predictions)):
        if list(predictions[i]).index(max(list(predictions[i]))) ==  list(labels[i]).index(1):
            total += 1
    error = (1- total/len(labels)) * 100
    return error


def log_stats(path, epochs_acc_train, epochs_intermediate_acc_train, epochs_loss_train, epochs_lr, epochs_acc_test, epochs_intermediate_acc_test, epochs_loss_test):

    with open(path + os.sep + "train_accuracies.txt", "a") as fp:
        for a in epochs_acc_train:
            fp.write("%.4f " % a)
        fp.write("\n")

    with open(path + os.sep + "train_intermediate_accuracies.txt", "a") as fp:
        for a in epochs_intermediate_acc_train:
            fp.write("%.4f " % a)
        fp.write("\n")    

    with open(path + os.sep + "train_losses.txt", "a") as fp:
        for loss in epochs_loss_train:
            fp.write("%.4f " % loss)
        fp.write("\n")

    with open(path + os.sep + "epochs_lr.txt", "a") as fp:
        for a in epochs_lr:
            fp.write("%.4f " % a)
        fp.write("\n")    

    with open(path + os.sep + "test_accuracies.txt", "a") as fp:
        for a in epochs_acc_test:
            fp.write("%.4f " % a)
        fp.write("\n")

    with open(path + os.sep + "test_intermediate_accuracies.csv", "a") as fp:
        wr = csv.writer(fp)
        wr.writerows(epochs_intermediate_acc_test)
    
    with open(path + os.sep + "test_losses.txt", "a") as fp:
        for loss in epochs_loss_test:
            fp.write("%.4f " % loss)
        fp.write("\n")    



class Plot():

    def __init__(self, args):
        self.imageStr = args.imageStr
        self.errors = args.errors


    def plot(self):
        """
        plot test
        """
        # fig, ax = plt.subplots(1, sharex=True)
        fig, (ax0, ax1) = plt.subplots(1, 1, sharey=True)
        colormap = plt.cm.tab20

    # plot f1-score
        plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 1, self.num_outputs)])

        for k in range(len(self.errors[0])):
            # Plots
            x = np.arange(len(self.errors)) + 1
            y = np.array(self.errors)[:, k]
            c_label = 'Layer ' + str(k)
            ax0.plot(x, y, label=c_label)

            # Legends
            y = self.errors[-1][k]
            x = len(self.errors)
            ax0.text(x, y, "%d" % k)
        
        ax0.set_ylabel(self.imageStr["ax0_set_ylabel"])
        ax0.set_xlabel('epoch')
        # title = "InceptionV3 test on CIFAR 100"
        # title2 = "ElasticNN-InceptionV3 test on CIFAR 100"
        ax0.set_title(self.imageStr["ax0_title"])
        ax0.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    # plot f1-score
        # fig_f_score, ax_f_score = plt.subplots(1, sharex=True)
        # plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 1, self.num_outputs)])

        # for k in range(len(self.f_score[0])):
        #     # Plots
        #     x = np.arange(len(self.f_score)) + 1
        #     y = np.array(self.f_score)[:, k]
        #     c_label = 'Layer ' + str(k)
        #     ax1.plot(x, y, label=c_label)

        #     # Legends
        #     y = self.f_score[-1][k]
        #     x = len(self.f_score)
        #     ax1.text(x, y, "%d" % k)
        
        # ax1.set_ylabel(self.imageStr["ax1_set_ylabel"])
        # ax1.set_xlabel('epoch')
        # # title = "InceptionV3 test on CIFAR 100"
        # # title2 = "f1-score ElasticNN-InceptionV3 test on CIFAR 100"
        # ax1.set_title(self.imageStr["ax1_title"])
        # ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        fig_size = plt.rcParams["figure.figsize"]

        fig_size[0] = 6.4
        # fig_size[1] = 6.4

        plt.rcParams["figure.figsize"] = fig_size

        #plt.tight_layout()

        # imagecaption = "accuracy_InceptionV3_CIFAR_100.pdf"
        # imagecaption2 = "accuracy_Elastic_InceptionV3_CIFAR_100.pdf"
        plt.savefig(path + os.sep + self.imageStr["save_fig"], bbox_inches="tight")
        plt.close("all")