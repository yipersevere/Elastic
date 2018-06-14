import matplotlib

matplotlib.use("PDF")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
"""
plot test
"""
errors = []
# error_1 = np.genfromtxt('./Train_InceptionV3/CIFAR100/Accuracy/InceptionV3_CIFAR100_2018-05-08-23-21-28/accuracies.txt', dtype=float, delimiter=' ', names=False) 
error_1 = pd.read_table('/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/trainModel_withCIFAR/inceptionV3/Train_InceptionV3/CIFAR100/Accuracy/InceptionV3_CIFAR100_2018-05-08-23-21-28/accuracies.txt', delim_whitespace=True, header=None)
error_2 = pd.read_table('/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/trainModel_withCIFAR/inceptionV3/Pretrained_ElasticNN/CIFAR100/AGE/InceptionV3_CIFAR100_2018-05-09-01-31-06/accuracies.txt', delim_whitespace=True, header=None) 
errors.append(list(error_2.iloc[:,11]))
errors.append(list(error_2.iloc[:,10]))
errors.append(list(error_2.iloc[:,6]))
errors.append(list(error_2.iloc[:,0]))
errors.append(list(error_1.iloc[:,0]))
fig, ax = plt.subplots(1, sharex=True)
colormap = plt.cm.tab20
layer_index = [11, 10, 6, 0]
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 1, len(errors))])

for k in range(len(errors)):
    # Plots
    x = np.arange(len(errors[k])) + 1
    if k < 4:
        c_label = 'Layer ' + str(layer_index[k])
    elif k == 4:
        c_label = 'original_inceptionV3'
    ax.plot(x, errors[k], label=c_label)

    # Legends
    y = k
    x = len(errors)
    ax.text(x, y, "%d" % k)

ax.set_ylabel('error rate on CIFAR-100')
ax.set_xlabel('epoch')
title = "InceptionV3 test on CIFAR 100"
title2 = "ElasticNN-InceptionV3 test on CIFAR 100"
ax.set_title(title)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

fig_size = plt.rcParams["figure.figsize"]

plt.rcParams["figure.figsize"] = fig_size

plt.tight_layout()

imagecaption = "accuracy_InceptionV3_CIFAR_100.pdf"
imagecaption2 = "accuracy_Elastic_InceptionV3_CIFAR_100.pdf"
plt.savefig(imagecaption, bbox_inches="tight")
plt.close("all")