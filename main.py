from funct import load_epoch, analysis
from os import listdir
from os.path import isdir, join
from confusion_matrix_pretty_print import pretty_plot_confusion_matrix
import mne as mne
import numpy as np
import seaborn as sn
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from read_open import read_file
from folder import read_folder
from os import listdir
from os.path import isfile, join
# def analysis(epoch, clf="linear", typec="mfcc", comparison="filterbank", filterbank=[], filterband=[], hilbert=False,
# filter=(None, None), time_slice=[(0, 500), (500, 1000), (1000, 1500)], csp=24, pca=30, step=2, cv=10, plot_clf=False):
mne.set_log_level(verbose=False)


filter1 = (0.1, 70)
filter2 = (8, 70)

dir = listdir("Scalp")
# NET = [f for f in dir if "NET" in f]
ST01 = [f for f in dir if "ST01" in f]
CJH = [f for f in dir if "CJH" in f]
ST02 = [f for f in dir if "ST02" in f]

# name = NET
# name.extend(ST01)
name = ST01
name.extend(CJH)
name.extend(ST02)
e = []
for i in range(len(name)):
    e.extend(read_folder(name[i], type='ieeg', scalp_only=True))
epoch = e#, epoch13, epoch14, epoch15, epoch16]
print(len(epoch))
# epoch = [epoch4]
s = []
for i in range(len(name)):
    s.extend(read_folder(name[i], type='seeg', scalp_only=True))
speechepoch = s
# epoch = mne.concatenate_epochs([epoch11])#, epoch5])
# epoch1.plot_psd()
# epoch2.plot_psd()
# epoch3.plot_psd()
# epoch4.plot_psd()
# epoch5.plot_psd()
# epoch.plot_psd()
# epoch11 = load_epoch("0108_speech_20t2c2s31ch_jwc_1.vhdr", typec="ieeg1", time=(-0.1, 1.9), baseline=(-0.1, 0))
# epoch21 = load_epoch("0108_speech_20t2c2s31ch_jwc_2.vhdr", typec="ieeg1", time=(-0.1, 1.9), baseline=(-0.1, 0))
# #
# # epoch12 = load_epoch("0108_speech_20t2c2s31ch_jwc_1.vhdr", typec="ieeg2", time=(-0.1, 1.9), baseline=(-0.1, 0))
# # epoch22 = load_epoch("0108_speech_20t2c2s31ch_jwc_2.vhdr", typec="ieeg2", time=(-0.1, 1.9), baseline=(-0.1, 0))
# epoch111 = mne.concatenate_epochs([epoch11, epoch21])
# epoch222 = mne.concatenate_epochs([epoch12, epoch22])
#
# fepoch = epoch111.copy().filter(8, 150).plot_psd(proj=True, normalization="full", average=True)
# fepoch.plot_image(cmap='interactive')
bank = [(None, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, 45), (45, 50), (50, 55), (55, 60), (65, 70), (70, 75), (75, 80), (80, 85), (85, 90)]#,(50,60),(60,None)]
band = ["[10,40]","[20,50]","[30,60]","[40,70]","[50,80]","[60,None]"]
# time = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 250), (250, 300), (300, 350), (350, 400), (400, 450),
#         (450, 500), (500, 550), (600, 650), (650, 700), (700, 750), (750, 800), (800, 850), (850, 900), (900, 950), (950, 1000)]
# ms = 1000
# tm = []
# for t in range(0, 2000, int(ms/2)):
    # tm.append((t, int(t+ms/2)))
# tm = [(0, 250), (125, 375), (250, 500), (375, 625), (500, 750), (625, 875), (750, 1000)]
# tm = [(0, 200), (100, 300), (200, 400), (300, 500), (400, 600), (500, 700), (600, 800), (700, 900), (800, 1000)]
# tm = [(0, 100), (50, 150), (100, 200), (150, 250), (200, 300), (250, 350), (300, 400), (350, 450), (400, 500)]
# tm = [(0, 250), (125, 375), (250, 500)]
tm = [(0, 500), (250, 750), (500, 1000)]
# tm = [(0, 50), (25, 75), (50, 100), (75, 125), (100, 150), (125, 175), (150, 200), (175, 225), (200, 250)]
# tm = [(0, 500), (100, 600), (200, 700), (300, 800), (400, 900), (500, 1000)]
# print("MFCC")
# analysis([epoch1, epoch2], typec="MFCC", comparison=None, filter=(20, 60), clf="SVC")
# print("MFCCPCA")
# analysis([epoch1, epoch2], typec="MFCCPCA", comparison=None, filter=(20, 50), clf="SVC", cv=10, csp=9, pca=10)
# print("TSclassifier")
# analysis(
#             [epoch1, epoch2], typec="TSc", comparison="filterbank", filter=(30, 60), filterbank=bank, filterband=band, csp=6,
#             step=1, pca=19, time_slice=tm, cv=10, clf="linear", plot_clf=False)

# for i     in range(10):
#     cvprob.append(np.mean(
#         analysis(
#             [epoch1, epoch2], typec="tCSPPCA", comparison="None", filter=(20, 50), filterbank=bank, filterband=band, csp=6,
#             step=1, pca=19, time_slice=tm, cv=10, clf="linear", plot_clf=False, shuffle=True)))

# for i in range(10):
#     cvprob.append(np.mean(
#         analysis(
#             [epoch1, epoch2], typec="tCSPPCA", comparison="None", filter=(20, 60), filterbank=bank, filterband=band, csp=6,
#             step=1, pca=19, time_slice=tm, cv=10, clf="SVC", plot_clf=False, shuffle=True)))
random_state = [43, 31, 25, 36, 5, 32, 22, 10, 21, 40]
methods = ["CSP", "filterbank", "vCSP", "tCSP", "tCSPPCA", "tCSPMIC"]
mAccuracy = []
mcAccuracy = []
# print(random_state)
j = "SINGLE"
# for j in methods:

clprob = []
var = []
clvar = []
cf = []
cf2 = []
kp = []
num_c = 5
ac = []
algorithm = "TSc"
classifiers = ["RVC"]#, 'RVC']
methods = ["TSc"]#, "CSP"]
for classifier in classifiers:
    for method in methods:
        clprob = []
        var = []
        clvar = []
        cf = []
        cf2 = []
        kp = []
        num_c = 5
        ac = []
        for e in range(len(epoch)):
            cvprob = []
            prob, cprob, conf, kappa = analysis(
                [epoch[e]], typec=method, comparison="None", filter=(0.1, 100), filterbank=bank,
                filterband=band, csp=6, step=1, pca=36, time_slice=tm, state=None, cv=10, clf=classifier,
                plot_clf=False, shuffle=False, total=False, confusion=True, use_feature=False, seeg=[speechepoch[e]])
            cvprob.append(np.mean(prob))
            var.append(np.var(prob))
            clprob.append(np.average(cprob, axis=0))
            clvar.append(np.var(cprob, axis=0))
            confusion = np.sum(conf, axis=0)
            kp.append(kappa)
            cf.append(confusion)
            ac.append(np.mean(cvprob))
        # print(np.sum(cf, axis=0))
        # df_cm = pd.DataFrame(np.sum(cf, axis=0), range(num_c), range(num_c))
            # plt.figure(figsize=(10,7))

        # for i in range(10):
        #     prob, cprob, conf = analysis(
        #         [epoch4, epoch5, epoch6], typec="evCSP", comparison="None", filter=(8, 70), filterbank=bank,
        #         filterband=band, csp=15, step=1, pca=30, time_slice=tm, state=random_state[i], cv=10, clf="SVC",
        #         plot_clf=False, shuffle=True, total=False, confusion=True)
        #     confusion = np.sum(conf, axis=0)
        #     cf2.append(confusion)
        # print(np.sum(cf2, axis=0))
        # df_cm2 = pd.DataFrame(np.sum(cf2, axis=0), range(4), range(4))
        print(ac)

        f = open(classifier + "_" + method + "_seeg_results.txt", "w")
        for i in range(len(name)):
            f.write(name[i] + "\n")
            data = ""
            for j in range(2):
                data = data + str(ac[i*2 + j]) + "\t"
            f.write(data+"\n")
        f.close()

for classifier in classifiers:
    for method in methods:
        clprob = []
        var = []
        clvar = []
        cf = []
        cf2 = []
        kp = []
        num_c = 5
        ac = []
        for e in range(len(epoch)):
            cvprob = []
            prob, cprob, conf, kappa = analysis(
                [epoch[e]], typec=method, comparison="None", filter=(0.1, 100), filterbank=bank,
                filterband=band, csp=6, step=1, pca=36, time_slice=tm, state=None, cv=10, clf=classifier,
                plot_clf=False, shuffle=False, total=False, confusion=True, use_feature=False, seeg=False)
            cvprob.append(np.mean(prob))
            var.append(np.var(prob))
            clprob.append(np.average(cprob, axis=0))
            clvar.append(np.var(cprob, axis=0))
            confusion = np.sum(conf, axis=0)
            kp.append(kappa)
            cf.append(confusion)
            ac.append(np.mean(cvprob))
        # print(np.sum(cf, axis=0))
        # df_cm = pd.DataFrame(np.sum(cf, axis=0), range(num_c), range(num_c))
        # plt.figure(figsize=(10,7))

        # for i in range(10):
        #     prob, cprob, conf = analysis(
        #         [epoch4, epoch5, epoch6], typec="evCSP", comparison="None", filter=(8, 70), filterbank=bank,
        #         filterband=band, csp=15, step=1, pca=30, time_slice=tm, state=random_state[i], cv=10, clf="SVC",
        #         plot_clf=False, shuffle=True, total=False, confusion=True)
        #     confusion = np.sum(conf, axis=0)
        #     cf2.append(confusion)
        # print(np.sum(cf2, axis=0))
        # df_cm2 = pd.DataFrame(np.sum(cf2, axis=0), range(4), range(4))
        print(ac)

        f = open(classifier + "_" + method + "_ieegonly_results.txt", "w")
        for i in range(len(name)):
            f.write(name[i] + "\n")
            data = ""
            for j in range(2):
                data = data + str(ac[i * 2 + j]) + "\t"
            f.write(data + "\n")
        f.close()

# for classifier in classifiers:
#     for method in methods:
#         for e in range(len(epoch)):
#             cvprob = []
#             prob, cprob, conf, kappa = analysis(
#                 [epoch[e]], typec=method, comparison="None", filter=(0.1, 100), filterbank=bank,
#                 filterband=band, csp=6, step=1, pca=36, time_slice=tm, state=None, cv=10, clf=classifier,
#                 plot_clf=False, shuffle=False, total=False, confusion=True, use_feature=False, seeg=[speechepoch[e]], seeg_only=True)
#             cvprob.append(np.mean(prob))
#             var.append(np.var(prob))
#             clprob.append(np.average(cprob, axis=0))
#             clvar.append(np.var(cprob, axis=0))
#             confusion = np.sum(conf, axis=0)
#             kp.append(kappa)
#             cf.append(confusion)
#             ac.append(np.mean(cvprob))
#         # print(np.sum(cf, axis=0))
#         # df_cm = pd.DataFrame(np.sum(cf, axis=0), range(num_c), range(num_c))
#         # plt.figure(figsize=(10,7))
#
#         # for i in range(10):
#         #     prob, cprob, conf = analysis(
#         #         [epoch4, epoch5, epoch6], typec="evCSP", comparison="None", filter=(8, 70), filterbank=bank,
#         #         filterband=band, csp=15, step=1, pca=30, time_slice=tm, state=random_state[i], cv=10, clf="SVC",
#         #         plot_clf=False, shuffle=True, total=False, confusion=True)
#         #     confusion = np.sum(conf, axis=0)
#         #     cf2.append(confusion)
#         # print(np.sum(cf2, axis=0))
#         # df_cm2 = pd.DataFrame(np.sum(cf2, axis=0), range(4), range(4))
#         print(ac)
#
#         f = open(classifier + "_" + method + "_seegonly_results.txt", "w")
#         for i in range(len(name)):
#             f.write(name[i] + "\n")
#             data = ""
#             for j in range(2):
#                 data = data + str(ac[i * 2 + j]) + "\t"
#             f.write(data + "\n")

# print("5x2-crossfold validation (%s)" % j)
# print(cvprob)
# print("Mean accuracy : %f" % np.mean(cvprob))
# print("Variance")
# print(var)
# print(np.var(cvprob))
# print("Class accuracy")
# print(clprob)
# print(np.mean(clprob, axis=0))
# print("Variance")
# print(clvar)
# print(np.var(clprob, axis=0))
# print("Kappa Scores")
# print(kp)
# print(np.mean(kp, axis=1))
# print(np.mean(kp))
# sn.set(font_scale=1.4)  # for label size
# pretty_plot_confusion_matrix(df_cm, annot=True)
# plt.show()
# # pretty_plot_confusion_matrix(df_cm2, annot=True)
# plt.show()
# mAccuracy.append(np.mean(cvprob))
# mcAccuracy.append(np.mean(clprob, axis=0))
# for k in range(len(mAccuracy)):
#     print("Average Accuracy for %s : %f" % (methods[k], mAccuracy[k]))
#     print("Average Class Accuracy for %s" % methods[k])
#     print(mcAccuracy[k])
# print("p-value")
# print(stats.binom_test(int(np.mean(cvprob)*800), n=800))











# ieeg : 20, 50, c:9, p:10
# print("CSP")
#CSP : 14, PCA : 7 40, 70
# analysis([epoch1, epoch2], typec="CSP", comparison=None, filter=(20, 60), clf="SVC")
# [ 7, 17, 47, 48, 75]
# [ 8, 46, 60, 65, 71]
# [ 2, 39, 46, 52, 65]
# [31, 33, 45, 60, 65]
# [ 5, 50, 51, 53, 77]
# [10, 12, 17, 66, 78]
# [ 2,  7, 24, 26, 71]
# [ 3, 24, 25, 50, 63, 66, 68, 78]
# [ 7, 27, 31, 33, 57, 65]
# [10, 19, 27, 47, 52, 78]
# [ 8, 28, 34, 41, 48, 52, 56]
# [35, 37, 49, 61, 70]
# [17, 19, 24, 31, 78]
# [19, 21, 24, 26, 66, 72]
# [ 7, 20, 22, 32, 44, 47, 48]
# [12, 18, 36, 37, 40, 42, 78]
# [24, 27, 45, 49, 55]

# m = [ 7, 17, 47, 48, 75, 8, 46, 60, 65, 71,
#  2, 39, 46, 52, 65,
# 31, 33, 45, 60, 65,
#  5, 50, 51, 53, 77,
# 10, 12, 17, 66, 78,
#  2,  7, 24, 26, 71,
#  3, 24, 25, 50, 63, 66, 68, 78,
#  7, 27, 31, 33, 57, 65,
# 10, 19, 27, 47, 52, 78,
#  8, 28, 34, 41, 48, 52, 56,
# 35, 37, 49, 61, 70,
# 17, 19, 24, 31, 78,
# 19, 21, 24, 26, 66, 72,
#  7, 20, 22, 32, 44, 47, 48,
# 12, 18, 36, 37, 40, 42, 78,
# 24, 27, 45, 49, 55]
#
# from scipy import stats
#
# mo = stats.mode(m)
# print(mo)

# [14 40 42 57 66 71 76 78]
# [14 42 57 66 71]
#

# 21, 33