import mne as mne
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, normalize
from sklearn.model_selection import ShuffleSplit, cross_val_score
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from scipy.spatial.distance import euclidean, cdist
from librosa.feature import mfcc
from fastdtw import fastdtw
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from pyriemann.classification import MDM, TSclassifier
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from pyriemann.estimation import Covariances
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from matplotlib import pyplot as plt
import seaborn as sns
from pyriemann.embedding import Embedding
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from read_open import extract_feature_set_one
# from skrvm import RVC
from pyriemann.classification import MDM, TSclassifier
cn = 5

def load_epoch(filename, notch=60, typec="ieeg", time=(-0.1, 2.9), baseline=(-0.1, 0), filter=(0.1, 100)):
    raw = mne.io.read_raw_brainvision(filename)
    raw.load_data()
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage)
    # raw = raw.resample(500)
    # fRaw = raw
    fRaw = raw.notch_filter([notch, notch*2, notch*3])
    fRaw = fRaw.filter(filter[0], filter[1])
    # fRaw.plot()
    # fRaw.plot()
    # fRaw = fRaw.notch_filter(notch*2)
    # fRaw = fRaw.notch_filter(notch*3)

    if typec == "veeg":
        eventid = {"['Stimulus']/11": 0, "['Stimulus']/21": 1, "['Stimulus']/31": 2, "['Stimulus']/41": 3}#, "['Stimulus']/3": 4}
    elif typec == "seeg":
        eventid = {"['Stimulus']/12": 0, "['Stimulus']/22": 1, "['Stimulus']/32": 2, "['Stimulus']/42": 3}#,"['Stimulus']/3": 4}#, "['Stimulus']/3": 4}
    elif typec == "ieeg":
        eventid = {"['Stimulus']/13": 0, "['Stimulus']/23": 1, "['Stimulus']/33": 2, "['Stimulus']/43": 3,"['Stimulus']/3": 4}
        # eventid = {"['Stimulus']/3": 4, "['Stimulus']/13": 0, "['Stimulus']/23": 1, "['Stimulus']/33": 2, "['Stimulus']/43": 3}


    event, eid = mne.events_from_annotations(fRaw, event_id=eventid)

    epoch = mne.Epochs(fRaw, event, event_id=eid, tmin=time[0], tmax=time[1], baseline=baseline)
    epoch.load_data()

    scale = mne.decoding.Scaler(epoch.info)
    x = scale.fit_transform(epoch.get_data())
    # print(x[0][0])
    epoch = mne.EpochsArray(x, info=epoch.info, events=event, event_id=eid)

    # epoch.drop_bad()
    # select = ['F5', 'C1', 'FC1', 'Fz', 'FT7', 'Cz', 'PO3', 'P7', 'CP1', 'F3', 'C3', 'Oz', 'T7', 'TP7', 'FC5']
    # ch = [epoch.ch_names[i] for i in select]
    ch = ['Fz', 'Cz', 'CPz', 'Pz', 'POz', 'Oz', 'PO3', 'F1', 'FC1', 'C1', 'CP1', 'P1']
    lh = ['Fp2','AF4','AF8','F2','F4','F6','F8','FC2','FC4','FC6','FT8','FT10','C2','C4','C6','T8','CP2','CP4','CP6','TP8','TP10','P2','P4','P6','P8','PO4','PO8','O2']
    rh = ['Fp1','AF7','AF3','F7','F5','F3','F1','FT9','FT7','FC5','FC3','FC1','T7','C5','C3','C1','TP9','TP7','CP5','CP3','CP1','P7','P5','P3','P1','PO7','PO3','O1']

    # print(ch)
    #
    # epoch.pick_channels(select)
    # epoch.drop_channels(['O1', 'O2'])
    # epoch.drop_channels(['Fp1'])
    # print(np.shape(epoch.get_data()))
    return epoch

# e = load_epoch("0527_ST01/0527_STIM_aligned_speech_10t5c2s32ch_ST01_1.vhdr")
# print(np.shape(e.get_data()))
# print(e.events[:, -1])
def run_cv(epoch, typec, c, p, cv, hilbert, time_slice, state, classifier="linear", plot_clf=False,
           filterbank=[(10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, 45), (45, 50), (50, 55), (55, 60), (60, 65), (65, 70)],
           shuffle=True, total=False, confusion=False, use_feature=False, temp=False, seeg=False, seeg_only=False):
    test_accuracy = []
    class_accuracy = []
    kappa = []
    if shuffle:
        if state == None:
            skf = StratifiedKFold(n_splits=cv, shuffle=False)
        else:
            skf = StratifiedKFold(n_splits=cv, shuffle=shuffle, random_state=state)
    else:
        skf = StratifiedKFold(n_splits=cv, shuffle=False)

    scale = mne.decoding.Scaler(info = epoch.info)
    X = epoch.get_data()
    # X = scale.fit_transform(X)
    print(np.shape(X))
    X = np.real(X)
    if seeg:
        x_seeg = seeg.get_data()
        y_seeg = seeg.events[:, -1]
    conf_list = []
    Y = epoch.events[:, -1]
    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        if total:
            Y_test = Y
            X_test = X
        if seeg:
            if seeg_only:
                X_train = x_seeg
                Y_train = y_seeg
            else:
                X_train = np.concatenate((X_train,x_seeg), axis=0)
                print(np.shape(X_train))
                Y_train = np.concatenate((Y_train,y_seeg))
        if classifier == "linear":
            clf = LinearDiscriminantAnalysis()

        elif classifier == "SVC":
            clf = SVC(probability=True, C=1)
        # elif classifier == "RVC":
        #     clf = RVC(alpha=1e-06, beta=1e-06, beta_fixed=False, bias_used=True, coef0=0.0,
        #               coef1=None, degree=3, kernel='linear', n_iter=3000,
        #               threshold_alpha=1000000000.0, tol=0.001, verbose=False)
        elif classifier == "RVC":
            from sklearn_rvm import EMRVC
            clf = EMRVC(n_iter_posterior=10)
        else:
            clf = LinearSVC()

        # if typec == "MFCC":
        #     # csp = mne.decoding.CSP(n_components=c, reg=None, transform_into='csp_space', norm_trace=False)
        #     # train_csp = csp.fit_transform(X_train, Y_train)
        #     mfcc_train = []
        #     for k in range(np.shape(X_train)[0]):
        #         mfcc1 = []
        #         for j in range(np.shape(X_train)[1]):
        #             mfcc2 = np.reshape(mfcc(X_train[k, j, :], sr=500, n_mfcc=13), [26])
        #             mfcc1 = np.concatenate((mfcc1, mfcc2))
        #         mfcc_train.append(mfcc1)
        #     clf.fit(mfcc_train, Y_train)
        #     # test_csp = csp.transform(X_test)
        #     mfcc_test = []
        #     for k in range(np.shape(X_test)[0]):
        #         mfcc1 = []
        #         for j in range(np.shape(X_test)[1]):
        #             mfcc2 = np.reshape(mfcc(X_test[k, j, :], sr=500, n_mfcc=13), [26])
        #             mfcc1 = np.concatenate((mfcc1, mfcc2))
        #         mfcc_test.append(mfcc1)
        #     predict = clf.predict(mfcc_test)
        #     # if confusion:
        #     #     disp = plot_confusion_matrix(clf, mfcc_test, Y_test)
        #     #     print(disp.confusion_matrix)
        # 
        # 
        # elif typec == "MFCCPCA":
        #     # csp = mne.decoding.CSP(n_components=c, reg=None, transform_into='csp_space', norm_trace=False)
        #     # pca = PCA(n_components=p, svd_solver="full")
        #     # train_csp = csp.fit_transform(X_train, Y_train)
        #     # mfcc_train = []
        #     # for k in range(np.shape(train_csp)[0]):
        #     #     mfcc1 = []
        #     #     for j in range(np.shape(train_csp)[1]):
        #     #         mfcc2 = np.reshape(mfcc(train_csp[k, j, :], sr=500, n_mfcc=20), [80])
        #     #         mfcc1 = np.concatenate((mfcc1, mfcc2))
        #     #     mfcc_train.append(mfcc1)
        #     # train_pca = pca.fit_transform(mfcc_train, Y_train)
        #     # clf.fit(train_pca, Y_train)
        #     # test_csp = csp.transform(X_test)
        #     # mfcc_test = []
        #     # for k in range(np.shape(test_csp)[0]):
        #     #     mfcc1 = []
        #     #     for j in range(np.shape(test_csp)[1]):
        #     #         mfcc2 = np.reshape(mfcc(test_csp[k, j, :], sr=500, n_mfcc=20), [80])
        #     #         mfcc1 = np.concatenate((mfcc1, mfcc2))
        #     #     mfcc_test.append(mfcc1)
        #     # test_pca = pca.transform(mfcc_test)
        #     # predict = clf.predict(test_pca)
        # 
        #     pca = PCA(n_components=p, svd_solver="full")
        #     mfcc_train = []
        #     for k in range(np.shape(X_train)[0]):
        #         mfcc1 = []
        #         for j in range(np.shape(X_train)[1]):
        #             mfcc2 = np.reshape(mfcc(X_train[k, j, :], sr=500, n_mfcc=13), [26])
        #             mfcc1 = np.concatenate((mfcc1, mfcc2))
        #         mfcc_train.append(mfcc1)
        #     train_pca = pca.fit_transform(mfcc_train, Y_train)
        #     clf.fit(train_pca, Y_train)
        #     mfcc_test = []
        #     for k in range(np.shape(X_test)[0]):
        #         mfcc1 = []
        #         for j in range(np.shape(X_test)[1]):
        #             mfcc2 = np.reshape(mfcc(X_test[k, j, :], sr=500, n_mfcc=13), [26])
        #             mfcc1 = np.concatenate((mfcc1, mfcc2))
        #         mfcc_test.append(mfcc1)
        #     test_pca = pca.transform(mfcc_test)
        #     predict = clf.predict(test_pca)
        #     # if confusion:
        #     #     disp = plot_confusion_matrix(clf, mfcc_test, Y_test)
        #     #     print(disp.confusion_matrix)

        if typec == "CSP":
            csp = mne.decoding.CSP(n_components=c, reg=None, log=True, norm_trace=False)
            if use_feature:
                X_train = extract_feature_set_one(X_train)
                X_test = extract_feature_set_one(X_test)
            train_csp = csp.fit_transform(X_train, Y_train)
            clf.fit(train_csp, Y_train)
            test_csp = csp.transform(X_test)
            predict = clf.predict(test_csp)
            if plot_clf:
                layout = mne.channels.read_layout('biosemi')
                csp.plot_filters(epoch.info, components=[0, 1, 2, 3, 4, 5], ch_type='eeg', units='Patterns (AU)', cmap='interactive', scalings=dict(eeg=1, grad=1, mag=1), layout=layout, size=1.5, show_names=True)
            # if confusion:
            #     disp = plot_confusion_matrix(clf, test_csp, Y_test)
            #     print(disp.confusion_matrix)
        # elif typec == "tCSP":
        #     csp = []
        #     train_csp = np.asarray([])
        #     for t in time_slice:
        #         csp_clf = mne.decoding.CSP(n_components=c, reg=None, log=True, norm_trace=False)
        #         if use_feature:
        #             train_feature = np.asarray(csp_clf.fit_transform(extract_feature_set_one(X_train[:, :, t[0]:t[1]]), Y_train))
        #         else:
        #             train_feature = np.asarray(csp_clf.fit_transform(X_train[:, :, t[0]:t[1]], Y_train))
        #         if len(train_csp) == 0:
        #             train_csp = train_feature.T
        #         else:
        #             train_csp = np.append(train_csp, train_feature.T, axis=0)
        #         csp.append(csp_clf)
        # 
        #     clf.fit(train_csp.T, Y_train)
        #     test_csp = np.asarray([])
        #     for i in range(np.shape(time_slice)[0]):
        #         csp_clf = csp[i]
        #         if use_feature:
        #             test_feature = np.asarray(csp_clf.transform(extract_feature_set_one(X_test[:, :, time_slice[i][0]:time_slice[i][1]])))
        #         else:
        #             test_feature = np.asarray(csp_clf.transform(X_test[:, :, time_slice[i][0]:time_slice[i][1]]))
        #         if len(test_csp) == 0:
        #             test_csp = test_feature.T
        #         else:
        #             test_csp = np.append(test_csp, test_feature.T, axis=0)
        #     predict = clf.predict(test_csp.T)
        #     # if confusion:
        #     #     disp = plot_confusion_matrix(clf, test_csp.T, Y_test)
        #     #     print(disp.confusion_matrix)
        # 
        # elif typec == "tCSPPCA":
        #     csp = []
        #     train_csp = np.asarray([])
        #     for t in time_slice:
        #         csp_clf = mne.decoding.CSP(n_components=c, reg=None, log=True, norm_trace=False)
        #         train_feature = np.asarray(csp_clf.fit_transform(X_train[:, :, t[0]:t[1]], Y_train))
        #         if len(train_csp) == 0:
        #             train_csp = train_feature.T
        #         else:
        #             train_csp = np.append(train_csp, train_feature.T, axis=0)
        #         csp.append(csp_clf)
        #     pca = PCA(n_components=p, svd_solver="full")
        #     train_pca = pca.fit_transform(train_csp.T, Y_train)
        #     clf.fit(train_pca, Y_train)
        #     test_csp = np.asarray([])
        #     for i in range(np.shape(time_slice)[0]):
        #         csp_clf = csp[i]
        #         test_feature = np.asarray(csp_clf.transform(X_test[:, :, time_slice[i][0]:time_slice[i][1]]))
        #         if len(test_csp) == 0:
        #             test_csp = test_feature.T
        #         else:
        #             test_csp = np.append(test_csp, test_feature.T, axis=0)
        #     test_pca = pca.transform(test_csp.T)
        #     predict = clf.predict(test_pca)
        #     print(predict)
        #     # if confusion:
        #     #     disp = plot_confusion_matrix(clf, test_pca, Y_test)
        #     #     print(disp.confusion_matrix)
        # 
        # elif typec == "tCSPMIC":
        #     csp = []
        #     train_csp = np.asarray([])
        #     for t in time_slice:
        #         csp_clf = mne.decoding.CSP(n_components=c, reg=None, log=True, norm_trace=False)
        #         train_feature = np.asarray(csp_clf.fit_transform(X_train[:, :, t[0]:t[1]], Y_train))
        #         if len(train_csp) == 0:
        #             train_csp = train_feature.T
        #         else:
        #             train_csp = np.append(train_csp, train_feature.T, axis=0)
        #         csp.append(csp_clf)
        #     kmeans = SelectKBest(mutual_info_classif, k=p)
        #     train_pca = kmeans.fit_transform(train_csp.T, Y_train)
        #     clf.fit(train_pca, Y_train)
        #     test_csp = np.asarray([])
        #     for i in range(np.shape(time_slice)[0]):
        #         csp_clf = csp[i]
        #         test_feature = np.asarray(csp_clf.transform(X_test[:, :, time_slice[i][0]:time_slice[i][1]]))
        #         if len(test_csp) == 0:
        #             test_csp = test_feature.T
        #         else:
        #             test_csp = np.append(test_csp, test_feature.T, axis=0)
        #     print("Selected Features")
        #     a = kmeans.get_support()
        #     print(np.floor(np.multiply(np.where(a == True), 1/c)))
        #     test_pca = kmeans.transform(test_csp.T)
        #     predict = clf.predict(test_pca)
        #     # if confusion:
        #     #     disp = plot_confusion_matrix(clf, test_pca, Y_test)
        #     #     print(disp.confusion_matrix)
        # 
        # elif typec == "filterbank":
        #     train_data = []
        #     test_data = []
        #     for f in filterbank:
        #         # print(f[0])
        #         if use_feature:
        #             train_data.append(extract_feature_set_one(epoch.copy().filter(f[0], f[1]).get_data()[train_index]))
        #             test_data.append(extract_feature_set_one(epoch.copy().filter(f[0], f[1]).get_data()[test_index]))
        #         else:
        #             train_data.append(epoch.copy().filter(f[0], f[1]).get_data()[train_index])
        #             test_data.append(epoch.copy().filter(f[0], f[1]).get_data()[test_index])
        #     csp = []
        #     train_feature = []
        #     test_feature = []
        #     for f in range(len(filterbank)):
        #         csp_clf = mne.decoding.CSP(n_components=c)
        #         feature = np.asarray(csp_clf.fit_transform(train_data[f], Y_train))
        #         tfeature = np.asarray(csp_clf.transform(test_data[f]))
        #         if len(train_feature) == 0:
        #             train_feature = feature.T
        #             test_feature = tfeature.T
        #         else:
        #             train_feature = np.append(train_feature, feature.T, axis=0)
        #             test_feature = np.append(test_feature, tfeature.T, axis=0)
        #     Kbest = SelectKBest(mutual_info_classif, k=p)
        #     print(np.shape(train_feature))
        #     train_csp = Kbest.fit_transform(train_feature.T, Y_train)
        #     print("Selected Features")
        #     a = Kbest.get_support()
        #     print(np.floor(np.divide(np.where(a == True), c)))
        #     clf.fit(train_csp, Y_train)
        #     test_csp = Kbest.transform(test_feature.T)
        #     predict = clf.predict(test_csp)
        #     # if confusion:
        #     #     disp = plot_confusion_matrix(clf, test_csp, Y_test)
        #     #     print(disp.confusion_matrix)
        # 
        # elif typec == "MDM":
        #     mdm = MDM(metric=dict(mean='riemann', distance='riemann'))
        #     cov = Covariances(estimator='lwf')
        #     cov_train = cov.fit_transform(X_train, Y_train)
        #     cov_test = cov.transform(X_test)
        #     mdm.fit(cov_train, Y_train)
        #     predict = mdm.predict(cov_test)
        #     # if confusion:
        #     #     disp = plot_confusion_matrix(mdm, cov_test, Y_test)
        #     #     print(disp.confusion_matrix)

        elif typec == "TSc":
            clf = TSclassifier(clf=clf)
            cov = Covariances(estimator='lwf')
            # print(np.shape(np.cov(X_train[0])))
            if temp:
                X_train = extract_feature_set_one(X_train, shape='etc')
                X_test = extract_feature_set_one(X_test, shape='etc')
            elif use_feature:
                X_train = extract_feature_set_one(X_train)
                X_test = extract_feature_set_one(X_test)
            cov_train = []
            cov_test = []
            # for co in range(np.shape(X_train)[0]):
            #     cov_train.append(np.cov(X_train[co]))
            # for cot in range(np.shape(X_test)[0]):
            #     cov_test.append(np.cov(X_test[cot]))
            print(np.shape(X_train))
            cov_train = cov.fit_transform(X_train, Y_train)
            cov_test = cov.transform(X_test)
            print(np.shape(cov_train))
            clf.fit(cov_train, Y_train)
            predict = clf.predict(cov_test)
            # print(predict)
            # print(Y_test)
            if plot_clf:
                tot_cov = np.append(cov_train, cov_test, axis=0)
                print("Convert")
                lapl = Embedding(metric='riemann', n_components=2)
                embd = lapl.fit_transform(tot_cov)
                fig, ax = plt.subplots(figsize=(7, 8), facecolor='white')
                print("Graph")
                for label in [0, 1]:
                    idx = (np.append(Y_train, Y_test) == label)
                    ax.scatter(embd[idx, 0], embd[idx, 1], s=36, label=label)
                idx = np.append([False]*72, [True]*8)
                ax.scatter(embd[idx, 0], embd[idx, 1], c='white', marker='.', s=16)
                ax.set_xlabel(r'$\varphi_1$', fontsize=16)
                ax.set_ylabel(r'$\varphi_2$', fontsize=16)
                ax.set_title('Spectral embedding of ERP recordings', fontsize=16)
                ax.legend()
        # elif typec == "eFBR":
        #     train_data = []
        #     test_data = []
        #     for f in filterbank:
        #         if use_feature:
        #             if temp:
        #                 train_data.append(
        #                     extract_feature_set_one(epoch.copy().filter(f[0], f[1]).get_data()[train_index], shape='etc'))
        #                 test_data.append(
        #                     extract_feature_set_one(epoch.copy().filter(f[0], f[1]).get_data()[test_index], shape='etc'))
        #             else:
        #                 train_data.append(extract_feature_set_one(epoch.copy().filter(f[0], f[1]).get_data()[train_index]))
        #                 test_data.append(extract_feature_set_one(epoch.copy().filter(f[0], f[1]).get_data()[test_index]))
        #         else:
        #             train_data.append(epoch.copy().filter(f[0], f[1]).get_data()[train_index])
        #             test_data.append(epoch.copy().filter(f[0], f[1]).get_data()[test_index])
        # 
        #     prob = [[0]*cn]*len(Y_test)
        #     for f in range(len(filterbank)):
        #         cov_clf = Covariances(estimator='lwf')
        #         t_clf = TSclassifier(clf=clf)
        #         cov_train = cov_clf.fit_transform(train_data[f], Y_train)
        #         t_clf.fit(cov_train, Y_train)
        #         cov_test = cov_clf.transform(test_data[f])
        #         prob = np.add(prob, t_clf.predict_proba(cov_test))
        #     predict = np.argmax(prob, axis=1)
        # 
        # elif typec == "FBR":
        #     train_data = []
        #     test_data = []
        #     for f in filterbank:
        #         if use_feature:
        #             train_data.append(extract_feature_set_one(epoch.copy().filter(f[0], f[1]).get_data()[train_index]))
        #             test_data.append(extract_feature_set_one(epoch.copy().filter(f[0], f[1]).get_data()[test_index]))
        #         else:
        #             train_data.append(epoch.copy().filter(f[0], f[1]).get_data()[train_index])
        #             test_data.append(epoch.copy().filter(f[0], f[1]).get_data()[test_index])
        #     csp = []
        #     train_feature = []
        #     test_feature = []
        #     for f in range(len(filterbank)):
        #         csp_clf = TangentSpace()
        #         cov_clf = Covariances(estimator='lwf')
        #         cov_train = cov_clf.fit_transform(train_data[f], Y_train)
        #         cov_test = cov_clf.transform(test_data[f])
        #         feature = np.asarray(csp_clf.fit_transform(cov_train, Y_train))
        #         tfeature = np.asarray(csp_clf.transform(cov_test))
        #         if len(train_feature) == 0:
        #             train_feature = feature.T
        #             test_feature = tfeature.T
        #         else:
        #             train_feature = np.append(train_feature, feature.T, axis=0)
        #             test_feature = np.append(test_feature, tfeature.T, axis=0)
        #     Kbest = SelectKBest(mutual_info_classif, k=p)
        #     train_csp = Kbest.fit_transform(train_feature.T, Y_train)
        #     print("Selected Features")
        #     a = Kbest.get_support()
        #     print(np.floor(np.divide(np.where(a == True), c)))
        #     clf.fit(train_csp, Y_train)
        #     test_csp = Kbest.transform(test_feature.T)
        #     predict = clf.predict(test_csp)
        # elif typec == "vTSc":
        #     clf = TSclassifier()
        #     cov = XdawnCovariances()
        #     cov_train = cov.fit_transform(X_train, Y_train)
        #     predict = []
        #     prob = []
        #     clf.fit(cov_train, Y_train)
        #     for s in range(np.shape(time_slice)[0]):
        #         cov_test = np.asarray(cov.transform(X_test[:, :, time_slice[s][0]:time_slice[s][1]]))
        #         if len(predict) == 0:
        #             predict = clf.predict(cov_test)
        #             prob = clf.predict_proba(cov_test)
        #         else:
        #             predict = np.add(predict, clf.predict(cov_test))
        #             prob = np.add(prob, clf.predict_proba(cov_test))
        #
        #     prob = np.multiply(prob, 1 / len(time_slice))
        #     print(prob)
        #     for p in range(len(predict)):
        #         if prob[p][0] >= 0.5:
        #             predict[p] = 0
        #         else:
        #             predict[p] = 1
        #     print(predict)

        # elif typec == "vCSP":
        #     print(np.shape(X))
        #     train_csp = np.asarray([])
        #     csp = mne.decoding.CSP(n_components=c, reg=None, log=True, norm_trace=False)
        #     train_feature = csp.fit_transform(X_train, Y_train)
        #     clf.fit(train_feature, Y_train)
        #     predict = []
        #     prob = []
        #     for s in range(np.shape(time_slice)[0]):
        #         test_feature = np.asarray(csp.transform(X_test[:, :, time_slice[s][0]:time_slice[s][1]]))
        #         if len(predict) == 0:
        #             predict = clf.predict(test_feature)
        #             prob = clf.predict_proba(test_feature)
        #         else:
        #             predict = np.add(predict, clf.predict(test_feature))
        #             prob = np.add(prob, clf.predict_proba(test_feature))
        #     # print(prob)
        #     prob = np.multiply(prob, 1/len(time_slice))
        #     print(prob)
        #     for p in range(len(predict)):
        #         print(np.where(prob[p] == (max(prob[p]))))
        #         predict[p] = np.where(prob[p] == (max(prob[p])))[0]
        #     print(predict)
        # elif typec == "evCSP":
        #     csp = []
        #     clf = []
        #     print(np.shape(X))
        #     train_csp = np.asarray([])
        #     for t in time_slice:
        #         csp_clf = mne.decoding.CSP(n_components=c, reg=None, log=True, norm_trace=False)
        #         c_clf = SVC(probability=True)
        #         train_feature = np.asarray(csp_clf.fit_transform(X_train[:, :, t[0]:t[1]], Y_train))
        #         c_clf.fit(train_feature, Y_train)
        #         clf.append(c_clf)
        #         csp.append(csp_clf)
        #     predict = []
        #     prob = []
        #     for s in range(np.shape(time_slice)[0]):
        #         test_feature = np.asarray(csp[s].transform(X_test[:, :, time_slice[s][0]:time_slice[s][1]]))
        #         if len(predict) == 0:
        #             predict = clf[s].predict(test_feature)
        #             prob = clf[s].predict_proba(test_feature)
        #         else:
        #             predict = np.add(predict, clf[s].predict(test_feature))
        #             prob = np.add(prob, clf[s].predict_proba(test_feature))
        #     prob = np.multiply(prob, 1 / len(time_slice))
        #     # for p in range(len(predict)):
        #     #     if predict[p] < len(clf)/2:
        #     #         predict[p] = 0
        #     #     else:
        #     #         predict[p] = 1
        #     for p in range(len(predict)):
        #         print(np.where(prob[p] == (max(prob[p]))))
        #         predict[p] = np.where(prob[p] == (max(prob[p])))[0]
        #     print(predict)
        else:
            print("Wrong Method")
            return []
        cnt = 0.0
        wrong = []
        cntc = [0.0]*cn
        clcnt = [0]*cn
        for i in range(len(predict)):
            clcnt[Y_test[i]] += 1
            if predict[i] == Y_test[i]:
                cnt += 1
                cntc[Y_test[i]] += 1
            else:
                wrong.append(i)
        test_accuracy.append(cnt / len(Y_test))
        for i in range(len(cntc)):
            cntc[i] = cntc[i]/clcnt[i]
        class_accuracy.append(cntc)
        print("Test Results")
        print(predict)
        print(Y_test)
        print(cntc)
        if not total:
            print(test_index)
            print(test_index[wrong])
        if confusion:
            conf_list.append(confusion_matrix(Y_test, predict))
        kappa.append(cohen_kappa_score(predict, Y_test))
    if plot_clf:
        plt.show()

    return test_accuracy, class_accuracy, conf_list, kappa


def analysis(epochlist, state, clf="linear", typec="mfcc", comparison="filterbank", filterbank=[], filterband=[],
             hilbert=False, filter=(20, 60), time_slice=[(0, 500), (500, 1000), (1000, 1500)], csp=24, pca=30, step=2,
             cv=10, plot_clf=False, shuffle=False, total=False, confusion=False, use_feature=False, temp=False, seeg=False, seeg_only=False):
    epoch = mne.concatenate_epochs(epochlist)
    epoch.load_data()
    if seeg:
        seeg = mne.concatenate_epochs(seeg)
        seeg.load_data()
    tAccuracy = []

    if comparison == "filterbank":
        for f in range(np.shape(filterbank)[0]):
            tepoch = epoch.copy()
            if hilbert:
                tepoch.apply_hilbert()
            accuracy, cAccuracy, conf, kappa = run_cv(tepoch, typec, csp, pca, cv, hilbert, time_slice, state, classifier=clf, shuffle=shuffle, total=total, confusion=confusion)
            print("Test Accuracy : %f at %s" % (np.mean(accuracy), filterband[f]))
            print(accuracy)
            tAccuracy.append(np.mean(accuracy))
        plt.plot(filterband, tAccuracy)
        plt.show()

    elif comparison == "CSP":
        for c in range(2, csp, step):
            tepoch = epoch.copy()
            if hilbert:
                tepoch.apply_hilbert()
            accuracy, cAccuracy, conf, kappa = run_cv(tepoch, typec, c, pca, cv, hilbert, time_slice, state, classifier=clf, shuffle=shuffle, total=total, confusion=confusion)
            print("Test Accuracy : %f at C=%f" % (np.mean(accuracy), c))
            print(accuracy)
            tAccuracy.append(np.mean(accuracy))
        plt.plot(range(2, csp, step), tAccuracy)
        plt.show()
    elif comparison == "PCA":
        for p in range(1, pca, step):
            tepoch = epoch.copy()
            if hilbert:
                tepoch.apply_hilbert()
            accuracy, cAccuracy, conf, kappa = run_cv(tepoch, typec, csp, p, cv, hilbert, time_slice, state, classifier=clf, shuffle=shuffle, total=total, confusion=confusion)
            print("Test Accuracy : %f at P=%f" % (np.mean(accuracy), p))
            print(accuracy)
            tAccuracy.append(np.mean(accuracy))
        plt.plot(range(1, pca, step), tAccuracy)
        plt.show()


    else:
        tepoch = epoch.copy()
        if hilbert:
            tepoch.apply_hilbert()
        accuracy, cAccuracy, conf, kappa = run_cv(tepoch, typec, csp, pca, cv, hilbert, time_slice, state, classifier=clf,
                                                  filterbank=filterbank, plot_clf=plot_clf, shuffle=shuffle, total=total, 
                                                  confusion=confusion, use_feature=use_feature, temp=temp, seeg=seeg, seeg_only=seeg_only)
        print("Test Accuracy : %f" % np.mean(accuracy))
        print(accuracy)

    return accuracy, cAccuracy, conf, kappa

