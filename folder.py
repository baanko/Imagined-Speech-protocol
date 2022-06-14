from os import listdir
from os.path import isfile, join
from funct import load_epoch
from read_open import read_file, read_band_features
import mne
# mypath = 'Ear/0525_ST01'
# file = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.split('.')[1] == 'vhdr']
# print([f for f in listdir(mypath) if isfile(join(mypath, f))])
mne.set_log_level(verbose=False)

def read_folder(mypath, t=(-0.1, 1.9), type="ieeg", b=(-0.1, 0), f=(0.1, 100), ear_only=False, scalp_only=False, band=None):
    brainvision_file = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f)) and f.split('.')[1] == 'vhdr']
    
    
    if type=="both":
        epoch1 = mne.concatenate_epochs([load_epoch(brainvision_file[0], typec='ieeg', time=t, baseline=b, filter=f),
                                         load_epoch(brainvision_file[0], typec='seeg', time=t, baseline=b, filter=f)])
        epoch2 = mne.concatenate_epochs([load_epoch(brainvision_file[1], typec='ieeg', time=t, baseline=b, filter=f),
                                         load_epoch(brainvision_file[1], typec='seeg', time=t, baseline=b, filter=f)])
    else:
        epoch1 = load_epoch(brainvision_file[0], typec=type, time=t, baseline=b, filter=f)
        epoch2 = load_epoch(brainvision_file[1], typec=type, time=t, baseline=b, filter=f)
    ear = join("Ear", mypath)


    ear_eeg_file = [join(ear, f) for f in listdir(ear) if isfile(join(ear, f)) if f.split('_')[2] == 'eeg']
    ear_time_file = [join(ear, f) for f in listdir(ear) if isfile(join(ear, f)) if f.split('_')[2] == 'aligned']
    print(brainvision_file)
    print(ear_eeg_file)
    print(ear_time_file)
    if band:
        epoch3 = read_band_features(ear_eeg_file[0], ear_time_file[0], filterband=band)
        epoch4 = read_band_features(ear_eeg_file[1], ear_time_file[1], filterband=band)
    else:
        epoch3 = read_file(ear_eeg_file[0], ear_time_file[0], filter=f)
        epoch4 = read_file(ear_eeg_file[1], ear_time_file[1], filter=f)
    if scalp_only:
        return [epoch1, epoch2]
    if ear_only:
        return [epoch3, epoch4]
    return [epoch1, epoch2, epoch3, epoch4]

# read_folder("0525_ST01")