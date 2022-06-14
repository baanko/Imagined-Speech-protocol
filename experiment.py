import expyriment
import time
import os
import datetime
from win32com.client import Dispatch

name = "speech"
folder = "Data"
date = datetime.datetime.now().strftime("%m%d")
data_type = "speech"
num_trials = 5
window_sec = 2.0
interim_time = 3.0
num_classes = 2
ch_num = 32

subject = ""
i = 1
fname = "{}/{}_{}_{}t{}c{}s{}ch_{}".format(folder, date, data_type, num_trials, num_classes, window_sec, ch_num, subject)
filename = "{}_{}.txt".format(fname, i)
while os.path.exists(filename):
    i += 1
    filename = "{}_{}.txt".format(fname, i)
file = open(filename, "w")
print("{}: Writing to {}".format(name, filename))

# brainrecorder = Dispatch("VisionRecorder.Application")
#excel = Dispatch("Excel.Application")
#excel_file = excel.Workbooks.Open(u'C:\\Users\\Choi\\PycharmProjects\\Experiment\\trial.xlsx')
#worksheet = excel_file.ActiveSheet
exp = expyriment.design.Experiment(name="Silent/Overt Speech")
expyriment.control.initialize(exp)
stim_cross = expyriment.stimuli.FixCross()
stim_cross.preload()
block_one = expyriment.design.Block(name="Right")
trial_one = expyriment.design.Trial()
stim = expyriment.stimuli.Video("slow.mpg")
stim.preload()
trial_one.add_stimulus(stim)
trial_two = expyriment.design.Trial()
stim = expyriment.stimuli.Video("slow.mpg")
stim.preload()
trial_two.add_stimulus(stim)
trial_three = expyriment.design.Trial()
stim = expyriment.stimuli.Video("slow.mpg")
stim.preload()
trial_three.add_stimulus(stim)
block_one.add_trial(trial_one)
block_one.add_trial(trial_two)
block_one.add_trial(trial_three)

block_two = expyriment.design.Block(name="Left")
trial_one = expyriment.design.Trial()
stim = expyriment.stimuli.Video("slow.mpg")
stim.preload()
trial_one.add_stimulus(stim)
trial_two = expyriment.design.Trial()
stim = expyriment.stimuli.Video("slow.mpg")
stim.preload()
trial_two.add_stimulus(stim)
trial_three = expyriment.design.Trial()
stim = expyriment.stimuli.Video("slow.mpg")
stim.preload()
trial_three.add_stimulus(stim)
block_two.add_trial(trial_one)
block_two.add_trial(trial_two)
block_two.add_trial(trial_three)

exp.add_block(block_one, num_trials)
exp.add_block(block_two, num_trials)
exp.shuffle_blocks()
stim_instruction = expyriment.stimuli.TextLine(text="Focus on the cross in the middle of the screen")
stim_instruction1 = expyriment.stimuli.TextLine(text="Listen to the word said")
stim_instruction2 = expyriment.stimuli.TextLine(text="When you see a circle, say the word out loud")
stim_instruction3 = expyriment.stimuli.TextLine(text="When you see a square, think the word inside your head")

video = expyriment.stimuli.Video("slow.mp4")
video.preload()
expyriment.control.start(auto_create_subject_id=True, skip_ready_screen=True)

# stim_instruction.present()
# exp.clock.wait(4000)
# stim_instruction1.present()
# exp.clock.wait(4000)
# stim_instruction2.present()
# exp.clock.wait(4000)
# stim_instruction3.present()
# exp.clock.wait(4000)
# stim_cross.present()

# brainrecorder.Acquisition.StartRecording(fname)
currenttime = expyriment.misc.Clock.monotonic_time()

btrigger = 0
trigger = 0
frame = 42
for block in exp.blocks:
    ttrigger = 1
    if block.name == "Right":
        btrigger = 1
    else:
        btrigger = 2
    for trial in block.trials:
        trigger = btrigger*10 + ttrigger
        ttrigger += 1
        print(trigger)
        file.write(str(time.time()) + " ," + str(trigger) + "\n")
        # brainrecorder.Acquisition.SetMarker(str(trigger), ['Stimulus'])
#        worksheet.Cells(1,1).Value = trigger
        video.present()
        video.play()
        print(video.length)
        print(video.is_playing)
        while video.is_playing and video.frame < frame:
            video.update()
        video.stop()
        while expyriment.misc.Clock.monotonic_time() < currenttime + window_sec:
            pass
        currenttime = expyriment.misc.Clock.monotonic_time()
        file.write(str(time.time()) + " ," + str(trigger) + "\n")
#        brainrecorder.Acquisition.SetMarker(str(trigger), ['Stimulus'])
        stim_cross.present()
#        exp.clock.wait(2000)
        while expyriment.misc.Clock.monotonic_time() < currenttime + interim_time:
            pass
        currenttime = expyriment.misc.Clock.monotonic_time()

expyriment.control.end()
file.close()
# brainrecorder.Acquisition.StopRecording()
#excel_file.Save()
#excel_file.Close()
#excel.Quit()