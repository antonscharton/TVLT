import torch
import torchaudio
from pathlib import Path
import os
import re
import cv2
import numpy as np

from demos import MOSEI_sentiment_model



# data flow:
#   log     ->      SEQUENCER   ->  zeros, ones      ->     AUGMENTATIONS        -> zeros, ones
#                                                           repeat every epoch
#   - > sampler frames from images (linear, random, log: gewichtung auf hinteren teil der sequenz)


class CUIX(torch.utils.data.Dataset):

    # dataset class for structure:
    #
    # - root_dataset_path/faces/
    #       - 1RTZ8_1693397155625_141
    #           - 000000.jpg
    #           - 000001.jpg
    #           - 000002.jpg
    #           - ...
    #       - 1RTZ8_1693397309803_7
    #       - 1RTZ8_1693397357086_286
    #       - ...
    # - root_dataset_path/logs/
    #       - 1RTZ8_1693397155625_141.txt
    #       - 1RTZ8_1693397309803_7.txt
    #       - 1RTZ8_1693397357086_286.txt
    #       - ...
    # - root_dataset_path/auto_label.txt
    # - root_dataset_path/fps_info.txt

    def __init__(self,
                 path_root,
                 include_audio = True,
                 include_video = True,
                 audio_samplingrate = 44100,
                 video_numframes = 8,
                 sample_mode = "linear",
                 split_mode = "action",
                 min_seq_duration = 2000,
                 seq_duration = None,
                 seq_sliding_delta = None
                 ):

        self.path_root = path_root
        self.audio_samplingrate = audio_samplingrate
        self.video_numframes = video_numframes
        self.include_audio = include_audio
        self.include_video = include_video

        # load fps information for each sequence
        with open(os.path.join(path_root, "fps_info.txt")) as f:
            self.fps_info = {line.split(" : ")[0].replace(".mp4", ""): float(line.split(" : ")[1].replace("\n", "")) for line in f.readlines()}

        # load auto label
        self.auto_labels = {}
        with open(os.path.join(path_root, "auto_labels.txt")) as f:
            auto_label_ = f.read()
        for line in auto_label_.split("\n")[:-1]:
            split = line.replace("\n", "").split(" : ")
            if split[1] != "None":
                self.auto_labels[split[0]] = int(split[1])
            else:
                self.auto_labels[split[0]] = None

        # load logs
        self.logs = {}
        for path in [f.path for f in os.scandir(os.path.join(path_root, "logs"))]:
            with open(path) as f:
                log = f.read()
            log = re.sub("[\(\[].*?[\)\]]", "", log)
            log = log.replace("  ", " ")
            self.logs[os.path.basename(path).replace(".txt", "")] = log
        self.keys = list(self.logs.keys())

        # prepare sequencing
        self.sequence_settings = {}
        self.sequence_method = None
        self.sample_method = None

        if split_mode == "slidingwindow":
            assert (seq_duration is not None) and (seq_sliding_delta is not None), "You need to specify 'seq_duration' and 'seq_sliding_delta' in order to sequence in 'slidingwindow' mode"
            self.sequence_settings["seq_duration"] = seq_duration
            self.sequence_settings["seq_sliding_delta"] = seq_sliding_delta
            self.sequence_method = self._sequence_slidingwindow
        elif split_mode == "action":
            self.sequence_settings["min_seq_duration"] = min_seq_duration
            self.sequence_method = self._sequence_action

        if sample_mode == "linear":
            self.sample_method = self._sample_linear

        self.sequence()

    # here the sequences are generated, this can be repeated
    # on start of a new epoch to allow for augmentation
    def sequence(self):
        # baseline sequencing
        self.sequences = []
        for key in self.keys:

            log = self.logs[key]
            label_auto = self.auto_labels[key]
            label_user = 1 if (".cantsolve" in log or ".confused" in log) else 0

            zeros, ones = self.sequence_method(log, label_auto, self.sequence_settings)

            for zero in zeros:
                self.sequences.append({"key": key, "start": zero[0], "end": zero[1], "auto_label": 0, "user_label": label_user})

            for one in ones:
                self.sequences.append({"key": key, "start": one[0], "end": one[1], "auto_label": 1, "user_label": label_user})


    @staticmethod
    def _sequence_slidingwindow(log, autolabel, settings):

        seq_duration = settings["seq_duration"]
        seq_sliding_delta = settings["seq_sliding_delta"]


    @staticmethod
    def _sequence_action(log, autolabel, settings):

        min_time = settings["min_seq_duration"]
        split_keys = [".button: ", ".driver speech starts"]

        split = log.split("\n\n")
        task = [line.replace("\n", "") for line in split[0].split("\n")]
        manipulations = [line.replace("\n", "").replace("DEMON ACTION: ", "") for line in split[1].split("\n")] if "DEMON ACTION:" in log else []
        actions = [line.replace("\n", "") for line in split[-1].split("\n")][1:-1]
        timestamps = [int(a.split(" ")[0]) for a in actions]
        actions = [actions[i].replace(str(timestamps[i]) + "  ", "") for i in range(len(actions))]

        splits = []
        for a, action in enumerate(actions):
            for key in split_keys:
                if key in action:
                    if (key == ".driver speech starts") and (".driver activated voice command." in actions[a-1]):
                        splits.append(timestamps[a-1])
                    else:
                        splits.append(timestamps[a])


        # require min length
        starts = [0] + splits
        ends = splits + [timestamps[-1]]
        lengths = np.array(ends) - np.array(starts)
        i = (lengths < min_time).nonzero()[0]


        while len(i) != 0:
            j = i[0]

            # new ending point
            if j == 0:
                starts.pop(j+1)
                ends.pop(j)

            # new start point
            elif j == len(lengths) - 1:
                starts.pop(j)
                ends.pop(j-1)

            elif lengths[j-1] < lengths[j+1]:
                starts.pop(j)
                ends.pop(j-1)

            elif lengths[j-1] > lengths[j+1]:
                starts.pop(j+1)
                ends.pop(j)

            lengths = np.array(ends) - np.array(starts)
            i = (lengths < min_time).nonzero()[0]


        # get label for this log
        if isinstance(autolabel, int):
            i = (np.array(ends) > autolabel).nonzero()[0][0]
            zeros = [[starts[j], ends[j]] for j in range(0, i)]
            ones = [[starts[j], ends[j]] for j in range(i, len(starts))]

        elif ".cantsolve" in log or ".confused" in log:
            i = (np.array(ends) > timestamps[-1]//2-1).nonzero()[0][0]
            zeros = [[starts[j], ends[j]] for j in range(0, i)]
            ones = [[starts[j], ends[j]] for j in range(i, len(starts))]

        else:
            zeros = [[starts[j], ends[j]] for j in range(len(starts))]
            ones = []

        return zeros, ones


    @staticmethod
    def _sample_linear(start, end, num_frames):
        return np.linspace(start, end, num_frames, endpoint=False).astype("int")


    @staticmethod
    def _load_audio(path, sr, timestamp):
        audio, org_sr = torchaudio.load(path)
        if org_sr != sr:
            audio = torchaudio.functional.resample(audio, orig_freq=org_sr, new_freq=sr)
        audio = audio.mean(0).numpy()

        start, end = int(sr * timestamp[0]), int(sr * timestamp[1])
        audio = audio[start: end]
        audio = audio - audio.mean()
        audio = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        audio = torch.from_numpy(pre_spec(audio)).unsqueeze(0)
        p = 16 - audio.shape[2]%16
        if p > 0:
            audio = torch.nn.functional.pad(audio, (0, p, 0, 0), "constant", -1.0)
        audio = audio.transpose(1,2)
        audio = audio[:, :1024].unsqueeze(0).float()
        return audio


    @staticmethod
    def _load_video(path, framerate, num_frames, timestamp, sample_method):
        paths = np.array([f.path for f in os.scandir(path)])

        # get indices
        s = np.linspace(0, len(paths)/framerate, len(paths), endpoint=False)
        start = np.argmin((s - timestamp[0])**2)
        end = np.argmin((s - timestamp[1])**2)
        downsamlp_indices = sample_method(start, end, num_frames)

        video = []
        for path in paths[downsamlp_indices]:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, (224, 224))
            video.append(img)
        video = np.array(video)

        #video = crop_image_only_outside(video)
        video = torch.from_numpy(video).permute(0, 3, 1, 2)
        video = (video/255.0-0.5)/0.5
        video = video.unsqueeze(0).float()
        return video


    def __getitem__(self, i):
        s = self.sequences[i]

        sample = {}
        sample["key"] = s["key"]
        sample["user_label"] = s["user_label"]
        sample["auto_label"] = s["auto_label"]

        timestamp = (s["start"]/1000, s["end"]/1000)

        if self.include_audio:
            sample["audio"] = self._load_audio(os.path.join(self.path_root, "audio", s["key"] + ".wav"),
                                               sr=self.audio_samplingrate,
                                               timestamp=timestamp)
        else:
            sample["audio"] = torch.zeros([1, 1, 400, 128]).float()

        if self.include_video:
            sample["video"] = self._load_video(os.path.join(self.path_root, "faces", s["key"]),
                                               framerate=self.fps_info[s["key"]],
                                               num_frames=self.video_numframes,
                                               timestamp=timestamp,
                                               sample_method=self.sample_method)
        else:
            sample["video"] = torch.zeros([1, self.video_numframes, 3, 224, 224]).float()

        return sample

    def __len__(self):
        return len(self.sequences)



gpu_id = 0
dataset = CUIX(Path(r"D:\datasets\lmmtm_faces_0509"), include_audio=False)
model = MOSEI_sentiment_model()


device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()



samples = [dataset[207], dataset[208], dataset[209]]

video = torch.cat([sample["video"] for sample in samples], dim=0).to(device)
audio = torch.cat([sample["audio"] for sample in samples], dim=0).to(device)

with torch.no_grad():
    encoder_last_hidden_outputs, *_ = model(video=video, audio=audio)
    sentiment_score = model.classifier(encoder_last_hidden_outputs).squeeze().data.cpu().numpy()


print('sentiment intensity:', np.round(sentiment_score, 3))
