import torchaudio
import torch
import numpy as np

from demos import MOSEI_sentiment_model
from decord import VideoReader, AudioReader

model = MOSEI_sentiment_model()

device  = 'cpu'
model.to(device)
model.eval()

from model.data.datasets.rawvideo_utils import load_audio, load_video, preprocess_audio, time_to_indices, crop_image_only_outside, _transform, center_crop

timestamp = (0.1, 4.7)
num_frames = 8


video_path = 'demo_samples/MOSEI/K0m1tO3Ybyc_2.mp4' # You can specify your own video file here
audio_path = 'demo_samples/MOSEI/K0m1tO3Ybyc_2.wav' # You can specify your own audio file here
video = load_video(video_path, num_frames=8)
audio = load_audio(audio_path, sr=44100)


# load audio, timestamp is tuple of floats representing seconds, example (1.2, 3.52)
def load_audio(path, sr=44100, timestamp=None):
    audio, org_sr = torchaudio.load(path)
    if org_sr != sr:
        audio = torchaudio.functional.resample(audio, orig_freq=org_sr, new_freq=sr)
    audio = audio.mean(0).numpy()      
    if timestamp is not None:
        start, end = int(sr * timestamp[0]), int(sr * timestamp[1])
        audio = audio[start: end]
    audio = preprocess_audio(audio, sr=sr)
    audio = audio[:, :1024]
    return audio.unsqueeze(0).float()


video = VideoReader(video_path)
framerate = video.get_avg_fps()
video_len = len(video)/framerate



def sampler(start, end, num_frames = 8):
    return np.linspace(start, end, num_frames, endpoint=False).astype("int")



if timestamp is not None:
    s = np.linspace(0, video_len, len(video), endpoint=False)
    start = np.argmin((s - timestamp[0])**2)
    end = np.argmin((s - timestamp[1])**2)
    downsamlp_indices = sampler(start, end, num_frames)

else:                       
    downsamlp_indices = sampler(0, len(video), num_frames)


downsamlp_indices
video = video.get_batch(downsamlp_indices).asnumpy()

# up to this point it is an array of numpy 3d arrays [h,w,c] with value range [0, 255]
# so also cv images can be loaded here and num_frames chosen
video.shape

np.max(video)

video = crop_image_only_outside(video)
min_shape = min(video.shape[1:3])
video = center_crop(video, min_shape, min_shape)
video = torch.from_numpy(video).permute(0, 3, 1, 2)
video = _transform(224)(video)
video = (video/255.0-0.5)/0.5
video = video.unsqueeze(0).float()