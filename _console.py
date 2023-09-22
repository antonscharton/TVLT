import torchaudio
import torch
import numpy as np

from demos import MOSEI_sentiment_model
from decord import VideoReader, AudioReader

model = MOSEI_sentiment_model()

device  = 'cuda:0'
model.to(device)
model.eval()

from model.data.datasets.rawvideo_utils import load_audio, load_video, preprocess_audio, time_to_indices, crop_image_only_outside, _transform, center_crop

def sampler_linear(start, end, num_frames):
    return np.linspace(start, end, num_frames, endpoint=False).astype("int")

timestamp = (0.1, 4.7)
num_frames = 8
sr = 44100
sampler = sampler_linear

video_path = 'demo_samples/MOSEI/K0m1tO3Ybyc_2.mp4' # You can specify your own video file here
audio_path = 'demo_samples/MOSEI/K0m1tO3Ybyc_2.wav' # You can specify your own audio file here
#video = load_video(video_path, num_frames=8)
audio = load_audio(audio_path, sr=sr, timestamp=timestamp)

audio.shape




# load video
video = VideoReader(video_path)
total_frames = len(video)
framerate = video.get_avg_fps()
video_len = total_frames/framerate


# get indices
s = np.linspace(0, video_len, total_frames, endpoint=False)

start = np.argmin((s - timestamp[0])**2)
end = np.argmin((s - timestamp[1])**2)

downsamlp_indices = sampler(start, end, num_frames)

downsamlp_indices

video = video.get_batch(downsamlp_indices).asnumpy()

# up to this point it is an array of numpy 3d arrays [h,w,c] with value range [0, 255] (num_frames,h,w,c)
# so also cv images can be loaded here and num_frames chosen

# images should be resized to (224, 224) by cv2


video = crop_image_only_outside(video)
video = torch.from_numpy(video).permute(0, 3, 1, 2)
video = (video/255.0-0.5)/0.5
video = video.unsqueeze(0).float()
