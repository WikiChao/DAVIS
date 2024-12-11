import os
import random
import numpy as np
import csv
from .base import BaseDataset
import clip


class AVEMixDataset(BaseDataset):
    def __init__(self, list_sample, opt, **kwargs):
        super(AVEMixDataset, self).__init__(
            list_sample, opt, **kwargs)
        self.fps = opt.frameRate
        self.num_mix = opt.num_mix
        self.audLen = opt.audLen
        self.model_type = opt.arch_frame

    def __getitem__(self, index):
        N = self.num_mix
        frames = [None for n in range(N)]
        audios = [None for n in range(N)]
        infos = [[] for n in range(N)]
        path_frames = [[] for n in range(N)]
        path_frames_ids = [[] for n in range(N)]
        path_frames_det = ['' for n in range(N)]
        path_audios = ['' for n in range(N)]
        center_frames = [0 for n in range(N)]
        class_list = []

        if self.split == 'train':
            # the first video
            infos[0] = self.list_sample[index]
            cls = infos[0][-1]
            class_list.append(cls)
            for n in range(1, N):
                indexN = random.randint(0, len(self.list_sample)-1)
                sample = self.list_sample[indexN]
                while sample[-1] in class_list:
                    indexN = random.randint(0, len(self.list_sample) - 1)
                    sample = self.list_sample[indexN]
                infos[n] = sample
                class_list.append(sample[-1])
        elif self.split == 'val':
            infos[0] = self.list_sample[index]
            cls = infos[0][-1]
            class_list.append(cls)
            if not self.split == 'train':
                random.seed(index)

            for n in range(1, N):
                indexN = random.randint(0, len(self.list_sample) - 1)
                sample = self.list_sample[indexN]
                while sample[-1] in class_list:
                    indexN = random.randint(0, len(self.list_sample) - 1)
                    sample = self.list_sample[indexN]
                infos[n] = sample
                class_list.append(sample[-1])
        else:
            csv_lis_path = "../data/AVE/test.csv"
            csv_lis = []
            for row in csv.reader(open(csv_lis_path, 'r'), delimiter=','):
                if len(row) < 2:
                    continue
                csv_lis.append(row)
            random.seed(index) # fixed
            samples = self.list_sample[index]
            for n in range(N):
                sample = samples[n].replace(" ", "")
                sample = sample.split('/')[2]

                for i in range(len(csv_lis)):
                    data = csv_lis[i]
                    if sample in data: 
                        infos[n] = data
                        break

        FPS = [None for n in range(N)]
        classes = [None for n in range(N)]
        texts = [None for n in range(N)]
        for n, infoN in enumerate(infos):
            path_audioN, path_frameN, count_framesN, class_name = infoN
            classes[n] = class_name
            texts[n] = clip.tokenize('an image of ' + class_name)

            fps = int(count_framesN) // 10
            FPS[n] = fps
            self.stride_frames = fps

            center_frameN = 6 
            center_frames[n] = center_frameN

            # absolute frame/audio paths
            for i in range(self.num_frames):
                idx_offset = (i - self.num_frames // 2) * self.stride_frames
                path_frames[n].append(
                    os.path.join("/YOUR_ROOT/AVE_Dataset/frames",
                        path_frameN,
                        '{:06d}.jpg'.format(center_frameN + idx_offset)))
                path_frames_ids[n].append(center_frameN + idx_offset)


            path_audios[n] = os.path.join("/YOUR_ROOT/AVE_Dataset/audio", path_audioN)

        # load frames and audios, STFT
        try:
            for n, infoN in enumerate(infos):
                if self.model_type != 'clip':
                    frames[n] = self._load_frames(path_frames[n])
                else:
                    frames[n] = self._load_frames_clip(path_frames[n])
                    
                # jitter audio
                center_timeN = 5
                audios[n] = self._load_audio(path_audios[n], center_timeN)
            mag_mix, mags, phase_mix, audio_mix = self._mix_n_and_stft(audios)

        except Exception as e:
            print('Failed loading frame/audio: {}'.format(e))
            # create dummy data
            mag_mix, mags, frames, audios, phase_mix = \
                self.dummy_mix_data(N)
            audio_mix = audios[0]

        ret_dict = {'mag_mix': mag_mix, 'frames': frames, 'mags': mags, 
        'audio_mix': audio_mix}
        ret_dict['audios'] = audios
        ret_dict['class'] = classes
        ret_dict['text'] = texts

        if self.split != 'train':
            # ret_dict['audios'] = audios
            ret_dict['phase_mix'] = phase_mix
            ret_dict['infos'] = infos

        return ret_dict
