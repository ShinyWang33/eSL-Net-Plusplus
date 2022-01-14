##gopro test dataset.

import os
from data.datasets import DatasetBase
from utils import cv_utils
import numpy as np
from collections import OrderedDict
import sys
from utils import event2frame
import operator

class Dataset(DatasetBase):
    def __init__(self, opt, is_for_train=False):
        super(Dataset, self).__init__(opt, is_for_train=is_for_train)
        self._name = 'goproval'
        print('Loading dataset...')
        self.opt = opt
        self.dataset_acc_num = [0]
        self.dataset_acc_num_e = [0]
        self._read_dataset_paths()

    def _read_dataset_paths(self):
        self.root_blur = os.path.expanduser(self.opt.input_blur_path)
        self.root = os.path.expanduser(self.opt.input_event_path)
        self.load_blur()
        self.load_event()
        if not operator.eq(self.dataset_acc_num, self.dataset_acc_num_e):
            print('The number of blurry images is not equal to the number of eventstream')
            sys.exit(1)

    def load_blur(self):
        self.blur_paths = OrderedDict()
        self.dataset_name = []
        for subroot in sorted(os.listdir(self.root_blur)):
            imgroot = os.path.join(self.root_blur, subroot)
            imglist = os.listdir(imgroot)
            imglist.sort(key=lambda x: float(x[:-4]))
            self.blur_paths[subroot] = imglist
            self.dataset_acc_num.append(len(imglist) + self.dataset_acc_num[-1])
            self.dataset_name.append(subroot)

    def load_event(self):
        self.event_paths = OrderedDict()
        for subroot in sorted(os.listdir(self.root)):
            eventroot = os.path.join(self.root, subroot)
            eventlist = os.listdir(eventroot)
            eventlist.sort(key=lambda x: float(x[5:-4]))
            self.event_paths[subroot] = eventlist
            self.dataset_acc_num_e.append(len(eventlist) + self.dataset_acc_num_e[-1])

    def __len__(self):
        return self.dataset_acc_num[-1]
        #return len(self.dataset_name)

    def __getitem__(self, index):
        ###################### TEST ########################
        dataset_idx = np.searchsorted(self.dataset_acc_num, index+1)-1
        img_idx = index - self.dataset_acc_num[dataset_idx]

        dataname = self.dataset_name[dataset_idx]
        # blurred images
        blur_paths = self.blur_paths.get(dataname)
        blur_path = os.path.join(self.root_blur, dataname, blur_paths[img_idx])
        blur = cv_utils.read_cv2_img(blur_path, input_nc=1)

        # event images
        event_paths = self.event_paths.get(dataname)
        event_path = os.path.join(self.root, dataname, event_paths[img_idx])
        section_event_timestamp = cv_utils.read_mat_gopro(event_path, 'section_event_timestamp')
        section_event_polarity = cv_utils.read_mat_gopro(event_path, 'section_event_polarity')
        section_event_x = cv_utils.read_mat_gopro(event_path, 'section_event_y')-1  # x,y exchange
        section_event_y = cv_utils.read_mat_gopro(event_path, 'section_event_x')-1  # x-->[1,m]  change  x-->[0,m-1]
        start_timestamp = cv_utils.read_mat_gopro(event_path, 'start_timestamp')
        end_timestamp = cv_utils.read_mat_gopro(event_path, 'end_timestamp')
        section_event = np.concatenate(
            (section_event_timestamp, section_event_polarity, section_event_x, section_event_y), axis=1)

        if self.opt.num_frames_for_blur < 2:
            num_frames = 2
        else:
            num_frames = self.opt.num_frames_for_blur

        event_img_lst = []
        for f in range(self.opt.num_frames_for_blur):
            events_lst = event2frame.split_events_by_time(section_event, f, num_frames, split_num=20,
                                                          start_ts=start_timestamp, end_ts=end_timestamp)
            for e_idx, events_split in enumerate(events_lst):
                event_img = event2frame.event_to_cnt_img(events_split, height=blur.shape[1],
                                                         width=blur.shape[2])
                event_img_lst.append(event_img)

        event_img_bins = np.concatenate(event_img_lst)



        sample = {'event_bins': event_img_bins,
                  'blurred': blur,
                  'dataname': dataname,
                  'img_idx': img_idx}

        return sample