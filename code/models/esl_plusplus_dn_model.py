## input: blurry image and eventstream
## output: sharp image

from models.models import BaseModel
from networks.networks import NetworksFactory
from utils import cv_utils
import os

class Model(BaseModel):
    def __init__(self, opt):
        super(Model, self).__init__(opt)
        self._name = 'eSL-Net-Plusplus_dn'
        self.channel = self._opt.channel
        self._init_create_networks()
        self._init_inputs()

    def _init_create_networks(self):

        self._G = NetworksFactory.get_by_name('eSL-Net-Plusplus_dn')
        if len(self._opt.load_G) > 0:
            self._load_network(self._G, self._opt.load_G)
        else:
            raise ValueError("Weights file not found.")
        if self._opt.cuda:
            self._G.cuda()

    def _init_inputs(self):
        self._input_blurred = self._Tensor()
        self._input_event = self._Tensor()

    def set_input(self, input):
        self._input_blurred.resize_(input['blurred'].size()).copy_(input['blurred'])
        self._input_event.resize_(input['event_bins'].size()).copy_(input['event_bins'])

        if self._opt.cuda:
            self._input_blurred = self._input_blurred.cuda()
            self._input_event = self._input_event.cuda()

        self.dataname = input['dataname'][0]
        self.img_idx = input['img_idx'][0].item()
        if self._opt.num_frames_for_blur < 2:
            self.num_frames = 2
        else:
            self.num_frames = self._opt.num_frames_for_blur

    def set_eval(self):
        self._G.eval()
        self._is_train = False

    def forward(self, keep_data_for_visuals=False, isTrain=True):
        for f in range(self._opt.num_frames_for_blur):
            self.cur_event = self._input_event[:, (f*80):(f+1)*80, :, :]
            self.est_sharp = self._G(self._input_blurred, self.cur_event, f/(self.num_frames-1))

            ############# save imgs #############
            path = os.path.join(self._opt.output_dir, self.dataname)
            if not os.path.exists(path):
                os.makedirs(path)
            path_i = os.path.join(path, str(self._opt.img_start_idx + self.img_idx * self._opt.img_inter_idx).zfill(4) + '_' + str(f).zfill(2) + '.png')
            cv_utils.debug_save_tensor(self.est_sharp, path_i, rela=False, rgb=self.channel == 3)
