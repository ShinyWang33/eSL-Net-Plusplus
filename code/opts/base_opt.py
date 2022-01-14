import argparse

class BaseOpt():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--name', type=str, default='eSL-Net')

        # dataset
        self.parser.add_argument('--dataset_mode', type=str, default='goproval',
                                 help='goproval, hqf')
        self.parser.add_argument('--num_frames_for_blur', type=int, default=5,
                                  help='number of reconstructed frames for a blurry image')

        # dataloader
        self.parser.add_argument('--n_threads', default=4, type=int, help='# threads for data')

        # model
        self.parser.add_argument('--model', type=str, default='eSL-Net_dn',
                                 help='model to run')
        self.parser.add_argument('--load_G', type=str, default='./pre_trained/model_dn_pretrained.pt', help='path of the pretrained model')


        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        self.opt.channel = 1

        self.opt.is_train = self.is_train

        return self.opt
