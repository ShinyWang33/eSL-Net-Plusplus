import os
import torch

class ModelsFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(model_name, *args, **kwargs):
        model = None
        ################ Ours #################
        if model_name == 'eSL-Net_dn':
            from models.esl_dn_model import Model
            model = Model(*args, **kwargs)
        elif model_name == 'eSL-Net_sr':
            from models.esl_sr_model import Model
            model = Model(*args, **kwargs)
        elif model_name == 'eSL-Net++_sr':
            from models.esl_plusplus_sr_model import Model
            model = Model(*args, **kwargs)
        elif model_name == 'eSL-Net++_dn':
            from models.esl_plusplus_dn_model import Model
            model = Model(*args, **kwargs)
        else:
            raise ValueError("Model %s not recognized." % model_name)
        print("Model %s was created" % model.name)
        #
        return model


class BaseModel(object):

    def __init__(self, opt):
        self._name = 'BaseModel'

        self._opt = opt
        self._is_train = opt.is_train
        if self._opt.cuda:
            self._Tensor = torch.cuda.FloatTensor
        else:
            self._Tensor = torch.FloatTensor


        self._save_dir = os.path.join('./checkpoints', opt.name)


    @property
    def name(self):
        return self._name

    @property
    def is_train(self):
        return self._is_train

    def set_input(self, input):
        assert False, "set_input not implemented"

    def set_eval(self):
        assert False, "set_eval not implemented"

    def forward(self, keep_data_for_visuals=False, isTrain=True):
        assert False, "forward not implemented"

    def _load_network(self, network, load_path):
        assert os.path.exists(
            load_path), 'Weights file not found. Have you trained a model!? We are not providing one' % load_path

        if self._opt.cuda:
            pretrained = torch.load(load_path)
        else:
            pretrained = torch.load(load_path, map_location=torch.device('cpu'))

        model_dict = network.state_dict()
        '''try:
            stereo_dict = {k: v for k, v in pretrained.items() if str(k) in model_dict}
        except:
            stereo_dict = {k.replace('module.',''): v for k, v in pretrained.items() if str(k) in model_dict}'''
        stereo_dict = {k.replace('module.', ''): v for k, v in pretrained.items() if str(k) in model_dict}
        if len(stereo_dict) == 0:
            stereo_dict = {k.replace('module.', ''): v for k, v in pretrained.items()}
        model_dict.update(stereo_dict)
        network.load_state_dict(model_dict)

        # network.load_state_dict(torch.load(load_path))
        print ('loaded net: %s' % load_path)
        print ('loaded paremeters: ', len(stereo_dict), 'in total ', len(model_dict))
