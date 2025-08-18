from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

class GetPretrainedFeatures:
    def __init__(self,trained_wts_file):
        self.trained_wts_file = trained_wts_file
        self.checkpoint = torch.load(self.trained_wts_file, map_location="cpu")
        self.model_info = self.checkpoint['model']

    def print_loaded_model_info(self):
        print('Loaded pretrained model from: ' + self.trained_wts_file)
        # print(type(self.model_info))
        for cp in self.model_info:
            print(cp, "\tlen:", self.model_info[cp].size())
        print("Done")

    def get_cls_token(self):
        return self.model_info['cls_token']

    def get_pos_embed(self):
        return self.model_info['pos_embed']

    def print_model_info(self, model):
        for key, value in model.state_dict().items():
            print(key, "\tlen:", value.size())
        print("Done")

    def get_patch_embed(self):
        return {'proj.weight': self.model_info['patch_embed.proj.weight'],
                'proj.bias': self.model_info['patch_embed.proj.bias']}

    '''
    Creates a new state dictionary by matching the common states 
    '''
    def create_new_state_dict(self, model):
        current_model = model.state_dict()

        new_state_dict = {k: v if v.size() == current_model[k].size() else current_model[k] for k, v in
                          zip(current_model.keys(), self.checkpoint['model'].values())
                          }
        print("Difference in sizes =>", len(current_model.items()), len(self.checkpoint['model'].items()))

        return new_state_dict

def test_get_pretrained_features():
    trained_wts_file = '../weights/mae_pretrain_vit_base.pth'
    gpf = GetPretrainedFeatures(trained_wts_file)
    gpf.print_model_info()
    # input_process = gpf.get_input_layer_params()
    # gpf.print_tensor_sizes(input_process)

if __name__ == '__main__':
    test_get_pretrained_features()