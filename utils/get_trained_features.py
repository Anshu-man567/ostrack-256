from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

class GetTrainedFeatures:
    def __init__(self, trained_wts_file, print_stats=0):
        self.trained_wts_file = trained_wts_file
        self.loaded_checkpoint = torch.load(self.trained_wts_file, map_location="cpu")
        # self.loaded_checkpoint = torch.load(self.trained_wts_file, map_location="cuda:0")
        self.loaded_model_info = self.loaded_checkpoint['net']
        self.print_stats = print_stats
        
    def get_model(self):
        return self.loaded_model_info
    
    def get_checkpoint(self):
        return self.loaded_checkpoint

    def print_loaded_model_info(self):
        print('Loaded pretrained model from: ' + self.trained_wts_file)
        for cp in self.loaded_model_info:
            print(cp, "\tlen:", self.loaded_model_info[cp].size())
        print("Done printing loaded model info")

    def get_cls_token(self):
        return self.loaded_model_info['cls_token']

    def get_pos_embed(self):
        return self.loaded_model_info['pos_embed']

    def print_model_info(self, model):
        for key, value in model.state_dict().items():
            print(key, "\tlen:", value.size())
        print("Done printing model info")

    def get_patch_embed(self):
        return {'proj.weight': self.loaded_model_info['patch_embed.proj.weight'],
                'proj.bias': self.loaded_model_info['patch_embed.proj.bias']}

    '''
    Creates a new state dictionary by matching the common states 
    '''
    def create_new_state_dict(self, model):
        current_model = model.state_dict()

        # Compact form, but works only if the models match exactly
        # new_state_dict = {k: v if v.size() == current_model[k].size() else current_model[k] for k, v in
        #                   zip(current_model.keys(), self.loaded_model_info.values())
        #                   }

        new_state_dict = OrderedDict()
        is_exact_copy = 1

        loaded_model_items = self.loaded_model_info
        loaded_model_keys = self.loaded_model_info.keys()

        print("Difference in sizes =>", len(current_model.items()), len(self.loaded_model_info.items()))

        for curr_model_key in current_model:
            # print("Checking for curr_model_key", curr_model_key)
            check_1 = (curr_model_key in loaded_model_keys)
            check_2 = (loaded_model_items[curr_model_key].shape == current_model[curr_model_key].shape)
            if check_1 and check_2:
                # print("Found an entry", type(loaded_model_items[curr_model_key]), loaded_model_items[curr_model_key].shape)
                new_state_dict[curr_model_key] = loaded_model_items[curr_model_key]
                # print(type(new_state_dict))
            else:
                is_exact_copy = 0
                if check_1 is False:
                    print("Skipping including weight from loaded model for:", curr_model_key)
                else:
                    print("Skipping including weight from loaded model due to shape mismatch for:", curr_model_key, "curr sz:", current_model[curr_model_key].shape, "loaded sz:", loaded_model_items[curr_model_key].shape)

        # print(new_state_dict.keys())

        if len(current_model.items()) != len(self.loaded_model_info.items()):
            new_state_dict_keys = new_state_dict.keys()
            for loaded_model_key in loaded_model_keys:
                if loaded_model_key not in new_state_dict_keys:
                    print("Had skipped including weight from loaded model for:", loaded_model_key)

        return new_state_dict, is_exact_copy

    def selectively_compare_models(self, current_model, loaded_model=None):
        '''
           Adapted from Source : https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351/6
           :param model_1: Your model implementation dict 1
           :param model_2: Model loaded using params
           :return:
        '''

        if loaded_model is None:
            loaded_model = self.loaded_model_info

        current_model_state_dict = current_model.state_dict()
        loaded_model_keys = loaded_model.keys()

        models_differ = 0
        mismatched_keys = []

        for curr_model_key in current_model_state_dict:
            # print("Checking for curr_model_key", curr_model_key)
            check_1 = curr_model_key in loaded_model_keys
            check_2 = torch.equal(current_model_state_dict[curr_model_key], loaded_model[curr_model_key])
            if check_1:
                if not check_2:
                    models_differ += 1
                    mismatched_keys.append([curr_model_key, current_model_state_dict[curr_model_key].shape, loaded_model[curr_model_key].shape])
            else:
                print("Don't know, but this model key is not present in the loaded model", curr_model_key)
                raise Exception

        if models_differ == 0:
            print('Models match perfectly! :)')
        else:
            print("Models mismatched", models_differ, "many times :(")
            for key in mismatched_keys:
                print("Mismatched keys", key)

        return (models_differ == 0)

    def compare_models(self, model_1, model_2=None):
        '''
        Source : https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351/6
        :param model_1: Your model implementation dict 1
        :param model_2: Model loaded using params
        :return:
        '''

        if model_2 is None:
            model_2 = self.get_model()

        models_differ = 0
        mismatched_keys = []
        for key_item_1, key_item_2 in zip(sorted(model_1.state_dict().items()), sorted(model_2.items())):
            print("Comparing the keys:", key_item_1[0], key_item_2[0])
            if torch.equal(key_item_1[1], key_item_2[1]):
                pass
            else:
                models_differ += 1
                if (key_item_1[0] == key_item_2[0]):
                    # print('Mismatch found at', key_item_1[0])
                    mismatched_keys.append([key_item_1[0], key_item_1[1].shape, key_item_2[0], key_item_2[1].shape])
                else:
                    print("Don't know what happened", key_item_1[0], key_item_2[0])
                    raise Exception
        if models_differ == 0:
            print('Models match perfectly! :)')
        else:
            print("Models mismatched", models_differ, "many times :(")
            for key in mismatched_keys:
                print(key)

        return (models_differ == 0)

def test_get_pretrained_features():
    trained_wts_file = '../weights/mae_pretrain_vit_base.pth'
    gtf = GetTrainedFeatures(trained_wts_file)
    gtf.print_loaded_model_info()
    # input_process = gtf.get_input_layer_params()
    # gtf.print_tensor_sizes(input_process)

if __name__ == '__main__':
    test_get_pretrained_features()