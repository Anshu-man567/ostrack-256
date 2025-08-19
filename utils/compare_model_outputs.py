import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class CompareModelOutputs:
    def __init__(self):
        self.seq_name ='GOT-10k_Test_000001_1'
        self.ostrack_out = torch.load(('/home/anshu-man567/PycharmProjects/ProjectIdeas/OSTrack/model_out_dict/'+self.seq_name+'_ostrack_out.pt'), map_location='cpu')
        self.my_ostrack_out = torch.load('my_ostrack_out.pt', map_location='cpu')

        self.input_str = 'x_patch_arr'
        self.vit_out = 'backbone_feat'
        self.score_map = 'score_map'
        self.in_x = 'in_x'
        self.in_z = 'in_z'
        self.patch_x = 'patch_x'
        self.patch_z = 'patch_z'
        self.patch_pos_x = 'patch_pos_x'
        self.patch_pos_z = 'patch_pos_z'

        self.backbone_attn_mat = 'attn_out_'
        self.backbone_mlp_out = 'mlp_out_'

    def compare_backbone_attn_mat(self, idx=0):
        # print(self.ostrack_out)
        text_str = self.backbone_attn_mat+str(idx)

        print(self.ostrack_out[text_str], self.my_ostrack_out[text_str])
        print(text_str, self.ostrack_out[text_str].shape, self.my_ostrack_out[text_str].shape)

        os_out = self.ostrack_out[text_str].round(decimals=1)
        my_os_out = self.my_ostrack_out[text_str].round(decimals=1)
        # os_out = self.my_ostrack_out[self.backbone_attn_mat+str(1)].round(decimals=1)

        self.image_viewer(os_out, "os_out")
        self.image_viewer(my_os_out, "my_os_out")

        out = (os_out - my_os_out).reshape(-1)
        out, _ = out.sort()
        # print("out", out.nonzero())
        non_z_ct = torch.count_nonzero(out)
        if non_z_ct > 0:
            print("SOMETHING FISHY", text_str, non_z_ct)
        else:
            print("THEY ARE GOOOD!", text_str)

    def compare_input(self):
        # print(self.ostrack_out)
        print(self.ostrack_out[self.input_str].shape, self.my_ostrack_out[self.input_str].shape)

        if torch.all(self.ostrack_out[self.input_str], self.my_ostrack_out[self.input_str]):
            print("THEY ARE GOOOD!", )
        else:
            print("SOMETHING FISHY")

    def compare_backbone_feat(self):
        text_str = self.vit_out

        print(self.ostrack_out[text_str], self.my_ostrack_out[text_str])
        print(text_str, self.ostrack_out[text_str].shape, self.my_ostrack_out[text_str].shape)

        os_out = self.ostrack_out[text_str]
        my_os_out = self.my_ostrack_out[text_str]

        self.image_viewer(os_out, "os_out")
        self.image_viewer(my_os_out, "my_os_out")

        out = os_out - my_os_out
        out = (os_out - my_os_out).reshape(-1)
        out, _ = out.sort()
        print("out", out)
        non_z_ct = torch.count_nonzero(out)
        if non_z_ct > 0:
            print("SOMETHING FISHY", text_str, non_z_ct)
        else:
            print("THEY ARE GOOOD!", text_str)

    def compare_in_x(self):
        text_str = self.in_x

        print(self.ostrack_out[text_str], self.my_ostrack_out[text_str])
        print(text_str, self.ostrack_out[text_str].shape, self.my_ostrack_out[text_str].shape)

        os_out = self.ostrack_out[text_str]
        my_os_out = self.my_ostrack_out[text_str]

        self.image_viewer(os_out.squeeze(dim=0), "os_out")
        self.image_viewer(my_os_out.squeeze(dim=0), "my_os_out")

        out = os_out - my_os_out
        non_z_ct = torch.count_nonzero(out)
        if non_z_ct > 0:
            print("SOMETHING FISHY", text_str, non_z_ct)
        else:
            print("THEY ARE GOOOD!", text_str)

    def compare_in_z(self):
        # print(self.ostrack_out)
        text_str = self.in_z
        print(self.ostrack_out[text_str], self.my_ostrack_out[text_str])
        print(text_str, self.ostrack_out[text_str].shape, self.my_ostrack_out[text_str].shape)

        os_out = self.ostrack_out[text_str]
        my_os_out = self.my_ostrack_out[text_str]

        self.image_viewer(os_out.squeeze(dim=0), "os_out")
        self.image_viewer(my_os_out.squeeze(dim=0), "my_os_out")

        out = os_out - my_os_out
        non_z_ct = torch.count_nonzero(out)
        if non_z_ct > 0:
            print("SOMETHING FISHY", text_str, non_z_ct)
        else:
            print("THEY ARE GOOOD!", text_str)

    def compare_x(self):
        text_str = self.patch_x
        print(self.ostrack_out[text_str], self.my_ostrack_out[text_str])
        print(text_str, self.ostrack_out[text_str].shape, self.my_ostrack_out[text_str].shape)

        os_out = self.ostrack_out[text_str]
        my_os_out = self.my_ostrack_out[text_str]

        self.image_viewer(os_out, "os_out")
        self.image_viewer(my_os_out, "my_os_out")

        out = os_out - my_os_out
        non_z_ct = torch.count_nonzero(out)
        if non_z_ct > 0:
            print("SOMETHING FISHY", text_str, non_z_ct)
        else:
            print("THEY ARE GOOOD!", text_str)

    def compare_z(self):
        # print(self.ostrack_out)
        text_str = self.patch_z

        print(self.ostrack_out[text_str], self.my_ostrack_out[text_str])
        print(text_str, self.ostrack_out[text_str].shape, self.my_ostrack_out[text_str].shape)

        os_out = self.ostrack_out[text_str]
        my_os_out = self.my_ostrack_out[text_str]

        self.image_viewer(os_out, "os_out")
        self.image_viewer(my_os_out, "my_os_out")

        out = os_out - my_os_out
        non_z_ct = torch.count_nonzero(out)
        if non_z_ct > 0:
            print("SOMETHING FISHY", text_str, non_z_ct)
        else:
            print("THEY ARE GOOOD!", text_str)

    def compare_pos_x(self):
        # print(self.ostrack_out)
        text_str = self.patch_pos_x

        print(self.ostrack_out[text_str], self.my_ostrack_out[text_str])
        print(text_str, self.ostrack_out[text_str].shape, self.my_ostrack_out[text_str].shape)

        os_out = self.ostrack_out[text_str].round(decimals=1)
        my_os_out = self.my_ostrack_out[text_str].round(decimals=1)

        self.image_viewer(os_out, "os_out")
        self.image_viewer(my_os_out, "my_os_out")

        out = (os_out - my_os_out).reshape(-1)
        out, _ = out.sort()
        # print("out", out.nonzero())
        non_z_ct = torch.count_nonzero(out)
        if non_z_ct > 0:
            print("SOMETHING FISHY", text_str, non_z_ct)
        else:
            print("THEY ARE GOOOD!", text_str)

    def compare_pos_z(self):
        # print(self.ostrack_out)
        text_str = self.patch_pos_z
        print(text_str, self.ostrack_out[text_str], self.my_ostrack_out[text_str])
        print(text_str, self.ostrack_out[text_str].shape, self.my_ostrack_out[text_str].shape)

        os_out = self.ostrack_out[text_str]
        my_os_out = self.my_ostrack_out[text_str]
        
        os_out = self.ostrack_out[text_str].round(decimals=1)
        my_os_out = self.my_ostrack_out[text_str].round(decimals=1)

        self.image_viewer(os_out, "os_out")
        self.image_viewer(my_os_out, "my_os_out")

        out = os_out - my_os_out
        out = (os_out - my_os_out).reshape(-1)
        out, _ = out.sort()
        print("out", out)
        non_z_ct = torch.count_nonzero(out)
        if non_z_ct > 0:
            print("SOMETHING FISHY", text_str, non_z_ct)
        else:
            print("THEY ARE GOOOD!", text_str)

    def image_viewer(self, image, text_str="NOSTRING",dump_img_flag=1):
        if dump_img_flag != 1:
            return
        print(image.shape)
        image = image.detach().clone()
        # if self.img_lib is ImageParseLib.TORCHVISION:
        plt.imshow(image.permute(1, 2, 0))  # matplotlib expects h, w, c; but tensors are c, h, w
        # h, w, c = self.get_image_size(image)
        plt.text(0.5, 0.05, text_str,          # f"Image size: {h} x {w}",
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=plt.gca().transAxes,
                 color='white',
                 fontsize=12,
                 bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))
        # Remove axis ticks
        plt.xticks([])
        plt.yticks([])
        plt.show()
        # elif self.img_lib is ImageParseLib.OPENCV:
        # cv.imshow("Display Image", image)
        # elif self.img_lib is ImageParseLib.PIL:
        #     image.show()
        # else:
        #     print("Did not set img lib :(")



def compare_models():
    cmp = CompareModelOutputs()
    # cmp.compare_input()
    # cmp.compare_in_x()
    # cmp.compare_in_z()
    # cmp.compare_pos_x()
    # cmp.compare_pos_z()
    # cmp.compare_x()
    # cmp.compare_z()
    # cmp.compare_backbone_feat()
    cmp.compare_backbone_attn_mat(0)

if __name__ == "__main__":
    compare_models()