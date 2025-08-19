import torch
from utils.image_utils import ImageUtils

def show_outputs_from_data(save_path, patch_img_dim):
    img_utils = ImageUtils(side_length=patch_img_dim)
    full_out_data = torch.load(save_path)
    folder_ct = 0
    for _, folder in full_out_data['folders'].items():
        print("Showing outputs for folder: ", folder, full_out_data[folder]['folder'])
        for curr_iter in range(full_out_data[folder]['num_iters']):
            # View classifier scores
            classifier_score_map = full_out_data[folder][str(curr_iter)]['classifier_score']
            img_utils.image_viewer(classifier_score_map, "classifer score")

            # View output on patch images
            search_image_path = full_out_data[folder][str(curr_iter)]['search_image_path']
            op_coord = full_out_data[folder][str(curr_iter)]['op_coord']
            bb_img = img_utils.show_bbox_on_img(search_image_path, op_coord, req_resz=True, des_img_sz=patch_img_dim, str=str(folder_ct))

            # View BB on full size image, only partially works for now
            # TODO (Anshu-man567): Fix it completely
            torch_image = img_utils.load_image_and_params(search_image_path)
            short_w, long_h = torch_image.shape[1], torch_image.shape[2]
            op_coord = torch.tensor(op_coord).squeeze(0).detach()
            pad = 10
            bb_resz = torch.tensor([
                (op_coord[0] - 0) * long_h / 256 - pad,
                (op_coord[1] - 56) * long_h / 256 - pad,
                (op_coord[2] - 0) * long_h / 256 + pad,
                (op_coord[3] - 56) * long_h / 256 + pad,
            ])
            img_utils.show_bbox_on_img(search_image_path, bb_resz.reshape(1,4), req_resz=False, str=str(folder_ct), width=5)

            # break
            # if curr_iter == 20:
            #     break
        folder_ct += 1
        # break

if __name__ == "__main__":
    iter = 30
    save_path = 'got10k_test_run_results_' + str(iter) + '.pth'
    show_outputs_from_data(save_path, patch_img_dim=256)