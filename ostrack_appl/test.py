import os
import torch
from collections import OrderedDict

from ostrack.ostrack_model import OSTrackModel
from ostrack_appl.seq_tracker_ostrack import SeqOSTrack
from ostrack_appl.seq_tracker_ostrack import TrackExecutionMode
from utils.image_utils import ImageParseLib
from utils.parse_cli_args import parse_args

def test_seq_ostrack_got10k(cli_args, iter_lim=-1):

    seq_ostrack = SeqOSTrack(exec_mode=TrackExecutionMode.TEST,
                             img_lib=ImageParseLib.TORCHVISION,
                             show_dumps=cli_args.show_dumps,
                             print_stats=cli_args.print_stats,
                             en_early_cand_elimn=cli_args.en_ece,
                             search_img_dim=cli_args.search_size,
                             tmpl_img_dim=cli_args.template_size,
                             size_D=cli_args.hidden_dim,
                             pretrained_weights=cli_args.pretrained_weights)

    got10k_test_dir = "/home/anshu-man567/PycharmProjects/ProjectIdeas/OSTrack/data/got10k/test/"
    full_out_data = OrderedDict()
    full_out_data['folders'] = OrderedDict()
    iter = 1
    running_avg_fps = 0.0
    for folder in sorted(os.listdir(got10k_test_dir)):
        if os.path.isdir(os.path.join(got10k_test_dir, folder)) is False:
            print("ERROR, this path is not a folder, skipping using it", os.path.join(got10k_test_dir, folder), folder)
            continue
        tmpl_img_file_path = os.path.join(got10k_test_dir, folder, "00000001.jpg")
        txt_path = os.path.join(got10k_test_dir, folder, "groundtruth.txt")
        search_image_folder_path = os.path.join(got10k_test_dir, folder)
        print(tmpl_img_file_path, txt_path, search_image_folder_path)
        image = seq_ostrack.img_utils.load_image_and_params(tmpl_img_file_path)
        bbox, crop = seq_ostrack.parse_gt_bbox_got10k(txt_path)
        seq_ostrack.img_utils.show_bbox_on_img(tmpl_img_file_path, bbox_coords=bbox)
        seq_ostrack.init_new_seq(tmpl_img_file_path=tmpl_img_file_path, req_resz=True, crop_box=crop)

        full_out_data['folders'][str(iter)] = folder
        full_out_data[folder] = seq_ostrack.track_seq_test(search_image_folder=search_image_folder_path, req_resz=True)
        running_avg_fps += full_out_data[folder]['fps_rate']
        print("Last FPS:", full_out_data[folder]['fps_rate'], "Current avg FPS:", running_avg_fps/iter)

        if iter == iter_lim:
            break
        else:
            iter += 1

    save_path = 'got10k_test_run_results_'+str(iter)+'.pth'
    torch.save(full_out_data, save_path)

    # show_outputs_from_data(save_path, img_dim=seq_ostrack.search_img_dim)


def test_seq_ostrack_got1_dynamic(cli_args):
    # search_image_file_path = "../test_image/search/angry_cat.jpg"
    # tmpl_img_file_path = "../test_image/tmpl/angry_cat_crop.jpg"
    # search_image_file_path = "../test_image/search/shapes.jpg"
    # tmpl_img_file_path = "../test_image/tmpl/shapes_crop.jpg"
    # search_image_file_path = "../test_image/search/dk_search.jpg"
    # tmpl_img_file_path = "../test_image/tmpl/dk_tmpl.jpg"
    # search_image_file_path = "../test_image/search/doggo.png"
    # tmpl_img_file_path = "../test_image/tmpl/doggo_crop.png"
    # search_image_file_path = "../test_image/search/x_patch_arr.jpg"
    # tmpl_img_file_path = "../test_image/tmpl/template.jpg"

    # search_image_file_path = "../test_image/search/bb_search.jpg"
    # tmpl_img_file_path = "../test_image/tmpl/bb_tmpl.jpg"

    # search_image_file_path = "../test_image/search/whale_sample.jpg"

    '''
    GOT 10k details:
    01 => dolphin
    02 => black boat in city river
    03 => blue boat
    23 => driffffffffting
    38 => some angel fish?
    52 => bullock cart battles
    90 => is the weirdest I cant comprehend
    62 => kangaroo fights with another animal (forgot its name)
    43 => another kangaroo running, but couldnt track it due to occlusion of the same color
    45 => template so smol, even I cant detect it
    62 +> kangaroo fights with a racoon
    '''

    seq_ostrack = SeqOSTrack(exec_mode=TrackExecutionMode.TEST,
                             img_lib=ImageParseLib.TORCHVISION,
                             show_dumps=cli_args.show_dumps,
                             save_outputs=cli_args.save_outputs,
                             print_stats=cli_args.print_stats,
                             en_early_cand_elimn=cli_args.en_ece,
                             search_img_dim=cli_args.search_size,
                             tmpl_img_dim=cli_args.template_size,
                             size_D=cli_args.hidden_dim,
                             pretrained_weights=cli_args.pretrained_weights)
    input_img_folder = "/home/anshu-man567/PycharmProjects/ProjectIdeas/OSTrack/data/got10k/test/GOT-10k_Test_000062/"

    tmpl_img_file_path = os.path.join(input_img_folder, "00000001.jpg")
    txt_path = os.path.join(input_img_folder, "groundtruth.txt")
    search_image_folder_path = input_img_folder
    print(tmpl_img_file_path, txt_path, search_image_folder_path)
    image = seq_ostrack.img_utils.load_image_and_params(tmpl_img_file_path)
    bbox, crop = seq_ostrack.parse_gt_bbox_got10k(txt_path)
    seq_ostrack.img_utils.show_bbox_on_img(tmpl_img_file_path, bbox_coords=bbox)
    seq_ostrack.init_new_seq(tmpl_img_file_path=tmpl_img_file_path, req_resz=True, crop_box=crop)
    seq_ostrack.track_seq_test(search_image_folder=search_image_folder_path, req_resz=True)


def test_seq_ostrack_single():
    # search_image_file_path = "../test_image/search/angry_cat.jpg"
    # tmpl_img_file_path = "../test_image/tmpl/angry_cat_crop.jpg"
    # search_image_file_path = "../test_image/search/shapes.jpg"
    # tmpl_img_file_path = "../test_image/tmpl/shapes_crop.jpg"
    # search_image_file_path = "../test_image/search/dk_search.jpg"
    # tmpl_img_file_path = "../test_image/tmpl/dk_tmpl.jpg"
    # search_image_file_path = "../test_image/search/doggo.png"
    # tmpl_img_file_path = "../test_image/tmpl/doggo_crop.png"
    # search_image_file_path = "../test_image/search/x_patch_arr.jpg"
    # tmpl_img_file_path = "../test_image/tmpl/template.jpg"

    search_image_file_path = "../test_image/search/bb_search.jpg"
    tmpl_img_file_path = "../test_image/tmpl/bb_tmpl.jpg"

    # search_image_file_path = "../test_image/search/whale_sample.jpg"
    # tmpl_img_file_path = "../test_image/tmpl/whale_tmpl.jpg"

    seq_ostrack = SeqOSTrack(exec_mode=TrackExecutionMode.TEST,
                             img_lib=ImageParseLib.TORCHVISION,
                             show_dumps=1)

    seq_ostrack.init_new_seq(tmpl_img_file_path=tmpl_img_file_path)
    seq_ostrack.track_seq_test(search_image_folder=None, search_image_paths=[search_image_file_path])


def test_ostrack_blocks():
    search_image_file_path = "../test_image/search/angry_cat.jpg"
    tmpl_img_file_path = "../test_image/tmpl/angry_cat_crop.jpg"
    # search_image_file_path = "../test_image/search/shapes.jpg"
    # tmpl_img_file_path = "../test_image/tmpl/shapes_crop.jpg"
    # search_image_file_path = "../test_image/search/dk_search.jpg"
    # tmpl_img_file_path = "../test_image/tmpl/dk_tmpl.jpg"
    # search_image_file_path = "../test_image/search/doggo.png"
    # tmpl_img_file_path = "../test_image/tmpl/doggo_crop.png"
    # search_image_file_path = "../test_image/search/x_patch_arr.jpg"
    # tmpl_img_file_path = "../test_image/tmpl/template.jpg"

    # search_image_file_path = "../test_image/search/bb_search.jpg"
    # tmpl_img_file_path = "../test_image/tmpl/bb_tmpl.jpg"

    # search_image_file_path = "../test_image/search/whale_sample.jpg"
    # tmpl_img_file_path = "../test_image/tmpl/whale_tmpl.jpg"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ostrack = OSTrackModel(print_stats=0, en_early_cand_elimn=True)
    ostrack = ostrack.eval()
    ostrack = ostrack.to(device)

    with torch.no_grad():
        ostrack_state_dict, is_exact_copy = ostrack.gtf_ostrack.create_new_state_dict(ostrack)
        print("Is it an exact copy?", is_exact_copy)

        missing_keys, unexpected_keys = ostrack.load_state_dict(ostrack_state_dict, strict=False)

        # ostrack.gtf_ostrack.print_model_info(ostrack)

        ostrack.create_tmpl_patches(template_img_file_path=tmpl_img_file_path)

        x = ostrack.try_on_input_tokens(search_img_file_path=search_image_file_path)

if __name__ == "__main__":
    args = parse_args()

    # test_seq_ostrack_single()
    test_seq_ostrack_got1_dynamic(args)
    # test_seq_ostrack_got10k(args, iter_lim=30)
    # show_outputs_from_data("got10k_test_run_results_180.pth", 256)