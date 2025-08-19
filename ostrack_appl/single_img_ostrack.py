import torch
from ostrack.ostrack_model import OSTrackModel

def test_ostrack_blocks():
    # search_image_file_path = "../test_image/search/search/angry_cat.jpg"
    # tmpl_img_file_path = "../test_image/tmpl/angry_cat_crop.jpg"
    # search_image_file_path = "../test_image/search/search/shapes.jpg"
    # tmpl_img_file_path = "../test_image/tmpl/shapes_crop.jpg"
    # search_image_file_path = "../test_image/search/search/dk_search.jpg"
    # tmpl_img_file_path = "../test_image/tmpl/dk_tmpl.jpg"
    search_image_file_path = "../test_image/search/doggo.png"
    template_image_file_patch = "../test_image/tmpl/doggo_crop.png"
    # search_image_file_path = "../test_image/search/search/x_patch_arr.jpg"
    # tmpl_img_file_path = "../test_image/tmpl/template.jpg"

    # search_image_file_path = "../test_image/search/search/bb_search.jpg"
    # tmpl_img_file_path = "../test_image/tmpl/bb_tmpl.jpg"

    # search_image_file_path = "../test_image/search/search/whale_sample.jpg"
    # tmpl_img_file_path = "../test_image/tmpl/whale_tmpl.jpg"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ostrack = OSTrackModel(print_stats=1, en_early_cand_elimn=True)
    ostrack = ostrack.eval()
    ostrack = ostrack.to(device)

    # with torch.no_grad():
    #     ostrack_state_dict, is_exact_copy = ostrack.gtf_ostrack.create_new_state_dict(ostrack)
    #     print("Is it an exact copy?", is_exact_copy)
    #
    #     missing_keys, unexpected_keys = ostrack.load_state_dict(ostrack_state_dict, strict=False)
    #
    #     # ostrack.gtf_ostrack.print_model_info(ostrack)
    #
    #     x = ostrack.try_on_input_tokens(search_img_file_path=search_image_file_path,
    #                                     template_img_file_path=template_image_file_patch)


    with torch.no_grad():
        ostrack_state_dict, is_exact_copy = ostrack.gtf_ostrack.create_new_state_dict(ostrack)
        print("Is it an exact copy?", is_exact_copy)

        missing_keys, unexpected_keys = ostrack.load_state_dict(ostrack_state_dict, strict=False)

        # ostrack.gtf_ostrack.print_model_info(ostrack)

        x = ostrack.try_on_input_tokens(search_img_file_path=search_image_file_path,
                                        template_img_file_path=template_image_file_patch)


if __name__ == "__main__":
    test_ostrack_blocks()