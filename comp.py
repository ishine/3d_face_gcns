import os
import cv2
import numpy as np
from torch import empty
from tqdm import tqdm

from options.options import Options
from audiodvp_utils.util import create_dir, get_file_list, get_max_crop_region, load_coef
from audiodvp_utils.rescale_image import rescale_and_paste

if __name__ == '__main__':
    opt = Options().parse_args()

    create_dir(os.path.join(opt.src_dir, 'comp'))

    foregrounds = get_file_list(os.path.join(opt.src_dir, 'images'), suffix='fake')
    backgrounds = get_file_list(os.path.join(opt.tgt_dir, 'full'))
    crop_region_list = load_coef(os.path.join(opt.tgt_dir, 'crop_region'))
    top, bottom, left, right = get_max_crop_region(crop_region_list)
    crop_region = [top, bottom, left, right]

    masks = get_file_list(os.path.join(opt.tgt_dir, 'mask'))
    H, W, _ = cv2.imread(backgrounds[0]).shape

    for i in tqdm(range(len(foregrounds))):
        # idx = i % (opt.offset_end - opt.offset_start ) + opt.offset_start
        idx = i
        fg = cv2.imread(foregrounds[i])
        bg = cv2.imread(backgrounds[idx])
        empty_img = np.zeros((H, W, 3), np.uint8)
        comp = rescale_and_paste(crop_region, bg, fg)

        cv2.imwrite(os.path.join(opt.src_dir, 'comp', '%05d.png' % (i+1)), comp)