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
        fg = cv2.imread(foregrounds[i])
        bg = cv2.imread(backgrounds[i + opt.offset])
        empty_img = np.zeros((H, W, 3), np.uint8)
        rescaled_fg = rescale_and_paste(crop_region, empty_img, fg)

        # First, create masking region of fg and mix fg and bg using mask_fg
        # Next, using actual mask region, mix cover_img and bg
        mask_fg = cv2.cvtColor(rescaled_fg.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        mask_fg = ((mask_fg > 0) * 255).astype(np.uint8)
        mask_fg = cv2.erode(mask_fg, np.ones((3,3), np.uint8), iterations=5)
        mask_fg = cv2.cvtColor(mask_fg, cv2.COLOR_GRAY2BGR)
        mask_fg = cv2.GaussianBlur(mask_fg, (5,5), cv2.BORDER_DEFAULT) / 255.0
        
        cover_img = mask_fg * rescaled_fg + (1 - mask_fg) * bg
    
        mask = cv2.imread(masks[i + opt.offset])
        empty_mask = np.zeros((H, W, 3), np.uint8)
        rescaled_mask = rescale_and_paste(crop_region, empty_mask, mask)
        rescaled_mask = cv2.erode(rescaled_mask, np.ones((3,3), np.uint8), iterations=5)
        rescaled_mask = cv2.GaussianBlur(rescaled_mask, (5,5), cv2.BORDER_DEFAULT) / 255.0
        
        comp = rescaled_mask * cover_img + (1 - rescaled_mask) * bg

        cv2.imwrite(os.path.join(opt.src_dir, 'comp', '%05d.png' % (i+1)), comp)

        if i >= opt.test_num:
            break
