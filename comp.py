import os
import cv2
import numpy as np
from tqdm import tqdm

from options.options import Options
from utils.util import create_dir, get_file_list


if __name__ == '__main__':
    opt = Options().parse_args()

    create_dir(os.path.join(opt.src_dir, 'comp'))

    foregrounds = get_file_list(os.path.join(opt.src_dir, 'images'), suffix='fake')
    backgrounds = get_file_list(os.path.join(opt.tgt_dir, 'crop'))
    masks = get_file_list(os.path.join(opt.tgt_dir, 'mask'))

    for i in tqdm(range(len(foregrounds))):
        fg = cv2.imread(foregrounds[i])
        bg = cv2.imread(backgrounds[i + opt.offset])
        
        # First, create masking region of fg and mix fg and bg using mask_fg
        # Next, using actual mask region, mix cover_img and bg
        mask_fg = cv2.cvtColor(fg.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        mask_fg = ((mask_fg > 0) * 255).astype(np.uint8)
        mask_fg = cv2.erode(mask_fg, np.ones((3,3), np.uint8), iterations=3)
        mask_fg = cv2.cvtColor(mask_fg, cv2.COLOR_GRAY2BGR)
        mask_fg = cv2.GaussianBlur(mask_fg, (5,5), cv2.BORDER_DEFAULT) / 255.0
        
        cover_img = mask_fg * fg + (1 - mask_fg) * bg
    
        mask = cv2.imread(masks[i + opt.offset])
        mask = cv2.erode(mask, np.ones((3,3), np.uint8), iterations=3)
        mask = cv2.GaussianBlur(mask, (5,5), cv2.BORDER_DEFAULT) / 255.0

        comp = mask * cover_img + (1 - mask) * bg

        cv2.imwrite(os.path.join(opt.src_dir, 'comp', '%05d.png' % (i+1)), comp)

        if i >= opt.test_num:
            break
