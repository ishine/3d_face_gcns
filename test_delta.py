import os
from tqdm import tqdm

from options.options import Options
from datasets import create_dataset
from audiodvp_utils.util import create_dir
from models import wav2delta_model

if __name__ == '__main__':
    opt = Options().parse_args()   # get training options
    # hard-code some parameters for test
    opt.isTrain = False
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.

    create_dir(os.path.join(opt.data_dir, 'reenact_delta'))

    dataset = create_dataset(opt)

    model = wav2delta_model.Wav2DeltaModel(opt)
    model.load_network()
    model.eval()

    for i, data in enumerate(tqdm(dataset)):
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        model.save_delta()

        if i >= opt.test_num - 1:
            break
