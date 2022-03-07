import os
import time
from tqdm import tqdm

from options.expression_extract_options import Options
from datasets import create_dataset
from audiodvp_utils.irls_solver import IRLSSolver

if __name__ == '__main__':
    opt = Options().parse_args()   # get training options

    dataset = create_dataset(opt)

    irls_solver = IRLSSolver(opt)


    for i, data in enumerate(dataset):  # inner loop within one epoch

        pass

