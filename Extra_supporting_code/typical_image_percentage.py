# coding=utf-8
import io
import os
from multiprocessing import Pool
import numpy as np
import zstandard as zstd

def process_file(file):
    filepath = os.path.join(data_dir, file)

    with open(filepath, 'rb') as f:
        compressed = f.read()

    dctx = zstd.ZstdDecompressor()
    raw_npz = dctx.decompress(compressed)

    buffer = io.BytesIO(raw_npz)
    with np.load(buffer) as data:
        predict = data['probability']
        predict = np.where(predict <= cutoff, 0, 1)
        return np.nansum(predict)


if __name__ == "__main__":
    data_dir = '/rstor/jmayhall/cataloging/nc_process/shear_distrubution_and_model/tcb_probs/tcb_probs/'
    cutoff = 0.02
    files = os.listdir(data_dir)

    count = 0
    tcb_pixels = 0
    tcb_max = 0
    tcb_min = float('inf')

    # IMPORTANT: reduce worker count — 128 is likely excessive
    with Pool(64) as pool:   # try 16–32 first
        for pixel_sum in pool.imap_unordered(process_file, files, chunksize=4):
            count += 1
            tcb_pixels += pixel_sum
            tcb_max = max(tcb_max, pixel_sum)
            tcb_min = min(tcb_min, pixel_sum)

    print(f'Average: {tcb_pixels / (count * 1024 * 1024)}')
    print(f'Max: {tcb_max / (1024 * 1024)}')
    print(f'Min: {tcb_min / (1024 * 1024)}')
