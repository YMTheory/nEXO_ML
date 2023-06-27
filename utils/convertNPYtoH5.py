import h5py as h5
import numpy as np
import sys
import re
import os

seed = sys.argv[1]
particle = sys.argv[2]
path = "/junofs/users/miaoyu/0nbb/nEXO_simulation/scripts/channelq_npy/"
st = f"{particle}_seed{seed}"
dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d.startswith(st)]

h5file = f"/junofs/users/miaoyu/0nbb/nEXO_simulation/scripts/h5files/nexo_{particle}_seed{seed}.h5"
with h5.File(h5file, 'a') as f:
    if particle not in f:
        g = f.create_group(f"{particle}")

    for d in dirs:
        print(d)
        numbers = re.findall(r"\d+", d)
        start, stop = int(numbers[-2]), int(numbers[-1])
    
        for ievt in range(start, stop, 1):
            arr = np.load(path+d+'/'+f"{particle}_seed{seed}_event{ievt}_image2d.npy")
            dataset_name = f"{particle}_seed{seed}_event{ievt}"
            g.create_dataset(dataset_name, data=arr)

