import time

###############################################################################
## Misc Functions
###############################################################################

def cpu_data_to_gpu(cpu_data, exclude_keys=None):
    if exclude_keys is None:
        exclude_keys = []

    gpu_data = {}
    for key, val in cpu_data.items():
        if key in exclude_keys:
            continue

        if isinstance(val, list):
            assert len(val) > 0
            if not isinstance(val[0], str): # ignore string instance
                gpu_data[key] = [x.cuda() for x in val]
        elif isinstance(val, dict):
            gpu_data[key] = {sub_k: sub_val.cuda() for sub_k, sub_val in val.items()}
        else:
            gpu_data[key] = val.cuda()

    return gpu_data


###############################################################################
## Timer
###############################################################################

class Timer():
    def __init__(self):
        self.curr_time = 0

    def begin(self):
        self.curr_time = time.time()

    def log(self):
        diff_time = time.time() - self.curr_time
        self.begin()
        return f"{diff_time:.2f} sec"
