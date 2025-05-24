import os
import argparse
import shutil
import json
import numpy as np
import torch
import random


def make_dir(dirs, check=True):
    if isinstance(dirs, str):
        dirs = [dirs]
    if check:
        check_dir(dirs)
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)


def check_dir(dirs):
    if isinstance(dirs, str):
        dirs = [dirs]
    for dir in dirs:
        if os.path.isdir(dir):
            str_input = input(f"\n{dir} \nSave Directory already exists, would you like to continue (y,n)?")
            if not str2bool(str_input):
                exit()
            else:
                empty_dir(dir) # clear out existing files


def empty_dir(dirs):
    if isinstance(dirs, str):
        dirs = [dirs]
    for dir in dirs:
        for filename in os.listdir(dir):
            file_path = os.path.join(dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print("Failed to delete %s. Reason: %s" % (file_path, e))
    

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def save_json_obj(obj, name):
    with open(name + ".json", "w") as fp:
        json.dump(obj, fp)


def load_json_obj(name):
    with open(name + ".json", "r") as fp:
        return json.load(fp)


def convert_json(obj):
    """Convert obj to a version which can be serialized with JSON."""
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v) for k, v in obj.items()}

        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)

        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]

        elif hasattr(obj, "__name__") and not ("lambda" in obj.__name__):
            return convert_json(obj.__name__)

        elif hasattr(obj, "__dict__") and obj.__dict__:
            obj_dict = {convert_json(k): convert_json(v) for k, v in obj.__dict__.items()}
            return {str(obj): obj_dict}

        return str(obj)


def is_json_serializable(v):
    try:
        json.dumps(v)
        return True
    except:
        return False


def numpy_collate(batch):
    '''
    Batch is list of len: batch_size
    Each element is dict {images: ..., labels: ...}
    Use Collate fn to ensure they are returned as np arrays.
    '''
    # list of arrays -> stacked into array
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)

    # list of lists/tuples -> recursive on each element
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]

    # list of dicts -> recursive returned as dict with same keys
    elif isinstance(batch[0], dict):
        return {key: numpy_collate([d[key] for d in batch]) for key in batch[0]}

    # list of non array element -> list of arrays
    else:
        return np.array(batch)
    