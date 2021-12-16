
import os
import pandas as pd
import numpy as np

def load_data_cvs_exclude(images_path, scores_path, is_train=False):

    data = pd.read_csv(scores_path, usecols=['image', 'glossiness', 'refsharp', 'contgloss', 'metallicness', 'lightness', 'anisotropy'])
    data = np.array(data)

    input_files = list(os.path.join(images_path, file_name) for file_name in data[:,0])
    gt_values = (data[:,1:7] - 1.0)/6.0
    return input_files, gt_values

def exists_or_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    else:
        return True
