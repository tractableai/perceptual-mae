from arguments import args
from time import time
from PIL import Image
import numpy as np
import pandas as pd 

import torch 
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.utils import save_image
import random

def read_df(path, cols=None):
    if path.endswith('.csv'):
        if cols!=None:
            return pd.read_csv(path, usecols=lambda s: s in cols)
        else:
            return pd.read_csv(path)
    else:
        if cols!=None:
            return pd.read_parquet(path, columns= cols)
        else:
            return pd.read_parquet(path)


def df_random_select(df_path, cache_dir):
    df = read_df(df_path)
    imbag_ids = list(df.imbag_id.unique())
    rand_imbag_id = random.choice(imbag_ids)
    # get a list of all image paths;
    image_list = list(df[df['imbag_id']==rand_imbag_id]['image_ids'])[0]
    db_name = db_names = list(df[df['imbag_id']==rand_imbag_id]['db_name'])[0]
    # select one;
    random_image_name = random.choice(image_list)
    # get the subfolder id (usually first 2 characters of image name)
    sub_dir = random_image_name[:2]
    filename = '{}/{}/raw/{}/{}.jpg'.format(cache_dir, db_name, sub_dir, random_image_name)
    print(filename)
    return filename

def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(image)
    plt.title(title, fontsize=16)
    plt.axis('off')
    return


def run_one_image(img, model):
    x= img
    #x = torch.tensor(img).squeeze()
    
    # make it a batch-like
    #x = x.unsqueeze(dim=0)
    #x = torch.einsum('nhwc->nchw', x)
    # run MAE
    y, mask = model(x.float())
    y = model.unpatchify(y.unsqueeze(0))
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    
    x= x.squeeze(2)
    x = torch.einsum('nchw->nhwc', x)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]

    plt.subplot(1, 4, 1)
    show_image(x[0], "original")

    plt.subplot(1, 4, 2)
    show_image(im_masked[0], "masked")

    plt.subplot(1, 4, 3)
    show_image(y[0], "reconstruction")

    plt.subplot(1, 4, 4)
    show_image(im_paste[0], "reconstruction + visible")

    plt.show()