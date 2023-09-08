import torch 
import torch.nn as nn 

# import pandas as pd 
import numpy as np
import os 
from omegaconf import OmegaConf
from pathlib import Path
from tqdm import tqdm

from src.common.data_types import Datasets
from src.models.base_model import BaseModel

class ToyPredictor():
    def __init__(self, config, datasets: Datasets, model: BaseModel, run_name):
        self.config = config 
        self.training_config = self.config.training
        self.datasets = datasets
        self.model = model

        # set model in eval mode;
        self.model.eval();

        print("Loading Datasets")
        self.test_loader = self.datasets.test_dataloader
        
        self.device = torch.device(self.training_config.device)
        print('using device: {}'.format(self.training_config.device))

        # put model on device
        self.model.to(self.device);
        # create an empty df;
        # self.output_df = pd.DataFrame(columns=self.useable_cols)

        home = str(Path.home())
        self.output_dir = '{}/predictions_{}'.format(home, run_name)
        if os.path.exists(self.output_dir)!=True:
            os.makedirs(self.output_dir)
        
        print('output dir: {} created!'.format(self.output_dir))

    def test(self):
        correct = 0
        total = 0

        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        preds_, gt_labels=[],[]

        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for i, data in enumerate(tqdm(self.test_loader)):
                images, labels = data
                # put the data on cuda;
                images = images.to(self.device)
                labels = labels.to(self.device)
                pad_mask = None
                # run inference;
                out = self.model(images, pad_mask)
                # the class with the highest energy is what we choose as prediction
                _, predictions = torch.max(out.data, 1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

                preds_.append(predictions.detach().cpu().numpy())
                gt_labels.apend(labels.cpu().numpy())
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1

            print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
        
        print('testing loop done! now converting outputs to arrays')
        
        preds_ = np.asarray(preds_)
        gt_labels = np.asarray(gt_labels)

        np.save('{}/pred_outputs.npy'.format(self.output_dir), preds_)
        np.save('{}/gt_labels.npy'.format(self.output_dir), gt_labels)
        print('saving numpy arrays as backup')

        print('saving jsons for dict as backup')