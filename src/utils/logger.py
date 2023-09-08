import importlib
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os
from src.utils.aws import get_current_time_string
import torchvision 

class TensorboardLogger():
    def __init__(self, config):
        #self.aws_client = aws_client
        self.config = config
        self.tb_output_dir = './runs/{}/'.format(get_current_time_string())
        if os.path.exists(self.tb_output_dir)!=True:
            os.makedirs(self.tb_output_dir)
        
        self.writer = SummaryWriter(self.tb_output_dir)

    def add_graph(self, model, inputs):
        # show model graph;
        self.writer.add_graph(model, inputs)
        self.writer.close()
    
    def add_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def add_scalars(self, scalar_dict, step):
        assert type(scalar_dict)==dict, "scalar_dict arg must be of type dictionary!"
        for tag, value in scalar_dict.items():
            self.writer.add_scalar(tag, value, step)

    def add_image(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def add_images(self, tag, images, step):
        # make a grid of images;
        img_grid = torchvision.utils.make_grid(images)
        self.writer.add_image(tag, img_grid, step)

    def add_text(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)


    def add_embedding(self, features, labels, metadata):
        pass


    def close(self):
        """
        Closes the tensorboard summary writer.
        """
        self.writer.close()