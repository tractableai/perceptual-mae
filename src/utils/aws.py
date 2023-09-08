import configparser
import boto3
from urllib.parse import urlparse
from botocore.exceptions import ClientError
from functools import partial, wraps
import tempfile 
import os
import logging
from io import BytesIO
import time 
from datetime import datetime, timedelta
import torch 
from omegaconf import OmegaConf

# Use S3Bucket helper + configuration from tcomms

log = logging.getLogger(__name__)

# AWS File IO
class S3Client():
    def __init__(self, config, *, remote_experiment_data_dir):
        self.config = config
        # setup aws env variables:
        self.aws_key_region = self.config.user_config.aws_key_region
        self.data_bucket_region = self.config.user_config.data_bucket_region
        self.image_bucket_region = self.config.user_config.image_bucket_region
        self.set_aws_env_variables(self.aws_key_region, self.data_bucket_region, self.image_bucket_region)
        
        self.s3 = self.set_up_boto_client()

        parts = remote_experiment_data_dir.replace('s3://', '').split('/')
        self.output_bucket = parts[0]
        self.output_dir = '/'.join(parts[1:])

    def set_aws_env_variables(self, key_region, data_region, images_region):
        aws_config_parser = configparser.ConfigParser()
        aws_config_parser.read("{}/.aws/credentials".format(os.environ["HOME"]))
        os.environ["AWS_ACCESS_KEY_ID"] = aws_config_parser.get(
            key_region, "aws_access_key_id"
        )
        os.environ["AWS_SECRET_ACCESS_KEY"] = aws_config_parser.get(
            key_region, "aws_secret_access_key"
        )
        os.environ["AWS_DATA_REGION"] = data_region
        os.environ["AWS_REGION"] = data_region
        os.environ["AWS_IMAGES_REGION"] = images_region
        os.environ["AWS_LOG_LEVEL"] = "3"
        os.environ["S3_ENDPOINT"] = f"s3.{data_region}.amazonaws.com"
        os.environ["S3_USE_HTTPS"] = "1"
        os.environ["S3_VERIFY_SSL"] = "1"

    def set_up_boto_client(self, region="eu"):
        aws_config_parser = configparser.ConfigParser()
        aws_config_parser.read("{}/.aws/credentials".format(os.environ["HOME"]))
        aws_access_key_id = aws_config_parser.get(region, "aws_access_key_id")
        aws_secret_access_key = aws_config_parser.get(region, "aws_secret_access_key")

        s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        return s3

    def upload_config_to_s3(self, config):
        with tempfile.NamedTemporaryFile(mode='w+b') as f:
            OmegaConf.save(config=config, f=f.name)
            self.s3.upload_file(os.path.join(tempfile.gettempdir(), f.name),
                                Bucket=self.output_bucket,
                                Key=os.path.join(self.output_dir,
                                                 f'{self.config.user_config.experiment_name}_config.yaml'))

    def upload_weights_to_s3(self, model, file_name):
        with tempfile.NamedTemporaryFile(mode='w+b') as f:
            torch.save(model.state_dict(), f)
            self.s3.upload_file(os.path.join(tempfile.gettempdir(), f.name),
                                Bucket=self.output_bucket,
                                Key=os.join(self.output_dir, f'{file_name}.path'))
        
    def upload_to_s3_from_disk(self, file, file_path):
        try:
            self.s3.put_object(Body=file,
                               Bucket=self.output_bucket,
                               Key=file_path)
        except Exception as e:
            print('file not uploaded, due to;')
            print(e)


# ----------------------------------------------- AWS utils -----------------------------------------------
def get_current_time_string():
    t = time.time()
    t = datetime.fromtimestamp(int(t))
    t = t + timedelta(hours=1)
    return datetime.strftime(t, "%y%m%d-%H%M%S")
