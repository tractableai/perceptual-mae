import argparse 

"""
The following method adds default args whilst giving the freedom to add new ones in your function call as opts.
These opts will be stored in the args namespace.
"""
class Args:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.add_core_args()

    def get_parser(self):
        return self.parser

    def add_core_args(self):
        self.parser.add_argument_group("Core Arguments")
        # TODO: Add Help flag here describing MMF Configuration
        # and point to configuration documentation
        self.parser.add_argument(
            "-co",
            "--config_override",
            type=str,
            default=None,
            help="Use to override config from command line directly",
        )
        self.parser.add_argument('--default_config_path', type=str, default='./configs/default.yaml', help='the path to default config')
        self.parser.add_argument('--model_config_path', type=str, default='./configs/models/masked_image.yaml', help='the path to model config')
        self.parser.add_argument('--dataset_config_path', type=str, default='./configs/datasets/public_vision.yaml', help='the path to dataset config')
        self.parser.add_argument('--user_config_path', type=str, default='./configs/user/sample.yaml', help='the path to user config')

        self.parser.add_argument('--model_variant_config_path', type=str, default=None, help='overrides to model config')
        self.parser.add_argument('--experiment_name_from_variant', type=bool, default=True,
                                 help='if set to true, experiment name is based on filename of model variant config path')
        self.parser.add_argument('--experiment_tag', type=str, default=None,
                                 help='optional text string pre-pended to experiment name')

        # This is needed to support torch.distributed.launch
        self.parser.add_argument(
            "--local_rank", type=int, default=None, help="Local rank of the argument"
        )
        self.parser.add_argument(
            "opts",
            default=None,
            nargs=argparse.REMAINDER,
            help="Modify config options from command line",
        )


args = Args()