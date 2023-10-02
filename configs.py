"""Defines configuration parameters for the whole model and dataset.
"""
import argparse
import json
import os
from collections import OrderedDict
from datetime import datetime


def get_current_datetime() -> str:
    """Converts the current datetime to string.
    Returns:
        String version of current datetime of the form: %y%m%d_%H%M%S.
    """
    return datetime.now().strftime("%y%m%d_%H%M%S")


def mkdirs(paths) -> None:
    """Creates directories represented by paths argument.
    Args:
        paths: Either list of paths or a single path.
    """
    if isinstance(paths, str):
        os.makedirs(paths, exist_ok=True)
    else:
        for path in paths:
            os.makedirs(path, exist_ok=True)


class Config:
    """Configuration class.
    Attributes:
        args: 命令行参数.
        root: json路径
        gpu_ids: list.
        params: 参数字典用于保存jsonjson.
        name: 实验名.
        phase: Either train or val.
        distributed: Whether the computation will be distributed among multiple GPUs or not.
        log: Path to logs.
        tb_logger: Tensorboard logging directory.
        results: Validation results directory.
        checkpoint: Model checkpoints directory.
        resume_state: The path to load the network.
        dataset_name: The name of dataset.
        dataroot: The path to dataset.
        batch_size: Batch size.
        num_workers: The number of processes for multi-process data loading.
        use_shuffle: Either to shuffle the training data or not.
        train_min_date: Minimum date starting from which to read the data for training.
        train_max_date: Maximum date until which to read the date for training.
        val_min_date: Minimum date starting from which to read the data for validation.
        val_max_date: Maximum date until which to read the date for validation.
        train_subset_min_date: Minimum date starting from which to read the data for model evaluation on train subset.
        train_subset_max_date: Maximum date starting until which to read the data for model evaluation on train subset.
        variables: A list of WeatherBench variables.
        finetune_norm: Whetehr to fine-tune or train from scratch.
        in_channel: The number of channels of input tensor of U-Net.
        out_channel: The number of channels of output tensor of U-Net.
        inner_channel: Timestep embedding dimension.
        norm_groups: The number of groups for group normalization.
        channel_multiplier: A tuple specifying the scaling factors of channels.
        attn_res: A tuple of spatial dimensions indicating in which resolutions to use self-attention layer.
        res_blocks: The number of residual blocks.
        dropout: Dropout probability.
        init_method: NN weight initialization method. One of normal, kaiming or orthogonal inisializations.
        train_schedule: Defines the type of beta schedule for training.
        train_n_timestep: Number of diffusion timesteps for training.
        train_linear_start: Minimum value of the linear schedule for training.
        train_linear_end: Maximum value of the linear schedule for training.
        val_schedule: Defines the type of beta schedule for validation.
        val_n_timestep: Number of diffusion timesteps for validation.
        val_linear_start: Minimum value of the linear schedule for validation.
        val_linear_end: Maximum value of the linear schedule for validation.
        test_schedule: Defines the type of beta schedule for inference.
        test_n_timestep: Number of diffusion timesteps for inference.
        test_linear_start: Minimum value of the linear schedule for inference.
        test_linear_end: Maximum value of the linear schedule for inference.
        conditional: Whether to condition on INTERPOLATED image or not.
        diffusion_loss: Either 'l1' or 'l2'.
        n_iter: Number of iterations to train.
        val_freq: Validation frequency.
        save_checkpoint_freq: Model checkpoint frequency.
        print_freq: The frequency of displaying training information.
        n_val_vis: Number of data points to visualize.
        val_vis_freq: Validation data points visualization frequency.
        sample_size: Numer of SR images to generate to calculate metrics.
        optimizer_type: The name of optimization algorithm. Supported values are 'adam', 'adamw'.
        amsgrad: Whether to use the AMSGrad variant of optimizer.
        lr: The learning rate.
        experiments_root: The path to experiment.
        tranform_monthly: Whether to apply transformation monthly or on the whole dataset.
        height: U-Net input tensor height value.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = self.args.config
        self.gpu_ids = self.args.gpu_ids
        # self.members=self.args.members
        self.params = {}
        self.experiments_root = None
        self.__parse_configs()
        self.name = self.params["name"]
        self.phase = self.params["phase"]
        self.gpu_ids = self.params["gpu_ids"]
        self.distributed = self.params["distributed"]
        self.log = self.params["path"]["log"]
        self.tb_logger = self.params["path"]["tb_logger"]
        self.results = self.params["path"]["results"]
        self.checkpoint = self.params["path"]["checkpoint"]
        self.resume_state = self.params["path"]["resume_state"]
        self.batch_size = self.params["data"]["batch_size"]
        self.num_workers = self.params["data"]["num_workers"]
        self.use_shuffle = self.params["data"]["use_shuffle"]
        self.height = self.params["data"]["height"]
        self.finetune_norm = self.params["model"]["finetune_norm"]
        self.in_channel = self.params["model"]["unet"]["in_channel"]
        self.out_channel = self.params["model"]["unet"]["out_channel"]
        self.inner_channel = self.params["model"]["unet"]["inner_channel"]
        self.norm_groups = self.params["model"]["unet"]["norm_groups"]
        self.channel_multiplier = self.params["model"]["unet"]["channel_multiplier"]
        self.attn_res = self.params["model"]["unet"]["attn_res"]
        self.res_blocks = self.params["model"]["unet"]["res_blocks"]
        self.dropout = self.params["model"]["unet"]["dropout"]
        self.init_method = self.params["model"]["unet"]["init_method"]
        self.train_schedule = self.params["model"]["beta_schedule"]["train"]["schedule"]
        self.train_n_timestep = self.params["model"]["beta_schedule"]["train"]["n_timestep"]
        self.train_linear_start = self.params["model"]["beta_schedule"]["train"]["linear_start"]
        self.train_linear_end = self.params["model"]["beta_schedule"]["train"]["linear_end"]
        self.val_schedule = self.params["model"]["beta_schedule"]["val"]["schedule"]
        self.val_n_timestep = self.params["model"]["beta_schedule"]["val"]["n_timestep"]
        self.val_linear_start = self.params["model"]["beta_schedule"]["val"]["linear_start"]
        self.val_linear_end = self.params["model"]["beta_schedule"]["val"]["linear_end"]
        self.test_schedule = self.params["model"]["beta_schedule"]["test"]["schedule"]
        self.test_n_timestep = self.params["model"]["beta_schedule"]["test"]["n_timestep"]
        self.test_linear_start = self.params["model"]["beta_schedule"]["test"]["linear_start"]
        self.test_linear_end = self.params["model"]["beta_schedule"]["test"]["linear_end"]
        self.conditional = self.params["model"]["diffusion"]["conditional"]
        self.diffusion_loss = self.params["model"]["diffusion"]["loss"]
        self.n_iter = self.params["training"]["epoch_n_iter"]
        self.val_freq = self.params["training"]["val_freq"]
        self.save_checkpoint_freq = self.params["training"]["save_checkpoint_freq"]
        self.print_freq = self.params["training"]["print_freq"]
        self.n_val_vis = self.params["training"]["n_val_vis"]
        self.val_vis_freq = self.params["training"]["val_vis_freq"]
        self.sample_size = self.params["training"]["sample_size"]
        self.optimizer_type = self.params["training"]["optimizer"]["type"]
        self.amsgrad = self.params["training"]["optimizer"]["amsgrad"]
        self.lr = self.params["training"]["optimizer"]["lr"]

    def __parse_configs(self):
        """Reads configureation json file and stores in params attribute."""
        json_str = ""
        with open(self.root, "r") as f:
            for line in f:
                json_str = f"{json_str}{line.split('//')[0]}\n"

        self.params = json.loads(json_str, object_pairs_hook=OrderedDict)

        if not self.params["path"]["resume_state"]:
            self.experiments_root = os.path.join("experiments", f"{self.params['name']}_{get_current_datetime()}")
        else:
            self.experiments_root = "/".join(self.params["path"]["resume_state"].split("/")[:-2])

        for key, path in self.params["path"].items():
            if not key.startswith("resume"):
                self.params["path"][key] = os.path.join(self.experiments_root, path)
                mkdirs(self.params["path"][key])

        if self.gpu_ids:
            self.params["gpu_ids"] = [int(gpu_id) for gpu_id in self.gpu_ids.split(",")]
            gpu_list = self.gpu_ids
        else:
            gpu_list = ",".join(str(x) for x in self.params["gpu_ids"])

        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
        self.params["distributed"] = True if len(gpu_list) > 1 else False

    def __getattr__(self, item):
        """Returns None when attribute doesn't exist.
        Args:
            item: Attribute to retrieve.
        Returns:
            None
        """
        return None

    def get_hyperparameters_as_dict(self):
        """Returns dictionary containg parsed configuration json file.
        """
        return self.params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="JSON file for configuration")
    parser.add_argument("-p", "--phase", type=str, choices=["train", "val"],
                        help="Run either training or validation(inference).", default="train")
    parser.add_argument("-gpu", "--gpu_ids", type=str, default=None)
    args = parser.parse_args()
    configs = Config(args)
    print(configs.name)