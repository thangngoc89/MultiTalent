#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
from nnunet.network_architecture.generic_modular_residual_UNet import (
    FabiansUNet, get_default_network_config)
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.data_augmentation.data_augmentation_moreDA import \
    get_moreDA_augmentation
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.learning_rate.poly_lr import poly_lr
from nnunet.training.network_training.nnUNet_variants.architectural_variants.nnUNetTrainerV2_ResencUNet_SimonsInit import \
    init_last_bn_before_add_to_0
from nnunet.training.network_training.nnUNet_variants.transformers.nnUNetTrainerV2_SwinUNETR_ddp import \
    nnUNetTrainerV2_swinunetr_adam_ddp_lr5e4
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.training.network_training.nnUNetTrainerV2_DDP import \
    nnUNetTrainerV2_DDP
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.set_n_proc_DA import get_allowed_n_proc_DA
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from torch import distributed
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import _LRScheduler

from batchgenerators.utilities.file_and_folder_operations import *


class nnUNetTrainerV2_warmup_increasing_lr_DDP(nnUNetTrainerV2_DDP):
    def __init__(self, plans_file, fold, local_rank, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, distribute_batch_size=False, fp16=False):
        super().__init__(plans_file, fold, local_rank, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, distribute_batch_size, fp16)
        self.warmup_duration = 50
        self.max_num_epochs = 1000 + self.warmup_duration

    def maybe_update_lr(self, epoch=None):
        if self.epoch < self.warmup_duration:
            # epoch 49 is max
            # we increase lr linearly from 0 to initial_lr
            lr = (self.epoch + 1) / self.warmup_duration * self.initial_lr
            self.optimizer.param_groups[0]['lr'] = lr
            self.print_to_log_file("epoch:", self.epoch, "lr:", lr)
        else:
            if epoch is not None:
                ep = epoch - (self.warmup_duration - 1)
            else:
                ep = self.epoch - (self.warmup_duration - 1)
            assert ep > 0, "epoch must be >0"
            return super().maybe_update_lr(ep)

    def on_epoch_end(self) -> bool:
        network = self.network.module if isinstance(self.network, DDP) else self.network
        self.print_to_log_file(network.conv_blocks_context[0].blocks[0].conv.weight[0, 0, 0])
        ret = super().on_epoch_end()
        return ret


class nnUNetTrainerV2_warmupsegheads_DDP(nnUNetTrainerV2_warmup_increasing_lr_DDP):
    def __init__(self, plans_file, fold, local_rank, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, distribute_batch_size=False, fp16=False):
        super().__init__(plans_file, fold, local_rank, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, distribute_batch_size, fp16)
        self.num_epochs_sgd_warmup = 10  # lin increase warmup
        self.warmup_max_lr = 5e-4  # for heads
        self.warmup_duration = 50  # this is for the seg heads
        self.max_num_epochs = 1000 + self.num_epochs_sgd_warmup + self.warmup_duration

    # def process_plans(self, plans):
    #     super().process_plans(plans)
    #     self.patch_size = [64,96,96]
    def initialize(self, training=True, force_load_plans=False):
        # here we call initialize_optimizer_and_scheduler with seg heads only
        super().initialize(training, force_load_plans)
        if training:
            self.initialize_optimizer_and_scheduler(True)

    def maybe_update_lr(self, epoch=None):
        print(self.warmup_duration)
        if self.epoch < self.warmup_duration:
            # we increase lr linearly from 0 to self.warmup_max_lr
            lr = (self.epoch + 1) / self.warmup_duration * self.warmup_max_lr
            self.optimizer.param_groups[0]['lr'] = lr
            self.lr = lr
            self.print_to_log_file("epoch:", self.epoch, "lr for heads:", lr)
        elif self.warmup_duration <= self.epoch < self.warmup_duration + self.num_epochs_sgd_warmup:
            # we increase lr linearly from 0 to self.initial_lr
            lr = (self.epoch - self.warmup_duration + 1) / self.num_epochs_sgd_warmup * self.initial_lr
            self.optimizer.param_groups[0]['lr'] = lr
            self.print_to_log_file("epoch:", self.epoch, "lr now lin increasing whole network:", lr)
        else:
            if epoch is not None:
                ep = epoch - (self.warmup_duration + self.num_epochs_sgd_warmup - 1)
            else:
                ep = self.epoch - (self.warmup_duration + self.num_epochs_sgd_warmup - 1)
            assert ep > 0, "epoch must be >0"

            self.optimizer.param_groups[0]['lr'] = poly_lr(ep,
                                                           self.max_num_epochs - self.num_epochs_sgd_warmup - self.warmup_duration,
                                                           self.initial_lr, 0.9)
            self.print_to_log_file("lr was set to:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def on_epoch_end(self) -> bool:
        network = self.network.module if isinstance(self.network, DDP) else self.network
        self.print_to_log_file(network.conv_blocks_context[0].blocks[0].conv.weight[0, 0, 0])
        if self.epoch == self.warmup_duration:
            self.print_to_log_file("now train whole network")
            self.initialize_optimizer_and_scheduler(seg_heads_only=False)
        ret = super().on_epoch_end()
        return ret

    def initialize_optimizer_and_scheduler(self, seg_heads_only=False):
        assert self.network is not None, "self.initialize_network must be called first"
        network = self.network.module if isinstance(self.network, DDP) else self.network
            

        if seg_heads_only:
            parameters = network.seg_outputs.parameters()
            self.optimizer = torch.optim.AdamW(parameters, 3e-3, weight_decay=self.weight_decay, amsgrad=True)
        else:
            parameters = network.parameters()
            self.optimizer = torch.optim.SGD(parameters, self.initial_lr, weight_decay=self.weight_decay,
                                             momentum=0.99, nesterov=True)

        self.lr_scheduler = None

    def load_checkpoint_ram(self, checkpoint, train=True):
        """
        we need to have the correct parameters in the optimizer  (warmup etc)

        :param checkpoint:
        :param train:
        :return:
        """
        if not self.was_initialized:
            self.initialize(train)

        network = self.network.module if isinstance(self.network, DDP) else self.network
        new_state_dict = OrderedDict()
        curr_state_dict_keys = list(network.state_dict().keys())
        # if state dict comes form nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        for k, value in checkpoint['state_dict'].items():
            key = k

            if key not in curr_state_dict_keys and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        if self.fp16:
            self._maybe_init_amp()
            if 'amp_grad_scaler' in checkpoint.keys():
                self.amp_grad_scaler.load_state_dict(checkpoint['amp_grad_scaler'])

        network.load_state_dict(new_state_dict)
        self.epoch = checkpoint['epoch']
        if train:
            # we need to have the correct parameters in the optimizer
            if self.epoch > self.warmup_duration: self.initialize_optimizer_and_scheduler(seg_heads_only=False)

            optimizer_state_dict = checkpoint['optimizer_state_dict']
            if optimizer_state_dict is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)

            if self.lr_scheduler is not None and hasattr(self.lr_scheduler, 'load_state_dict') and checkpoint[
                'lr_scheduler_state_dict'] is not None:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

            if issubclass(self.lr_scheduler.__class__, _LRScheduler):
                self.lr_scheduler.step(self.epoch)

        self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode, self.all_val_eval_metrics = checkpoint[
            'plot_stuff']

        # load best loss (if present)
        if 'best_stuff' in checkpoint.keys():
            self.best_epoch_based_on_MA_tr_loss, self.best_MA_tr_loss_for_patience, self.best_val_eval_criterion_MA = \
                checkpoint[
                    'best_stuff']

        # after the training is done, the epoch is incremented one more time in my old code. This results in
        # self.epoch = 1001 for old trained models when the epoch is actually 1000. This causes issues because
        # len(self.all_tr_losses) = 1000 and the plot function will fail. We can easily detect and correct that here
        if self.epoch != len(self.all_tr_losses):
            self.print_to_log_file("WARNING in loading checkpoint: self.epoch != len(self.all_tr_losses). This is "
                                   "due to an old bug and should only appear when you are loading old models. New "
                                   "models should have this fixed! self.epoch is now set to len(self.all_tr_losses)")
            self.epoch = len(self.all_tr_losses)
            self.all_tr_losses = self.all_tr_losses[:self.epoch]
            self.all_val_losses = self.all_val_losses[:self.epoch]
            self.all_val_losses_tr_mode = self.all_val_losses_tr_mode[:self.epoch]
            self.all_val_eval_metrics = self.all_val_eval_metrics[:self.epoch]

        self._maybe_init_amp()