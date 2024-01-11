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
from typing import List, Tuple

import numpy as np
import torch
from nnunet.network_architecture.generic_modular_residual_UNet import (
    FabiansUNet, get_default_network_config)
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.data_augmentation.custom_transforms import (
    Convert2DTo3DTransform, Convert3DTo2DTransform,
    ConvertSegmentationToRegionsTransform, MaskTransform)
from nnunet.training.data_augmentation.data_augmentation_moreDA import \
    get_moreDA_augmentation
from nnunet.training.data_augmentation.default_data_augmentation import \
    get_patch_size
from nnunet.training.data_augmentation.downsampling import \
    DownsampleSegForDSTransform2
from nnunet.training.data_augmentation.pyramid_augmentations import (
    ApplyRandomBinaryOperatorTransform, MoveSegAsOneHotToData,
    RemoveRandomConnectedComponentFromOneHotEncodingTransform)
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.learning_rate.poly_lr import poly_lr
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.training.network_training.nnUNetTrainerV2_DDP import \
    nnUNetTrainerV2_DDP
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.set_n_proc_DA import get_allowed_n_proc_DA
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from torch import distributed, nn
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import _LRScheduler

from batchgenerators.dataloading.nondet_multi_threaded_augmenter import \
    NonDetMultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import (AbstractTransform,
                                                            Compose)
from batchgenerators.transforms.channel_selection_transforms import \
    SegChannelSelectionTransform
from batchgenerators.transforms.color_transforms import (
    BrightnessTransform, ContrastAugmentationTransform, GammaTransform)
from batchgenerators.transforms.local_transforms import (
    BrightnessGradientAdditiveTransform, LocalGammaTransform)
from batchgenerators.transforms.noise_transforms import (
    BlankRectangleTransform, GaussianBlurTransform, GaussianNoiseTransform,
    MedianFilterTransform, SharpeningTransform)
from batchgenerators.transforms.resample_transforms import \
    SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import (
    MirrorTransform, Rot90Transform, SpatialTransform, TransposeAxesTransform)
from batchgenerators.transforms.utility_transforms import (
    NumpyToTensor, OneOfTransform, RemoveLabelTransform, RenameTransform)
from batchgenerators.utilities.file_and_folder_operations import *
from batchgenerators.utilities.file_and_folder_operations import (
    join, maybe_mkdir_p)


class nnUNetTrainerV2_DA5_DDP(nnUNetTrainerV2_DDP):
    def __init__(self, plans_file, fold, local_rank, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, distribute_batch_size=False, fp16=False):
        super().__init__(plans_file, fold, local_rank, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, distribute_batch_size, fp16)
        self.do_mirroring = True
        self.mirror_axes = None
        proc = get_allowed_n_proc_DA()
        self.num_proc_DA = proc if proc is not None else 12
        self.num_cached = 4
        self.regions_class_order = self.regions = None

    def setup_DA_params(self):
        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]

        self.data_aug_params = dict()
        self.data_aug_params['scale_range'] = (0.7, 1.43)
        self.data_aug_params['num_threads'] = self.num_proc_DA
        # we need this because this is adapted in the cascade
        self.data_aug_params['selected_seg_channels'] = None
        self.data_aug_params["move_last_seg_chanel_to_data"] = False

        if self.threeD:
            if self.do_mirroring:
                self.mirror_axes = (0, 1, 2)
                self.data_aug_params['do_mirror'] = True  # needed for inference
                self.data_aug_params['mirror_axes'] = (0, 1, 2)  # needed for inference
            else:
                self.data_aug_params['mirror_axes'] = tuple()
                self.data_aug_params['do_mirror'] = False

            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)

            if self.do_dummy_2D_aug:
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["dummy_2D"] = True
                self.data_aug_params["rotation_x"] = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
        else:
            if self.do_mirroring:
                self.mirror_axes = (0, 1)
                self.data_aug_params['mirror_axes'] = (0, 1)  # needed for inference
                self.data_aug_params['do_mirror'] = True  # needed for inference
            else:
                self.data_aug_params['mirror_axes'] = tuple()
                self.data_aug_params['do_mirror'] = False  # needed for inference


            self.do_dummy_2D_aug = False

            self.data_aug_params['rotation_x'] = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-0. / 360 * 2. * np.pi, 0. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-0. / 360 * 2. * np.pi, 0. / 360 * 2. * np.pi)

        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])

    def get_train_transforms(self) -> List[AbstractTransform]:
        # used for transpost and rot90
        matching_axes = np.array([sum([i == j for j in self.patch_size]) for i in self.patch_size])
        valid_axes = list(np.where(matching_axes == np.max(matching_axes))[0])

        tr_transforms = []

        if self.data_aug_params['selected_seg_channels'] is not None:
            tr_transforms.append(SegChannelSelectionTransform(self.data_aug_params['selected_seg_channels']))

        if self.do_dummy_2D_aug:
            ignore_axes = (0,)
            tr_transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = self.patch_size[1:]
        else:
            patch_size_spatial = self.patch_size
            ignore_axes = None

        tr_transforms.append(
            SpatialTransform(
                patch_size_spatial,
                patch_center_dist_from_border=None,
                do_elastic_deform=False,
                do_rotation=True,
                angle_x=self.data_aug_params["rotation_x"],
                angle_y=self.data_aug_params["rotation_y"],
                angle_z=self.data_aug_params["rotation_z"],
                p_rot_per_axis=0.5,
                do_scale=True,
                scale=self.data_aug_params['scale_range'],
                border_mode_data="constant",
                border_cval_data=0,
                order_data=3,
                border_mode_seg="constant",
                border_cval_seg=-1,
                order_seg=1,
                random_crop=False,
                p_el_per_sample=0.2,
                p_scale_per_sample=0.2,
                p_rot_per_sample=0.4,
                independent_scale_for_each_axis=True,
            )
        )

        if self.do_dummy_2D_aug:
            tr_transforms.append(Convert2DTo3DTransform())

        if np.any(matching_axes > 1):
            tr_transforms.append(
                Rot90Transform(
                    (0, 1, 2, 3), axes=valid_axes, data_key='data', label_key='seg', p_per_sample=0.5
                ),
            )

        if np.any(matching_axes > 1):
            tr_transforms.append(
                TransposeAxesTransform(valid_axes, data_key='data', label_key='seg', p_per_sample=0.5)
            )

        tr_transforms.append(OneOfTransform([
            MedianFilterTransform(
                (2, 8),
                same_for_each_channel=False,
                p_per_sample=0.2,
                p_per_channel=0.5
            ),
            GaussianBlurTransform((0.3, 1.5),
                                  different_sigma_per_channel=True,
                                  p_per_sample=0.2,
                                  p_per_channel=0.5)
        ]))

        tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))

        tr_transforms.append(BrightnessTransform(0,
                                                 0.5,
                                                 per_channel=True,
                                                 p_per_sample=0.1,
                                                 p_per_channel=0.5
                                                 )
                             )

        tr_transforms.append(OneOfTransform(
            [
                ContrastAugmentationTransform(
                    contrast_range=(0.5, 2),
                    preserve_range=True,
                    per_channel=True,
                    data_key='data',
                    p_per_sample=0.2,
                    p_per_channel=0.5
                ),
                ContrastAugmentationTransform(
                    contrast_range=(0.5, 2),
                    preserve_range=False,
                    per_channel=True,
                    data_key='data',
                    p_per_sample=0.2,
                    p_per_channel=0.5
                ),
            ]
        ))

        tr_transforms.append(
            SimulateLowResolutionTransform(zoom_range=(0.25, 1),
                                           per_channel=True,
                                           p_per_channel=0.5,
                                           order_downsample=0,
                                           order_upsample=3,
                                           p_per_sample=0.15,
                                           ignore_axes=ignore_axes
                                           )
        )

        tr_transforms.append(
            GammaTransform((0.7, 1.5), invert_image=True, per_channel=True, retain_stats=True, p_per_sample=0.1))
        tr_transforms.append(
            GammaTransform((0.7, 1.5), invert_image=True, per_channel=True, retain_stats=True, p_per_sample=0.1))

        if self.do_mirroring:
            tr_transforms.append(MirrorTransform(self.mirror_axes))

        tr_transforms.append(
            BlankRectangleTransform([[max(1, p // 10), p // 3] for p in self.patch_size],
                                    rectangle_value=np.mean,
                                    num_rectangles=(1, 5),
                                    force_square=False,
                                    p_per_sample=0.4,
                                    p_per_channel=0.5
                                    )
        )

        tr_transforms.append(
            BrightnessGradientAdditiveTransform(
                lambda x, y: np.exp(np.random.uniform(np.log(x[y] // 6), np.log(x[y]))),
                (-0.5, 1.5),
                max_strength=lambda x, y: np.random.uniform(-5, -1) if np.random.uniform() < 0.5 else np.random.uniform(1, 5),
                mean_centered=False,
                same_for_all_channels=False,
                p_per_sample=0.3,
                p_per_channel=0.5
            )
        )

        tr_transforms.append(
            LocalGammaTransform(
                lambda x, y: np.exp(np.random.uniform(np.log(x[y] // 6), np.log(x[y]))),
                (-0.5, 1.5),
                lambda: np.random.uniform(0.01, 0.8) if np.random.uniform() < 0.5 else np.random.uniform(1.5, 4),
                same_for_all_channels=False,
                p_per_sample=0.3,
                p_per_channel=0.5
            )
        )

        tr_transforms.append(
            SharpeningTransform(
                strength=(0.1, 1),
                same_for_each_channel=False,
                p_per_sample=0.2,
                p_per_channel=0.5
            )
        )

        if any(self.use_mask_for_norm.values()):
            tr_transforms.append(MaskTransform(self.use_mask_for_norm, mask_idx_in_seg=0, set_outside_to=0))

        tr_transforms.append(RemoveLabelTransform(-1, 0))

        if self.data_aug_params["move_last_seg_chanel_to_data"]:
            all_class_labels = np.arange(1, self.num_classes)
            tr_transforms.append(MoveSegAsOneHotToData(1, all_class_labels, 'seg', 'data'))
            if self.data_aug_params["cascade_do_cascade_augmentations"]:
                tr_transforms.append(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(all_class_labels), 0)),
                        p_per_sample=0.4,
                        key="data",
                        strel_size=(1, 8),
                        p_per_label=1
                    )
                )

                tr_transforms.append(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(all_class_labels), 0)),
                        key="data",
                        p_per_sample=0.2,
                        fill_with_other_class_p=0.15,
                        dont_do_if_covers_more_than_X_percent=0
                    )
                )

        tr_transforms.append(RenameTransform('seg', 'target', True))

        if self.regions is not None:
            tr_transforms.append(ConvertSegmentationToRegionsTransform(self.regions, 'target', 'target'))

        if self.deep_supervision_scales is not None:
            tr_transforms.append(
                DownsampleSegForDSTransform2(self.deep_supervision_scales, 0, input_key='target',
                                             output_key='target')
            )

        tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
        return tr_transforms

    def get_val_transforms(self) -> List[AbstractTransform]:
        val_transforms = list()
        val_transforms.append(RemoveLabelTransform(-1, 0))

        if self.data_aug_params['selected_seg_channels'] is not None:
            val_transforms.append(SegChannelSelectionTransform(self.data_aug_params['selected_seg_channels']))

        if self.data_aug_params["move_last_seg_chanel_to_data"]:
            all_class_labels = np.arange(1, self.num_classes)
            val_transforms.append(MoveSegAsOneHotToData(1, all_class_labels, 'seg', 'data'))
        val_transforms.append(RenameTransform('seg', 'target', True))

        if self.regions is not None:
            val_transforms.append(ConvertSegmentationToRegionsTransform(self.regions, 'target', 'target'))

        if self.deep_supervision_scales is not None:
            val_transforms.append(
                DownsampleSegForDSTransform2(
                    self.deep_supervision_scales, 0, input_key='target',
                    output_key='target')
            )

        val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
        return val_transforms

    def wrap_transforms(self, dataloader_train, dataloader_val, train_transforms, val_transforms, train_seeds=None, val_seeds=None):
        tr_gen = NonDetMultiThreadedAugmenter(dataloader_train,
                                              Compose(train_transforms),
                                              self.num_proc_DA,
                                              self.num_cached,
                                              seeds=train_seeds,
                                              pin_memory=self.pin_memory)
        val_gen = NonDetMultiThreadedAugmenter(dataloader_val,
                                               Compose(val_transforms),
                                               self.num_proc_DA // 2,
                                               self.num_cached,
                                               seeds=val_seeds,
                                               pin_memory=self.pin_memory)
        return tr_gen, val_gen

    def initialize(self, training=True, force_load_plans=False):

        """
        :param training:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    if self.local_rank == 0:
                        print("unpacking dataset")
                        unpack_dataset(self.folder_with_preprocessed_data)
                        print("done")
                    distributed.barrier()
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                # setting weights for deep supervision losses
                net_numpool = len(self.net_num_pool_op_kernel_sizes)

                # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
                # this gives higher resolution outputs more weight in the loss
                weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

                # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
                mask = np.array([True if i < net_numpool - 1 else False for i in range(net_numpool)])
                weights[~mask] = 0
                weights = weights / weights.sum()
                self.ds_loss_weights = weights

                seeds_train = np.random.random_integers(0, 99999, self.data_aug_params.get('num_threads'))
                seeds_val = np.random.random_integers(0, 99999, max(self.data_aug_params.get('num_threads') // 2, 1))
                print("seeds train", seeds_train)
                print("seeds_val", seeds_val)

                tr_transforms = self.get_train_transforms()
                val_transforms = self.get_val_transforms()
                self.tr_gen, self.val_gen = self.wrap_transforms(self.dl_tr, 
                                                                 self.dl_val, 
                                                                 tr_transforms, 
                                                                 val_transforms, 
                                                                 train_seeds=seeds_train, 
                                                                 val_seeds=seeds_val)

                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()
            self.network = DDP(self.network, device_ids=[self.local_rank])

        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True


class nnUNetTrainerV2_DA5_warmup_increasing_lr_DDP(nnUNetTrainerV2_DA5_DDP):
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


class nnUNetTrainerV2_DA5_warmupsegheads_DDP(nnUNetTrainerV2_DA5_warmup_increasing_lr_DDP):
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