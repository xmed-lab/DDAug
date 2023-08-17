import os
import numpy as np
import random
from copy import deepcopy
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.channel_selection_transforms import (
    SegChannelSelectionTransform,
    DataChannelSelectionTransform,
)
from batchgenerators.transforms.color_transforms import (
    BrightnessMultiplicativeTransform,
    BrightnessTransform,
    ContrastAugmentationTransform,
    GammaTransform,
)
from batchgenerators.transforms.noise_transforms import (
    GaussianBlurTransform,
    GaussianNoiseTransform,
)
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform, SpatialTransform
from batchgenerators.transforms.utility_transforms import (
    NumpyToTensor,
    RemoveLabelTransform,
    RenameTransform,
)

try:
    import nnunet
except:
    import nnformer as nnunet

from nnunet.training.data_augmentation.custom_transforms import (
    Convert3DTo2DTransform,
    Convert2DTo3DTransform,
)

from nnunet.training.data_augmentation.new_augments import (
    Affine,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    BezierCurveTransform_similar,
    FourierMixTransform,
    AdaptiveHistogramEqualizationImageFilter,
    LaplacianImageFilter,
    SobelEdgeDetectionImageFilter,
    InvertIntensityImageFilter,
    InterpolationTransform,
)
from nnunet.training.data_augmentation.downsampling import (
    DownsampleSegForDSTransform2,
    DownsampleSegForDSTransform3,
)

# change augment, A can have ABC, B can have AB, C can have nothign

rotate_left_bound = -30.0
rotate_right_bound = 30.0
tree_depth = 4

# prune_every_n_epochs = set([200, 400, 600, 800, 1000])
prune_every_n_epochs = set([200, 400, 600, 800, 1000])
prune_percentage = 0.25
prune_search_increment = 5  # this is in percentage
visited_times_threshold_to_uct_sample = [4, 1, 1, 1]
delete_epoch_range = [5, 2, 1, 1]

# visited_times_threshold_to_uct_sample = [4, 1, 1, 1, 1, 1, 1, 1, 1, 1,]
# delete_epoch_range = [5, 2, 2, 1, 1, 1, 1, 1, 1]

# 15, 200, 2000, 20000
# [20, 5, 2, 2]

# smaller will focus on *val change*, larger will smooth improving effect
eq3_beta = 0.3
# inverse scaling factor for exploitation, smaller to prioritize exploitation 0.0025
eq6_tau = 0.5
# ratio value, larger focus on tree layer same type nodeq communication, smaller focus on current nodeq
eq7_lambda = 0.5

"""
RUN 2 
eq6_tau = 0.6
eq7_lambda = 0

RUN 3 
eq6_tau = 0.6
eq7_lambda = 0
eq8_c1 = 1.6
"""

# larger for exploration
eq8_c1 = np.sqrt(2)
# scale node communication contribution
eq8_c2 = 0.0

search_space_population = [
    (
        ContrastAugmentationTransform,
        (("contrast_range", (0.5, 0.75), (0.75, 1), (1, 1.25), (1.25, 1.5)),),
        {"p_per_channel": 1, "p_per_sample": 0.8},
    ),
    (
        GammaTransform,
        (("gamma_range", (0.5, 0.75), (0.75, 1), (1, 1.25), (1.25, 1.5)),),
        {"per_channel": False, "retain_stats": True, "p_per_sample": 0.8},
    ),
    (
        BrightnessMultiplicativeTransform,
        (("multiplier_range", (0.5, 0.75), (0.75, 1), (1, 1.25), (1.25, 1.5)),),
        {"p_per_sample": 0.8},
    ),
    (
        GaussianNoiseTransform,
        (("noise_variance", (0, 0.1)),),
        {"p_per_channel": 1, "per_channel": False, "p_per_sample": 0.8},
    ),
    (
        GaussianBlurTransform,
        (("blur_sigma", (0.5, 0.75), (0.75, 1), (1, 1.25), (1.25, 1.5)),),
        {"p_per_channel": 1, "p_per_sample": 0.8},
    ),
    (
        SimulateLowResolutionTransform,
        (("zoom_range", (0.5, 0.75), (0.75, 1)),),
        {"per_channel": False, "p_per_channel": 1, "p_per_sample": 0.8},
    ),
    (
        Affine,  # this replaces zoom transform
        (("scale_factor", (0.5, 0.75), (0.75, 1), (1, 1.25), (1.25, 1.5)),),
        {"shear": 0, "rotate": 0, "translate_percent": 0},
    ),
    (OpticalDistortion,),
    (ElasticTransform,),
    (GridDistortion,),
    # (
    #     'expand',
    #     BezierCurveTransform_similar,
    #     FourierMixTransform,
    #     AdaptiveHistogramEqualizationImageFilter,
    #     LaplacianImageFilter,
    #     SobelEdgeDetectionImageFilter,
    #     InvertIntensityImageFilter
    # )
    # (InterpolationTransform,),
    # need CLAHE
]


class Node:
    def __init__(self, opt, level, **kwargs):
        self.node_q = 1
        self.delete_epoch_scope = delete_epoch_range[level] if level != -1 else 100
        self.children = []
        self.parent = None
        self.kwargs = kwargs
        self.call_record = []
        self.can_delete = False
        self.visited_times = 0
        self.opt = [opt] if not isinstance(opt, list) else opt
        self.encoding = f"{str(opt.__class__.__name__)}|{str(kwargs)}"

    def update_call_record(self, score_change):
        # score_change less than 0 is getting better
        # score_change greater than 0 is getting worse
        self.call_record.append(score_change)
        self.call_record = self.call_record[-self.delete_epoch_scope :]
        self.can_delete = (sum(self.call_record) >= 0) and len(self.call_record) == self.delete_epoch_scope
        return None

    def __str__(self):
        return f"node opt: {str(self.opt[0].__class__.__name__):<35} with: {self.kwargs}"

    def __call__(self, arg):
        return self.opt(arg)

    def to_dict(self):
        this_json = {
            "node_q": self.node_q,
            "visited_times": self.visited_times,
            "opt": str(self.opt[0]),
            "args": self.kwargs,
            "record": self.call_record,
            "encoding": self.encoding,
            "children": [each.to_dict() for each in self.children],
        }
        return this_json


class MCTS_Augment:
    def __init__(self, patch_size, param, deep_supervision_scales=None):
        self.depth = tree_depth
        self.eq3_beta = eq3_beta
        self.eq6_tau = eq6_tau
        self.eq8_c1 = eq8_c1
        self.eq8_c2 = eq8_c2
        self.eq7_lambda = eq7_lambda
        self.deep_supervision_scales = deep_supervision_scales
        self.patch_size = patch_size
        self.epochs = 0
        self.last_score = 2
        root_transform = []

        # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!

        if param.get("selected_data_channels") is not None:
            root_transform.append(DataChannelSelectionTransform(param.get("selected_data_channels")))
        if param.get("selected_seg_channels") is not None:
            root_transform.append(SegChannelSelectionTransform(param.get("selected_seg_channels")))
        if param.get("dummy_2D"):
            self.sim_low_res_args = True
            self.ignore_axes = (0,)
            patch_size = patch_size[1:]
            root_transform.append(Convert3DTo2DTransform())
        else:
            self.sim_low_res_args = False

        root_transform.extend(
            [
                # SegChannelSelectionTransform([0]),
                MirrorTransform((0, 1, 2), p_per_sample=0.5),
                SpatialTransform(
                    patch_size,
                    do_elastic_deform=False,
                    p_el_per_sample=-1,
                    do_scale=False,
                    p_scale_per_sample=-1,
                    # 4 basic augmentations
                    do_rotation=False,
                    p_rot_per_sample=0.5,
                    p_rot_per_axis=1,
                    angle_x=(
                        rotate_left_bound / 360 * 2.0 * np.pi,
                        rotate_right_bound / 360 * 2.0 * np.pi,
                    ),
                    angle_y=(
                        rotate_left_bound / 360 * 2.0 * np.pi,
                        rotate_right_bound / 360 * 2.0 * np.pi,
                    ),
                    angle_z=(
                        rotate_left_bound / 360 * 2.0 * np.pi,
                        rotate_right_bound / 360 * 2.0 * np.pi,
                    ),
                    random_crop=True,
                    patch_center_dist_from_border=[each // 3 for each in patch_size],
                    border_mode_data="constant",
                    # do we use 0 to fill background values?
                    border_cval_data=0,
                    order_data=3,
                    border_mode_seg="constant",
                    border_cval_seg=-1,
                    order_seg=1,
                ),
            ]
        )

        if param.get("dummy_2D"):
            print("enabled dummy2d on SimulateLowResolutionTransform")
            root_transform.append(Convert2DTo3DTransform())

        self.root = Node(root_transform, level=-1)
        self.tail = Node(
            [
                RemoveLabelTransform(-1, 0),
                RenameTransform("seg", "target", True),
                NumpyToTensor(["data", "target"], "float"),
            ],
            level=-1,
        )

        if self.deep_supervision_scales:
            print(f"MCTS using deep supervision, scales: {deep_supervision_scales}")
            self.tail.opt.insert(
                2,
                DownsampleSegForDSTransform2(
                    deep_supervision_scales, 0, input_key="target", output_key="target"
                ),
            )
        else:
            print("MCTS not using deep supervision")
        # now here we build tree
        self.expand_children(self.root, self.depth, search_space_population)
        self.traverse_path = [0 for _ in range(self.depth)]

    def to_dict(self):
        return self.root.to_dict()

    def build_train_gen(self, dataloader_train):
        train_gen = MultiThreadedAugmenter(
            dataloader_train,
            self.build_callable(),
            # this num_processes will keep value between 20 to 40
            num_processes=8,
            # num_processes=10,
            num_cached_per_queue=1,
            seeds=None,
            pin_memory=True,
        )
        return train_gen

    def build_callable(self):
        result = self.root.opt.copy()
        current_node = self.root
        for each_index in self.traverse_path:
            current_node = current_node.children[each_index]
            current_node.visited_times += 1
            result.extend(current_node.opt)
        result.extend(self.tail.opt)
        return Compose(result)

    def expand_children(self, node, num_recursion, population):
        for each_opt_index, each_opt in enumerate(population):
            # this is for next layer population, always remove current opt to avoid duplicates
            new_pop = population.copy()
            new_pop.pop(each_opt_index)

            if len(each_opt) == 1:
                each_opt[0].patch_size = self.patch_size
                each_opt = each_opt[0]
                new_node = Node(each_opt(), level=tree_depth - num_recursion)
                # only go down when num_recursion > 1, meaning still layers to build
                if num_recursion > 1:
                    self.expand_children(new_node, num_recursion - 1, new_pop)
                # record children with list
                node.children.append(new_node)
                new_node.parent = node
            elif isinstance(each_opt[0], str) and each_opt[0] == "expand":
                for each in each_opt[1:]:
                    each.patch_size = self.patch_size
                    new_node = Node(each(), level=tree_depth - num_recursion)
                    # only go down when num_recursion > 1, meaning still layers to build
                    if num_recursion > 1:
                        self.expand_children(new_node, num_recursion - 1, new_pop)
                    # record children with list
                    node.children.append(new_node)
                    new_node.parent = node
            else:
                each_opt[0].patch_size = self.patch_size
                each_opt, mag_space, kwargs = each_opt
                for (mag_name, *all_mag_space) in mag_space:
                    for each_bound in all_mag_space:
                        new_kwargs = {mag_name: each_bound}
                        new_kwargs.update(kwargs)
                        if self.sim_low_res_args and each_opt is SimulateLowResolutionTransform:
                            new_kwargs["ignore_axes"] = self.ignore_axes
                        new_node = Node(
                            each_opt(**new_kwargs), level=tree_depth - num_recursion, **new_kwargs
                        )
                        if num_recursion > 1:
                            self.expand_children(new_node, num_recursion - 1, new_pop)
                        node.children.append(new_node)
                        new_node.parent = node

    def find_all_children_from_list(self, list_of_nodes, filter_by_visited_times=False):
        result = []
        for each_node in list_of_nodes:
            children = each_node.children
            if filter_by_visited_times:
                children = filter(lambda each: each.visited_times > 0, children)
            result.extend(children)
        return result

    def find_same_type_mapping(self, list_of_nodes):
        same_type_mapping = {}
        for each_node in list_of_nodes:
            encoding = each_node.encoding
            if encoding not in same_type_mapping.keys():
                same_type_mapping[encoding] = list(
                    filter(lambda each_layer_node: encoding == each_layer_node.encoding, list_of_nodes)
                )
        return same_type_mapping

    def calculate_uct(self, node, same_type_mapping):
        if node.visited_times:
            part1 = node.node_q / node.visited_times
        else:
            part1 = 0.1

        if node.parent.visited_times and node.visited_times:
            part2 = self.eq8_c1 * np.sqrt(np.log10(node.parent.visited_times) / node.visited_times)
        else:
            part2 = 0.1

        same_type = same_type_mapping.get(node.encoding, [])
        if len(same_type) > 0:
            part3_g = sum(map(lambda each: each.node_q, same_type)) / len(same_type)
            part3 = self.eq7_lambda * part3_g + (1 - self.eq7_lambda) * node.node_q
            part3 *= self.eq8_c2
        else:
            part3 = 0.1

        return part1 + part2 + part3

    def create_mem_set(self, tree, all_tree_nodes, cut_off):
        do_not_prune_set = set()
        prune_node_mem_address = set()

        layer1 = [(each, self.calculate_uct(each, {})) for each in tree.root.children]
        layer1_to_prune = list(filter(lambda each: each[1] < cut_off, layer1))
        # when layer1 will have less than 5 node, only prune the lowest one
        if len(layer1) - len(layer1_to_prune) < 5:
            do_not_prune_set.update(map(lambda each: each[0], layer1_to_prune))

        # first use filter to get those with uct smaller than cutoff
        filtered_nodes = filter(
            lambda each: each[1] < cut_off and each[0] not in do_not_prune_set, all_tree_nodes
        )
        # then use map to get the mem id
        mem_ids = map(lambda each: hex(id(each[0])), filtered_nodes)
        prune_node_mem_address.update(mem_ids)
        return prune_node_mem_address

    def update_epoch(self, score):
        # score itself is ce - DICE, to simulate loss converging to 0, we use 1+score
        # would be same as ce + (1-DICE)
        out_str = "\n"
        self.epochs += 1

        # eq 3
        new_score = self.eq3_beta * self.last_score + (1 - self.eq3_beta) * score
        # eq 4 -> node score
        node_q = new_score / score

        current_node = self.root
        for layer_index, each_index in enumerate(self.traverse_path, 1):
            # update current node, and node Q
            current_node = current_node.children[each_index]
            current_node.node_q = node_q

            # score should get smaller
            # therefore if current - last > 0, means getting worse
            # if current - last < 0, means getting better
            current_node.update_call_record(score - self.last_score)

            # delete node from current layer if necessary
            node_layer_population = current_node.parent.children
            if len(node_layer_population) > 5 and current_node.can_delete:
                bad_node = node_layer_population.pop(each_index)
                out_str += (
                    f"removed node at layer: {layer_index} with index: {each_index}, object: {bad_node}\n"
                )
            elif current_node.can_delete:
                out_str += f"wanted to remove node but cannot, layer: {layer_index} with index: {each_index}, object: {current_node}\n"

        tree_layer_population_count = []
        new_traverse_path = []
        node_layer_population = self.root.children
        tree_layer_population = [self.root]
        for layer_index in range(self.depth):
            # uct need all the node of the TREE from this layer

            tree_layer_population = self.find_all_children_from_list(tree_layer_population)
            tree_layer_population_count.append(len(tree_layer_population))
            # it is possible that this layer doesnt have any children due to pruning
            if len(node_layer_population) == 0:
                break

            # only sample when average visited time of children is larger than threshold
            mean_visited_times = sum(map(lambda each: each.visited_times, node_layer_population)) / len(
                node_layer_population
            )

            if mean_visited_times > visited_times_threshold_to_uct_sample[layer_index]:
                raw = [
                    self.calculate_uct(each, self.find_same_type_mapping(tree_layer_population))
                    / self.eq6_tau
                    for each in node_layer_population
                ]
                exps = np.exp(raw)
                prob = exps / exps.sum()
                new_index = np.random.choice(np.arange(len(node_layer_population)), p=prob)
                out_str += f"    uct sampling at layer {layer_index+1}, tree have {len(tree_layer_population)} nodes\n"
            else:
                new_index = np.random.choice(np.arange(len(node_layer_population)))
                out_str += f"uniform sampling at layer {layer_index+1}, tree have {len(tree_layer_population)} nodes\n"

            new_traverse_path.append(new_index)
            node_layer_population = node_layer_population[new_index].children

        assert len(node_layer_population) == 0
        # node_info = "\n".join([f"node index |{index:^5}| -> {node}" for index, node in enumerate(layer_children)])
        # out_str += f"\n{node_info}\ntable updated with result from path: {self.traverse_path}\n"
        # out_str += "      node index: |" + "|".join([f"{index:^5}" for index in range(len(raw))]) + "|\n"
        # out_str += "       raw score: |" + "|".join([f"{float(each):>5.2f}" for each in raw]) + "|\n"
        # out_str += "       exp score: |" + "|".join([f"{float(each):>5.2f}" for each in exps]) + "|\n"
        # out_str += "       node prob: |" + "|".join([f"{float(each):>5.0%}" for each in prob]) + "|\n"
        # out_str += "node visit times: |" + "|".join([f"{node.visited_times:>5}" for node in layer_children]) + "|\n"

        self.traverse_path = new_traverse_path
        self.last_score = score
        return out_str.rstrip(), tree_layer_population_count


if __name__ == "__main__":
    GridDistortion.patch_size = (100, 100)
    node = Node(GridDistortion())
    print(node.encoding)
