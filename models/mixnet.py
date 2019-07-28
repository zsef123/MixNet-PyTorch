import re
import collections

import torch
import torch.nn as nn

from models.layers import SEModule
from models.layers import Swish
from models.layers import Flatten

from models.mdconv import MDConv


class MixBlock(nn.Module):
    def __init__(self, dw_ksize, expand_ksize, project_ksize,
                 in_channels, out_channels, expand_ratio, id_skip,
                 strides, se_ratio, swish, dilated):
        super().__init__()

        self.id_skip = id_skip and all(s == 1 for s in strides) and in_channels == out_channels

        act_fn = lambda : Swish() if swish else nn.ReLU(True)

        layers = []
        expaned_ch = in_channels * expand_ratio
        if expand_ratio != 1:
            expand = nn.Sequential(
                nn.Conv2d(in_channels, expaned_ch, expand_ksize, bias=False),
                nn.BatchNorm2d(expaned_ch),
                act_fn(),
            )
            layers.append(expand)

        depthwise = nn.Sequential(
            MDConv(expaned_ch, dw_ksize, strides, bias=False),
            nn.BatchNorm2d(expaned_ch),
            act_fn(),
        )
        layers.append(depthwise)

        if se_ratio > 0:
            se = SEModule(expaned_ch, int(expaned_ch * se_ratio))
            layers.append(se)

        project = nn.Sequential(
            nn.Conv2d(expaned_ch, out_channels, project_ksize, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        layers.append(project)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        if self.id_skip:
            out = out + x
        return out


class MixModule(nn.Module):
    def __init__(self, dw_ksize, expand_ksize, project_ksize, num_repeat,
                 in_channels, out_channels, expand_ratio, id_skip,
                 strides, se_ratio, swish, dilated):
        super().__init__()
        layers = [MixBlock(dw_ksize, expand_ksize, project_ksize,
                           in_channels, out_channels, expand_ratio, id_skip,
                           strides, se_ratio, swish, dilated)]
        for _ in range(num_repeat - 1):
            layers.append(MixBlock(dw_ksize, expand_ksize, project_ksize,
                                   in_channels, out_channels, expand_ratio, id_skip,
                                   [1, 1], se_ratio, swish, dilated))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


def round_filters(filters, depth_multiplier, depth_divisor, min_depth):
    """Round number of filters based on depth depth_multiplier.
    TODO : ref link
    """
    if not depth_multiplier:
        return filters

    filters *= depth_multiplier
    min_depth = min_depth or depth_divisor
    new_filters = max(min_depth, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return new_filters


class MixNet(nn.Module):
    def __init__(self, stem, blocks_args, head, dropout_rate, num_classes=1000):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, stem, 3, 2, 1, bias=False),
            nn.BatchNorm2d(stem),
            nn.ReLU(True)
        )

        self.blocks = nn.Sequential(*[MixModule(*args) for args in blocks_args])

        self.classifier = nn.Sequential(
            nn.Conv2d(blocks_args[-1].out_channels, head, 1, bias=False),
            nn.BatchNorm2d(head),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(head, num_classes)
        )

    def forward(self, x):
        # print("Input : ", x.shape)
        stem = self.stem(x)
        # print("Stem : ", x.shape)
        feature = self.blocks(stem)
        # print("feature : ", feature.shape)
        out = self.classifier(feature)
        return out


BlockArgs = collections.namedtuple('BlockArgs', [
    'dw_ksize', 'expand_ksize', 'project_ksize', 'num_repeat',
    'in_channels', 'out_channels', 'expand_ratio', 'id_skip',
    'strides', 'se_ratio', 'swish', 'dilated',
])


class MixnetDecoder:
    """A class of Mixnet decoder to get model configuration."""

    @staticmethod
    def _decode_block_string(block_string, depth_multiplier, depth_divisor, min_depth):
        """Gets a mixnet block through a string notation of arguments.

        E.g. r2_k3_a1_p1_s2_e1_i32_o16_se0.25_noskip: r - number of repeat blocks,
        k - kernel size, s - strides (1-9), e - expansion ratio, i - input filters,
        o - output filters, se - squeeze/excitation ratio

        Args:
        block_string: a string, a string representation of block arguments.

        Returns:
        A BlockArgs instance.
        Raises:
        ValueError: if the strides option is not correctly specified.
        """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        if 's' not in options or len(options['s']) != 2:
            raise ValueError('Strides options should be a pair of integers.')

        def _parse_ksize(ss):
            ks = [int(k) for k in ss.split('.')]
            return ks if len(ks) > 1 else ks[0]

        return BlockArgs(num_repeat=int(options['r']),
                         dw_ksize=_parse_ksize(options['k']),
                         expand_ksize=_parse_ksize(options['a']),
                         project_ksize=_parse_ksize(options['p']),
                         strides=[int(options['s'][0]), int(options['s'][1])],
                         expand_ratio=int(options['e']),
                         in_channels=round_filters(int(options['i']), depth_multiplier, depth_divisor, min_depth),
                         out_channels=round_filters(int(options['o']), depth_multiplier, depth_divisor, min_depth),
                         id_skip=('noskip' not in block_string),
                         se_ratio=float(options['se']) if 'se' in options else 0,
                         swish=('sw' in block_string),
                         dilated=('dilated' in block_string)
                         )

    @staticmethod
    def _encode_block_string(block):
        """Encodes a Mixnet block to a string."""

        def _encode_ksize(arr):
            return '.'.join([str(k) for k in arr])

        args = [
            'r%d' % block.num_repeat,
            'k%s' % _encode_ksize(block.dw_ksize),
            'a%s' % _encode_ksize(block.expand_ksize),
            'p%s' % _encode_ksize(block.project_ksize),
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.in_channels,
            'o%d' % block.out_channels
        ]

        if (block.se_ratio is not None and block.se_ratio > 0 and block.se_ratio <= 1):
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        if block.swish:
            args.append('sw')
        if block.dilated:
            args.append('dilated')
        return '_'.join(args)

    @staticmethod
    def decode(string_list, depth_multiplier, depth_divisor, min_depth):
        """Decodes a list of string notations to specify blocks inside the network.

        Args:
        string_list: a list of strings, each string is a notation of Mixnet
        block.build_model_base

        Returns:
        A list of namedtuples to represent Mixnet blocks arguments.
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(MixnetDecoder._decode_block_string(block_string, depth_multiplier, depth_divisor, min_depth))
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        """Encodes a list of Mixnet Blocks to a list of strings.

        Args:
        blocks_args: A list of namedtuples to represent Mixnet blocks arguments.
        Returns:
        a list of strings, each string is a notation of Mixnet block.
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(MixnetDecoder._encode_block_string(block))
        return block_strings


def mixnet_s(depth_multiplier=1, depth_divisor=8, min_depth=None):
    """
    Creates mixnet-s model.

    Args:
        depth_multiplier: depth_multiplier to number of filters per layer.
    """
    stem = round_filters(16,   depth_multiplier, depth_divisor, min_depth)
    head = round_filters(1536, depth_multiplier, depth_divisor, min_depth)
    dropout = 0.2

    blocks_args = [
        'r1_k3_a1_p1_s11_e1_i16_o16',
        'r1_k3_a1.1_p1.1_s22_e6_i16_o24',
        'r1_k3_a1.1_p1.1_s11_e3_i24_o24',

        'r1_k3.5.7_a1_p1_s22_e6_i24_o40_se0.5_sw',
        'r3_k3.5_a1.1_p1.1_s11_e6_i40_o40_se0.5_sw',

        'r1_k3.5.7_a1_p1.1_s22_e6_i40_o80_se0.25_sw',
        'r2_k3.5_a1_p1.1_s11_e6_i80_o80_se0.25_sw',

        'r1_k3.5.7_a1.1_p1.1_s11_e6_i80_o120_se0.5_sw',
        'r2_k3.5.7.9_a1.1_p1.1_s11_e3_i120_o120_se0.5_sw',

        'r1_k3.5.7.9.11_a1_p1_s22_e6_i120_o200_se0.5_sw',
        'r2_k3.5.7.9_a1_p1.1_s11_e6_i200_o200_se0.5_sw',
    ]

    blocks_args = MixnetDecoder.decode(blocks_args, depth_multiplier, depth_divisor, min_depth)
    print("-----------")
    print("Mixnet S")
    for a in blocks_args:
        print(a)
    print("-----------")
    return MixNet(stem, blocks_args, head, dropout)


if __name__ == "__main__":
    mixnet_s()
