# %%
from dataclasses import dataclass
from functools import partial

import torch

from multimae.input_adapters import PatchedInputAdapter
from multimae.output_adapters import ConvNeXtAdapter, DPTOutputAdapter
from MULTIMAE_UTILS import create_model
from MULTIMAE_UTILS.pos_embed import interpolate_pos_embed_multimae

# Parameters
pretrained = True
model_name = 'mae'

@dataclass
class DefaultArgs:
    patch_size = 16
    input_size = 224  # 256
    model = 'multivit_base'
    in_domains = ['rgb']
    out_domains = ['depth']
    output_adapter = 'dpt'
    decoder_main_tasks = ['rgb']
    head_type = 'regression'


WEIGHTS_DICT = {
    'mae': {
        'url': 'https://github.com/EPFL-VILAB/MultiMAE/releases/download/pretrained-weights/mae-b_dec512d8b_1600e_multivit-c477195b.pth',
        'args': DefaultArgs(),
    },
    'multimae': {
        'url': 'https://github.com/EPFL-VILAB/MultiMAE/releases/download/pretrained-weights/multimae-b_98_rgb+-depth-semseg_1600e_multivit-afff3f8c.pth',
        'args': DefaultArgs(),
    },
}

DOMAIN_CONF = {
    'rgb': {
        'channels': 3,
        'stride_level': 1,
        'input_adapter': partial(PatchedInputAdapter, num_channels=3),
        'aug_type': 'image',
    },
    'depth': {
        'channels': 1,
        'stride_level': 1,
        'input_adapter': partial(PatchedInputAdapter, num_channels=1),
        'aug_type': 'mask',
    },
    'mask_valid': {
        'stride_level': 1,
        'aug_type': 'mask',
    },
}

# Load
model_dict = WEIGHTS_DICT[model_name]
args: DefaultArgs = model_dict['args']

input_adapters = {
    domain: DOMAIN_CONF[domain]['input_adapter'](
        stride_level=DOMAIN_CONF[domain]['stride_level'],
        patch_size_full=args.patch_size,
        image_size=args.input_size,
    )
    for domain in args.in_domains
}

# DPT settings are fixed for ViT-B. Modify them if using a different backbone.
if args.model != 'multivit_base' and args.output_adapter == 'dpt':
    raise NotImplementedError('Unsupported backbone: DPT head is fixed for ViT-B.')

adapters_dict = {
    'dpt': partial(DPTOutputAdapter, head_type=args.head_type),
    'convnext': partial(ConvNeXtAdapter, preds_per_patch=64),
}

output_adapters = {
    domain: adapters_dict[args.output_adapter](
        num_classes=DOMAIN_CONF[domain]['channels'],
        stride_level=DOMAIN_CONF[domain]['stride_level'],
        patch_size=args.patch_size,
        main_tasks=args.decoder_main_tasks
    )
    for domain in args.out_domains
}

model = create_model(
    args.model,
    input_adapters=input_adapters,
    output_adapters=output_adapters,
    drop_path_rate=0.0,
)

# %%

if pretrained:
    url = model_dict['url']
    if url.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(url, map_location='cpu')
    else:
        checkpoint = torch.load(url, map_location='cpu')
    checkpoint_model = checkpoint['model']

    # Interpolate position embedding
    interpolate_pos_embed_multimae(model, checkpoint_model)

    # Load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    
    # Check
    if model_name == 'mae':
        # These are expected for this model
        assert all(k.startswith('output_adapters.depth') for k in msg.missing_keys)
        assert all(k.startswith('decoder_encoder') for k in msg.unexpected_keys)
    else:
        print('Loaded state dict with errors. Here are the missing and unexpected keys:', msg)

# %%

# Example
x = torch.randn(1, 3, 224, 224)
o = model(x)
print(o.shape)

# %%
