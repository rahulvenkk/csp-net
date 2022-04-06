import torch
import torch.distributions as dist
from torch import nn
import os
from ..encoder import encoder_dict
from .. import data
from .. import config
from . import models, training, generation

def get_model(cfg, device=None, dataset=None, **kwargs):
    ''' Return the Occupancy Network model.

    Args:
        cfg (dict): imported yaml config 
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    decoder = cfg['model']['decoder']
    encoder = cfg['model']['encoder']
    encoder_latent = cfg['model']['encoder_latent']
    dim = cfg['data']['dim']
    z_dim = cfg['model']['z_dim']
    c_dim = cfg['model']['c_dim']
    udf_res = cfg['model']['udf_res']
    eps_udf = cfg['model']['eps_udf']
    decoder_kwargs = cfg['model']['decoder_kwargs']
    encoder_kwargs = cfg['model']['encoder_kwargs']
    encoder_latent_kwargs = cfg['model']['encoder_latent_kwargs']

    # local positional encoding
    if 'local_coord' in cfg['model'].keys():
        encoder_kwargs['local_coord'] = cfg['model']['local_coord']
        decoder_kwargs['local_coord'] = cfg['model']['local_coord']
    if 'pos_encoding' in cfg['model']:
        encoder_kwargs['pos_encoding'] = cfg['model']['pos_encoding']
        decoder_kwargs['pos_encoding'] = cfg['model']['pos_encoding']

    decoder = models.decoder_dict[decoder](
        dim=dim, c_dim=c_dim, out_size=3,
        **decoder_kwargs
    )
  

    if encoder == 'idx':
        encoder = nn.Embedding(len(dataset), c_dim)
        # encoder = nn.Embedding(1, c_dim)
    elif encoder is not None:
        enc = encoder
        encoder = encoder_dict[encoder](dim=dim,
            c_dim=c_dim,
            **encoder_kwargs
        )
        
        encoder_pretrained = encoder_dict[enc](dim=dim,
            c_dim=c_dim,
            **encoder_kwargs
        )
    else:
        encoder = None
    
    model = models.OccupancyNetwork(decoder, encoder, encoder_latent, \
                                    device=device, udf_res=udf_res, \
                                    eps_udf=eps_udf, \
                                    encoder_pretrained=encoder_pretrained)

    return model


def get_trainer(model, optimizer, cfg, device, scheduler=None,scheduler_type=None, **kwargs):
    ''' Returns the trainer object.

    Args:
        model (nn.Module): the Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    threshold = cfg['test']['threshold']
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')
    input_type = cfg['data']['input_type']

    trainer = training.Trainer(
        model, optimizer,
        device=device, input_type=input_type,
        vis_dir=vis_dir, threshold=threshold,
        eval_sample=cfg['training']['eval_sample']
    )

    return trainer


def get_generator(model, cfg, device, **kwargs):
    ''' Returns the generator object.

    Args:
        model (nn.Module): Occupancy Network model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    preprocessor = config.get_preprocessor(cfg, device=device)

    generator = generation.Generator3D(
        model,
        device=device,
        threshold=cfg['test']['threshold'],
        resolution0=cfg['generation']['resolution_0'],
        upsampling_steps=cfg['generation']['upsampling_steps'],
        sample=cfg['generation']['use_sampling'],
        refinement_step=cfg['generation']['refinement_step'],
        simplify_nfaces=cfg['generation']['simplify_nfaces'],
        preprocessor=preprocessor,
    )
    return generator


def get_prior_z(cfg, device, **kwargs):
    ''' Returns prior distribution for latent code z.

    Args:
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    z_dim = cfg['model']['z_dim']
    p0_z = dist.Normal(
        torch.zeros(z_dim, device=device),
        torch.ones(z_dim, device=device)
    )

    return p0_z


def get_data_fields(mode, cfg):
    ''' Returns the data fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    '''
    points_transform = data.SubsamplePointsUDF(cfg['data']['points_subsample'], cfg['data']['pointcloud_n'])
    with_transforms = cfg['model']['use_camera']

    fields = {}
    fields['points'] = data.uDFField(
        cfg['data']['points_file'], points_transform,
        with_transforms=with_transforms,
    )
   
    return fields
