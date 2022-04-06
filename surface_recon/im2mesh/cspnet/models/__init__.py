import torch
import torch.nn as nn
from torch import distributions as dist
from . import encoder_latent, decoder
import numpy as np

# Encoder latent dictionary
encoder_latent_dict = {
    'simple': encoder_latent.Encoder,
}

# Decoder dictionary
decoder_dict = {
    'simple': decoder.Decoder,
    'simple_local': decoder.LocalDecoder
}


class OccupancyNetwork(nn.Module):
    ''' Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        encoder_latent (nn.Module): latent encoder network
        p0_z (dist): prior distribution for latent code z
        device (device): torch device
    '''

    def __init__(self, decoder, encoder=None, encoder_latent=None, p0_z=None,
                 device=None, udf_res=10, eps_udf=0.01, encoder_pretrained=None):
        super().__init__()
        if p0_z is None:
            p0_z = dist.Normal(torch.tensor([]), torch.tensor([]))

        self.decoder = decoder.to(device)        
        self.udf_res = udf_res
        self.eps_udf = eps_udf
        
        if encoder_latent is not None:
            self.encoder_latent = encoder_latent.to(device)
        else:
            self.encoder_latent = None
            
        if encoder_pretrained is not None:
            self.encoder = encoder_pretrained.to(device)
        else:
            self.encoder = None

        self._device = device
        self.p0_z = p0_z
        
        self.loss_2l = torch.nn.MSELoss(reduction="mean")
    
    def get_encoding(self, inputs):
        c = self.encoder(inputs)
        return c
        
    def forward(self, p, p_surface, inputs, **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
            sample (bool): whether to sample for z
        '''
        c = self.encoder(inputs)
        csp = self.decode(p, p_surface, c, **kwargs)
        return csp
    
    def loss(self, point_pred, point_gt):
        loss = self.loss_2l(point_pred, point_gt)
        return loss
  

    def encode_inputs(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''
        c = self.encoder(inputs)
        return c
    
    def get_sampled_local_grid(self, p, c, **kwargs):
        get_cuda_device = p.get_device()
        pts = torch.linspace(-self.eps_udf, self.eps_udf, self.udf_res).to(get_cuda_device)
        
        perturb = torch.meshgrid(pts, pts, pts)
        perturb = torch.stack(perturb, 3).to(get_cuda_device)
        perturb = perturb.view(-1, 3)
        perturb = torch.unsqueeze(torch.unsqueeze(perturb, 0), 0)

        p = torch.unsqueeze(p, 2)
        
        perturb_p = p + perturb
        perturb_p = perturb_p.view(p.shape[0], -1, 3)  #batch, n_points_udf*10*10*10, 3 
        
        with torch.no_grad():
            points_pred = self.decoder(perturb_p, c, **kwargs)
            
        points_pred = points_pred.view(perturb_p.shape[0], p.shape[1], 3*self.udf_res**3) # batch, n_points, 1000*3 (udf vals)
    
        perturb_p = perturb_p.view(perturb_p.shape[0], p.shape[1], 3*self.udf_res**3) # batch, n_points, udf_res**3*3
        
        p = p[:, :, 0, :] #batch, n_points, 3
        concat = torch.cat([p, points_pred, perturb_p], -1)
        return concat

    def decode_csp(self, points, encoding, **kwargs):
        return self.decoder(points, encoding, **kwargs)

    def decode_udf(self, points, encoding, **kwargs):
        points_pred = self.decode_csp(points, encoding, **kwargs)
        udf = torch.sqrt(torch.sum((points_pred - points)**2, -1))
        
        return udf
    
    def decode_nvf(self, points, encoding, **kwargs):        
        pts = self.decode_csp(points, encoding, **kwargs)
        nvf = points - pts
        nvf = (nvf.permute(2, 0, 1) / torch.sqrt(torch.sum(nvf ** 2, -1) + 1e-9)).permute(1, 2, 0)
        
        return nvf
    
    def decode(self, points, encoding, **kwargs):
        ''' Returns udf predicted for sampled points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''
        udf = self.decode_udf(points, encoding, return_feat=True, **kwargs)
        return udf
    
    def normalize(self, norm_xyz, eps=1e-7):
        return ((norm_xyz.t()) / torch.sqrt(torch.sum(norm_xyz ** 2, 1) + eps * 0.1)).t()
    
    def get_numerically_obtained_normals_torch(self, input_points, encoding_udf):
        """
        get numerically obtained normals for the given input points in world coordinates
        Parameters
        ----------
        input_points : pytorch tensor [N, 3]
            points for which normals are to be estimated
        net : torch nn.Module
            model (sDF + nF)
        eps : float
            some small number
        Returns
        -------
        norm_xyz: pytorch tensor [N, 3]
            estimated normals
        """
        with torch.enable_grad():
            input_points = input_points.detach().cpu().numpy()
            X = torch.from_numpy(input_points).cuda().float().requires_grad_(True)
            Y = self.decode_udf(X, encoding_udf)
            grad_outputs = torch.ones_like(Y)
            grad = torch.autograd.grad(Y, [X], grad_outputs=grad_outputs)[0]
            normals_pred_numerical = grad

            curr_shape = normals_pred_numerical.shape
            normals_pred_numerical = torch.reshape(normals_pred_numerical,(-1,3))
            normals_pred_numerical = self.normalize(normals_pred_numerical)
            normals_pred_numerical = torch.reshape(normals_pred_numerical,curr_shape)
        return normals_pred_numerical