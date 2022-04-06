import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import os
import argparse
import time
import matplotlib; matplotlib.use('Agg')
from im2mesh import config, data
from im2mesh.checkpoints import CheckpointIO
import torch.nn as nn
import _thread as thread
import warnings
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Arguments
np.random.seed(0)

parser = argparse.ArgumentParser(
    description='Train a 3D reconstruction model.'
)
parser.add_argument('--config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--max-val', type=int, default=-1, \
                    help='Number of validation samples to evaluate on. \
                          Default: -1 (all samples)')


args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')

is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda:0" if is_cuda else "cpu")
cfg['training']['max_val'] = args.max_val

# Set t0
t0 = time.time()

# Shorthands
out_dir = cfg['training']['out_dir']

# copy config
max_val = cfg['training']['max_val']
load_dir = cfg['training']['load_dir_']


batch_size = cfg['training']['batch_size']


# Output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

os.system('cp ' + args.config + ' ' + out_dir + '/')

method = cfg['method']
dataset_type = cfg['data']['dataset']
dataset_folder = cfg['data']['path']
categories = cfg['data']['classes']

if categories is None:
    categories = os.listdir(dataset_folder)
    categories = [c for c in categories
                  if os.path.isdir(os.path.join(dataset_folder, c))]

np.random.seed(0)    

val_dataset = config.get_dataset('test', cfg)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, num_workers=0, shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

# For visualizations
vis_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=12, shuffle=True,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)
data_vis = next(iter(vis_loader))

# Model
model = config.get_model(cfg, device=device)
checkpoint_io_load = CheckpointIO(load_dir, model=model)

try:
    scalar_dict, model_dict, opt_dict = checkpoint_io_load.load('model_best.pt')
except FileExistsError:
    scalar_dict, model_dict, opt_dict = dict(), None, None

epoch_it = scalar_dict.get('epoch_it', -1)
it = scalar_dict.get('it', -1)

if model_dict:
    model_dict = {k: v for k, v in model_dict.items() if '_nvf' not in k}
    model_dict = {k.replace('_points', ''): v for k, v in model_dict.items()}
    
    model.load_state_dict(model_dict)
    print('Model loaded successfully!')

model = nn.DataParallel(model)
model.to(device)

trainer = config.get_trainer(model, None, cfg, \
                            device=device, max_val=max_val)

val_logger = SummaryWriter(os.path.join(out_dir, 'logs_val'))

t = time.time()

############# DATALOADER ##############
def enum_dataset(dataset):
    dataset.no_return=True
    for data in enumerate(dataset):
        print("fetching")
        x = 1
        
    dataset.no_return=False
    return


###############################################################################
# Evaluation
###############################################################################
print('Running Evaluation')
print('Evaluation samples: ', len(val_loader))
print('Evaluating on %d samples. Specify --max-val flag to control this.' % cfg['training']['max_val'])
eval_dict = trainer.evaluate(val_loader)

for k, v in eval_dict.items():
    if 'viz_image' not in k:
        print('%s: %.4f' % (k, v))
