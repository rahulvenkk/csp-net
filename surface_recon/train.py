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
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of seconds'
                         'with exit code 2.')
parser.add_argument('--eval-only', action='store_true', help='Only evaluation')


args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda:0" if is_cuda else "cpu")

# Set t0
t0 = time.time()

# Shorthands
out_dir = cfg['training']['out_dir']

# copy config
max_val = cfg['training']['max_val']
load_dir = cfg['training']['load_dir_']
load_optim = cfg['training']['load_optim']


batch_size = cfg['training']['batch_size']
backup_every = cfg['training']['backup_every']
exit_after = args.exit_after

model_selection_metric = cfg['training']['model_selection_metric']
if cfg['training']['model_selection_mode'] == 'maximize':
    model_selection_sign = 1
elif cfg['training']['model_selection_mode'] == 'minimize':
    model_selection_sign = -1
else:
    raise ValueError('model_selection_mode must be '
                     'either maximize or minimize.')

# Output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

os.system('cp ' + args.config + ' ' + out_dir + '/')

n_max_ram = cfg['training']['max_ram']
method = cfg['method']
dataset_type = cfg['data']['dataset']
dataset_folder = cfg['data']['path']
categories = cfg['data']['classes']
n_internal_epochs = cfg['training']['n_internal_epochs'] * \
                    min(n_max_ram, batch_size)
if categories is None:
    categories = os.listdir(dataset_folder)
    categories = [c for c in categories
                  if os.path.isdir(os.path.join(dataset_folder, c))]

model_list = []
for c_idx, c in enumerate(categories):
    subpath = os.path.join(dataset_folder, c)
    if not os.path.isdir(subpath):
        warnings.warn('Category %s does not exist in dataset.' % c, UserWarning)

    split_file = os.path.join(subpath, 'train.lst')
    with open(split_file, 'r') as f:
        models_c = f.read().split('\n')

    model_list += [
        {'category': c, 'model': m}
        for m in models_c
    ]


np.random.seed(0)    
np.random.shuffle(model_list)

val_dataset = config.get_dataset('test', cfg)

train_dataset = config.get_dataset('train', cfg, model_list=model_list, \
                                   n_internal_epochs=n_internal_epochs)
train_loader = torch.utils.data.DataLoader(train_dataset, \
                                           batch_size=batch_size, \
                                           num_workers=0, shuffle=True, \
                                           collate_fn=data.collate_remove_none,\
                                           worker_init_fn=data.worker_init_fn, \
                                           pin_memory=True)

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

optimizer = optim.Adam([{'params':model.decoder.parameters(), 'lr':1e-4},\
                        {'params':model.encoder.parameters(), 'lr':1e-4}])

if load_optim:
    checkpoint_io_load = CheckpointIO(load_dir, model=model, \
                                      optimizer=optimizer)
else:
    checkpoint_io_load = CheckpointIO(load_dir, model=model)

try:
    scalar_dict, model_dict, opt_dict = checkpoint_io_load.load('model_latest.pt')
except FileExistsError:
    scalar_dict, model_dict, opt_dict = dict(), None, None

epoch_it = scalar_dict.get('epoch_it', -1)
it = scalar_dict.get('it', -1)

if model_dict:
    model_dict = {k: v for k, v in model_dict.items() if '_nvf' not in k}
    model_dict = {k.replace('_points', ''): v for k, v in model_dict.items()}
    
    model.load_state_dict(model_dict)
    print('Model loaded successfully!')

if opt_dict:
    optimizer.load_state_dict(opt_dict)
    print('Optimizer loaded successfully!')

checkpoint_io_save = CheckpointIO(out_dir, model=model, optimizer=optimizer)

model = nn.DataParallel(model)
model.to(device)

trainer = config.get_trainer(model, optimizer, cfg, \
                            device=device, max_val=max_val)

metric_val_best = dict().get(
    'loss_val_best', -model_selection_sign * np.inf)

if metric_val_best == np.inf or metric_val_best == -np.inf:
    metric_val_best = -model_selection_sign * np.inf

print('Current best validation metric (%s): %.8f'
      % (model_selection_metric, metric_val_best))

if os.path.exists(os.path.join(out_dir, 'logs_train')) \
    and cfg['training']['delete_logs']:
    os.system('rm -rf ' + os.path.join(out_dir, 'logs_train'))
    os.system('rm -rf ' + os.path.join(out_dir, 'logs_val'))

train_logger = SummaryWriter(os.path.join(out_dir, 'logs_train'))
val_logger = SummaryWriter(os.path.join(out_dir, 'logs_val'))

# Shorthands
print_every = cfg['training']['print_every']
checkpoint_every = cfg['training']['checkpoint_every']
validate_every = cfg['training']['validate_every']
visualize_every = cfg['training']['visualize_every']

# Print model
nparameters = sum(p.numel() for p in model.parameters())
#print(model)
print('Total number of parameters: %d' % nparameters)
t = time.time()

############# DATALOADER ##############
def enum_dataset(dataset):
    dataset.no_return=True
    for data in enumerate(dataset):
        print("fetching")
        x = 1
        
    dataset.no_return=False
    return


print("Total models: ", len(model_list))
np.random.shuffle(model_list)
print("shuffled")

###############################################################################
# Evaluation
###############################################################################
if args.eval_only:
    print('Running Evaluation')
    print('Evaluation samples: ', len(val_loader))
    eval_dict = trainer.evaluate(val_loader)
    metric_val = eval_dict[model_selection_metric]
    print('Validation metric (%s): %.4f'
            % (model_selection_metric, metric_val))

    for k, v in eval_dict.items():
        if 'viz_image' not in k:
            print('%s: %.4f' % (k, v))
    exit()


###############################################################################
# Training loop
###############################################################################
global_it = -1

while True:
    epoch_it += 1
    it = -1

    for batch in train_loader:
        it += 1
        global_it += 1

        loss = trainer.train_step(batch)
        train_logger.add_scalar('loss', loss, global_it)

        print('[Epoch %02d] it=%03d, loss=%.6f' % (epoch_it, it, loss))

        # Save checkpoint
        if (checkpoint_every > 0 and (it % checkpoint_every) == 0):
            print('Saving checkpoint')
            checkpoint_io_save.save('model.pt', epoch_it=epoch_it, it=it,
                                    loss_val_best=metric_val_best)

        # Backup if necessary
        if (backup_every > 0 and (it % backup_every) == 0):
            print('Backup checkpoint')
            checkpoint_io_save.save('model_latest.pt', epoch_it=epoch_it, \
                                    it=it, loss_val_best=metric_val_best)

        # Run validation
        if validate_every > 0 and (it % validate_every) == 0:
            print('Running validation')
            eval_dict = trainer.evaluate(val_loader)
            metric_val = eval_dict[model_selection_metric]
            print('Validation metric (%s): %.4f'
                    % (model_selection_metric, metric_val))
    
            for k, v in eval_dict.items():
                if 'viz_image' in k:
                    val_logger.add_image('%s' % k, \
                                        v[:, :, :3].transpose([2, 0, 1]), global_it)
                else:
                    val_logger.add_scalar('%s' % k, v, global_it)

            if model_selection_sign * (metric_val - metric_val_best) > 0:
                metric_val_best = metric_val
                print('New best model (Val Metric %.6f)' % metric_val_best)
                checkpoint_io_save.save('model_best.pt', epoch_it=epoch_it, \
                                        it=it, loss_val_best=metric_val_best)

        # Exit if necessary
        if exit_after > 0 and (time.time() - t0) >= exit_after:
            print('Time limit reached. Exiting.')
            checkpoint_io_save.save('model.pt', epoch_it=epoch_it, it=it,
                                loss_val_best=metric_val_best)
            exit(3)

        if train_dataset.completed:
            for field_name, field in train_dataset.fields.items():
                if field_name == 'points':
                    print("completed inner", "nepochs:", field.epoch)
                    field.cleanup()
            break

    if train_dataset.completed:
        break
