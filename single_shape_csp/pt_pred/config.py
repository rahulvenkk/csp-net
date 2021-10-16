class Config:
    # default configs
    num_iters = 1000000
    batch_size = 1024
    val_after = 5000
    train_perc = 0.9
    save_after = 5000
    res = 256

    flip_axis = True

    omega = 30
    is_first = False

    mesh_file_path = "./single_shape_csp/data/lion.off"
    load_weights = False
    load_iter = 190000
    delete_logs = True
    exp_name = 'lion'
    load_exp_name = ''
    root_path = './single_shape_csp'

    resample = False
    
    sphere_tracing_thresh = 0.008
    sphere_tracing_thresh_gt = 0.008
    sphere_tracing_max_iters = 1000

    eps = 1e-7

    gt_pt_thresh = 6e-3

    def __init__(self):
        return
    
    def make_paths(self):
        exp_name = self.exp_name
        load_exp_name = self.load_exp_name
        root_path = self.root_path
        self.exp_path = root_path + '/experiments/' + exp_name
        self.train_writer_path = root_path +'/experiments/' + exp_name + '_train'
        self.val_writer_path = root_path + '/experiments/' + exp_name + '_val'
        self.weights_path = root_path + '/weights/' + exp_name
        self.load_weights_path = root_path + '/weights/' + load_exp_name
        self.videos_path = root_path + '/videos/' + exp_name
