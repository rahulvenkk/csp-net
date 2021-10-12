class Config:
    # default configs
    num_iters = 1000000
    batch_size = 1024
    val_after = 5000
    train_perc = 0.9
    save_after = 5000
    res = 512

    flip_axis = True

    omega = 30
    is_first = False

    mesh_file_path = "./single_shape_csp/data/lion.off"
    load_weights = False
    load_iter = 190000
    delete_logs = True
    exp_name = 'lion'
    load_exp_name = ''
    exp_path = './single_shape_csp/experiments/' + exp_name
    train_writer_path = './single_shape_csp/experiments/' + exp_name + '_train'
    val_writer_path = './single_shape_csp/experiments/' + exp_name + '_val'
    weights_path = './single_shape_csp/weights/' + exp_name
    load_weights_path = './single_shape_csp/weights/' + load_exp_name
    videos_path = './single_shape_csp/videos/' + exp_name

    resample = False
    
    sphere_tracing_thresh = 0.008
    sphere_tracing_thresh_gt = 0.008
    sphere_tracing_max_iters = 1000

    eps = 1e-7

    gt_pt_thresh = 6e-3

    if load_weights:
        load_weights_path = load_weights_path + '/' + str(load_iter) + '.pth'

    def __init__(self):
        return
