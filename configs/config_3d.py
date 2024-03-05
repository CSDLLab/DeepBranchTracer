import argparse

parser = argparse.ArgumentParser(description='Hyper-parameters management')

# Hardware options
parser.add_argument('--n_threads', type=int, default=20,help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',help='use cpu only')
parser.add_argument('--gpu_id', type=int, default=0, help='use gpu only')
parser.add_argument("--local_rank", type=int, default=-1)

# parser.add_argument('--seed', default=0, type=int)
# parser.add_argument('--init_method', default='tcp://localhost:52013', type=str)
# parser.add_argument('--rank', default=0, type=int)
# parser.add_argument('--world_size', default=0, type=int)
# parser.add_argument('--device_id_list_str', default='0,1,2,3', type=str)


parser.add_argument("--resize_radio", type=float, default=1.0) # FMOST
parser.add_argument("--r_resize", type=float, default=10)

parser.add_argument('--device_id', default='0', type=str)


# Datasets parameters FMOST DIEDAM FMOST_mpgan
parser.add_argument('--dataset_name', default = 'FMOST',help='datasets name')
parser.add_argument('--dataset_img_path', default = '/4T/liuchao/deepneutracing/deepbranchtracer_3d/FMOST/training_data_/training_datasets/',help='Train datasets image root path')
parser.add_argument('--dataset_img_test_path', default = '/4T/liuchao/deepneutracing/deepbranchtracer_3d/FMOST/training_data_/test_datasets/',help='Train datasets label root path')
parser.add_argument('--test_data_path', default = '/4T/liuchao/deepneutracing/deepbranchtracer_3d/FMOST/test/images/',help='Test datasets root path')
# parser.add_argument('--test_data_mask_path', default = '/4T/liuchao/deepneutracing/deepbranchtracer_3d/FMOST/temp/mask/',help='Test datasets mask root path')

# parser.add_argument('--gold_centerline_path', default = '/4T/liuchao/deepneutracing/deepbranchtracer_3d/FMOST/temp/centerline/',help='Saved centerline result root path')
# parser.add_argument('--gold_seed_path', default = '/4T/liuchao/deepneutracing/deepbranchtracer_3d/FMOST_mpgan/temp/seed/',help='swc seed root path')

parser.add_argument('--predict_seed_path', default = '/4T/liuchao/deepneutracing/deepbranchtracer_3d/FMOST/temp/seed/',help='Seed root path')
parser.add_argument('--predict_centerline_path', default = '/4T/liuchao/deepneutracing/deepbranchtracer_3d/FMOST/results/pre_centerline_test/',help='Saved centerline result root path')
parser.add_argument('--predict_swc_path', default = '/4T/liuchao/deepneutracing/deepbranchtracer_3d/FMOST/results/pre_swc_test/',help='Saved swc result root path')



# parser.add_argument('--test_data_brain_path', default = '/12T/data1/liuchao/Brain_202206/CH1_cut/',help='Test datasets root path')
# parser.add_argument('--brain_map_path', default = '/12T/data1/liuchao/Brain_202206/202206-whole.signal.tif',help='Test datasets root path')
# parser.add_argument('--predict_centerline_brain_path', default = '/12T/data1/liuchao/Brain_202206/CH1_prediction_supersived/',help='Saved centerline result root path')
# parser.add_argument('--predict_swc_brain_path', default = '/12T/data1/liuchao/Brain_202206/CH1_swc/deepbranchtracer_merge/',help='Saved swc result root path')

# 10_20 mpgan
parser.add_argument('--batch_size', type=int, default=16, help='batch size of trainset')
parser.add_argument('--valid_rate', type=float, default=0.1, help='')
parser.add_argument('--data_shape', type=list, default=[16,64,64], help='')

parser.add_argument('--test_patch_height', default=64)
parser.add_argument('--test_patch_width', default=64)
parser.add_argument('--test_patch_depth', default=16)
parser.add_argument('--stride_height', default=48)
parser.add_argument('--stride_width', default=48)
parser.add_argument('--stride_depth', default=12)


# data in/out and dataset
parser.add_argument('--model_save_dir', default='./model/model_',help='save path of trained model')
parser.add_argument('--log_save_dir', default='./log/log_',help='save path of trained log')

# train
parser.add_argument('--epochs', type=int, default=30, metavar='N',help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',help='learning rate (default: 0.0001)')
# parser.add_argument('--early-stop', default=6, type=int, help='early stopping (default: 30)')
# parser.add_argument('--crop_size', type=int, default=48)
# parser.add_argument('--val_crop_max_size', type=int, default=96)
# parser.add_argument('--hidden_layer_size', type=int, default=1)
parser.add_argument('--vector_bins', type=int, default=50)
parser.add_argument('--train_seg', default=False, type=bool)

# test
parser.add_argument('--print_info', default=False, type=bool)
parser.add_argument('--tracing_strategy_mode', default='anglecenterline', type=str) # centerline  angle   anglecenterlined
# parser.add_argument('--use_amp', default=False, type=bool)
parser.add_argument('--train_or_test')
parser.add_argument('--to_restore', default=False, type=bool)


args = parser.parse_args()

