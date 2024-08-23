import os
import glob
import configargparse
from datetime import datetime
from os.path import join, dirname, abspath

from src.utils import ensure_dirs

class Config(object):
    def __init__(self, phase):
        self.is_train = phase == "train"

        # init hyperparameters and parse from command-line
        parser, args = self.parse()
        self.num_classes = 9

        # set as attributes
        print("----Experiment Configuration-----")
        for k, v in sorted(args.__dict__.items()):
            self.__setattr__(k, v)
            print(f"{k:20}: {v}")

        # processing
        self.cont = self.cont_ckpt is not None

        if self.is_train:
            if self.debug:
                self.exp_detail, self.date = 'debug', 'debug'
            elif self.cont:
                # continue training
                self.exp_detail, self.date, self.ckpt = self.cont_ckpt.split('/')
            else:
                # new training
                self.exp_detail = self.get_exp_detail()
                self.date = datetime.now().strftime('%b%d_%H%M%S')
        else:
            self.exp_detail, self.date, self.ckpt = self.test_ckpt.split('/')

        print(f'exp name: {self.exp_name}, {self.exp_detail}')

        # log folder
        proj_root_cur = dirname(os.path.abspath(__file__))
        proj_root = join(proj_root_cur, "../exps")
        print(f'proj root: {proj_root}')
        self.log_dir = join(proj_root, self.exp_name, self.exp_detail, self.date)
        self.model_dir = join(proj_root, self.exp_name, self.exp_detail, self.date)

        if not self.is_train or self.cont:
            assert os.path.exists(self.log_dir), f'Log dir {self.log_dir} does not exist'
            assert os.path.exists(self.model_dir), f'Model dir {self.model_dir} does not exist'
        else:
            ensure_dirs([self.log_dir, self.model_dir])

        if self.is_train:
            # save all the configurations and code
            log_name = f"log_cont_{datetime.now().strftime('%b%d_%H%M%S')}.txt" if self.cont else 'log.txt'
            py_list = sorted(glob.glob(join(dirname(abspath(__file__)), '**/*.py'), recursive=True))

            with open(join(self.log_dir, log_name), 'w') as log:
                for k, v in sorted(self.__dict__.items()):
                    log.write(f'{k:20}: {v}\n')
                log.write('\n\n')
                for py in py_list:
                    with open(py, 'r') as f_py:
                        log.write(f'\n*****{f_py.name}*****\n')
                        log.write(f_py.read())
                        log.write('================================================'
                                  '===============================================\n')

        # GPU usage
        if args.gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)


    def parse(self):
        parser = configargparse.ArgumentParser(default_config_files=['settings/ssl.yml'])
        parser.add_argument('--config', is_config_file=True, help='config file path')
        self._add_basic_config_(parser)
        self._add_dataset_config_(parser)
        self._add_network_config_(parser)
        self._add_training_config_(parser)
        self._add_ssl_config_(parser)
        if not self.is_train:
            self._add_test_config_(parser)
        args = parser.parse_args()
        return parser, args

    def _add_basic_config_(self, parser):
        group = parser.add_argument_group('basic')
        group.add_argument('--exp_name', type=self.str2type)
        group.add_argument('--exp_detail', type=str, help='more details about the default exp_name')
        group.add_argument('--ss_ratio', type=float, help='supervised data ratio')
        group.add_argument('--is_full_range', action='store_true', help='is full range or not (front range)')
        return group

    def _add_dataset_config_(self, parser):
        group = parser.add_argument_group('dataset')
        group.add_argument('--data_dir_300WLP', type=str)
        group.add_argument('--data_dir_AFLWFace', type=str)
        group.add_argument('--data_dir_AFLW2000', type=str)
        group.add_argument('--data_dir_BIWItrain', type=str)
        group.add_argument('--data_dir_BIWItest', type=str)
        
        group.add_argument('--data_dir_WiderFace', type=str)
        group.add_argument('--data_dir_CrowdHuman', type=str)
        group.add_argument('--data_dir_DAD3DHeads', type=str)
        group.add_argument('--data_dir_COCOHead', type=str)
        group.add_argument('--data_dir_WildHead', type=str)

        group.add_argument('--train_labeled', type=str)
        group.add_argument('--train_unlabeled', type=str)
        group.add_argument('--test_set', type=str)
        return group

    def _add_network_config_(self, parser):
        """add hyperparameters for network architecture"""
        group = parser.add_argument_group('network')
        group.add_argument("--network", type=str, choices=['mobilenet', 'resnet18', 'resnet50', 'repvgg', 'effinetv2', 'effinet', 'tinyvit'])
        pass
        return group

    def _add_training_config_(self, parser):
        group = parser.add_argument_group('training')
        group.add_argument('--lr', type=float, help="initial learning rate")
        group.add_argument('--batch_size', type=int, help="batch size")
        group.add_argument('--num_workers', type=int, help="number of workers for data loading")
        group.add_argument('--stage1_iteration', type=int, help='#iters of stage1')
        group.add_argument('--max_iteration', type=int, help="total number of iterations to train for supervised & merge, "
                                                             "For SSL, it is the relative number for stage2")
        group.add_argument('--log_frequency', type=int, help="visualize output every x iterations")
        group.add_argument('--val_frequency', type=int, help="run validation every x iterations")
        group.add_argument('--save_frequency', type=int, help="save models every x iterations")
        group.add_argument('--cont_ckpt', type=str, help="continue from checkpoint")
        group.add_argument('-g', '--gpu_ids', type=str)
        group.add_argument('--debug', action='store_true', help='debugging mode to avoid generating log files')
        return group

    def _add_test_config_(self, parser):
        group = parser.add_argument_group('test')
        group.add_argument('test_ckpt', type=str)
        group.add_argument('--hist_low', type=int, default=10)
        group.add_argument('--hist_high', type=int, default=150)
        return group

    def _add_ssl_config_(self, parser):
        group = parser.add_argument_group('ssl')
        group.add_argument('--SSL_lambda', type=float, help="loss = super_loss + \lambda * unsuper_loss")
        group.add_argument('--conf_thres', type=float, help="confidence threshold of the Fisher entropy")
        group.add_argument('--dynamic_thres', action='store_true', help="dynamic threshold of Fisher entropy")  # workable
        group.add_argument('--std_ratio', type=float, default=3.0, help="std scale for dynamic threshold")  # defective
        group.add_argument('--left_ratio', type=float, default=0.95, help="left ratio for dynamic threshold")  # workable
        group.add_argument('--ulb_batch_ratio', type=float, help='the ratio of unlabeled to labeld data in mini-batch')
        group.add_argument('--is_ema', type=self.str2type, help='teacher parameters are EMA of student parameters'
                                                                'or identical to student model')
        group.add_argument('--ema_decay', type=float, help='ema variable decay rate (default: 0.999)')
        group.add_argument('--eman', action='store_true', help='Exponential Moving Average Normalization')  # not work
        group.add_argument('--type_unsuper', type=str, help='unsupervised loss', choices=['ce', 'nll'])
        group.add_argument('--distribution', type=str, help='rotation distribution', choices=['matrixFisher', 'RotationLaplace'])
        group.add_argument('--cutout_aug', action='store_true', help='whether to use CutOut strong_aug')  # workable
        group.add_argument('--cutmix_aug', action='store_true', help='whether to use CutMix strong_aug')  # workable
        group.add_argument('--rotate_aug', action='store_true', help='whether to use Rotation strong_aug')  # workable
        group.add_argument('--save_feat', action='store_true', help='whether to use t-SNE for visualization')  # workable
        return group

    def get_exp_detail(self):
        if self.exp_detail is not None:
            exp_detail = self.exp_detail
        else:
            name_thre = 'Dyna'+str(self.left_ratio) if self.dynamic_thres else str(self.conf_thres)
            name_ema = '_ema' if self.is_ema else ''
            name_ema = '_eman' if self.eman else name_ema
            name_range = '_full' if self.is_full_range else ''
            # name_aug += '_RL' if self.distribution=='RotationLaplace' else ''
            name_aug = '_RO' if self.rotate_aug else ''
            name_aug += '_CO' if self.cutout_aug else ''
            name_aug += '_CM' if self.cutmix_aug else ''
            name_vis = '_tSNE' if self.save_feat else ''
            exp_detail = f'SSL{self.SSL_lambda}_r{self.ss_ratio}_{self.type_unsuper}_{self.network}' \
                         f'_t{name_thre}_b{self.batch_size}{name_ema}{name_aug}{name_range}{name_vis}'
        return exp_detail

    @staticmethod
    def str2type(s):
        if str(s).lower() == 'true':
            return True
        elif str(s).lower() == 'false':
            return False
        elif str(s).lower() == 'none':
            return None
        else:
            return s


def get_config(phase):
    config = Config(phase)
    return config
