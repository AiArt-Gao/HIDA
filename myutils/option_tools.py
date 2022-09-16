import argparse
import os
import pickle
import torch


class ModelOption():
    def __init__(self):
        self.initialized = False

    def _initialize_(self, parser):
        parser = self.initialize(parser)
        self.initialized = True
        return parser

    def initialize(self, parser):
        raise NotImplementedError()

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self._initialize_(parser)

        # get the basic options
        opt, unknown = parser.parse_known_args()

        # modify model-related parser options
        # model_name = opt.model
        # model_option_setter = model.get_option_setter(model_name)
        # parser = model_option_setter(parser, self.isTrain)

        # modify dataset-related parser options
        # dataset_mode = opt.dataset_mode
        # dataset_option_setter = data.get_option_setter(dataset_mode)
        # parser = dataset_option_setter(parser, self.isTrain)

        # opt, unknown = parser.parse_known_args()

        # if there is opt_file, load it.
        # The previous default options will be overwritten
        # if opt.load_from_opt_file:
        #     parser = self.update_options_from_file(parser, opt)

        opt = parser.parse_args()
        self.parser = parser
        return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def option_file_path(self, opt, makedir=False):
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if makedir:
            try:
                os.makedirs(expr_dir)
            except FileExistsError:
                print('path already existed')
                pass

        file_name = os.path.join(expr_dir, 'opt')
        return file_name

    def save_options(self, opt):
        file_name = self.option_file_path(opt, makedir=True)
        with open(file_name + '.txt', 'wt') as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

        with open(file_name + '.pkl', 'wb') as opt_file:
            pickle.dump(opt, opt_file)

    def update_options_from_file(self, parser, opt):
        new_opt = self.load_options(opt)
        for k, v in sorted(vars(opt).items()):
            if hasattr(new_opt, k) and v != getattr(new_opt, k):
                new_val = getattr(new_opt, k)
                parser.set_defaults(**{k: new_val})
        return parser

    def load_options(self, opt):
        file_name = self.option_file_path(opt, makedir=False)
        new_opt = pickle.load(open(file_name + '.pkl', 'rb'))
        return new_opt

    def parse(self, save=False):

        opt = self.gather_options()
        opt.isTrain = self.isTrain  # train or test

        self.print_options(opt)
        if opt.isTrain:
            self.save_options(opt)

        # Set semantic_nc based on the option.
        # This will be convenient in many places
        # opt.semantic_nc = opt.label_nc + \
        #     (1 if opt.contain_dontcare_label else 0) + \
        #     (0 if opt.no_instance else 1)

        # set gpu ids
        str_ids = opt.gpu.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        assert len(opt.gpu_ids) == 0 or opt.bsize % len(opt.gpu_ids) == 0, \
            "Batch size %d is wrong. It must be a multiple of # GPUs %d." \
            % (opt.bsize, len(opt.gpu_ids))

        self.opt = opt
        self.after_parse()
        return self.opt

    def after_parse(self):
        pass
