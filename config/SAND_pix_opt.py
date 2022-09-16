from myutils.option_tools import ModelOption

class BaseOptions(ModelOption):
    def initialize(self, parser):
        # experiment specifics
        parser.add_argument('--name', type=str, default='DISC',
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--root', type=str, default='/data/yifan/FS2K', help='path of dataset')
        parser.add_argument('--checkpoints_dir', type=str, default='/data/yifan/rsync/SPADE2_checkpoint', help='models are saved here')
        parser.add_argument('--log_dir', type=str, default='/data/yifan/rsync/SPADE2_logger', help='logger are saved here')
        parser.add_argument('--log_name', type=str, default='orggan', help='same version different name')
        parser.add_argument('--result_img_dir', type=str, default='/data/yifan/rsync/SPADE_result_temp', help='models are saved here')
        parser.add_argument('--img_save_feq', type=int, default=50, help='fequence that save gen results')
        parser.add_argument('--gpu', type=str, default='3', help='number of using gpus')
        parser.add_argument('--ver', type=float, default=14.3, help='models version')
        parser.add_argument('--model', type=str, default='gauparsing', help='which model to use')
        parser.add_argument('--norm_G', type=str, default='spectralspadebatch3x3',
                            help='instance normalization or batch normalization')
        parser.add_argument('--norm_D', type=str, default='spectralinstance',
                            help='instance normalization or batch normalization')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--use_amp', action='store_true', help='fp16 trainning')
        parser.add_argument('--use_en_feature', default=True, help='cat encoder feature to parsing')
        parser.add_argument('--x_parsing', default=True, help='x_parsing or y_parsing')
        parser.add_argument('--use_gmsd', default=True, help='use gmsd loss')
        parser.add_argument('--use_res', default=True, help='use restruction')
        parser.add_argument('--no_par_gaussian', action='store_true', help='fp16 trainning')
        parser.add_argument('--gan_learn_par', action='store_true', help='parsing use gan mechanism to regress ')
        parser.add_argument('--generate_par', action='store_true', help='generator generate parsing')
        # input/output sizes
        parser.add_argument('--bsize', type=int, default=4, help='input batch size')
        parser.add_argument('--img_w', type=int, default=250, help='original img width')
        parser.add_argument('--img_h', type=int, default=250, help='original img height')
        parser.add_argument('--input_size', type=int, default=256, help='network input size')
        parser.add_argument('--FS2K', default=True, help='use gmsd loss')
        # for setting inputs
        parser.add_argument('--affine_type', type=str, default='normal')
        parser.add_argument('--no_flip', action='store_true',
                            help='if specified, do not flip the images for data argumentation')
        parser.add_argument('--load_from_opt_file', action='store_true',
                            help='load the options from checkpoints and use that as default')
        # for generator
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        parser.add_argument('--spade_mode', type=str, default="res2", choices=('org', 'res', 'concat', 'res2'),
                            help='type of spade shortcut connection : |org|res|concat|res2|')
        parser.add_argument('--init_type', type=str, default='xavier',
                            help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_variance', type=float, default=0.02,
                            help='variance of the initialization distribution')
        parser.add_argument('--image_nc', type=int, default=3, help='# of input image channels')
        parser.add_argument('--depth_nc', type=int, default=1, help='# of depth map channels')
        parser.add_argument('--style_nc', type=int, default=3, help='# of style map channels')
        parser.add_argument('--parsing_nc', type=int, default=1,
                            help='# of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.')
        parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels')
        # for discriminator
        parser.add_argument('--netD_subarch', type=str, default='n_layer',
                            help='architecture of each discriminator,n_layer|n_layer_style')
        parser.add_argument('--num_D', type=int, default=1,
                            help='number of discriminators to be used in multiscale')
        parser.add_argument('--n_layers_D', type=int, default=4,
                            help='# layers in each discriminator')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--netD', type=str, default='multiscale', help='(n_layers|multiscale|image)')
        parser.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)')
        return parser

    def after_parse(self):
        if self.opt.img_w > 256 and self.opt.img_h > 256:
            self.opt.input_size = 512
        else:
            self.opt.input_size = 256


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        # for displays
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')
        # for training
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--which_epoch', type=str, default='latest',
                            help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--niter', type=int, default=400,
                            help='# of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is niter + niter_decay')
        parser.add_argument('--niter_decay', type=int, default=400,
                            help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--every_val_epoch', type=int, default=25, help='# of iter to run test while training')
        parser.add_argument('--train_epoch', type=int, default=800, help='total train epoch')
        parser.add_argument('--optimizer', type=str, default='adam')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        parser.add_argument('--no_opti_part_D', action='store_true',
                            help='D optimized part -> add [Instance,LeaklyRelu] before last conv')
        parser.add_argument('--d_opti', action='store_true',
                            help='use optimized d')
        parser.add_argument('--D_steps_per_G', type=int, default=1,
                            help='number of discriminator iterations per generator iterations.')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")
        # Lpix
        parser.add_argument('--lambda_pix', type=float, default=4, help='weight for pixel matching loss')
        # Lgeo
        parser.add_argument('--lambda_par', type=float, default=4, help='weight for paring matching loss')
        # Ltex
        parser.add_argument('--lambda_gmsd', type=float, default=1, help='weight for gmsd loss')
        # Lgan
        parser.add_argument('--lambda_gan', type=float, default=1, help='weight for pixel matching loss')
        parser.add_argument('--lambda_panalty', type=float, default=2, help='weight for hinge loss')
        parser.add_argument('--no_TTUR', action='store_true', help='dont TTUR training scheme')
        parser.add_argument('--g_lr', type=float, default=0.0002)
        parser.add_argument('--d_lr', type=float, default=0.0002)
        self.isTrain = True
        return parser


class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--results_dir', type=str, default='./result/test', help='saves results here.')
        parser.add_argument('--style',default=1,help="sytle type:0/1/2/3")
        parser.add_argument('--data_dir', type=str, default='/data/yifan/rsync/data/0517/ali_250_nobg',
                            help='saves results here.')
        parser.add_argument('--depth_dir', type=str, default='/data/yifan/rsync/data/0517/depth_250',
                            help='saves results here.')
        parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/_ckpt_epoch_749.ckpt',
                            help='saves results here.')
        parser.add_argument('--bisenet_dir', type=str, default='./model/parsing/cp/79999_iter.pth', help='saves results here.')
        parser.add_argument('--how_many', type=int, default=float("inf"), help='how many test images to run')
        parser.set_defaults(no_flip=True)
        parser.set_defaults(phase='test')
        self.isTrain = False
        return parser
