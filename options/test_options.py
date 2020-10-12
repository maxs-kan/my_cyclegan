from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--img_dir', type=str, default='/all_data/hdd/un_depth/results/max/', help='saves results here.')
        parser.add_argument('--save_img', type=bool, default=False, help='save image?')
#         parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
#         parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
#         parser.add_argument('--num_test', type=int, default=10, help='how many test images to run')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(name='no_mean_blur_20k_upconv')
        parser.set_defaults(epoch='last')
#         parser.set_defaults(load_size_h=parser.get_default('crop_size'))
#         parser.set_defaults(load_size_w=parser.get_default('crop_size'))
        parser.set_defaults(data_shuffle=False)
        parser.set_defaults(batch_size=5)
#         parser.set_defaults(num_workers=4)
        self.isTrain = False
        return parser
