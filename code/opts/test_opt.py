from opts.base_opt import BaseOpt

class TestOpt(BaseOpt):
    def initialize(self):
        BaseOpt.initialize(self)
        self.parser.add_argument('--input_event_path', type=str, default='../../benchmark_dataset_proj/E_blur/E_blur_5_eventstream_val/')
        self.parser.add_argument('--input_blur_path', type=str, default='../../benchmark_dataset_proj/E_blur/E_blur_5_images_val/')
        self.parser.add_argument('--img_start_idx', type=int, default=2, help='image start index')
        self.parser.add_argument('--img_inter_idx', type=int, default=5, help='image interval index')
        # augmentation.
        self.parser.add_argument('--test_batch_size', type=int, default=1, help='input batch size')
        self.parser.add_argument('--output_dir', type=str, default='../method_results/eSL-Net/gopro_val/E_blur_5_results', help='output_path')

        # use cuda
        self.parser.add_argument("--cuda", action="store_true", help="Use cuda?")

        self.is_train = False
