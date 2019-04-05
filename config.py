import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing images and text")
# Added features mode to extract features for retrieval
parser.add_argument("--mode", required=True, choices=["train", "test", "features"])
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=30, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=0, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")

parser.add_argument("--separable_conv", action="store_true", help="use separable convolutions in the generator")
parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
# parser.add_argument("--lab_colorization", action="store_true", help="split input image into brightness (A) and color (B)")
parser.add_argument("--batch_size", type=int, default=8, help="number of images in batch")
parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=256, help="scale images to this size before cropping to 256x256")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=False)
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")

# Cross-modal-disen new arugments
parser.add_argument("--gan_exclusive_weight", type=float, default=0.1, help="weight on GAN term for exclusive generator gradient")
parser.add_argument("--noise", type=float, default=0.1, help="Stddev for noise input into representation")

# export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])

class Config:
    
    def __init__(self):
        self.a = parser.parse_args()

    def __getattr__(self, name):
        if name=="a":
            return a
        else:
            return getattr(a, name)
    

sys.modules[__name__] = Config()

# input_dir = "/home/ubuntu/everything/mscoco/output_dir"

# batch_size = 64

# image_height = 256
# image_width = 256

# ngfI = 64
# ngfT = 64

# exclusive_size = 128

# max_length = 32

