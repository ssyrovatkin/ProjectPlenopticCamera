from skimage import io
import skimage
import argparse
import numpy as np

from ParallaxProcessing import Parallax
from ThirdDimension import ThirdDimension

parser = argparse.ArgumentParser()

parser.add_argument('path', help="Path to input file")
parser.add_argument('mode', help="Determines the processing mode")
parser.add_argument('-o', '--output', help="Path to saved folder")

args = parser.parse_args()

print("Input path:", args.path)

if args.output:
    print("Files will be saved here:", args.output)


if args.mode == 'parallax':
    processor = Parallax(args.path, destination=args.output, presentation_mode=True)
elif args.mode == '3d':
    processor = ThirdDimension(args.path, destination=args.output, presentation_mode=True)
else:
    raise ValueError(f"{args.mode} is not implemented\n")

