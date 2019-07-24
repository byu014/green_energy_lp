import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir','-id', default=sys.path[0] + "/test_plate/",
                        type=str, help='Input generated image directory')
    parser.add_argument('--num_imgs','-ni', default=1000,
                        type=int, help='Number of images to be generated')
    parser.add_argument('--resample','-r', default=4,
                        type=int, help="Shrinks image by ImageSize/resample and re-enlarges it")
    parser.add_argument('--gaussian','-g', default=20,
                        type=int,help="Range of gaussian blur")
    parser.add_argument('--noise','-n', default=10,
                        type=int,help="range of noise strength")
    args = parser.parse_args()
    return args

