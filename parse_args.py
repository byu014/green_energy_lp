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
    parser.add_argument('--rank_blur','-rb', default=3,
                        type=int,help="range of kernel size for rank blur (odd integer)")
    parser.add_argument('--motion_blur','-mb', default=5,
                        type=int,help="range of motion blur")
    parser.add_argument('--brightness','-b', default=0.2,
                        type=float,help="range of brightness")
    args = parser.parse_args()
    return args

