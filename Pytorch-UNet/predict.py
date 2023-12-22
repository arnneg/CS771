import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE', help='Specify the file in which the model is stored')
    parser.add_argument('--input-file', '-f', metavar='FILE', help='File containing the list of input images', required=True)
    parser.add_argument('--output-dir', '-d', metavar='DIR', help='Directory for output images', required=True)
    parser.add_argument('--viz', '-v', action='store_true', help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5, help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
    return parser.parse_args()

def get_output_filename(input_file, output_dir):
    base_name = os.path.basename(input_file)
    return os.path.join(output_dir, os.path.splitext(base_name)[0] + '_OUT.gif')


def mask_to_image(mask: np.ndarray, mask_values):
    # Convert the mask to a binary format if necessary
    # Here, assuming mask is already a binary array with values 0 and 1
    mask = mask.astype(np.uint8) * 255  # Convert 1 to 255 for PIL compatibility

    # Convert numpy array to a PIL Image
    image = Image.fromarray(mask)

    # Define a custom palette: first entry is white (0), second entry is black (1)
    # Palette is a list of 768 values (256 * 3) representing RGB values
    palette = [255, 255, 255, 0, 0, 0] + [0, 0, 0] * 254

    # Convert image to 'P' mode and apply the custom palette
    image = image.convert('P')
    image.putpalette(palette)

    return image



if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Read the input file to get the list of image paths
    with open(args.input_file, 'r') as file:
        in_files = file.read().splitlines()

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    for filename in in_files:
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)

        mask = predict_img(net=net, full_img=img, device=device, scale_factor=args.scale, out_threshold=args.mask_threshold)

        if not args.no_save:
            out_filename = get_output_filename(filename, args.output_dir)
            result = mask_to_image(mask, mask_values)
            result.save(out_filename, format='GIF')
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)