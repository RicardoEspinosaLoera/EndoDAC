# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import json
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import os
import cv2
import torch
from torchvision import transforms

import models.endodac as endodac
import models.hadepth as hadepth


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for ManyDepth models.')

    parser.add_argument('--images_path', type=str,
                        help='path to a test image to predict for', required=True)
    parser.add_argument('--load_weights_folder', type=str,
                        help='path to a folder of weights to load', required=True)

    parser.add_argument('--output_path', type=str,
                        help='path to save depths', required=True)
    
    parser.add_argument('--model_type', type=str, default='monovit', 
                        choices=('endodac', 'hadepth', 'monovit'),
                        help='Type of model to use: endodac, hadepth, or monovit',
                        required=False)
                        
    parser.add_argument('--mode', type=str, default='multi', choices=('multi', 'mono'),
                        help='"multi" or "mono". If set to "mono" then the network is run without '
                             'the source image, e.g. as described in Table 5 of the paper.',
                        required=False)
    
    # Additional arguments for EndoDAC and HaDepth
    parser.add_argument('--lora_rank', type=int, default=4,
                        help='LoRA rank for EndoDAC and HaDepth models', required=False)
    parser.add_argument('--lora_type', type=str, default='lora',
                        help='LoRA type (lora, dora, flora)', required=False)
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='Path to pretrained weights for EndoDAC and HaDepth', required=False)
    parser.add_argument('--residual_block_indexes', type=int, nargs='+', default=[2, 5, 8, 11],
                        help='Residual block indexes for EndoDAC and HaDepth', required=False)
    parser.add_argument('--include_cls_token', action='store_true',
                        help='Include CLS token in EndoDAC and HaDepth models', required=False)
    
    parser.add_argument('--min_depth', type=float, default=0.1,
                        help='Minimum depth for disp_to_depth conversion', required=False)
    parser.add_argument('--max_depth', type=float, default=150.0,
                        help='Maximum depth for disp_to_depth conversion', required=False)
    
    parser.add_argument('--dataset', type=str, default='hamlyn',
                        choices=['hamlyn', 'endovis', 'c3vd'],
                        help='Dataset type (hamlyn, endovis, c3vd) for default depth bounds', required=False)
    
    parser.add_argument('--scale', type=float, default=52.864,
                        help='Image depth scaling. For Hamlyn dataset the weighted average baseline is 52.864',
                        required=False)
    
    parser.add_argument('--saturation_depth', type=float, default=300.0,
                        help='Saturation depth of the estimated depth images. For Hamlyn dataset it is 300 mm by default',
                        required=False)
    
    return parser.parse_args()

def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def load_and_preprocess_image(image_path, resize_width, resize_height):
    image = pil.open(image_path).convert('RGB')
    original_width, original_height = image.size
    image = image.resize((resize_width, resize_height), pil.LANCZOS)
    image = transforms.ToTensor()(image).unsqueeze(0)
    if torch.cuda.is_available():
        return image.cuda(), (original_height, original_width)
    return image, (original_height, original_width)


def load_model(model_type, weights_folder, args):
    """Load the appropriate depth estimation model"""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    if model_type == 'hadepth':
        print("-> Loading HaDepth model")
        depther_path = os.path.join(weights_folder, "depth_model.pth")
        depther_dict = torch.load(depther_path, map_location=device)
        
        depther = hadepth.hadepth(
            backbone_size="base",
            r=args.lora_rank,
            lora_type="dora",
            image_shape=(224, 280),
            pretrained_path=args.pretrained_path,
            residual_block_indexes=args.residual_block_indexes,
            include_cls_token=args.include_cls_token)
        model_dict = depther.state_dict()
        depther.load_state_dict({k: v for k, v in depther_dict.items() if k in model_dict})
        depther.cuda()
        depther.eval()
        return depther, (224, 280), 'hadepth'
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def process_image(image, model, model_type, output_size):
    """Process image and return depth map"""
    with torch.no_grad():
        if model_type == 'monovit':
            encoder, depth_decoder = model
            output = depth_decoder(encoder(image))[("disp", 0)]
        else:  # endodac, hadepth
            # For EndoDAC and HaDepth, the model returns depth directly
            output = model(image)
    
    return output, model_type


def test_simple(args):
    """Function to predict for a single image or folder of images
    """

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("-> Loading model from ", args.load_weights_folder)
    print("-> Model type: ", args.model_type)
    print("-> Dataset: ", args.dataset)

    # Set saturation depth based on dataset if using defaults
    if args.dataset == 'hamlyn':
        if args.saturation_depth == 300.0:  # Using default
            args.saturation_depth = 300.0  # Hamlyn typically has max depth around 300mm
            print("-> Using Hamlyn-specific saturation_depth: 300.0 mm")
    
    print(f"-> Depth scale: {args.scale}, saturation_depth: {args.saturation_depth}")
    
    # Loading pretrained model
    print("   Loading pretrained model")
    
    model, input_size, model_type = load_model(args.model_type, args.load_weights_folder, args)
    
    # Load input data
    dir_list = os.listdir(args.images_path)
    depth_stats = {'min': [], 'max': [], 'mean': [], 'unique_values': []}
    
    for idx, i in enumerate(dir_list):
        if not i.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        print(f"Processing {idx+1}/{len(dir_list)}: {i}")
        
        # Load and preprocess image based on model type
        if model_type in ['endodac', 'hadepth']:
            # EndoDAC/HaDepth will internally resize to 224x280, so pass image at reasonable size
            HEIGHT, WIDTH = 384, 512
            input_image, original_size = load_and_preprocess_image(
                os.path.join(args.images_path, i), 
                resize_width=WIDTH, 
                resize_height=HEIGHT)
            original_height, original_width = original_size
            
            with torch.no_grad():
                # Model internally resizes to 224x280
                output = model(input_image)
                
                # Extract disparity from output dict
                if isinstance(output, dict):
                    disp = output[("disp", 0)]
                else:
                    disp = output
                
                # Resize disparity to ORIGINAL image dimensions
                disp_resized = torch.nn.functional.interpolate(
                    disp, (original_height, original_width), mode="bilinear", align_corners=False)
                
                # Convert disparity to depth using min/max depth bounds
                disp_resized_np = disp_resized.squeeze().cpu().numpy()
                
                # DEBUG: Check raw disparity range
                if idx < 1:
                    print(f"   RAW OUTPUT RANGE: [{np.min(disp_resized_np):.6f}, {np.max(disp_resized_np):.6f}]")
                    print(f"   MEAN: {np.mean(disp_resized_np):.6f}, STD: {np.std(disp_resized_np):.6f}")
                    print(f"   Using scale: {args.scale}, saturation_depth: {args.saturation_depth}")
                
                # For HaDepth: use raw output directly as depth (skip disp_to_depth conversion)
                depth = disp_resized_np * args.scale
                
                # Only clip if values exceed saturation depth
                if np.max(depth) > args.saturation_depth:
                    depth[depth > args.saturation_depth] = args.saturation_depth

        # Save depth as uint16 PNG (keeping original output format)
        im_depth = depth.astype(np.uint16)
        
        # Collect statistics
        depth_stats['min'].append(np.min(depth))
        depth_stats['max'].append(np.max(depth))
        depth_stats['mean'].append(np.mean(depth))
        unique_vals = len(np.unique(depth))
        depth_stats['unique_values'].append(unique_vals)
        
        if idx < 3 or idx % 50 == 0:  # Print stats for first 3 and every 50th image
            print(f"   Image {i}: depth range=[{np.min(depth):.2f}, {np.max(depth):.2f}], "
                  f"mean={np.mean(depth):.2f}, unique_values={unique_vals}")
        
        im = pil.fromarray(im_depth)
        output_name = i.replace(".jpg","")
        output_file = os.path.join(args.output_path, "{}.png".format(output_name))
        im.save(output_file)

    print('-> Done!')
    
    # Print summary statistics
    if depth_stats['min']:
        print("\n--- Depth Statistics Summary ---")
        print(f"Overall depth range: [{np.mean(depth_stats['min']):.2f}, {np.mean(depth_stats['max']):.2f}] mm")
        print(f"Average mean depth: {np.mean(depth_stats['mean']):.2f} mm")
        print(f"Average unique values per image: {np.mean(depth_stats['unique_values']):.0f}")
        
        if np.mean(depth_stats['unique_values']) < 100:
            print("\n⚠️ WARNING: Very few unique depth values detected!")
            print("This suggests the depth predictions may be poorly distributed.")
            print("Try adjusting --min_depth and --max_depth parameters.")
            print(f"Current: --min_depth {args.min_depth} --max_depth {args.max_depth}")


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
