from __future__ import absolute_import, division, print_function

import glob
import os
import random
import numpy as np
import logging
from PIL import Image  # using pillow-simd for increased speed
from PIL import ImageFile
import cv2

import torch
import torch.utils.data as data
from torchvision import transforms

try:
    import tifffile
except ImportError:
    tifffile = None

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Testing scenes for C3VD dataset
TESTING_SCENES = ['trans_t1_a', 'trans_t1_b', 'trans_t2_a', 'trans_t2_b', 'trans_t2_c', 'trans_t3_a', 'trans_t3_b']

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class C3VDDataset(data.Dataset):
    """C3VD Dataset for depth estimation

    Supports loading C3VD dataset with RGB images and TIFF depth maps.
    Follows the structure from Cv3dDepth class with proper depth scaling.

    Args:
        data_path: Path to the C3VD dataset root
        height: Target image height
        width: Target image width
        frame_idxs: Frame indices to load
        num_scales: Number of scales for multi-scale training
        is_train: Whether in training mode
        split: Dataset split ('train', 'val', 'test')
        disparity: If True, return disparity instead of depth
    """
    def __init__(self,
                 data_path,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 split='val',
                 disparity=False):
        super(C3VDDataset, self).__init__()

        self.data_path = data_path
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.LANCZOS

        self.frame_idxs = frame_idxs
        self.is_train = is_train
        self.split = split
        self.disparity = disparity

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        
        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)
        
        # Prepare scene list and scans
        self.scans = []
        self._prepare_dataset()
        
        # Default crop box for C3VD
        self.box = (200, 180, 1150, 900)
        print("Prepared C3VD dataset with %d sets of RGB and depth images (split: %s)." % (len(self.scans), split))

    def _prepare_dataset(self):
        """Prepare dataset by traversing scene directories and collecting image-depth pairs"""
        video_files = sorted([d for d in os.listdir(self.data_path) 
                             if os.path.isdir(os.path.join(self.data_path, d))])
        
        # Filter scenes based on split
        filtered_scenes = self._get_filtered_scenes(video_files)
        
        for scene in filtered_scenes:
            scene_path = os.path.join(self.data_path, scene)
            color_images = sorted([f for f in os.listdir(scene_path) if f.endswith('_color.png')])
            
            for color_img in color_images:
                img_path = os.path.join(scene_path, color_img)
                depth_name = color_img.replace('_color.png', '_depth.tiff')
                depth_path = os.path.join(scene_path, depth_name)
                
                if os.path.exists(depth_path):
                    self.scans.append({
                        "image": img_path,
                        "depth": depth_path,
                        "sequence": scene,
                        "index": color_img[:-10],
                    })
    
    def _get_filtered_scenes(self, all_scenes):
        """Get scenes based on split configuration"""
        if self.split == 'val':
            # Validation uses sigmoid scenes but not testing scenes
            return [s for s in all_scenes if s not in TESTING_SCENES 
                   and os.path.isdir(os.path.join(self.data_path, s))]
        elif self.split == 'train':
            # Training uses sigmoid and testing scenes
            return [s for s in all_scenes if os.path.isdir(os.path.join(self.data_path, s))]
        elif self.split == 'test':
            # Test uses only testing scenes
            return [s for s in all_scenes if s in TESTING_SCENES 
                   and os.path.isdir(os.path.join(self.data_path, s))]
        return []



    def __len__(self):
        return len(self.scans)

    def get_color(self, path, do_flip):
        """Load RGB image"""
        color = self.loader(path)
        if do_flip:
            color = color.transpose(Image.FLIP_LEFT_RIGHT)
        return color

    def get_depth(self, path, do_flip):
        """Load depth from TIFF file
        
        Depth frame: depth along the camera frame's z-axis, clamped from 0-100 millimeters.
        Values are linearly scaled and encoded as a 16-bit grayscale image.
        """
        if tifffile is not None:
            depth_gt = tifffile.imread(path).astype(float)
        else:
            # Fallback to cv2 if tifffile not available
            depth_gt = cv2.imread(path, cv2.IMREAD_ANYDEPTH).astype(float)
        
        # Scale from 16-bit [0, 65535] to millimeters [0, 100]
        depth_gt = depth_gt / 65535.0 * 100.0
        
        # Create output based on disparity flag
        output_depth = np.zeros_like(depth_gt)
        valid_mask = depth_gt > 0
        
        if self.disparity:
            # 1/depth for disparity (with scaling)
            output_depth[valid_mask] = 1000.0 / depth_gt[valid_mask]
        else:
            # Keep depth as is
            output_depth[valid_mask] = depth_gt[valid_mask]
        
        if do_flip:
            output_depth = np.fliplr(output_depth)
        
        return output_depth


    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        scan = self.scans[index]
        image_path = scan["image"]
        depth_path = scan["depth"]
        
        inputs = {}
        
        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        # Load color image and depth
        inputs[("color", 0, 0)] = self.get_color(image_path, do_flip)
        inputs["depth_gt"] = self.get_depth(depth_path, do_flip)

        # Crop images (default C3VD crop box)
        inputs[("color", 0, 0)] = inputs[("color", 0, 0)].crop(self.box)
        inputs["depth_gt"] = inputs["depth_gt"][180:900, 200:1150]
        
        # Resize to target dimensions
        inputs[("color", 0, 0)] = self.resize[0](inputs[("color", 0, 0)])
        inputs[("color", 0, 0)] = self.to_tensor(inputs[("color", 0, 0)])
        
        # Convert depth to tensor
        inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"]).float()
        
        return inputs


