from __future__ import absolute_import, division, print_function
import os
import cv2
import numpy as np
from tqdm import tqdm
import time
import torch
from torch.utils.data import DataLoader
import scipy.stats as st
import wandb
import matplotlib.pyplot as plt

from utils.layers import disp_to_depth
from utils.utils import readlines, compute_errors
from options import MonodepthOptions
from datasets.scared_dataset import SCAREDRAWDataset
from datasets.hamlyn_dataset import HamlynDataset
from datasets.c3vd_dataset import C3VDDataset
import models.encoders as encoders
import models.decoders as decoders
import models.endodac as endodac
import models.hadepth as hadepth
import models.endosfmlearner as endosfmlearner
import models.monovit as monovit

cv2.setNumThreads(0)

# Initialize jet colormap for visualization (blue = low, red = high)
_ERROR_COLORMAP = plt.get_cmap('jet', 256)
_DEPTH_COLORMAP = plt.get_cmap('plasma', 256)  # for plotting

splits_dir = os.path.join(os.path.dirname(__file__), "splits")

def draw_dashed_rectangle(image, pt1, pt2, color, thickness=1, dash_length=5):
    """
    Draw a dashed rectangle on image.
    Args:
        image: input image
        pt1: top-left corner (x1, y1)
        pt2: bottom-right corner (x2, y2)
        color: line color (B, G, R)
        thickness: line thickness
        dash_length: length of each dash
    """
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Top line
    for i in range(x1, x2, dash_length * 2):
        cv2.line(image, (i, y1), (min(i + dash_length, x2), y1), color, thickness)
    
    # Bottom line
    for i in range(x1, x2, dash_length * 2):
        cv2.line(image, (i, y2), (min(i + dash_length, x2), y2), color, thickness)
    
    # Left line
    for i in range(y1, y2, dash_length * 2):
        cv2.line(image, (x1, i), (x1, min(i + dash_length, y2)), color, thickness)
    
    # Right line
    for i in range(y1, y2, dash_length * 2):
        cv2.line(image, (x2, i), (x2, min(i + dash_length, y2)), color, thickness)
    
    return image


def get_brightness_mask(rgb_image, threshold=100):
    """
    Create mask for bright regions in the image.
    Args:
        rgb_image: input RGB image (0-255)
        threshold: brightness threshold (0-255)
    Returns:
        binary mask where bright regions = 1
    """
    # Convert to grayscale
    if len(rgb_image.shape) == 3:
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = rgb_image
    
    # Create brightness mask
    bright_mask = gray > threshold
    return bright_mask


def apply_brightness_mask(error_map, rgb_image, threshold=100):
    """
    Apply brightness mask to error map - keep only bright regions.
    Dark regions become black/blue.
    """
    bright_mask = get_brightness_mask(rgb_image, threshold)
    
    # Apply mask - keep error map only where image is bright
    error_map_masked = error_map.copy()
    # Use 2D mask directly - numpy will broadcast to all 3 channels
    error_map_masked[~bright_mask] = [0, 0, 255]  # Dark regions become blue
    
    return error_map_masked


def get_brightest_region(rgb_image, region_size=100):
    """
    Find the brightest region in the image and return bounding box.
    Ensures bounding box is always the same size (region_size x region_size).
    """
    if len(rgb_image.shape) == 3:
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = rgb_image
    
    h, w = gray.shape
    
    # Find brightest point
    brightest_y, brightest_x = np.unravel_index(np.argmax(gray), gray.shape)
    
    # Create bounding box around brightest point with fixed size
    # Center on brightest point
    y1 = brightest_y - region_size // 2
    y2 = y1 + region_size
    x1 = brightest_x - region_size // 2
    x2 = x1 + region_size
    
    # Clamp to image bounds while maintaining size
    if y1 < 0:
        y1 = 0
        y2 = region_size
    if y2 > h:
        y2 = h
        y1 = max(0, h - region_size)
    
    if x1 < 0:
        x1 = 0
        x2 = region_size
    if x2 > w:
        x2 = w
        x1 = max(0, w - region_size)
    
    return (y1, y2, x1, x2)


def create_zoomed_with_marker(error_map, rgb_image, region_size=100):
    """
    Create error map with zoomed region marked by dashed box.
    """
    y1, y2, x1, x2 = get_brightest_region(rgb_image, region_size)
    
    # Zoom into the error map
    zoomed_error = error_map[y1:y2, x1:x2]
    
    # Create marked version with dashed bounding box on full image
    marked = error_map.copy()
    draw_dashed_rectangle(marked, (x1, y1), (x2, y2), (0, 0, 255), thickness=2, dash_length=4)
    
    return zoomed_error, marked, (y1, y2, x1, x2)


def visualize_depth_map(depth, percentile=95):
    depth = depth.astype(np.float32)

    # ✅ mask valid depth
    valid = depth > 1e-6
    if np.sum(valid) == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    depth_valid = depth[valid]

    # robust stats ONLY on valid pixels
    vmax = np.percentile(depth_valid, percentile)
    vmin = np.percentile(depth_valid, 5)

    if vmax - vmin < 1e-6:
        depth_norm = np.zeros_like(depth)
    else:
        depth_norm = (depth - vmin) / (vmax - vmin)

    depth_norm = np.clip(depth_norm, 0, 1)

    # optional: set invalid to 0
    depth_norm[~valid] = 0

    depth_color = _DEPTH_COLORMAP(depth_norm)
    depth_viz = (depth_color[:, :, :3] * 255).astype(np.uint8)

    return cv2.cvtColor(depth_viz, cv2.COLOR_RGB2BGR)


def visualize_error_map(gt_depth, pred_depth, percentile=75):
    """
    Visualize pixel-wise Abs Rel error map.
    |pred - gt| / |gt| normalized with percentile-based scaling for adaptive range.
    Returns error map image and mean Abs Rel metric.
    """
    valid = gt_depth > 1e-6
    
    # Initialize output
    error_norm = np.zeros_like(gt_depth, dtype=np.float32)
    
    if np.sum(valid) == 0:
        error_color = _ERROR_COLORMAP(error_norm)
        error_map = (error_color[:, :, :3] * 255).astype(np.uint8)
        return cv2.cvtColor(error_map, cv2.COLOR_RGB2BGR), 0.0
    
    # Calculate pixel-wise Abs Rel error ONLY on valid regions
    gt_valid = gt_depth[valid]
    pred_valid = pred_depth[valid]
    abs_rel_valid = np.abs(gt_valid - pred_valid) / (np.abs(gt_valid) + 1e-8)
    
    # Compute percentiles on valid data only (adaptive scaling)
    vmin = np.percentile(abs_rel_valid, 5)
    vmax = np.percentile(abs_rel_valid, percentile)
    
    # Compute mean Abs Rel for logging
    abs_rel_error_map = np.mean(abs_rel_valid)
    
    # Normalize valid pixels only to [0, 1]
    if vmax - vmin > 1e-6:
        # Blue = low error (0), Red = high error (1)
        normalized_valid = np.clip((abs_rel_valid - vmin) / (vmax - vmin), 0, 1)
        error_norm[valid] = normalized_valid
    else:
        error_norm[valid] = 0.5  # Uniform error if no variance
    
    # Set invalid pixels to 0 (blue = no error info)
    error_norm[~valid] = 0
    
    # Slight smoothing to preserve detail
    error_norm = cv2.GaussianBlur(error_norm, (3, 3), 0.5)
    
    error_color = _ERROR_COLORMAP(error_norm)  # Use jet colormap
    error_map = (error_color[:, :, :3] * 255).astype(np.uint8)
    
    print(f"  Abs Rel: range=[{vmin:.4f}, {vmax:.4f}], mean={abs_rel_error_map:.4f}")

    return cv2.cvtColor(error_map, cv2.COLOR_RGB2BGR), abs_rel_error_map


class DepthModelFactory:
    """Factory class to load different depth estimation models"""
    
    @staticmethod
    def load_model(model_type, opt):
        """Load model based on type"""
        if model_type == 'endodac':
            return DepthModelFactory._load_endodac(opt)
        elif model_type == 'hadepth':
            return DepthModelFactory._load_hadepth(opt)
        elif model_type == 'afsfm':
            return DepthModelFactory._load_afsfm(opt)
        elif model_type == 'endosfml':
            return DepthModelFactory._load_endosfmlearner(opt)
        elif model_type == 'monovit':
            return DepthModelFactory._load_monovit(opt)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def _load_endodac(opt):
        """Load EndoDAC model"""
        depther_path = os.path.join(opt.load_weights_folder, "depth_model.pth")
        depther_dict = torch.load(depther_path)
        
        depther = endodac.endodac(
            backbone_size="base", r=opt.lora_rank, lora_type=opt.lora_type,
            image_shape=(224, 280), pretrained_path=opt.pretrained_path,
            residual_block_indexes=opt.residual_block_indexes,
            include_cls_token=opt.include_cls_token)
        model_dict = depther.state_dict()
        depther.load_state_dict({k: v for k, v in depther_dict.items() if k in model_dict}, strict=False)
        depther.cuda()
        depther.eval()
        return depther, 'endodac'
    
    @staticmethod
    def _load_hadepth(opt):
        """Load HADepth model"""
        depther_path = os.path.join(opt.load_weights_folder, "depth_model.pth")
        depther_dict = torch.load(depther_path)

        depther_dict = torch.load(depther_path)
        residual_keys = [k for k in depther_dict.keys() if 'residual_' in k]
        print(f"Residual keys in checkpoint: {len(residual_keys)}")
        print(residual_keys[:5])
        
        depther = hadepth.hadepth(
            backbone_size="base",
            r=opt.lora_rank,
            lora_type=opt.lora_type,
            image_shape=(224, 280),
            pretrained_path=opt.pretrained_path,
            residual_block_indexes=[2, 5, 8, 11],  # explicitly hardcode this
            include_cls_token=opt.include_cls_token
)
        model_dict = depther.state_dict()
        depther.load_state_dict({k: v for k, v in depther_dict.items() if k in model_dict})
        #depther.load_state_dict({k: v for k, v in depther_dict.items() if k in model_dict}, strict=False)
        

                
        depther.cuda()
        depther.eval()
        return depther, 'hadepth'
    
    @staticmethod
    def _load_afsfm(opt):
        """Load AFSFM model (ResNet + DepthDecoder)"""
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
        encoder_dict = torch.load(encoder_path)
        
        encoder = encoders.ResnetEncoder(opt.num_layers, False)
        depth_decoder = decoders.DepthDecoder(encoder.num_ch_enc, scales=range(4))
        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))
        
        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()
        
        depther = lambda image: depth_decoder(encoder(image))
        return depther, 'afsfm'
    
    @staticmethod
    def _load_afslearner(opt):
        """Load AFSLearner model (ResNet + DepthDecoder)"""
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
        encoder_dict = torch.load(encoder_path)
        
        encoder = encoders.ResnetEncoder(opt.num_layers, False)
        depth_decoder = decoders.DepthDecoder(encoder.num_ch_enc, scales=range(4))
        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))
        
        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()
        
        depther = lambda image: depth_decoder(encoder(image))
        return depther, 'afslearner'
    
    @staticmethod
    def _load_endosfmlearner(opt):

        dispnet_path = os.path.join(opt.load_weights_folder, "dispnet_model_best.pth.tar")
        weights = torch.load(dispnet_path)
        
        # Create model
        disp_net = endosfmlearner.DispResNet(num_layers=opt.num_layers, pretrained=False)
        
        # Load weights
        if isinstance(weights, dict) and 'state_dict' in weights:
            disp_net.load_state_dict(weights['state_dict'])
        else:
            disp_net.load_state_dict(weights)
        
        # Move to GPU and set to eval mode
        disp_net.cuda()
        disp_net.eval()
        
        return disp_net, 'endosfmlearner'
    
    @staticmethod
    def _load_monovit(opt):
        """Load MonoViT model (MPViT encoder + DepthDecoderT)"""
        
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
        encoder_dict = torch.load(encoder_path)
    
        # Create encoder
        encoder = monovit.mpvit_small()
        encoder.num_ch_enc = [64, 128, 216, 288, 288]
        
        # Create decoder for transformer-based model
        depth_decoder = monovit.DepthDecoderT()
        
        # Load weights
        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))
        
        # Move to GPU and set to eval mode
        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()
        
        # Create depther function
        depther = lambda image: depth_decoder(encoder(image))
        return depther, 'monovit'


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set"""
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 150

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    print("-> Loading weights from {}".format(opt.load_weights_folder))

    # Load model using factory
    depther, model_name = DepthModelFactory.load_model(opt.model_type, opt)
    print(f"-> Loaded {model_name} model")

    # Load dataset based on eval_split
    if opt.eval_split == 'endovis':
        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        dataset = SCAREDRAWDataset(opt.data_path, filenames,
                                        opt.height, opt.width,
                                        [0], 4, is_train=False)
    elif opt.eval_split == 'hamlyn':
        dataset = HamlynDataset(opt.data_path, opt.height, opt.width,
                                            [0], 4, is_train=False)
    elif opt.eval_split == 'c3vd':
        dataset = C3VDDataset(opt.data_path, opt.height, opt.width,
                              [0], 4, is_train=False, split='test')
        MAX_DEPTH = 100
    else:
        raise ValueError(f"Unknown eval_split: {opt.eval_split}")

    dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=opt.num_workers,
                            pin_memory=True, drop_last=False)

    # Load ground truth
    if opt.eval_split == 'endovis':
        gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
        gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]

    print("-> Evaluating on {} split".format(opt.eval_split))
    
    # Initialize WandB for logging
    #run_name = opt.load_weights_folder.split(os.sep)[-1] if opt.load_weights_folder else "depth_eval"
    #wandb.init(project="endodac-depth-eval", name=run_name, config=vars(opt))
    wandb.init(project="II-Testing-zoomed", entity="respinosa")

    if opt.eval_stereo:
        print("   Stereo evaluation")
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = 1.0
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []
    inference_times = []
    abs_rel_error_maps = []

    print("-> Computing predictions with size {}x{}".format(opt.width, opt.height))

    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader)):
            input_color = data[("color", 0, 0)].cuda()
            
            if opt.post_process:
                # Post-processed results require each image to have two forward passes
                input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

            # Forward pass
            time_start = time.time()
            output = depther(input_color)
            inference_time = time.time() - time_start
            inference_times.append(inference_time)            

            # Extract disparity
            if isinstance(output, dict):
                output_disp = output[("disp", 0)]
            else:
                output_disp = output
            
            pred_disp, _ = disp_to_depth(output_disp, opt.min_depth, opt.max_depth)
            pred_disp = pred_disp.cpu()[:, 0].numpy()
            pred_disp = pred_disp[0] if pred_disp.shape[0] == 1 else pred_disp

            # Get ground truth
            if opt.eval_split == 'endovis':
                gt_depth = gt_depths[i]
                #sequence = str(np.array(data['sequence'][0]))
                #keyframe = str(np.array(data['keyframe'][0]))
                frame_id = "{:06d}".format(data['frame_id'][0])
            elif opt.eval_split == 'hamlyn' or opt.eval_split == 'c3vd':
                gt_depth = data["depth_gt"].squeeze().numpy()
            
            # Handle 3D gt_depth (extract first channel if needed)
            # if gt_depth.ndim == 3:
            #    gt_depth = gt_depth[:, :, 0]

            # Resize prediction to match ground truth
            gt_height, gt_width = gt_depth.shape[:2]
            pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
            pred_disp = np.clip(pred_disp, 1e-6, None)
            pred_depth = 1.0 / pred_disp
            
            # Save full 2D versions for visualization BEFORE masking
            pred_depth_full = pred_depth.copy()
            gt_depth_full = gt_depth.copy().astype(np.float32)  # Convert to float for consistency

            # Create mask for valid regions
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            # Extract valid regions
            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

            # Scale prediction
            pred_depth *= opt.pred_depth_scale_factor
            
            # Compute median scaling ratio
            scale_ratio = 1.0
            if not opt.disable_median_scaling:
                ratio = np.median(gt_depth) / np.median(pred_depth)
                if not np.isnan(ratio).all():
                    ratios.append(ratio)
                    scale_ratio = ratio
                pred_depth *= ratio
                # Debug: print scaling info to identify model differences
                if i % 5 == 0:
                    print(f"Frame {i}: GT range=[{np.min(gt_depth):.4f}, {np.max(gt_depth):.4f}], "
                          f"Pred raw range=[{np.min(pred_depth/ratio):.4f}, {np.max(pred_depth/ratio):.4f}], "
                          f"Scale ratio={ratio:.4f}")
            
            # Apply same scaling to full depth maps for consistent visualization
            pred_depth_full *= opt.pred_depth_scale_factor * scale_ratio
            gt_depth_full *= 1.0  # GT doesn't need scaling

            # Clip to valid range
            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

            # Compute errors
            error = compute_errors(gt_depth, pred_depth)
            if not np.isnan(error).all():
                errors.append(error)
            
            # Log to WandB (sample every 5 frames to avoid rate limiting)
            if i % 5 == 0 and len(pred_depth) > 0:
                # Visualize depth using FULL 2D maps before masking (better visualization)
                # Resize full maps to smaller size for faster upload
                h_viz, w_viz = int(gt_height / 4), int(gt_width / 4)
                #gt_depth_resized = cv2.resize(gt_depth_full, (w_viz, h_viz))
                #pred_depth_resized = cv2.resize(pred_depth_full, (w_viz, h_viz))

                gt_depth_resized = cv2.resize(gt_depth_full, (w_viz, h_viz), interpolation=cv2.INTER_NEAREST)
                pred_depth_resized = cv2.resize(pred_depth_full, (w_viz, h_viz), interpolation=cv2.INTER_LINEAR)
                
                try:
                    # Visualize depth: clip at 95th percentile
                    depth_pred_viz = visualize_depth_map(pred_depth_resized, percentile=95)
                    depth_gt_viz = visualize_depth_map(gt_depth_resized, percentile=95)
                    
                    # Create error map - pixel-wise Abs Rel with percentile-based adaptive scaling
                    error_map, abs_rel_error_map = visualize_error_map(gt_depth_resized, pred_depth_resized, percentile=75)
                    
                    # Convert BGR to RGB for WandB
                    depth_pred_rgb = cv2.cvtColor(depth_pred_viz, cv2.COLOR_BGR2RGB)
                    depth_gt_rgb = cv2.cvtColor(depth_gt_viz, cv2.COLOR_BGR2RGB)
                    error_map_rgb = cv2.cvtColor(error_map, cv2.COLOR_BGR2RGB)

                    rgb = data[("color", 0, 0)].cpu().numpy()[0].transpose(1,2,0)
                    rgb = (rgb * 255).astype(np.uint8)

                    # Resize RGB to match error map for consistent logging
                    rgb_resized = cv2.resize(rgb, (error_map.shape[1], error_map.shape[0]))
                    rgb_resized_rgb = cv2.cvtColor(rgb_resized, cv2.COLOR_BGR2RGB)
                    
                    # Apply brightness mask to error map (keep only bright regions)
                    error_map_bright = apply_brightness_mask(error_map, rgb_resized, threshold=100)
                    error_map_bright_rgb = cv2.cvtColor(error_map_bright, cv2.COLOR_BGR2RGB)
                    
                    # Create zoomed region with marking
                    zoomed_error, marked_error, bbox = create_zoomed_with_marker(error_map, rgb_resized, region_size=100)
                    marked_error_rgb = cv2.cvtColor(marked_error, cv2.COLOR_BGR2RGB)
                    zoomed_error_rgb = cv2.cvtColor(zoomed_error, cv2.COLOR_BGR2RGB)
                    
                    # Mark input image with dashed bounding box
                    marked_rgb = rgb_resized.copy()
                    y1, y2, x1, x2 = bbox
                    draw_dashed_rectangle(marked_rgb, (x1, y1), (x2, y2), (0, 0, 255), thickness=2, dash_length=4)
                    marked_rgb_rgb = cv2.cvtColor(marked_rgb, cv2.COLOR_BGR2RGB)
                    
                    wandb.log({
                        "input_image": wandb.Image(rgb_resized_rgb),
                        "input_image_marked": wandb.Image(marked_rgb_rgb),
                        "depth_pred": wandb.Image(depth_pred_rgb),
                        "depth_gt": wandb.Image(depth_gt_rgb),
                        "error_map": wandb.Image(error_map_rgb),
                        "error_map_bright_regions": wandb.Image(error_map_bright_rgb),
                        "error_map_marked": wandb.Image(marked_error_rgb),
                        "error_map_zoomed": wandb.Image(zoomed_error_rgb),
                        "frame_idx": i,
                        "abs_error": np.mean(np.abs(gt_depth - pred_depth)),
                        "abs_rel_error_map": abs_rel_error_map
                    }, commit=True)
                    abs_rel_error_maps.append(abs_rel_error_map)
                except Exception as e:
                    print(f"Warning: Could not log frame {i} to WandB: {e}")

                #print("GT valid %:", np.mean(gt_depth_resized > 0))
                #print("Pred range:", pred_depth_resized.min(), pred_depth_resized.max())

    # Print results
    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    errors = np.array(errors)
    mean_errors = np.mean(errors, axis=0)
    
    # Compute 95% confidence intervals
    cls = []
    for i in range(len(mean_errors)):
        cl = st.t.interval(alpha=0.95, df=len(errors)-1, loc=mean_errors[i], scale=st.sem(errors[:, i]))
        cls.append(cl[0])
        cls.append(cl[1])
    cls = np.array(cls)

    print("\n       " + ("{:>11}      | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print("mean:" + ("&{: 12.3f}      " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("cls: " + ("& [{: 6.3f}, {: 6.3f}] " * 7).format(*cls.tolist()) + "\\\\")
    avg_inference_time = np.mean(np.array(inference_times)) * 1000
    print("average inference time: {:0.1f} ms".format(avg_inference_time))
    
    # Compute mean Abs Rel for error maps
    mean_abs_rel_error_map = np.mean(abs_rel_error_maps) if len(abs_rel_error_maps) > 0 else 0.0
    print(f"mean abs_rel error map (max=0.2): {mean_abs_rel_error_map:.6f}")
    
    # Log final metrics to WandB
    wandb.log({
        "abs_rel": mean_errors[0],
        "sq_rel": mean_errors[1],
        "rmse": mean_errors[2],
        "rmse_log": mean_errors[3],
        "a1": mean_errors[4],
        "a2": mean_errors[5],
        "a3": mean_errors[6],
        "avg_inference_time_ms": avg_inference_time,
        "mean_abs_rel_error_map": mean_abs_rel_error_map
    })
    
    wandb.finish()
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
