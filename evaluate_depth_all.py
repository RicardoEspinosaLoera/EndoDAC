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

# Jet colormap for error visualization: blue (low error) -> red (high error)
_JET_COLORMAP = plt.get_cmap('jet', 256)

splits_dir = os.path.join(os.path.dirname(__file__), "splits")

def draw_dashed_rectangle(image, pt1, pt2, color, thickness=2, dash_length=5):
    """
    Draw a dashed rectangle on image (red dashed box for ROI marking).
    Args:
        image: input image (BGR)
        pt1: top-left corner (x, y)
        pt2: bottom-right corner (x, y)
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


def find_brightest_roi(gray_image, roi_size=80):
    """
    Find the brightest region (ROI) in the image.
    Args:
        gray_image: grayscale image
        roi_size: size of ROI (roi_size x roi_size)
    Returns:
        (y1, y2, x1, x2) bounding box coordinates
    """
    h, w = gray_image.shape
    
    # Find brightest point using max intensity
    brightest_y, brightest_x = np.unravel_index(np.argmax(gray_image), gray_image.shape)
    
    # Create fixed-size ROI centered on brightest point
    y1 = max(0, brightest_y - roi_size // 2)
    y2 = min(h, y1 + roi_size)
    x1 = max(0, brightest_x - roi_size // 2)
    x2 = min(w, x1 + roi_size)
    
    # Adjust if ROI exceeds bounds while maintaining size
    if y2 - y1 < roi_size:
        if y1 == 0:
            y2 = min(h, roi_size)
        else:
            y1 = max(0, y2 - roi_size)
    
    if x2 - x1 < roi_size:
        if x1 == 0:
            x2 = min(w, roi_size)
        else:
            x1 = max(0, x2 - roi_size)
    
    return (y1, y2, x1, x2)


def visualize_depth_map(depth, percentile=95):
    """Visualize depth map with inferno colormap and percentile normalization."""
    depth = depth.astype(np.float32)
    valid = depth > 1e-6
    
    if np.sum(valid) == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    
    depth_valid = depth[valid]
    vmin = np.percentile(depth_valid, 5)
    vmax = np.percentile(depth_valid, percentile)
    
    if vmax - vmin < 1e-6:
        depth_norm = np.zeros_like(depth)
    else:
        depth_norm = (depth - vmin) / (vmax - vmin)
    
    depth_norm = np.clip(depth_norm, 0, 1)
    depth_norm[~valid] = 0
    
    inferno = plt.get_cmap('inferno', 256)
    depth_colored = inferno(depth_norm)
    depth_map = (depth_colored[:, :, :3] * 255).astype(np.uint8)
    
    return cv2.cvtColor(depth_map, cv2.COLOR_RGB2BGR)


def visualize_error_map(gt_depth, pred_depth, max_abs_rel=0.2):
    """
    Visualize pixel-wise Abs Rel error map using jet colormap.
    Blue = low error, Red = high error
    
    Args:
        gt_depth: ground truth depth (2D)
        pred_depth: predicted depth (2D, must be depth not disparity)
        max_abs_rel: reference scale for normalization
    
    Returns:
        error_map: BGR error map image (H, W, 3)
        mean_abs_rel: mean Abs Rel error on valid pixels
    """
    gt_depth = gt_depth.astype(np.float32)
    pred_depth = pred_depth.astype(np.float32)
    
    # Create mask for valid pixels
    valid = gt_depth > 1e-6
    
    if np.sum(valid) == 0:
        error_norm = np.zeros_like(gt_depth, dtype=np.float32)
        error_colored = _JET_COLORMAP(error_norm)
        error_map = (error_colored[:, :, :3] * 255).astype(np.uint8)
        return cv2.cvtColor(error_map, cv2.COLOR_RGB2BGR), 0.0
    
    # Initialize output
    error_norm = np.zeros_like(gt_depth, dtype=np.float32)
    
    # Calculate Abs Rel ONLY on valid pixels: |pred - gt| / |gt|
    gt_valid = gt_depth[valid]
    pred_valid = pred_depth[valid]
    abs_rel_valid = np.abs(pred_valid - gt_valid) / (np.abs(gt_valid) + 1e-8)
    
    # Mean Abs Rel for logging
    mean_abs_rel = np.mean(abs_rel_valid)
    
    # Normalize to [0, 1] using absolute scale [0, max_abs_rel]
    # This ensures fair comparison across models
    error_norm[valid] = np.clip(abs_rel_valid / max_abs_rel, 0.0, 1.0)
    # Invalid pixels stay 0 (blue)
    
    # Apply slight smoothing
    error_norm = cv2.GaussianBlur(error_norm, (3, 3), 0.5)
    
    # Apply jet colormap
    error_colored = _JET_COLORMAP(error_norm)
    error_map = (error_colored[:, :, :3] * 255).astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV
    error_map = cv2.cvtColor(error_map, cv2.COLOR_RGB2BGR)
    
    print(f"  Abs Rel - Mean: {mean_abs_rel:.4f}, Min: {np.min(abs_rel_valid):.4f}, Max: {np.max(abs_rel_valid):.4f}")
    
    return error_map, mean_abs_rel


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
    abs_rel_errors = []

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
            # ✓ CRITICAL FIX: Convert disparity to depth (was using disparity as depth!)
            pred_depth = 1.0 / np.clip(pred_disp, 1e-6, None)

            # Save full 2D versions for visualization
            pred_depth_full = pred_depth.copy()
            gt_depth_full = gt_depth.copy()

            # Create mask for valid regions
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            # Extract valid pixel values using mask
            # ✓ CRITICAL FIX: Use pred_depth not pred_disp (was extracting with wrong array!)
            pred_depth_masked = pred_depth[mask]
            gt_depth_masked = gt_depth[mask]

            # Scale prediction
            pred_depth_masked = pred_depth_masked * opt.pred_depth_scale_factor

            scale_ratio = 1.0
            if not opt.disable_median_scaling:
                ratio = np.median(gt_depth_masked) / np.median(pred_depth_masked)
                if not np.isnan(ratio).all():
                    ratios.append(ratio)
                    scale_ratio = ratio
                pred_depth_masked = pred_depth_masked * ratio

            # Clip to valid range
            pred_depth_masked = np.clip(pred_depth_masked, MIN_DEPTH, MAX_DEPTH)
            gt_depth_masked = np.clip(gt_depth_masked, MIN_DEPTH, MAX_DEPTH)

            # Compute errors
            error = compute_errors(gt_depth_masked, pred_depth_masked)
            if not np.isnan(error).all():
                errors.append(error)

            # Apply scaling to full depth maps for visualization consistency
            pred_depth_full = pred_depth_full * opt.pred_depth_scale_factor * scale_ratio
            
            #  Log to WandB (sample frames to avoid rate limiting)
            if i % 5 == 0:
                try:
                    # Downscale for faster WandB upload
                    scale = 4
                    h_viz, w_viz = gt_height // scale, gt_width // scale
                    gt_depth_viz = cv2.resize(gt_depth_full, (w_viz, h_viz), interpolation=cv2.INTER_NEAREST)
                    pred_depth_viz = cv2.resize(pred_depth_full, (w_viz, h_viz), interpolation=cv2.INTER_LINEAR)

                    # Get input image
                    rgb = data[("color", 0, 0)].cpu().numpy()[0].transpose(1, 2, 0)
                    rgb = (rgb * 255).astype(np.uint8)
                    rgb_viz = cv2.resize(rgb, (w_viz, h_viz))
                    rgb_viz_rgb = cv2.cvtColor(rgb_viz, cv2.COLOR_BGR2RGB)

                    # Visualize depth maps (inferno colormap)
                    depth_gt_map = visualize_depth_map(gt_depth_viz, percentile=95)
                    depth_pred_map = visualize_depth_map(pred_depth_viz, percentile=95)
                    depth_gt_rgb = cv2.cvtColor(depth_gt_map, cv2.COLOR_BGR2RGB)
                    depth_pred_rgb = cv2.cvtColor(depth_pred_map, cv2.COLOR_BGR2RGB)

                    # Create error map (jet colormap: blue = low error, red = high error)
                    error_map, mean_abs_rel = visualize_error_map(
                        gt_depth_viz, pred_depth_viz, max_abs_rel=0.2
                    )
                    error_map_rgb = cv2.cvtColor(error_map, cv2.COLOR_BGR2RGB)

                    # Find and mark ROI (brightest region with red dashed box)
                    gray_rgb = cv2.cvtColor(rgb_viz, cv2.COLOR_BGR2GRAY)
                    y1, y2, x1, x2 = find_brightest_roi(gray_rgb, roi_size=80)

                    # Create error map with ROI marked (red dashed rectangle)
                    error_marked = error_map.copy()
                    draw_dashed_rectangle(error_marked, (x1, y1), (x2, y2),
                                        color=(0, 0, 255), thickness=2, dash_length=4)
                    error_marked_rgb = cv2.cvtColor(error_marked, cv2.COLOR_BGR2RGB)

                    # Extract zoomed error map region
                    error_zoomed = error_map[y1:y2, x1:x2] if (y2 > y1 and x2 > x1) else error_map
                    error_zoomed_rgb = cv2.cvtColor(error_zoomed, cv2.COLOR_BGR2RGB) if error_zoomed.size > 0 else error_zoomed

                    # Mark input image with ROI (red dashed rectangle)
                    rgb_marked = rgb_viz.copy()
                    draw_dashed_rectangle(rgb_marked, (x1, y1), (x2, y2),
                                        color=(0, 0, 255), thickness=2, dash_length=4)
                    rgb_marked_rgb = cv2.cvtColor(rgb_marked, cv2.COLOR_BGR2RGB)

                    # Log to WandB (following Figure 5 visualization)
                    wandb.log({
                        "input_image": wandb.Image(rgb_viz_rgb),
                        "input_marked_roi": wandb.Image(rgb_marked_rgb),
                        "depth_gt": wandb.Image(depth_gt_rgb),
                        "depth_pred": wandb.Image(depth_pred_rgb),
                        "error_map_jet": wandb.Image(error_map_rgb),
                        "error_map_with_roi": wandb.Image(error_marked_rgb),
                        "error_map_roi_detail": wandb.Image(error_zoomed_rgb),
                        "frame_idx": i,
                        "abs_rel": mean_abs_rel,
                    }, commit=True)
                    
                    # Collect for statistics
                    abs_rel_errors.append(mean_abs_rel)

                except Exception as e:
                    print(f"Warning: Could not log frame {i} to WandB: {e}")

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
    print("average inference time: {:0.1f} ms".format(np.mean(np.array(inference_times)) * 1000))

    # Final WandB logging
    mean_abs_rel = np.mean(abs_rel_errors) if len(abs_rel_errors) > 0 else 0.0
    wandb.log({
        "final/abs_rel": mean_errors[0],
        "final/sq_rel": mean_errors[1],
        "final/rmse": mean_errors[2],
        "final/rmse_log": mean_errors[3],
        "final/a1": mean_errors[4],
        "final/a2": mean_errors[5],
        "final/a3": mean_errors[6],
        "final/avg_inference_time_ms": np.mean(np.array(inference_times)) * 1000,
    })

    wandb.finish()
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
