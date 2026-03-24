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
_DEPTH_COLORMAP = plt.get_cmap('jet', 256)

splits_dir = os.path.join(os.path.dirname(__file__), "splits")

def render_depth(disp):
    disp = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
    disp = disp.astype(np.uint8)
    disp_color = cv2.applyColorMap(disp, cv2.COLORMAP_INFERNO)
    return disp_color


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


def visualize_error_map(gt_depth, pred_depth, percentile=95):
    error = np.abs(gt_depth - pred_depth).astype(np.float32)

    valid = gt_depth > 1e-6
    if np.sum(valid) == 0:
        return np.zeros((*error.shape, 3), dtype=np.uint8)

    error_valid = error[valid]

    vmax = np.percentile(error_valid, percentile)
    error_clipped = np.clip(error, 0, vmax)

    error_norm = error_clipped / (vmax + 1e-8)

    # optional smoothing (highly recommended)
    error_norm = cv2.GaussianBlur(error_norm, (5,5), 0)

    error_color = plt.get_cmap('inferno')(error_norm)
    error_map = (error_color[:, :, :3] * 255).astype(np.uint8)

    # mark invalid as gray (VERY important for papers)
    error_map[~valid] = [128, 128, 128]

    return cv2.cvtColor(error_map, cv2.COLOR_RGB2BGR)


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
    wandb.init(project="II-Testing", entity="respinosa")

    if opt.eval_stereo:
        print("   Stereo evaluation")
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = 1.0
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []
    inference_times = []

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
            gt_depth_full = gt_depth.copy()

            # Create mask for valid regions
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            # Extract valid regions
            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

            # Scale prediction
            pred_depth *= opt.pred_depth_scale_factor
            
            if not opt.disable_median_scaling:
                ratio = np.median(gt_depth) / np.median(pred_depth)
                if not np.isnan(ratio).all():
                    ratios.append(ratio)
                pred_depth *= ratio

            # Clip to valid range
            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

            # Compute errors
            error = compute_errors(gt_depth, pred_depth)
            if not np.isnan(error).all():
                errors.append(error)
            
            # Log to WandB (sample every 25 frames to avoid rate limiting)
            if i % 25 == 0 and len(pred_depth) > 0:
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
                    
                    # Create error map using the visualization function
                    error_map = visualize_error_map(gt_depth_resized, pred_depth_resized, percentile=95)
                    
                    # Convert BGR to RGB for WandB
                    depth_pred_rgb = cv2.cvtColor(depth_pred_viz, cv2.COLOR_BGR2RGB)
                    depth_gt_rgb = cv2.cvtColor(depth_gt_viz, cv2.COLOR_BGR2RGB)
                    error_map_rgb = cv2.cvtColor(error_map, cv2.COLOR_BGR2RGB)

                    rgb = data[("color", 0, 0)].cpu().numpy()[0].transpose(1,2,0)
                    rgb = (rgb * 255).astype(np.uint8)

                    # Resize RGB to match error map
                    rgb_resized = cv2.resize(rgb, (error_map.shape[1], error_map.shape[0]))
                    overlay = cv2.addWeighted(rgb_resized, 0.6, error_map, 0.4, 0)
                    
                    wandb.log({
                        "depth_pred": wandb.Image(depth_pred_rgb),
                        "depth_gt": wandb.Image(depth_gt_rgb),
                        "error_map": wandb.Image(error_map_rgb),
                        "error_overlay": wandb.Image(overlay),
                        "frame_idx": i,
                        "abs_error": np.mean(np.abs(gt_depth - pred_depth))
                    }, commit=True)
                except Exception as e:
                    print(f"Warning: Could not log frame {i} to WandB: {e}")

                print("GT valid %:", np.mean(gt_depth_resized > 0))
                print("Pred range:", pred_depth_resized.min(), pred_depth_resized.max())

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
    
    # Log final metrics to WandB
    wandb.log({
        "abs_rel": mean_errors[0],
        "sq_rel": mean_errors[1],
        "rmse": mean_errors[2],
        "rmse_log": mean_errors[3],
        "a1": mean_errors[4],
        "a2": mean_errors[5],
        "a3": mean_errors[6],
        "avg_inference_time_ms": avg_inference_time
    })
    
    wandb.finish()
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
