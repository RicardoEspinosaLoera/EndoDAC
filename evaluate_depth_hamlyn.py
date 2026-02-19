from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
import scipy.stats as st

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.layers import disp_to_depth
from utils.utils import readlines, compute_errors
from options import MonodepthOptions
from datasets.scared_dataset import SCAREDRAWDataset
import models.endodac as endodac

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

splits_dir = os.path.join(os.path.dirname(__file__), "splits")

def evaluate(opt):
    """Evaluates endodac model on HAMLYN dataset
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 150

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    print("-> Loading weights from {}".format(opt.load_weights_folder))

    # Load checkpoint
    depther_path = os.path.join(opt.load_weights_folder, "depth_model.pth")
    depther_dict = torch.load(depther_path)

    # Load dataset
    filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
    dataset = SCAREDRAWDataset(opt.data_path, filenames,
                                    opt.height, opt.width,
                                    [0], 4, is_train=False)
    dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=opt.num_workers,
                            pin_memory=True, drop_last=False)

    # Create model
    depther = endodac.endodac(
        backbone_size="base", r=4, lora_type="dvlora",
        image_shape=(224, 280), pretrained_path=None,
        residual_block_indexes=[2, 5, 8, 11],
        include_cls_token=True)

    model_dict = depther.state_dict()
    depther.load_state_dict({k: v for k, v in depther_dict.items() if k in model_dict}, strict=False)
    depther.cuda()
    depther.eval()

    # Load ground truth
    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]

    print("-> Evaluating on {} split".format(opt.eval_split))

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
            import time
            time_start = time.time()
            output = depther(input_color)
            inference_time = time.time() - time_start
            inference_times.append(inference_time)

            # Extract disparity
            output_disp = output[("disp", 0)]
            pred_disp, _ = disp_to_depth(output_disp, opt.min_depth, opt.max_depth)
            pred_disp = pred_disp.cpu()[:, 0].numpy()
            pred_disp = pred_disp[0] if pred_disp.shape[0] == 1 else pred_disp

            # Get ground truth
            gt_depth = gt_depths[i]
            
            # Handle 3D gt_depth (extract first channel if needed)
            if gt_depth.ndim == 3:
                gt_depth = gt_depth[:, :, 0]

            # Resize prediction to match ground truth
            gt_height, gt_width = gt_depth.shape[:2]
            pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
            pred_depth = 1 / pred_disp

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
    print("\n-> Done!")

if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
