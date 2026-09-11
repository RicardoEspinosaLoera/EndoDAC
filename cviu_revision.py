#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""cviu_revision.py: every new experiment of the MonoIIF CVIU revision, in one script.

Stages (each is idempotent: finished work is skipped unless --force is given):

  train         launch the ablation grid (E/R/C runs x seeds) as train_end_to_end.py subprocesses
  predict       per-frame depth metrics for every trained run and external method -> per_frame.csv
  illum-fit     model-free test of the affine illumination model on ground-truth geometry (SCARED)
  illum-params  statistics of the learned (c, b) calibration maps of a trained run
  illum-sens    response of the calibration to synthetic gain/bias, and depth robustness to it
  da3           zero-shot Depth Anything 3 rows
  stats         sequence-level aggregation, bootstrap CIs, paired tests, LaTeX tables
  report        markdown report and figures
  all           predict, illum-fit, illum-params, illum-sens, da3, stats, report (not train)
  full          train, then everything in `all`; stops if any training job failed

Examples:
  python cviu_revision.py full --gpus 0 1 2 3          # the whole revision in one go (run it in tmux)
  python cviu_revision.py train --gpus 0 1 2 3 --dry_run
  python cviu_revision.py train --gpus 0 --only E3 E8 --extra_flags "--num_epochs 1"
  python cviu_revision.py predict --datasets scared
  python cviu_revision.py stats && python cviu_revision.py report

The rationale of every experiment is in CVIU_REVISION_PLAN.md. All settings live in
cviu_config.yaml (missing keys fall back to DEFAULT_CONFIG below).
"""
from __future__ import absolute_import, division, print_function

import argparse
import copy
import csv
import glob
import json
import os
import queue
import shlex
import shutil
import subprocess
import sys
import threading
import time
import traceback

import numpy as np

try:  # stats/report run without torch; everything else needs it
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None

ROOT = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(ROOT, "splits", "endovis")
METRICS = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
LOWER_IS_BETTER = {"abs_rel": True, "sq_rel": True, "rmse": True, "rmse_log": True,
                   "a1": False, "a2": False, "a3": False}
MIN_DEPTH = 1e-3
MAX_DEPTH = {"scared": 150.0, "hamlyn": 150.0, "c3vd": 100.0}
H, W = 256, 320            # training / evaluation resolution of the repo
DEPTH_RANGE = (0.1, 150.0)  # --min_depth / --max_depth defaults of options.py

# --------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "data": {"scared": "/mnt/data-hdd2/Beilei/Dataset/SCARED",
             "hamlyn": "/mnt/data-hdd2/Beilei/Dataset/Hamlyn",
             "c3vd": "/mnt/data-hdd2/Beilei/Dataset/C3VD"},
    "log_dir": "./logs",
    "out_dir": "./results/cviu",
    "pretrained_path": "./pretrained_model",
    "python": sys.executable,
    "num_workers": 4,
    "train": {
        "common_flags": "--num_epochs 20 --batch_size 8 --learn_intrinsics True --wandb_mode offline",
        "seeds": [314],
        "multi_seeds": [314, 1, 2],
        "multi_seed_runs": ["E3", "E8", "C0", "C1", "R1", "R2", "D3"],
        "checkpoint": "best",
        "runs": {},
        "skip_runs": [],
    },
    "proposed": "E8",
    "datasets": ["scared", "hamlyn", "c3vd"],
    "save_pred": True,
    "methods": {},
    "da3": {"models": ["da3-base", "depth-anything/da3mono-large"], "affine_rows": True},
    "illum": {"runs": ["E8"], "seed": 314, "sequences": ["sequence1", "sequence2"],
              "patch_sizes": [64, 32, 16, 8], "alpha": 0.10, "beta": 0.05, "ridge": 0.01,
              "sens_stride": 5, "gains": [0.8, 0.9, 1.0, 1.1, 1.2],
              "biases": [-0.05, -0.025, 0.0, 0.025, 0.05]},
    "stats": {"n_boot": 10000, "alpha": 0.05, "paired_metrics": ["abs_rel", "sq_rel", "rmse", "a1"],
              "table_methods": None},
}

# The training grid of CVIU_REVISION_PLAN.md section 3. Flags come after common_flags.
GRID = {
    # E-grid: component build-up and leave-one-out
    "E1": {"group": "E", "desc": "DA frozen + DPT output heads, standard loss",
           "flags": "--lora_type none --residual_block_indexes --illum_calib none --illumination_invariant 0 --photometric standard"},
    "E2": {"group": "E", "desc": "E1 + Conv-neck residual blocks",
           "flags": "--lora_type none --illum_calib none --illumination_invariant 0 --photometric standard"},
    "E3": {"group": "E", "desc": "E2 + DV-LoRA (EndoDAC baseline)",
           "flags": "--illum_calib none --illumination_invariant 0 --photometric standard"},
    "E4": {"group": "E", "desc": "E3 + local calibration",
           "flags": "--illumination_invariant 0 --photometric standard"},
    "E5": {"group": "E", "desc": "E3 + IIF loss",
           "flags": "--illum_calib none --photometric standard"},
    "E6": {"group": "E", "desc": "E3 + highlight-aware loss",
           "flags": "--illum_calib none --illumination_invariant 0"},
    "E7": {"group": "E", "desc": "E3 + calibration + IIF",
           "flags": "--photometric standard"},
    "E8": {"group": "E", "desc": "MonoIIF (full)", "flags": ""},
    "E8-IIF": {"group": "E", "desc": "full minus IIF", "flags": "--illumination_invariant 0"},
    "C0": {"group": "EC", "desc": "full minus calibration (no lighting model)", "flags": "--illum_calib none"},
    "E8-DVLoRA": {"group": "E", "desc": "full with plain LoRA", "flags": "--lora_type lora"},
    # R-grid: same losses on a weak backbone
    "R1": {"group": "R", "desc": "ResNet-18 + standard loss",
           "flags": "--depth_backbone resnet18 --illum_calib none --illumination_invariant 0 --photometric standard"},
    "R2": {"group": "R", "desc": "ResNet-18 + calibration + IIF + highlight",
           "flags": "--depth_backbone resnet18"},
    # C-grid: illumination model
    "C1": {"group": "C", "desc": "global affine calibration", "flags": "--illum_calib global"},
    # D-grid: Depth Anything 3 encoder (its own DinoV2 with QK-norm/RoPE) under the same recipe
    "D3": {"group": "D", "desc": "MonoIIF with DA3-Base encoder", "flags": "--backbone_weights da3"},
    "D3-EndoDAC": {"group": "D", "desc": "EndoDAC recipe with DA3-Base encoder",
                   "flags": "--backbone_weights da3 --illum_calib none --illumination_invariant 0 --photometric standard"},
    "N0": {"group": "D", "desc": "MonoIIF with a randomly initialised encoder (no foundation weights)",
           "flags": "--backbone_weights none"},
}
ABLATION_ORDER = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E8-IIF", "C0", "E8-DVLoRA", "R1", "R2",
                  "N0", "D3-EndoDAC", "D3"]
CALIB_ORDER = [("C0", "none"), ("C1", "global affine"), ("E8", "local affine (MonoIIF)")]


def deep_update(base, upd):
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path and os.path.exists(path):
        with open(path) as f:
            text = f.read()
        try:
            import yaml
            user = yaml.safe_load(text) or {}
        except ImportError:
            user = json.loads(text)
        deep_update(cfg, user)
    elif path:
        print("[config] {} not found, using DEFAULT_CONFIG".format(path))
    return cfg


def all_runs(cfg):
    runs = copy.deepcopy(GRID)
    for name, spec in (cfg["train"].get("runs") or {}).items():
        runs.setdefault(name, {"group": "X", "desc": name, "flags": ""}).update(spec)
    for name in cfg["train"].get("skip_runs") or []:
        runs.pop(name, None)
    return runs


def run_seeds(cfg, run):
    t = cfg["train"]
    return list(t["multi_seeds"] if run in t["multi_seed_runs"] else t["seeds"])


def run_name(run, seed):
    return "cviu_{}_s{}".format(run, seed)


def weights_folder(cfg, run, seed):
    d = os.path.join(cfg["log_dir"], run_name(run, seed), "models")
    if cfg["train"]["checkpoint"] == "last":
        p = os.path.join(d, "weights_last")
        return p if os.path.isdir(p) else None
    best, best_ep = None, -1
    for p in glob.glob(os.path.join(d, "weights_*")):
        tail = os.path.basename(p).split("_")[-1]
        if tail.isdigit() and int(tail) > best_ep:
            best, best_ep = p, int(tail)
    return best


def git_hash():
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=ROOT) != 0
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def out_path(cfg, *parts):
    p = os.path.join(cfg["out_dir"], *parts)
    ensure_dir(os.path.dirname(p))
    return p


def readlines(path):
    with open(path) as f:
        return f.read().splitlines()


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def append_rows(path, rows, fieldnames):
    if not rows:
        return
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def device_of(args):
    if torch is None:
        raise RuntimeError("this stage needs torch")
    return torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")


# --------------------------------------------------------------------------------------
# stage: train
# --------------------------------------------------------------------------------------

def preflight(cfg):
    """Everything a training subprocess needs, checked before the grid is launched."""
    problems = []
    dp = cfg["data"]["scared"]
    if not os.path.isdir(dp):
        problems.append("data.scared is not a directory: {}".format(dp))
    else:
        folder, frame = readlines(os.path.join(SPLITS, "train_files.txt"))[0].split()[:2]
        img = os.path.join(dp, folder, "data", frame + ".jpg")  # SCAREDRAWDataset.get_image_path
        if not os.path.exists(img):
            problems.append("first training image not found: {} (dataset layout or data.scared)".format(img))
    pw = os.path.join(cfg["pretrained_path"], "depth_anything_vitb14.pth")
    if not os.path.exists(pw):
        problems.append("Depth Anything weights not found: {} (README: pretrained_model/)".format(pw))
    gt = os.path.join(SPLITS, "gt_depths.npz")
    if not os.path.exists(gt):
        problems.append("{} missing: python export_gt_depth.py --data_path {} --split endovis --useage eval".format(gt, dp))
    py = cfg["python"]
    if shutil.which(py) is None and not os.path.exists(py):
        problems.append("python interpreter not found: {}".format(py))
    else:
        missing = []
        for mod in ("torch", "kornia", "wandb", "tensorboardX", "skimage", "cv2", "matplotlib",
                    "yaml", "scipy", "fvcore", "PIL"):
            if subprocess.call([py, "-c", "import " + mod], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
                missing.append(mod)
        if missing:
            pip_names = {"skimage": "scikit-image", "cv2": "opencv-python-headless", "yaml": "pyyaml", "PIL": "pillow"}
            problems.append("'{}' cannot import {}: {} -m pip install {}".format(
                py, missing, py, " ".join(pip_names.get(m, m) for m in missing)))
        elif subprocess.call([py, "-c", "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"],
                             cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
            problems.append("'{}' has torch but torch.cuda.is_available() is False (CPU-only wheel or the "
                            "driver is not visible): {} -c \"import torch; print(torch.__version__, torch.version.cuda)\"".format(py, py))
    return problems


DA3_URL = "https://huggingface.co/depth-anything/DA3-BASE/resolve/main/model.safetensors"


def da3_weights_path(cfg):
    """DA3-Base checkpoint used by the D-grid and the zero-shot rows (ported code, no package)."""
    return os.path.join(cfg["pretrained_path"], "da3_base.safetensors")


def tail(path, n=25):
    try:
        with open(path, errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return ""


def stage_train(cfg, args):
    runs = all_runs(cfg)
    selected = [r for r in runs if not args.only or r in args.only]
    problems = preflight(cfg)
    for p in problems:
        print("[train] PREFLIGHT: " + p)
    if problems and not args.dry_run and not args.skip_preflight:
        print("[train] fix the above (or pass --skip_preflight) before launching")
        sys.exit(1)
    # DA3-encoder rows need the DA3-Base checkpoint (ported code, same Python env as the rest).
    # Without it they are left out so the rest of the grid still runs; rerun train after the download.
    da3_runs = [r for r in selected if "--backbone_weights da3" in runs[r]["flags"]]
    if da3_runs and not os.path.exists(da3_weights_path(cfg)):
        print("[train] {} need the DA3-Base weights: wget -O {} {}".format(
            da3_runs, da3_weights_path(cfg), DA3_URL))
        if not args.dry_run and not args.skip_preflight:
            print("[train] skipping {} for now".format(da3_runs))
            selected = [r for r in selected if r not in da3_runs]
    jobs = []
    for run, spec in runs.items():
        if run not in selected:
            continue
        for seed in run_seeds(cfg, run):
            name = run_name(run, seed)
            done = os.path.join(cfg["log_dir"], name, "cviu_done.txt")
            if os.path.exists(done) and not args.force:
                print("[train] {} done, skipping".format(name))
                continue
            cmd = [cfg["python"], os.path.join(ROOT, "train_end_to_end.py"),
                   "--data_path", cfg["data"]["scared"], "--log_dir", cfg["log_dir"],
                   "--pretrained_path", cfg["pretrained_path"],
                   "--model_name", name, "--seed", str(seed)]
            cmd += shlex.split(cfg["train"]["common_flags"]) + shlex.split(spec["flags"])
            cmd += shlex.split(args.extra_flags or "")
            jobs.append({"run": run, "seed": seed, "name": name, "desc": spec["desc"], "cmd": cmd})

    manifest = {"git": git_hash(), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "jobs": jobs}
    with open(out_path(cfg, "train_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("[train] git {}: {} job(s)".format(manifest["git"], len(jobs)))
    for j in jobs:
        print("  {:<22} {}".format(j["name"], " ".join(shlex.quote(c) for c in j["cmd"])))
    if args.dry_run or not jobs:
        return 0

    gpus = [str(g) for g in (args.gpus or ["0"])]
    q = queue.Queue()
    failed = []
    for j in jobs:
        q.put(j)
    log_dir = ensure_dir(os.path.join(cfg["out_dir"], "train_logs"))

    def worker(gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        while True:
            try:
                j = q.get_nowait()
            except queue.Empty:
                return
            log = os.path.join(log_dir, j["name"] + ".log")
            print("[train] gpu {} -> {} (log: {})".format(gpu, j["name"], log))
            t0 = time.time()
            with open(log, "a") as lf:
                lf.write("# {}\n# {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S"), " ".join(j["cmd"])))
                lf.flush()
                rc = subprocess.call(j["cmd"], cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
            secs = time.time() - t0
            if rc == 0:
                with open(os.path.join(cfg["log_dir"], j["name"], "cviu_done.txt"), "w") as f:
                    f.write("git {}\nhours {:.2f}\ncmd {}\n".format(manifest["git"], secs / 3600.0, " ".join(j["cmd"])))
                print("[train] {} finished in {:.1f} h".format(j["name"], secs / 3600.0))
            else:
                failed.append(j["name"])
                print("[train] {} FAILED (rc={}) after {:.0f} s, see {}\n----- log tail -----\n{}--------------------".format(
                    j["name"], rc, secs, log, tail(log)))
                if secs < 180:  # died before training started: a setup error, not worth repeating 25 times
                    print("[train] failed within 3 minutes: aborting the remaining jobs; fix the error and relaunch "
                          "(finished runs are skipped)")
                    while True:
                        try:
                            q.get_nowait()
                            q.task_done()
                        except queue.Empty:
                            break
            q.task_done()

    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if failed:
        print("[train] {} job(s) failed: {}".format(len(failed), failed))
    return len(failed)


# --------------------------------------------------------------------------------------
# datasets and per-frame evaluation
# --------------------------------------------------------------------------------------

def compute_errors(gt, pred):
    """monodepth2 metrics (identical to utils.utils.compute_errors)."""
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    rmse = np.sqrt(((gt - pred) ** 2).mean())
    rmse_log = np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def evaluate_prediction(pred, kind, align, gt, max_depth):
    """pred: (h,w) disparity or depth at any resolution. Returns (metrics dict, ratio)."""
    import cv2
    gh, gw = gt.shape[:2]
    pred = cv2.resize(pred.astype(np.float32), (gw, gh))
    mask = np.logical_and(gt > MIN_DEPTH, gt < max_depth)
    g = gt[mask]
    if kind == "disp":
        disp = np.clip(pred, 1e-6, None)[mask]
    else:
        disp = 1.0 / np.clip(pred, 1e-6, None)[mask]
    if align == "affine":  # disparity shift + scale (mytest_da.py), for relative-depth models
        gd = 1.0 / g
        t_gt, t_p = np.median(gd), np.median(disp)
        s_gt, s_p = np.mean(np.abs(gd - t_gt)), np.mean(np.abs(disp - t_p)) + 1e-12
        p = 1.0 / np.clip((disp - t_p) * (s_gt / s_p) + t_gt, 1e-6, None)
        ratio = float("nan")
    else:
        p = 1.0 / disp
        ratio = float(np.median(g) / np.median(p))
        p = p * ratio
    p = np.clip(p, MIN_DEPTH, max_depth)
    m = compute_errors(g, p)
    return dict(zip(METRICS, [float(x) for x in m])), ratio


def make_dataset(cfg, name):
    """Returns (dataset, meta) with meta(i, batch) -> (sequence, frame, gt_depth)."""
    if name == "scared":
        from datasets.scared_dataset import SCAREDRAWDataset
        lines = readlines(os.path.join(SPLITS, "test_files.txt"))
        ds = SCAREDRAWDataset(cfg["data"]["scared"], lines, H, W, [0], 4, is_train=False)
        gts = np.load(os.path.join(SPLITS, "gt_depths.npz"), fix_imports=True, encoding="latin1")["data"]

        def meta(i, data):
            folder, frame = lines[i].split()[:2]
            return folder, frame, gts[i]
    elif name == "hamlyn":
        from datasets.hamlyn_dataset import HamlynDataset
        ds = HamlynDataset(cfg["data"]["hamlyn"], H, W, [0], 4, is_train=False)

        def meta(i, data):
            return ("rectified{:02d}".format(int(data["sequence"][0])), str(int(data["index"][0])),
                    data["depth_gt"][0].numpy())
    elif name == "c3vd":
        from datasets.c3vd_dataset import C3VDDataset
        ds = C3VDDataset(cfg["data"]["c3vd"], H, W, [0], 4, is_train=False, split="test")

        def meta(i, data):
            return str(data["sequence"][0]), str(i), data["depth_gt"][0].numpy()
    else:
        raise ValueError("unknown dataset " + name)
    return ds, meta


def iterate_dataset(cfg, name):
    from torch.utils.data import DataLoader
    ds, meta = make_dataset(cfg, name)
    loader = DataLoader(ds, 1, shuffle=False, num_workers=cfg["num_workers"], pin_memory=True)
    for i, data in enumerate(loader):
        seq, frame, gt = meta(i, data)
        gt = np.asarray(gt, dtype=np.float32)
        if gt.ndim == 3:
            gt = gt[:, :, 0]
        yield i, data[("color", 0, 0)], seq, frame, gt


def _pretrained_or_none(cfg):
    p = cfg["pretrained_path"]
    return p if os.path.exists(os.path.join(p, "depth_anything_vitb14.pth")) else None


def load_depth_model_from_run(cfg, run, seed, device):
    """Depth network of a grid run (endodac or resnet18), from its saved opt.json + weights."""
    wf = weights_folder(cfg, run, seed)
    if wf is None:
        raise FileNotFoundError("no weights for {} seed {}".format(run, seed))
    with open(os.path.join(os.path.dirname(wf), "opt.json")) as f:
        opt = json.load(f)
    sd = torch.load(os.path.join(wf, "depth_model.pth"), map_location="cpu")
    if opt.get("depth_backbone", "endodac") == "resnet18":
        from models.resnet_depth import ResnetDepth
        model = ResnetDepth(opt["num_layers"], False, opt["scales"])
    else:
        import models.endodac as endodac
        # pretrained_path=None: the checkpoint holds every weight, no need to reload DA v1
        model = endodac.endodac(backbone_size="base", r=opt["lora_rank"], lora_type=opt["lora_type"],
                                image_shape=(224, 280), pretrained_path=None,
                                residual_block_indexes=opt["residual_block_indexes"],
                                include_cls_token=opt.get("include_cls_token", True),
                                backbone_weights=opt.get("backbone_weights", "da1"))
    md = model.state_dict()
    model.load_state_dict({k: v for k, v in sd.items() if k in md}, strict=False)
    return model.to(device).eval(), opt, wf


def build_predictor(cfg, name, spec, device):
    """Returns (predict(color)->np array, kind, align) or ('npy', arrays, kind, align)."""
    if spec["type"] == "npy":
        arrays = {ds: np.load(p) for ds, p in spec["files"].items()}
        return ("npy", arrays, spec.get("kind", "disp"), spec.get("align", "median"))
    if spec["type"] == "run":
        model, _, _ = load_depth_model_from_run(cfg, spec["run"], spec["seed"], device)
    else:
        from evaluate_depth_all import DepthModelFactory
        ns = argparse.Namespace(load_weights_folder=spec["weights"], model_type=spec["type"],
                                lora_rank=spec.get("lora_rank", 4), lora_type=spec.get("lora_type", "dvlora"),
                                pretrained_path=_pretrained_or_none(cfg),
                                residual_block_indexes=spec.get("residual_block_indexes", [2, 5, 8, 11]),
                                include_cls_token=spec.get("include_cls_token", True),
                                num_layers=spec.get("num_layers", 18))
        model, _ = DepthModelFactory.load_model(spec["type"], ns)
    from utils.layers import disp_to_depth

    def predict(color):
        with torch.no_grad():
            out = model(color.to(device))
        disp = out[("disp", 0)] if isinstance(out, dict) else out
        scaled, _ = disp_to_depth(disp, *DEPTH_RANGE)
        return scaled[0, 0].cpu().numpy()
    return ("model", predict, "disp", "median")


def predict_methods(cfg, args):
    """All (method name, seed, spec) triples the predict stage should cover."""
    items = []
    for run in all_runs(cfg):
        for seed in run_seeds(cfg, run):
            if weights_folder(cfg, run, seed) is not None:
                items.append((run, seed, {"type": "run", "run": run, "seed": seed}))
    for name, spec in (cfg.get("methods") or {}).items():
        items.append((name, 0, spec))
    if args.only:
        items = [it for it in items if it[0] in args.only]
    return items


def stage_predict(cfg, args):
    device = device_of(args)
    per_frame = out_path(cfg, "per_frame.csv")
    done = {(r["method"], r["seed"], r["dataset"]) for r in read_csv(per_frame)}
    fields = ["method", "seed", "dataset", "sequence", "frame"] + METRICS + ["ratio", "infer_ms", "git"]
    datasets = args.datasets or cfg["datasets"]
    g = git_hash()
    for method, seed, spec in predict_methods(cfg, args):
        todo = [d for d in datasets if (method, str(seed), d) not in done or args.force]
        if spec["type"] == "npy":
            todo = [d for d in todo if d in spec["files"]]
        if not todo:
            continue
        print("[predict] {} seed {} on {}".format(method, seed, todo))
        try:
            pred = build_predictor(cfg, method, spec, device)
        except Exception as e:
            print("[predict] cannot load {}: {}".format(method, e))
            continue
        for ds in todo:
            rows, disps = [], []
            for i, color, seq, frame, gt in iterate_dataset(cfg, ds):
                t0 = time.time()
                if pred[0] == "npy":
                    p = pred[1][ds][i]
                    p = p[0] if p.ndim == 3 else p
                else:
                    p = pred[1](color)
                ms = (time.time() - t0) * 1000.0
                m, ratio = evaluate_prediction(p, pred[2], pred[3], gt, MAX_DEPTH[ds])
                row = {"method": method, "seed": seed, "dataset": ds, "sequence": seq, "frame": frame,
                       "ratio": ratio, "infer_ms": ms, "git": g}
                row.update(m)
                rows.append(row)
                if cfg["save_pred"] and pred[0] != "npy":
                    disps.append(p.astype(np.float16))
            if args.force:  # drop stale rows of this (method, seed, dataset) before appending
                keep = [r for r in read_csv(per_frame)
                        if not (r["method"] == method and r["seed"] == str(seed) and r["dataset"] == ds)]
                write_csv(per_frame, keep, fields)
            append_rows(per_frame, rows, fields)
            if disps:
                np.save(out_path(cfg, "pred", "{}_s{}_{}.npy".format(method, seed, ds)), np.stack(disps))
            mean = {k: float(np.mean([r[k] for r in rows])) for k in METRICS}
            print("[predict] {:<20} {:<7} frames={:<4} abs_rel={:.4f} rmse={:.3f} a1={:.4f}".format(
                method, ds, len(rows), mean["abs_rel"], mean["rmse"], mean["a1"]))


# --------------------------------------------------------------------------------------
# illumination model: shared torch helpers
# --------------------------------------------------------------------------------------

def gaussian_blur(x, k=11):
    """Depthwise Gaussian blur with LightingDecoder's kernel/sigma heuristic."""
    sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
    ax = torch.arange(k, device=x.device, dtype=x.dtype) - (k - 1) / 2.0
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    k2 = torch.outer(g, g)[None, None].expand(x.shape[1], 1, k, k).contiguous()
    pad = k // 2
    return F.conv2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), k2, groups=x.shape[1])


def _patch_sums(x, P):
    """(B,C,H,W) -> (B,1,H/P,W/P) summed over channels and each PxP patch; P=None: whole image."""
    if P is None:
        return x.sum((1, 2, 3), keepdim=True)
    B, C, Hh, Ww = x.shape
    return x.reshape(B, C, Hh // P, P, Ww // P, P).sum((1, 3, 5)).unsqueeze(1)


def fit_affine(w, t, v, P, ridge):
    """Weighted least-squares affine fit t ~ c*w + b, per PxP patch (P=None: per image).

    w, t: (B,3,H,W) warped source and target; v: (B,1,H,W) weights (0/1 validity).
    The ridge lambda = ridge * n pulls (c, b) toward (1, 0), so patches without texture
    keep the identity instead of fitting noise. Returns c, b maps at pixel resolution.
    """
    vw = v * w
    n = _patch_sums(v.expand_as(w), P)
    Sw, St = _patch_sums(vw, P), _patch_sums(v * t, P)
    Sww, Swt = _patch_sums(vw * w, P), _patch_sums(vw * t, P)
    lam = ridge * n.clamp_min(1.0)
    a11, a12, a22 = Sww + lam, Sw, n + lam
    r1, r2 = Swt + lam, St
    det = (a11 * a22 - a12 * a12).clamp_min(1e-12)
    c = (r1 * a22 - a12 * r2) / det
    b = (a11 * r2 - a12 * r1) / det
    if P is None:
        return c.expand_as(v), b.expand_as(v)
    return (c.repeat_interleave(P, 2).repeat_interleave(P, 3),
            b.repeat_interleave(P, 2).repeat_interleave(P, 3))


def masked_l1(x, t, e):
    return ((x - t).abs() * e).sum() / (e.sum() * x.shape[1] + 1e-6)


def checkerboard(like):
    B, _, Hh, Ww = like.shape
    yy, xx = torch.meshgrid(torch.arange(Hh, device=like.device), torch.arange(Ww, device=like.device), indexing="ij")
    par = ((yy + xx) % 2).to(like.dtype)[None, None]
    return par, 1.0 - par


def affine_residuals(w, t, v, patch_sizes, alpha, beta, ridge, ssim=None):
    """Cross-validated residuals of the illumination models on one batch of warped pairs.

    Fits on one checkerboard half of the valid pixels and evaluates on the other (both ways),
    so extra degrees of freedom cannot win by construction. Returns a dict of scalars.
    """
    par0, par1 = checkerboard(v)
    halves = [(v * par0, v * par1), (v * par1, v * par0)]
    res = {"r_none": masked_l1(w, t, v).item(), "valid_frac": v.mean().item()}
    if ssim is not None:
        res["ssim_none"] = (ssim(w, t).mean(1, True) * v).sum().item() / (v.sum().item() + 1e-6)
    for name, P in [("global", None)] + [("local{}".format(P), P) for P in patch_sizes]:
        r, rb = 0.0, 0.0
        for fit_v, ev_v in halves:
            c, b = fit_affine(w, t, fit_v, P, ridge)
            r += masked_l1(c * w + b, t, ev_v).item()
            if P is not None:  # what the LightingDecoder is allowed to express
                cb = gaussian_blur(c.clamp(1 - alpha, 1 + alpha))
                bb = gaussian_blur(b.clamp(-beta, beta))
                rb += masked_l1(cb * w + bb, t, ev_v).item()
        res["r_" + name] = r / 2
        if P is not None:
            res["r_bounded{}".format(P)] = rb / 2
        # in-sample parameters (all valid pixels) for plausibility statistics
        c, b = fit_affine(w, t, v, P, ridge)
        if P is None:
            res["c_global"], res["b_global"] = c.mean().item(), b.mean().item()
        cv, bv = c[v > 0], b[v > 0]
        if cv.numel() > 0:
            res["c_{}_p1".format(name)] = torch.quantile(cv, 0.01).item()
            res["c_{}_p99".format(name)] = torch.quantile(cv, 0.99).item()
            res["b_{}_p1".format(name)] = torch.quantile(bv, 0.01).item()
            res["b_{}_p99".format(name)] = torch.quantile(bv, 0.99).item()
            res["out_of_bounds_{}".format(name)] = (((cv - 1).abs() > alpha) | (bv.abs() > beta)).float().mean().item()
        if ssim is not None and name in ("global", "local16"):
            res["ssim_" + name] = (ssim(c * w + b, t).mean(1, True) * v).sum().item() / (v.sum().item() + 1e-6)
    return res


class Warper(object):
    """GT/predicted-geometry warping of a source frame into the target view (repo layers)."""

    def __init__(self, device):
        from utils.layers import BackprojectDepth, Project3D
        self.device = device
        self._bp, self._pr = {}, {}
        self.BackprojectDepth, self.Project3D = BackprojectDepth, Project3D

    def _mods(self, B):
        if B not in self._bp:
            self._bp[B] = self.BackprojectDepth(B, H, W).to(self.device)
            self._pr[B] = self.Project3D(B, H, W).to(self.device)
        return self._bp[B], self._pr[B]

    def warp(self, source, depth_t, K, inv_K, T, depth_s=None, tol=0.05):
        """Returns warped source (B,3,H,W), validity (B,1,H,W) and the z-buffer test rate."""
        B = source.shape[0]
        bp, pr = self._mods(B)
        cam_points = bp(depth_t, inv_K)
        pix = pr(cam_points, K, T)
        warped = F.grid_sample(source, pix, padding_mode="border", align_corners=True)
        valid = ((pix.abs() <= 1).all(-1).unsqueeze(1) & (depth_t > 0)).to(source.dtype)
        occl = None
        if depth_s is not None:  # depth-consistency occlusion check with the source GT depth
            z = torch.matmul(T, cam_points)[:, 2].view(B, 1, H, W)
            d_s = F.grid_sample(depth_s, pix, mode="nearest", padding_mode="border", align_corners=True)
            ok = ((d_s > 0) & ((z - d_s).abs() < tol * d_s.clamp_min(1e-6))).to(source.dtype)
            occl = (ok * valid).sum() / (valid.sum() + 1e-6)
            valid = valid * ok
        return warped, valid, occl


def photometric_map(ssim, pred, target):
    return 0.85 * ssim(pred, target).mean(1, True) + 0.15 * (pred - target).abs().mean(1, True)


def highlight_mask(image):
    """HADepth highlight test without kornia: S = (max-min)/max, V = max; 1 = highlight."""
    v, _ = image.max(1, keepdim=True)
    mn, _ = image.min(1, keepdim=True)
    s = (v - mn) / (v + 1e-6)
    return ((s < 0.1) & (v > 0.9)).to(image.dtype)


def pearson(a, b):
    """Per-image Pearson correlation of two (B,1,H,W) maps -> (B,)"""
    a = a.flatten(1) - a.flatten(1).mean(1, keepdim=True)
    b = b.flatten(1) - b.flatten(1).mean(1, keepdim=True)
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-8)


def sequence_pairs(cfg, seq):
    """Consecutive-frame pairs of a SCARED pose sequence: (lines, pair indices, gt T per line)."""
    lines = readlines(os.path.join(SPLITS, "test_files_{}.txt".format(seq)))
    pairs = []
    for i in range(len(lines) - 1):
        f0, f1 = lines[i].split(), lines[i + 1].split()
        if f0[0] == f1[0] and int(f1[1]) == int(f0[1]) + 1:
            pairs.append(i)
    gt_T = None
    p = os.path.join(SPLITS, "curve", "gt_poses_{}.npz".format(seq))
    if os.path.exists(p):
        gt_T = np.load(p)["data"]
    return lines, pairs, gt_T


def gt_relative_pose(data_path, folder, frame):
    """T = P(frame+1) @ pinv(P(frame)) from SCARED frame_data json (export_gt_pose.py)."""
    sequence = folder[7]
    split = "train" if int(sequence) < 8 else "test"
    poses = []
    for idx in (frame - 1, frame):
        with open(os.path.join(data_path, split, folder, "data", "frame_data", "frame_data{:06d}.json".format(idx))) as f:
            poses.append(np.array(json.load(f)["camera-pose"]))
    return (poses[1] @ np.linalg.pinv(poses[0])).astype(np.float32)


# --------------------------------------------------------------------------------------
# stage: illum-fit (model free)
# --------------------------------------------------------------------------------------

def stage_illum_fit(cfg, args):
    import cv2
    from datasets.scared_dataset import SCAREDRAWDataset
    from utils.layers import SSIM
    device = device_of(args)
    ic = cfg["illum"]
    warper = Warper(device)
    ssim = SSIM().to(device)
    for seq in ic["sequences"]:
        path = out_path(cfg, "illum", "fit_{}.csv".format(seq))
        if os.path.exists(path) and not args.force:
            print("[illum-fit] {} exists, skipping".format(path))
            continue
        lines, pairs, gt_T = sequence_pairs(cfg, seq)
        ds = SCAREDRAWDataset(cfg["data"]["scared"], lines, H, W, [0, 1], 4, is_train=False)
        rows, convention = [], None

        def load_pair(i):
            folder, frame, side = lines[i].split()
            frame = int(frame)
            item = ds[i]
            I_t = item[("color", 0, 0)][None].to(device)
            I_s = item[("color", 1, 0)][None].to(device)
            K, inv_K = item[("K", 0)][None].to(device), item[("inv_K", 0)][None].to(device)
            d_t = cv2.resize(ds.get_depth(folder, frame, side, False).astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
            d_s = cv2.resize(ds.get_depth(folder, frame + 1, side, False).astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
            d_t = torch.from_numpy(d_t)[None, None].to(device)
            d_s = torch.from_numpy(d_s)[None, None].to(device)
            d_t = d_t * ((d_t > MIN_DEPTH) & (d_t < MAX_DEPTH["scared"])).to(d_t.dtype)
            T = gt_T[i] if gt_T is not None else gt_relative_pose(cfg["data"]["scared"], folder, frame)
            T = torch.from_numpy(np.asarray(T, dtype=np.float32))[None].to(device)
            return folder, frame, I_t, I_s, K, inv_K, d_t, d_s, T

        # the pose json convention is checked empirically on the first pairs: the direction
        # whose GT warp has the lower photometric residual is used for the whole sequence
        scores = {"direct": 0.0, "inverse": 0.0}
        for i in pairs[:20]:
            _, _, I_t, I_s, K, inv_K, d_t, d_s, T = load_pair(i)
            for name, Tx in (("direct", T), ("inverse", torch.inverse(T))):
                wpd, v, _ = warper.warp(I_s, d_t, K, inv_K, Tx, d_s)
                scores[name] += masked_l1(wpd, I_t, v).item()
        convention = min(scores, key=scores.get)
        print("[illum-fit] {}: pose convention '{}' (residuals {})".format(seq, convention, scores))

        t0 = time.time()
        for n, i in enumerate(pairs):
            folder, frame, I_t, I_s, K, inv_K, d_t, d_s, T = load_pair(i)
            if convention == "inverse":
                T = torch.inverse(T)
            wpd, v, occl = warper.warp(I_s, d_t, K, inv_K, T, d_s)
            with torch.no_grad():
                res = affine_residuals(wpd, I_t, v, ic["patch_sizes"], ic["alpha"], ic["beta"], ic["ridge"], ssim)
            res.update({"sequence": folder, "frame": frame, "r_identity": masked_l1(I_s, I_t, v).item(),
                        "zbuffer_pass": float(occl), "convention": convention})
            rows.append(res)
            if n % 100 == 0:
                print("[illum-fit] {} {}/{}  none={:.4f} global={:.4f} local16={:.4f} bounded16={:.4f} ({:.0f}s)".format(
                    seq, n, len(pairs), res["r_none"], res["r_global"], res["r_local16"], res["r_bounded16"], time.time() - t0))
        fields = ["sequence", "frame", "convention", "valid_frac", "zbuffer_pass", "r_identity"] + \
                 sorted(k for k in rows[0] if k not in ("sequence", "frame", "convention", "valid_frac", "zbuffer_pass", "r_identity"))
        write_csv(path, rows, fields)
        print("[illum-fit] wrote {} ({} pairs)".format(path, len(rows)))


# --------------------------------------------------------------------------------------
# stage: illum-params / illum-sens (learned calibration)
# --------------------------------------------------------------------------------------

def load_run_models(cfg, run, seed, device):
    """Depth, pose, intrinsics and lighting networks of a grid run."""
    import models.decoders as decoders
    from models.encoders import ResnetEncoder
    depth, opt, wf = load_depth_model_from_run(cfg, run, seed, device)
    models = {"depth": depth}
    pe = ResnetEncoder(opt["num_layers"], False, num_input_images=2)
    pe.load_state_dict(torch.load(os.path.join(wf, "pose_encoder.pth"), map_location="cpu"))
    models["pose_encoder"] = pe.to(device).eval()
    pd = decoders.PoseDecoder(pe.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)
    pd.load_state_dict(torch.load(os.path.join(wf, "pose.pth"), map_location="cpu"))
    models["pose"] = pd.to(device).eval()
    if opt.get("learn_intrinsics", True) and os.path.exists(os.path.join(wf, "intrinsics_head.pth")):
        ih = decoders.IntrinsicsHead(pe.num_ch_enc)
        ih.load_state_dict(torch.load(os.path.join(wf, "intrinsics_head.pth"), map_location="cpu"))
        models["intrinsics"] = ih.to(device).eval()
    calib = opt.get("illum_calib", "local")
    lp = os.path.join(wf, "lighting.pth")
    if calib != "none" and os.path.exists(lp):
        if calib == "global":
            lt = decoders.GlobalLightingHead(pe.num_ch_enc, opt["scales"])
        else:
            lt = decoders.LightingDecoder(pe.num_ch_enc, opt["scales"])
        lt.load_state_dict(torch.load(lp, map_location="cpu"))
        models["lighting"] = lt.to(device).eval()
    models["_opt"] = opt
    return models


def run_pair(models, warper, ssim, target, source, K_in, inv_K_in):
    """One forward pass of the full pipeline on (target, source) batches. No grad."""
    from utils.layers import disp_to_depth, transformation_from_parameters
    with torch.no_grad():
        disp = models["depth"](target)[("disp", 0)]
        disp = F.interpolate(disp, [H, W], mode="bilinear", align_corners=True)
        _, depth = disp_to_depth(disp, *DEPTH_RANGE)
        feats = models["pose_encoder"](torch.cat([source, target], 1))
        axisangle, translation, inter = models["pose"]([feats])
        if "intrinsics" in models:
            K = models["intrinsics"](inter, W, H)
            inv_K = torch.inverse(K)
        else:
            K, inv_K = K_in, inv_K_in
        T = transformation_from_parameters(axisangle[:, 0], translation[:, 0])
        warped, valid, _ = warper.warp(source, depth, K, inv_K, T)
        if "lighting" in models:
            lo = models["lighting"](feats)
            c = F.interpolate(lo[("contrast", 0)], [H, W], mode="bilinear", align_corners=False)
            b = F.interpolate(lo[("brightness", 0)], [H, W], mode="bilinear", align_corners=False)
        else:
            c, b = torch.ones_like(valid), torch.zeros_like(valid)
        automask = (photometric_map(ssim, warped, target) < photometric_map(ssim, source, target)).to(target.dtype) * valid
    return {"depth": depth, "T": T, "K": K, "warped": warped, "valid": valid, "c": c, "b": b, "automask": automask}


def stage_illum_params(cfg, args):
    from torch.utils.data import DataLoader
    from datasets.scared_dataset import SCAREDRAWDataset
    from utils.layers import SSIM
    device = device_of(args)
    ic = cfg["illum"]
    warper, ssim = Warper(device), SSIM().to(device)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
    radial = torch.sqrt(yy ** 2 + xx ** 2)[None, None].to(device)
    runs = args.only or ic["runs"]
    for run in runs:
        for seed in ([ic["seed"]] if not args.all_seeds else run_seeds(cfg, run)):
            try:
                models = load_run_models(cfg, run, seed, device)
            except Exception as e:
                print("[illum-params] cannot load {} s{}: {}".format(run, seed, e))
                continue
            lt = models.get("lighting")
            alpha = getattr(lt, "alpha", ic["alpha"])
            beta = getattr(lt, "beta", ic["beta"])
            hist = {"c": np.zeros(200), "b": np.zeros(200)}
            c_edges, b_edges = np.linspace(1 - 2 * alpha, 1 + 2 * alpha, 201), np.linspace(-2 * beta, 2 * beta, 201)
            for seq in ic["sequences"]:
                path = out_path(cfg, "illum", "params_{}_s{}_{}.csv".format(run, seed, seq))
                if os.path.exists(path) and not args.force:
                    print("[illum-params] {} exists, skipping".format(path))
                    continue
                lines, _, gt_T = sequence_pairs(cfg, seq)
                # frames whose both neighbours exist in the split
                idx = [i for i in range(1, len(lines) - 1)
                       if lines[i - 1].split()[0] == lines[i].split()[0] == lines[i + 1].split()[0]
                       and int(lines[i + 1].split()[1]) - int(lines[i - 1].split()[1]) == 2]
                sub = [lines[i] for i in idx]
                ds = SCAREDRAWDataset(cfg["data"]["scared"], sub, H, W, [0, -1, 1], 4, is_train=False)
                loader = DataLoader(ds, 8, shuffle=False, num_workers=cfg["num_workers"], drop_last=False)
                rows, prev_c_fwd, prev_frame = [], None, None
                for bi, data in enumerate(loader):
                    target = data[("color", 0, 0)].to(device)
                    K_in, inv_K_in = data[("K", 0)].to(device), data[("inv_K", 0)].to(device)
                    outs = {f: run_pair(models, warper, ssim, target, data[("color", f, 0)].to(device), K_in, inv_K_in) for f in (-1, 1)}
                    hl = highlight_mask(target)
                    B = target.shape[0]
                    for f in (-1, 1):
                        o = outs[f]
                        c, b, am = o["c"], o["b"], o["automask"]
                        refined = c * o["warped"] + b
                        # least-squares oracle (patch 16) on the same pixels, for reference
                        c_or, b_or = fit_affine(o["warped"], target, am, 16, ic["ridge"])
                        gc = (c[:, :, :, 1:] - c[:, :, :, :-1]).abs().mean((1, 2, 3)) + (c[:, :, 1:] - c[:, :, :-1]).abs().mean((1, 2, 3))
                        gb = (b[:, :, :, 1:] - b[:, :, :, :-1]).abs().mean((1, 2, 3)) + (b[:, :, 1:] - b[:, :, :-1]).abs().mean((1, 2, 3))
                        stats = {
                            "c_mean": c.mean((1, 2, 3)), "c_std": c.flatten(1).std(1), "b_mean": b.mean((1, 2, 3)), "b_std": b.flatten(1).std(1),
                            "c_p1": torch.quantile(c.flatten(1), 0.01, dim=1), "c_p99": torch.quantile(c.flatten(1), 0.99, dim=1),
                            "b_p1": torch.quantile(b.flatten(1), 0.01, dim=1), "b_p99": torch.quantile(b.flatten(1), 0.99, dim=1),
                            "sat_c": ((c - 1).abs() > 0.95 * alpha).float().mean((1, 2, 3)), "sat_b": (b.abs() > 0.95 * beta).float().mean((1, 2, 3)),
                            "grad_c": gc, "grad_b": gb,
                            "corr_c_radial": pearson(c, radial.expand_as(c)), "corr_b_radial": pearson(b, radial.expand_as(b)),
                            "corr_c_highlight": pearson(c, hl), "corr_b_highlight": pearson(b, hl),
                            "highlight_frac": hl.mean((1, 2, 3)), "automask_frac": am.mean((1, 2, 3)),
                            "res_before": ((o["warped"] - target).abs().mean(1, True) * am).sum((1, 2, 3)) / (am.sum((1, 2, 3)) + 1e-6),
                            "res_after": ((refined - target).abs().mean(1, True) * am).sum((1, 2, 3)) / (am.sum((1, 2, 3)) + 1e-6),
                            "res_oracle16": ((c_or * o["warped"] + b_or - target).abs().mean(1, True) * am).sum((1, 2, 3)) / (am.sum((1, 2, 3)) + 1e-6),
                            "tz_pred": o["T"][:, 2, 3],
                        }
                        stats = {k: v.detach().cpu().numpy() for k, v in stats.items()}
                        for j in range(B):
                            li = idx[bi * 8 + j]
                            row = {"run": run, "seed": seed, "sequence": lines[li].split()[0], "frame": int(lines[li].split()[1]), "source": f}
                            row.update({k: float(v[j]) for k, v in stats.items()})
                            if gt_T is not None:
                                row["tz_gt"] = float(gt_T[li if f == 1 else li - 1][2, 3])
                            rows.append(row)
                    # temporal consistency and forward/backward reciprocity between consecutive frames
                    c_fwd, c_bwd = outs[1]["c"], outs[-1]["c"]
                    for j in range(B):
                        li = idx[bi * 8 + j]
                        cur = c_fwd[j:j + 1]
                        if prev_c_fwd is not None and prev_frame == int(lines[li].split()[1]) - 1:
                            tcorr = pearson(prev_c_fwd, cur).item()
                            recip = (prev_c_fwd * c_bwd[j:j + 1] - 1).abs().mean().item()
                            for r in rows[-2 * B:]:
                                if r["frame"] == int(lines[li].split()[1]):
                                    r["temporal_corr_c"], r["reciprocity_err"] = tcorr, recip
                        prev_c_fwd, prev_frame = cur, int(lines[li].split()[1])
                    hist["c"] += np.histogram(c_fwd.detach().cpu().numpy(), bins=c_edges)[0]
                    hist["b"] += np.histogram(outs[1]["b"].detach().cpu().numpy(), bins=b_edges)[0]
                fields = sorted(set(k for r in rows for k in r), key=lambda k: (k not in ("run", "seed", "sequence", "frame", "source"), k))
                write_csv(path, rows, fields)
                print("[illum-params] wrote {} ({} rows)".format(path, len(rows)))
            np.savez(out_path(cfg, "illum", "hist_{}_s{}.npz".format(run, seed)), c=hist["c"], b=hist["b"],
                     c_edges=c_edges, b_edges=b_edges, alpha=alpha, beta=beta)


def stage_illum_sens(cfg, args):
    from torch.utils.data import DataLoader
    from datasets.scared_dataset import SCAREDRAWDataset
    from utils.layers import SSIM, disp_to_depth
    device = device_of(args)
    ic = cfg["illum"]
    warper, ssim = Warper(device), SSIM().to(device)
    runs = args.only or ic["runs"]
    for run in runs:
        seed = ic["seed"]
        path = out_path(cfg, "illum", "sens_{}_s{}.csv".format(run, seed))
        if os.path.exists(path) and not args.force:
            print("[illum-sens] {} exists, skipping".format(path))
            continue
        try:
            models = load_run_models(cfg, run, seed, device)
        except Exception as e:
            print("[illum-sens] cannot load {} s{}: {}".format(run, seed, e))
            continue
        rows = []
        # (a) calibration response: perturb the source frame, read the predicted (c, b)
        seq = ic["sequences"][0]
        lines, pairs, _ = sequence_pairs(cfg, seq)
        sub = [lines[i] for i in pairs[::ic["sens_stride"]]]
        ds = SCAREDRAWDataset(cfg["data"]["scared"], sub, H, W, [0, 1], 4, is_train=False)
        loader = DataLoader(ds, 8, shuffle=False, num_workers=cfg["num_workers"])
        for data in loader:
            target, source = data[("color", 0, 0)].to(device), data[("color", 1, 0)].to(device)
            K_in, inv_K_in = data[("K", 0)].to(device), data[("inv_K", 0)].to(device)
            for g in ic["gains"]:
                for bias in ic["biases"]:
                    if g != 1.0 and bias != 0.0:
                        continue  # one factor at a time
                    o = run_pair(models, warper, ssim, target, (g * source + bias).clamp(0, 1), K_in, inv_K_in)
                    am = o["automask"]
                    for j in range(target.shape[0]):
                        rows.append({"run": run, "seed": seed, "kind": "calib", "gain": g, "bias": bias,
                                     "frame": int(data["frame_id"][j]),
                                     "c_mean": o["c"][j].mean().item(), "b_mean": o["b"][j].mean().item(),
                                     "res_before": ((o["warped"][j] - target[j]).abs().mean(0, True) * am[j]).sum().item() / (am[j].sum().item() + 1e-6),
                                     "res_after": ((o["c"][j] * o["warped"][j] + o["b"][j] - target[j]).abs().mean(0, True) * am[j]).sum().item() / (am[j].sum().item() + 1e-6)})
        # (b) depth robustness: perturb the frame fed to the depth network, score against GT
        gts = np.load(os.path.join(SPLITS, "gt_depths.npz"), fix_imports=True, encoding="latin1")["data"]
        tlines = readlines(os.path.join(SPLITS, "test_files.txt"))
        tds = SCAREDRAWDataset(cfg["data"]["scared"], tlines, H, W, [0], 4, is_train=False)
        tl = DataLoader(tds, 1, shuffle=False, num_workers=cfg["num_workers"])
        settings = [(g, 0.0) for g in ic["gains"]] + [(1.0, b) for b in ic["biases"] if b != 0.0]
        for i, data in enumerate(tl):
            color = data[("color", 0, 0)].to(device)
            for g, bias in settings:
                with torch.no_grad():
                    disp = models["depth"]((g * color + bias).clamp(0, 1))[("disp", 0)]
                    scaled, _ = disp_to_depth(disp, *DEPTH_RANGE)
                m, _ = evaluate_prediction(scaled[0, 0].cpu().numpy(), "disp", "median", gts[i].astype(np.float32), MAX_DEPTH["scared"])
                row = {"run": run, "seed": seed, "kind": "depth", "gain": g, "bias": bias,
                       "sequence": tlines[i].split()[0], "frame": int(tlines[i].split()[1])}
                row.update(m)
                rows.append(row)
        fields = ["run", "seed", "kind", "gain", "bias", "sequence", "frame", "c_mean", "b_mean", "res_before", "res_after"] + METRICS
        write_csv(path, rows, fields)
        print("[illum-sens] wrote {} ({} rows)".format(path, len(rows)))


# --------------------------------------------------------------------------------------
# stage: da3 (zero-shot Depth Anything 3)
# --------------------------------------------------------------------------------------

def _da3_package_infer(model, img_uint8):
    """Single-view inference through the depth_anything_3 package -> depth (h, w) float32."""
    pred = model.inference([img_uint8])
    depth = getattr(pred, "depth", None)
    if depth is None and isinstance(pred, dict):
        depth = pred["depth"]
    depth = np.asarray(depth.cpu().numpy() if hasattr(depth, "cpu") else depth, dtype=np.float32)
    return depth[0] if depth.ndim == 3 else depth


def _da3_predictor(cfg, variant, device):
    """Zero-shot DA3 predictor: color (1,3,h,w) in [0,1] -> depth (h', w') numpy, or None.

    "da3-base" runs the ported encoder + DualDPT main branch (models/endodac/da3_zeroshot.py,
    verified bit-exact against the original code) from <pretrained_path>/da3_base.safetensors,
    so it needs no extra package. Any other id goes through the depth_anything_3 package
    (Python >= 3.9, torch >= 2) and is skipped when that is not installed.
    """
    if variant in ("da3-base", "depth-anything/da3-base", "depth-anything/DA3-BASE"):
        path = da3_weights_path(cfg)
        if not os.path.exists(path):
            print("[da3] missing {}: wget -O {} {}".format(path, path, DA3_URL))
            return None
        from models.endodac.da3_zeroshot import DA3BaseMono
        model = DA3BaseMono(weights_path=path).to(device).eval()

        def predict(color):
            with torch.no_grad():
                return model(color.to(device))[0, 0].float().cpu().numpy()
        return predict
    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError:
        print("[da3] {} needs the depth_anything_3 package, not installed here: skipped".format(variant))
        return None
    model = DepthAnything3.from_pretrained(variant).to(device).eval()

    def predict(color):
        return _da3_package_infer(model, (color[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    return predict


def stage_da3(cfg, args):
    device = device_of(args)
    per_frame = out_path(cfg, "per_frame.csv")
    done = {(r["method"], r["seed"], r["dataset"]) for r in read_csv(per_frame)}
    fields = ["method", "seed", "dataset", "sequence", "frame"] + METRICS + ["ratio", "infer_ms", "git"]
    g = git_hash()
    datasets = args.datasets or cfg["datasets"]
    for variant in cfg["da3"]["models"]:
        method = "DA3_" + variant.split("/")[-1].replace("-", "_").lower()
        todo = [d for d in datasets if (method, "0", d) not in done or args.force]
        if not todo:
            continue
        print("[da3] loading {}".format(variant))
        predict = _da3_predictor(cfg, variant, device)
        if predict is None:
            continue
        for ds in todo:
            rows, rows_aff, preds = [], [], []
            for i, color, seq, frame, gt in iterate_dataset(cfg, ds):
                t0 = time.time()
                depth = predict(color)
                ms = (time.time() - t0) * 1000.0
                preds.append(depth.astype(np.float16))
                for align, target, name in (("median", rows, method), ("affine", rows_aff, method + "_affine")):
                    m, ratio = evaluate_prediction(depth, "depth", align, gt, MAX_DEPTH[ds])
                    row = {"method": name, "seed": 0, "dataset": ds, "sequence": seq, "frame": frame,
                           "ratio": ratio, "infer_ms": ms, "git": g}
                    row.update(m)
                    target.append(row)
            append_rows(per_frame, rows, fields)
            if cfg["da3"].get("affine_rows", True):
                append_rows(per_frame, rows_aff, fields)
            if cfg["save_pred"]:
                np.save(out_path(cfg, "pred", "{}_s0_{}.npy".format(method, ds)), np.stack(preds))
            print("[da3] {:<24} {:<7} abs_rel={:.4f}".format(method, ds, float(np.mean([r["abs_rel"] for r in rows]))))


# --------------------------------------------------------------------------------------
# stage: stats
# --------------------------------------------------------------------------------------

def group_mean(rows, keys, values):
    """Mean of `values` over rows sharing the same `keys` (insertion ordered)."""
    acc = {}
    for r in rows:
        k = tuple(r[x] for x in keys)
        a = acc.setdefault(k, {v: [] for v in values})
        for v in values:
            a[v].append(float(r[v]))
    out = []
    for k, a in acc.items():
        row = dict(zip(keys, k))
        row.update({v: float(np.mean(a[v])) for v in values})
        row["n"] = len(a[values[0]])
        out.append(row)
    return out


def bootstrap_ci(x, n_boot, alpha, rng, stat=np.mean):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    s = stat(x[idx], axis=1)
    return float(np.percentile(s, 100 * alpha / 2)), float(np.percentile(s, 100 * (1 - alpha / 2)))


def t_ci(x, alpha):
    from scipy import stats as st
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float("nan"), float("nan")
    h = st.t.ppf(1 - alpha / 2, len(x) - 1) * st.sem(x)
    return float(x.mean() - h), float(x.mean() + h)


def paired_tests(d):
    """d = method - proposed per sequence. Returns dict of p-values and effect size."""
    from scipy import stats as st
    d = np.asarray(d, dtype=float)
    res = {"n": len(d), "mean_diff": float(d.mean()), "dz": float(d.mean() / (d.std(ddof=1) + 1e-12)) if len(d) > 1 else float("nan"),
           "wins": int((d < 0).sum()), "losses": int((d > 0).sum())}
    if len(d) > 1 and np.any(d != 0):
        res["p_t"] = float(st.ttest_rel(d, np.zeros_like(d)).pvalue)
        try:
            res["p_wilcoxon"] = float(st.wilcoxon(d, method="exact").pvalue)
        except TypeError:
            res["p_wilcoxon"] = float(st.wilcoxon(d, mode="exact").pvalue)
        except ValueError:
            res["p_wilcoxon"] = float("nan")
        k, n = int((d != 0).sum()), int(min((d < 0).sum(), (d > 0).sum()))
        try:
            res["p_sign"] = float(st.binomtest(n, k, 0.5).pvalue) if k else float("nan")
        except AttributeError:
            res["p_sign"] = float(st.binom_test(n, k, 0.5)) if k else float("nan")
    else:
        res.update({"p_t": float("nan"), "p_wilcoxon": float("nan"), "p_sign": float("nan")})
    return res


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    adj = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.sum() == 0:
        return adj
    idx = np.argsort(p[ok])
    m = ok.sum()
    sorted_p = p[ok][idx]
    a = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * sorted_p))
    tmp = np.empty(m)
    tmp[idx] = a
    adj[ok] = tmp
    return adj


def fmt(x, nd=3):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else "{:.{}f}".format(x, nd)


def stage_stats(cfg, args):
    sc = cfg["stats"]
    rows = read_csv(out_path(cfg, "per_frame.csv"))
    if not rows:
        print("[stats] per_frame.csv is empty; run predict first")
        return
    rng = np.random.default_rng(0)
    # 1) frame -> (method, seed, dataset, sequence); 2) -> (method, dataset, sequence) over seeds
    per_seed_seq = group_mean(rows, ["method", "seed", "dataset", "sequence"], METRICS)
    per_seq = group_mean(per_seed_seq, ["method", "dataset", "sequence"], METRICS)
    write_csv(out_path(cfg, "per_sequence.csv"), per_seq, ["method", "dataset", "sequence", "n"] + METRICS)
    # seed variability of the overall mean (multi-seed runs)
    per_seed = group_mean(per_seed_seq, ["method", "seed", "dataset"], METRICS)
    seed_sd = {}
    for (m, d), grp in _groupby(per_seed, ["method", "dataset"]).items():
        if len(grp) > 1:
            seed_sd[(m, d)] = {k: float(np.std([g[k] for g in grp], ddof=1)) for k in METRICS}
    # 3) summary per (method, dataset)
    summary = []
    for (m, d), grp in _groupby(per_seq, ["method", "dataset"]).items():
        row = {"method": m, "dataset": d, "n_seq": len(grp), "n_seeds": len({r["seed"] for r in rows if r["method"] == m and r["dataset"] == d})}
        for k in METRICS:
            x = np.array([g[k] for g in grp])
            row[k] = float(x.mean())
            row[k + "_sd"] = float(x.std(ddof=1)) if len(x) > 1 else float("nan")
            row[k + "_tlo"], row[k + "_thi"] = t_ci(x, sc["alpha"])
            row[k + "_blo"], row[k + "_bhi"] = bootstrap_ci(x, sc["n_boot"], sc["alpha"], rng)
            row[k + "_seed_sd"] = seed_sd.get((m, d), {}).get(k, float("nan"))
        summary.append(row)
    sfields = ["method", "dataset", "n_seq", "n_seeds"] + [k + s for k in METRICS for s in ("", "_sd", "_tlo", "_thi", "_blo", "_bhi", "_seed_sd")]
    write_csv(out_path(cfg, "summary.csv"), summary, sfields)
    # 4) paired comparisons against the proposed method, per dataset and metric
    prop = cfg["proposed"]
    paired = []
    by_md = _groupby(per_seq, ["method", "dataset"])
    for d in sorted({r["dataset"] for r in per_seq}):
        if (prop, d) not in by_md:
            print("[stats] proposed method {} has no rows on {}".format(prop, d))
            continue
        ref = {r["sequence"]: r for r in by_md[(prop, d)]}
        block = []
        for (m, dd), grp in by_md.items():
            if dd != d or m == prop:
                continue
            common = [r for r in grp if r["sequence"] in ref]
            if len(common) < 2:
                continue
            for k in METRICS:
                diff = [r[k] - ref[r["sequence"]][k] for r in common]
                if not LOWER_IS_BETTER[k]:
                    diff = [-x for x in diff]  # so that negative = proposed is better, consistently
                res = paired_tests(diff)
                res["diff_blo"], res["diff_bhi"] = bootstrap_ci(diff, sc["n_boot"], sc["alpha"], rng)
                res.update({"dataset": d, "method": m, "metric": k})
                block.append(res)
        for k in METRICS:  # Holm across methods within (dataset, metric)
            sel = [r for r in block if r["metric"] == k]
            for key in ("p_t", "p_wilcoxon", "p_sign"):
                adj = holm([r[key] for r in sel])
                for r, a in zip(sel, adj):
                    r[key + "_holm"] = float(a)
        paired += block
    pfields = ["dataset", "method", "metric", "n", "mean_diff", "diff_blo", "diff_bhi", "dz", "wins", "losses",
               "p_t", "p_wilcoxon", "p_sign", "p_t_holm", "p_wilcoxon_holm", "p_sign_holm"]
    write_csv(out_path(cfg, "paired.csv"), paired, pfields)
    write_tables(cfg, summary, paired, per_seq)
    print("[stats] wrote summary.csv, per_sequence.csv, paired.csv and tables/*.tex")


def _groupby(rows, keys):
    out = {}
    for r in rows:
        out.setdefault(tuple(r[k] for k in keys), []).append(r)
    return out


def _method_order(cfg, methods):
    order = cfg["stats"].get("table_methods")
    prop = cfg["proposed"]
    if order:
        return [m for m in order if m in methods]
    runs = [m for m in ABLATION_ORDER if m in methods and m != prop]
    ext = sorted(m for m in methods if m not in GRID and m != prop)
    return ext + runs + ([prop] if prop in methods else [])


def write_tables(cfg, summary, paired, per_seq):
    tdir = ensure_dir(os.path.join(cfg["out_dir"], "tables"))
    by = {(r["method"], r["dataset"]): r for r in summary}
    pb = {(r["dataset"], r["method"], r["metric"]): r for r in paired}
    sc = cfg["stats"]
    for d in sorted({r["dataset"] for r in summary}):
        methods = _method_order(cfg, [m for (m, dd) in by if dd == d])
        best = {}
        for k in METRICS:
            vals = [(by[(m, d)][k], m) for m in methods]
            best[k] = (min if LOWER_IS_BETTER[k] else max)(vals)[1]
        lines = ["% sequence-level results on {}: mean $\\pm$ SD over n={} video sequences".format(d, by[(methods[0], d)]["n_seq"]),
                 "\\begin{tabular}{l" + "c" * len(METRICS) + "}", "\\toprule",
                 "Method & " + " & ".join(k.replace("_", "\\_") for k in METRICS) + " \\\\", "\\midrule"]
        for m in methods:
            r = by[(m, d)]
            cells = []
            for k in METRICS:
                s = "{} $\\pm$ {}".format(fmt(r[k]), fmt(r[k + "_sd"]))
                cells.append("\\textbf{" + s + "}" if best[k] == m else s)
            lines.append("{} & {} \\\\".format(m.replace("_", "\\_"), " & ".join(cells)))
        lines += ["\\bottomrule", "\\end{tabular}"]
        with open(os.path.join(tdir, "main_{}.tex".format(d)), "w") as f:
            f.write("\n".join(lines) + "\n")
        # paired table
        pm = sc["paired_metrics"]
        lines = ["% paired differences (method $-$ {}) over sequences; negative = proposed better; bootstrap 95\\% CI; Holm-adjusted exact Wilcoxon".format(cfg["proposed"]),
                 "\\begin{tabular}{l" + "c" * len(pm) + "}", "\\toprule",
                 "Method & " + " & ".join("$\\Delta$" + k.replace("_", "\\_") + " [CI], $p_W$" for k in pm) + " \\\\", "\\midrule"]
        for m in methods:
            if m == cfg["proposed"]:
                continue
            cells = []
            for k in pm:
                r = pb.get((d, m, k))
                if r is None:
                    cells.append("--")
                else:
                    cells.append("{} [{}, {}], {} ({}/{})".format(fmt(r["mean_diff"]), fmt(r["diff_blo"]), fmt(r["diff_bhi"]),
                                                                  fmt(r.get("p_wilcoxon_holm"), 3), r["wins"], r["n"]))
            lines.append("{} & {} \\\\".format(m.replace("_", "\\_"), " & ".join(cells)))
        lines += ["\\bottomrule", "\\end{tabular}"]
        with open(os.path.join(tdir, "paired_{}.tex".format(d)), "w") as f:
            f.write("\n".join(lines) + "\n")
        # supplementary per-sequence abs_rel
        seqs = sorted({r["sequence"] for r in per_seq if r["dataset"] == d})
        ps = {(r["method"], r["sequence"]): r["abs_rel"] for r in per_seq if r["dataset"] == d}
        lines = ["% per-sequence Abs Rel on {}".format(d), "\\begin{tabular}{l" + "c" * len(seqs) + "}", "\\toprule",
                 "Method & " + " & ".join(s.replace("_", "\\_") for s in seqs) + " \\\\", "\\midrule"]
        for m in methods:
            lines.append("{} & {} \\\\".format(m.replace("_", "\\_"), " & ".join(fmt(ps.get((m, s))) for s in seqs)))
        lines += ["\\bottomrule", "\\end{tabular}"]
        with open(os.path.join(tdir, "per_sequence_{}.tex".format(d)), "w") as f:
            f.write("\n".join(lines) + "\n")
    # ablation and calibration tables (SCARED full metrics + Abs Rel on the other sets)
    runs = all_runs(cfg)
    others = [d for d in cfg["datasets"] if d != "scared"]
    for fname, order in (("ablation.tex", [(r, runs[r]["desc"]) for r in ABLATION_ORDER if r in runs]),
                         ("calibration.tex", CALIB_ORDER)):
        lines = ["\\begin{tabular}{ll" + "c" * (len(METRICS) + len(others)) + "}", "\\toprule",
                 "Run & Variant & " + " & ".join(k.replace("_", "\\_") for k in METRICS) + "".join(" & {} Abs Rel".format(d) for d in others) + " \\\\", "\\midrule"]
        for run, desc in order:
            if (run, "scared") not in by:
                continue
            r = by[(run, "scared")]
            cells = ["{} $\\pm$ {}".format(fmt(r[k]), fmt(r[k + "_sd"])) for k in METRICS]
            cells += [fmt(by[(run, d)]["abs_rel"]) if (run, d) in by else "--" for d in others]
            lines.append("{} & {} & {} \\\\".format(run, desc, " & ".join(cells)))
        lines += ["\\bottomrule", "\\end{tabular}"]
        with open(os.path.join(tdir, fname), "w") as f:
            f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------------------
# stage: report
# --------------------------------------------------------------------------------------

def _md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _try_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def stage_report(cfg, args):
    ic = cfg["illum"]
    md = ["# MonoIIF CVIU revision: results", "", "git: `{}`  generated: {}".format(git_hash(), time.strftime("%Y-%m-%d %H:%M")), ""]
    plt = _try_plt()
    fdir = ensure_dir(os.path.join(cfg["out_dir"], "figures"))
    manifest = out_path(cfg, "train_manifest.json")
    if os.path.exists(manifest):
        with open(manifest) as f:
            mf = json.load(f)
        md += ["Training grid launched at git `{}` ({} jobs).".format(mf["git"], len(mf["jobs"])), ""]

    summary = read_csv(out_path(cfg, "summary.csv"))
    paired = read_csv(out_path(cfg, "paired.csv"))
    per_seq = read_csv(out_path(cfg, "per_sequence.csv"))
    if summary:
        md += ["## Sequence-level results (mean ± SD over sequences; bootstrap 95% CI of the mean)", ""]
        for d in sorted({r["dataset"] for r in summary}):
            rows = [r for r in summary if r["dataset"] == d]
            order = _method_order(cfg, [r["method"] for r in rows])
            by = {r["method"]: r for r in rows}
            md += ["### {} (n = {} sequences)".format(d, rows[0]["n_seq"]), ""]
            md.append(_md_table(["method", "seeds"] + METRICS, [
                [m, by[m]["n_seeds"]] + ["{} ± {} [{}, {}]".format(fmt(float(by[m][k])), fmt(float(by[m][k + "_sd"])), fmt(float(by[m][k + "_blo"])), fmt(float(by[m][k + "_bhi"]))) for k in METRICS]
                for m in order]))
            md.append("")
            pr = [r for r in paired if r["dataset"] == d and r["metric"] in cfg["stats"]["paired_metrics"]]
            if pr:
                md += ["Paired vs {} (negative = proposed better): mean diff [bootstrap CI], Wilcoxon p (Holm), wins/n".format(cfg["proposed"]), ""]
                md.append(_md_table(["method", "metric", "diff [CI]", "p_t", "p_W (Holm)", "p_sign", "wins/n", "d_z"], [
                    [r["method"], r["metric"], "{} [{}, {}]".format(fmt(float(r["mean_diff"])), fmt(float(r["diff_blo"])), fmt(float(r["diff_bhi"]))),
                     fmt(float(r["p_t"])), fmt(float(r["p_wilcoxon_holm"])), fmt(float(r["p_sign"])), "{}/{}".format(r["wins"], r["n"]), fmt(float(r["dz"]), 2)]
                    for r in pr]))
                md.append("")
            if plt is not None and per_seq:
                fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(order)), 3.2))
                data = [[float(r["abs_rel"]) for r in per_seq if r["dataset"] == d and r["method"] == m] for m in order]
                ax.boxplot(data, labels=order, showmeans=True)
                ax.set_ylabel("Abs Rel per sequence")
                ax.set_title(d)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
                fig.tight_layout()
                fig.savefig(os.path.join(fdir, "per_sequence_{}.pdf".format(d)))
                plt.close(fig)
                md += ["![per-sequence {}](figures/per_sequence_{}.pdf)".format(d, d), ""]

    # illum-fit
    fits = {seq: read_csv(out_path(cfg, "illum", "fit_{}.csv".format(seq))) for seq in ic["sequences"]}
    fits = {k: v for k, v in fits.items() if v}
    if fits:
        md += ["## Model-free illumination test (GT geometry, cross-validated L1 residual)", "",
               "Explained fraction = 1 - r_model / r_none. Bounded = clipped to the decoder's ranges and blurred.", ""]
        models = ["global"] + ["local{}".format(P) for P in ic["patch_sizes"]] + ["bounded{}".format(P) for P in ic["patch_sizes"]]
        rows = []
        curves = {}
        for seq, fr in fits.items():
            r_none = np.array([float(r["r_none"]) for r in fr])
            r_id = np.array([float(r["r_identity"]) for r in fr])
            row = [seq, len(fr), fmt(float(np.mean([float(r["zbuffer_pass"]) for r in fr])), 2), fmt(r_id.mean(), 4), fmt(r_none.mean(), 4)]
            for m in models:
                rm = np.array([float(r["r_" + m]) for r in fr])
                ef = 1 - rm / r_none
                lo, hi = bootstrap_ci(ef, 2000, 0.05, np.random.default_rng(0))
                row.append("{} [{}, {}]".format(fmt(ef.mean(), 3), fmt(lo, 3), fmt(hi, 3)))
                curves[(seq, m)] = ef.mean()
            oob = np.mean([float(r.get("out_of_bounds_local16", "nan")) for r in fr])
            row.append(fmt(oob, 3))
            rows.append(row)
        md.append(_md_table(["sequence", "pairs", "z-buffer pass", "r identity", "r none"] + ["EF " + m for m in models] + ["oracle16 outside bounds"], rows))
        md.append("")
        if plt is not None:
            fig, ax = plt.subplots(figsize=(4.5, 3))
            for seq in fits:
                ax.plot(ic["patch_sizes"], [curves[(seq, "local{}".format(P))] for P in ic["patch_sizes"]], "o-", label="{} local".format(seq))
                ax.plot(ic["patch_sizes"], [curves[(seq, "bounded{}".format(P))] for P in ic["patch_sizes"]], "s--", label="{} bounded".format(seq))
                ax.axhline(curves[(seq, "global")], ls=":", lw=1)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("patch size (px)")
            ax.set_ylabel("explained fraction of residual")
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(os.path.join(fdir, "explained_vs_patch.pdf"))
            plt.close(fig)
            md += ["![explained vs patch](figures/explained_vs_patch.pdf)", ""]

    # illum-params
    pfiles = sorted(glob.glob(os.path.join(cfg["out_dir"], "illum", "params_*.csv")))
    if pfiles:
        md += ["## Learned calibration parameters (per-frame statistics, mean over frames)", ""]
        cols = ["c_mean", "c_std", "c_p1", "c_p99", "b_mean", "b_std", "b_p1", "b_p99", "sat_c", "sat_b", "grad_c", "grad_b",
                "corr_c_radial", "corr_b_radial", "corr_c_highlight", "corr_b_highlight", "temporal_corr_c", "reciprocity_err",
                "res_before", "res_after", "res_oracle16"]
        rows = []
        for p in pfiles:
            fr = read_csv(p)
            name = os.path.basename(p)[7:-4]
            vals = []
            for c in cols:
                x = [float(r[c]) for r in fr if r.get(c) not in (None, "")]
                vals.append(fmt(float(np.mean(x)), 4) if x else "--")
            tz = [(float(r["tz_gt"]), float(r["b_mean"])) for r in fr if r.get("tz_gt") not in (None, "") and r["source"] == "1"]
            corr_tz = fmt(float(np.corrcoef([t for t, _ in tz], [b for _, b in tz])[0, 1]), 3) if len(tz) > 2 else "--"
            rows.append([name, len(fr)] + vals + [corr_tz])
        md.append(_md_table(["run/seq", "rows"] + cols + ["corr(b, gt tz)"], rows))
        md.append("")
        if plt is not None:
            for hp in sorted(glob.glob(os.path.join(cfg["out_dir"], "illum", "hist_*.npz"))):
                h = np.load(hp)
                fig, axes = plt.subplots(1, 2, figsize=(7, 2.6))
                for ax, key, bounds in ((axes[0], "c", (1 - float(h["alpha"]), 1 + float(h["alpha"]))), (axes[1], "b", (-float(h["beta"]), float(h["beta"])))):
                    e = h[key + "_edges"]
                    ax.bar(0.5 * (e[1:] + e[:-1]), h[key] / max(h[key].sum(), 1), width=e[1] - e[0])
                    for bnd in bounds:
                        ax.axvline(bnd, color="k", ls="--", lw=0.8)
                    ax.set_xlabel(key)
                fig.tight_layout()
                name = os.path.basename(hp)[5:-4]
                fig.savefig(os.path.join(fdir, "cb_hist_{}.pdf".format(name)))
                plt.close(fig)
                md += ["![c/b histogram {}](figures/cb_hist_{}.pdf)".format(name, name), ""]

    # illum-sens
    sfiles = sorted(glob.glob(os.path.join(cfg["out_dir"], "illum", "sens_*.csv")))
    if sfiles:
        md += ["## Sensitivity to synthetic gain / bias", ""]
        rows = []
        for p in sfiles:
            fr = read_csv(p)
            name = os.path.basename(p)[5:-4]
            cal = [r for r in fr if r["kind"] == "calib"]
            g_rows = [(float(r["gain"]), float(r["c_mean"])) for r in cal if float(r["bias"]) == 0.0]
            b_rows = [(float(r["bias"]), float(r["b_mean"])) for r in cal if float(r["gain"]) == 1.0]
            # the warp is c * (g * I + bias) + b: the calibration should respond with c ~ 1/g, b ~ -bias
            sg = np.polyfit([1 / g for g, _ in g_rows], [c for _, c in g_rows], 1)[0] if len(set(g for g, _ in g_rows)) > 1 else float("nan")
            sb = np.polyfit([-b for b, _ in b_rows], [bm for _, bm in b_rows], 1)[0] if len(set(b for b, _ in b_rows)) > 1 else float("nan")
            dep = [r for r in fr if r["kind"] == "depth"]
            base = np.mean([float(r["abs_rel"]) for r in dep if float(r["gain"]) == 1.0 and float(r["bias"]) == 0.0]) if dep else float("nan")
            worst = max([np.mean([float(r["abs_rel"]) for r in dep if float(r["gain"]) == g and float(r["bias"]) == b])
                         for g, b in {(float(r["gain"]), float(r["bias"])) for r in dep}]) if dep else float("nan")
            rows.append([name, fmt(sg, 3), fmt(sb, 3), fmt(base, 4), fmt(worst, 4)])
        md.append(_md_table(["run", "slope c vs 1/gain", "slope b vs -bias", "Abs Rel clean", "Abs Rel worst perturbation"], rows))
        md += ["", "Slopes near 1 mean the calibration head tracks the injected illumination change.", ""]

    with open(out_path(cfg, "report.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("[report] wrote {}".format(out_path(cfg, "report.md")))


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

STAGES = {"train": stage_train, "predict": stage_predict, "illum-fit": stage_illum_fit,
          "illum-params": stage_illum_params, "illum-sens": stage_illum_sens, "da3": stage_da3,
          "stats": stage_stats, "report": stage_report}
ALL_ORDER = ["predict", "illum-fit", "illum-params", "illum-sens", "da3", "stats", "report"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=list(STAGES) + ["all", "full"])
    ap.add_argument("--config", default=os.path.join(ROOT, "cviu_config.yaml"))
    ap.add_argument("--python", help="interpreter for the training subprocesses (overrides the config)")
    ap.add_argument("--gpus", nargs="*", help="GPU ids for train (one subprocess per GPU)")
    ap.add_argument("--only", nargs="*", help="restrict to these runs / methods")
    ap.add_argument("--seeds", nargs="*", type=int, help="train: override the seeds of every selected run")
    ap.add_argument("--datasets", nargs="*", help="restrict predict/da3 to these datasets")
    ap.add_argument("--extra_flags", default="", help="appended to every training command (e.g. \"--num_epochs 1\")")
    ap.add_argument("--dry_run", action="store_true", help="train: print the commands only")
    ap.add_argument("--skip_preflight", action="store_true", help="train: launch even if the pre-flight checks fail")
    ap.add_argument("--force", action="store_true", help="redo work whose outputs exist")
    ap.add_argument("--all_seeds", action="store_true", help="illum-params: every seed of the run")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.python:
        cfg["python"] = args.python
    if args.seeds:
        cfg["train"]["seeds"] = cfg["train"]["multi_seeds"] = list(args.seeds)
    ensure_dir(cfg["out_dir"])
    if args.stage == "all":
        stages = ALL_ORDER
    elif args.stage == "full":
        stages = ["train"] if args.dry_run else ["train"] + ALL_ORDER
    else:
        stages = [args.stage]
    failed = []
    for s in stages:
        print("=" * 20, s, "=" * 20)
        try:
            n_failed = STAGES[s](cfg, args)
            if s == "train" and n_failed and len(stages) > 1:
                print("[full] {} training job(s) failed; not continuing to the analysis stages. "
                      "Fix the error and rerun 'full' (finished runs are skipped).".format(n_failed))
                sys.exit(1)
        except Exception:
            traceback.print_exc()
            failed.append(s)
            if len(stages) == 1:
                sys.exit(1)
    if failed:
        print("FAILED stages: {}".format(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
