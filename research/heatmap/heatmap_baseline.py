#!/usr/bin/env python3
"""Swin/FPN spatial heatmap baseline. No VQ, codebook or landmark token head.

Train on the original training partition only. Checkpoint selection uses a
fixed internal training-image split; optional (default) refit uses all training
images for the selected epoch count. Evaluation never selects a checkpoint.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import timm
from torchvision import transforms as T


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def angle(points):
    p = np.asarray(points, dtype=np.float64)
    u = p[..., 1, :] - p[..., 0, :]
    v = p[..., 2, :] - p[..., 3, :]
    w = p[..., 4, :] - p[..., 3, :]
    def theta(a, b):
        dot = np.sum(a * b, axis=-1)
        cross = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
        return np.abs(np.degrees(np.arctan2(cross, dot)))
    return np.stack((theta(u, v), theta(w, u)), axis=-1)


def training_partition(root, fraction, seed):
    names = (root / 'multi_train_list.txt').read_text().splitlines()
    heldout = set((root / 'multi_val_list.txt').read_text().splitlines())
    if len(names) != 2180 or len(heldout) != 242 or set(names) & heldout:
        raise ValueError('Expected the original disjoint 2,180/242 partition')
    ordered = sorted(names, key=lambda n: hashlib.sha256(
        f'heatmap-training-only-split-v1|{seed}|{n}'.encode()).hexdigest())
    n_val = max(1, round(len(names) * fraction))
    validation = set(ordered[:n_val])
    return names, [n for n in names if n not in validation], [n for n in names if n in validation]


class Speckle:
    def __call__(self, x):
        if torch.rand(()) < 0.3:
            x = (x + x * torch.randn_like(x) * .05).clamp(0, 1)
        return x


def image_transform(size, augment):
    ops = [T.Resize((size, size), interpolation=T.InterpolationMode.BILINEAR)]
    if augment:
        ops += [T.ColorJitter(.3, .3, .3, .05), T.RandomAutocontrast(.4),
                T.RandomAdjustSharpness(2., .4), T.GaussianBlur(3, (.1, 1.5))]
    ops += [T.ToTensor()]
    if augment:
        ops += [Speckle(), T.RandomErasing(.4, scale=(.02, .1), ratio=(.3, 3.3))]
    ops += [T.Normalize((.485, .456, .406), (.229, .224, .225))]
    return T.Compose(ops)


class DDHImages(Dataset):
    def __init__(self, root, names, size=448, augment=False):
        self.root, self.names = Path(root), list(names)
        all_names = (self.root / 'multi_image_list.txt').read_text().splitlines()
        if len(set(all_names)) != len(all_names):
            raise ValueError('Duplicate names in index')
        self.index = {n: i for i, n in enumerate(all_names)}
        self.raters = np.stack([np.loadtxt(self.root / f'landmarks_{r}.txt').reshape(-1, 5, 2)
                               for r in 'abcde'], axis=1)
        self.valid = np.any(self.raters > 0, axis=(2, 3))
        self.valid[:, :2] = True
        if self.raters.shape[0] != len(all_names):
            raise ValueError('Annotation/index mismatch')
        self.transform = image_transform(size, augment)
        self.mean_raters = float(np.mean([self.valid[self.index[n]].sum() for n in self.names]))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        row = self.index[name]
        with Image.open(self.root / 'images' / name) as image:
            image = image.convert('RGB')
            width, height = image.size
            tensor = self.transform(image)
        if (width, height) != (617, 563):
            raise ValueError(f'Unexpected source grid for {name}: {(width, height)}')
        points = self.raters[row].copy()
        mask = self.valid[row].copy()
        if not np.isfinite(points[mask]).all():
            raise ValueError(f'Nonfinite annotation: {name}')
        scale = np.array([width, height])
        mean = points[mask].mean(axis=0)
        return {'image': tensor, 'raters_norm': torch.tensor(points / scale, dtype=torch.float32),
                'raters_px': torch.tensor(points, dtype=torch.float64),
                'valid': torch.tensor(mask), 'gt_px': torch.tensor(mean, dtype=torch.float64),
                'wh': torch.tensor(scale, dtype=torch.float64), 'name': name}


class CachedDDH(Dataset):
    """Read shared frozen-encoder maps. Never executes the encoder or reads images."""
    def __init__(self, cache_root, names):
        root = Path(cache_root)
        z = np.load(root / 'meta.npz', allow_pickle=False)
        self.meta = {k: z[k] for k in z.files}
        self.features = [np.load(root / f'level{i}.npy', mmap_mode='r') for i in range(4)]
        cache_names = self.meta['img_names'].astype(str)
        index = {n: i for i, n in enumerate(cache_names)}
        if len(index) != len(cache_names):
            raise ValueError('Duplicate cached image names')
        self.names = list(names)
        self.indices = [index[n] for n in names]
        self.mean_raters = float(self.meta['rater_valid'][self.indices].sum(1).mean())
        if any(len(a) != len(cache_names) for a in self.features):
            raise ValueError('Cache arrays disagree on row count')

    def __len__(self): return len(self.names)

    def __getitem__(self, i):
        row = self.indices[i]
        points = self.meta['rater_px'][row]
        valid = self.meta['rater_valid'][row].astype(bool)
        norm_key = 'rater_norm' if 'rater_norm' in self.meta else 'raters_norm'
        return {'image': [torch.tensor(np.array(a[row]), dtype=torch.float32) for a in self.features],
                'raters_norm': torch.tensor(self.meta[norm_key][row], dtype=torch.float32),
                'raters_px': torch.tensor(points, dtype=torch.float64), 'valid': torch.tensor(valid),
                'gt_px': torch.tensor(points[valid].mean(0), dtype=torch.float64),
                'wh': torch.tensor(self.meta['wh'][row], dtype=torch.float64), 'name': self.names[i]}


def dataset(args, names, augment=False):
    if getattr(args, 'feature_cache', None):
        return CachedDDH(args.feature_cache, names)
    return DDHImages(args.data_root, names, args.image_size, augment)


def verify_feature_cache(cache_root, expected_backbone_hash):
    manifest_path = Path(cache_root) / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('status') != 'complete':
        raise ValueError('Shared feature cache is not complete')
    if expected_backbone_hash not in manifest.get('source_hashes', {}).values():
        raise ValueError('Shared cache must use exactly the requested source encoder checkpoint')
    if int(manifest['img_size']) != 448 or manifest.get('feature_dtype') != 'float16':
        raise ValueError('Expected the declared 448-pixel float16 shared feature cache')
    if digest(Path(cache_root) / 'meta.npz') != manifest['labels_sha256']:
        raise ValueError('Shared cache label metadata failed its digest check')
    return digest(manifest_path)


def feature_state(path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    sd = payload.get('state_dict', payload)
    prefix = 'guide_backbone.model.'
    if any(k.startswith(prefix) for k in sd):
        return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}, 'original-stage1-encoder'
    if payload.get('format') == 'timm-feature-backbone-state-v1':
        return sd, 'imagenet-feature-backbone-artifact'
    raise ValueError('Provide original Stage 1 checkpoint or declared timm feature-backbone artifact')


class SpatialHeatmap(nn.Module):
    def __init__(self, size=448, heatmap_size=112, hidden=128, freeze_encoder=True):
        super().__init__()
        self.encoder = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False,
                                         features_only=True, out_indices=(0, 1, 2, 3), img_size=size)
        self.channels = self.encoder.feature_info.channels()
        self.heatmap_size = heatmap_size
        self.freeze_encoder = freeze_encoder
        self.laterals = nn.ModuleList([nn.Conv2d(c, hidden, 1) for c in self.channels])
        self.smooth = nn.ModuleList([nn.Sequential(nn.Conv2d(hidden, hidden, 3, padding=1),
                                       nn.GroupNorm(8, hidden), nn.GELU()) for _ in self.channels])
        self.head = nn.Sequential(nn.Conv2d(hidden, hidden, 3, padding=1),
                                  nn.GroupNorm(8, hidden), nn.GELU(), nn.Dropout2d(.1), nn.Conv2d(hidden, 5, 1))
        if freeze_encoder:
            self.encoder.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            features = x
        elif self.freeze_encoder:
            with torch.no_grad():
                features = self.encoder(x)
        else:
            features = self.encoder(x)
        features = [f if f.shape[1] == c else f.permute(0, 3, 1, 2).contiguous()
                    for f, c in zip(features, self.channels)]
        pyramid = None
        for i in range(len(features) - 1, -1, -1):
            lateral = self.laterals[i](features[i])
            if pyramid is not None:
                lateral = lateral + F.interpolate(pyramid, size=lateral.shape[-2:],
                                                  mode='bilinear', align_corners=False)
            pyramid = self.smooth[i](lateral)
        logits = self.head(pyramid)
        return F.interpolate(logits, (self.heatmap_size, self.heatmap_size),
                             mode='bilinear', align_corners=False)


def spatial_grid(size, device, dtype):
    axis = (torch.arange(size, device=device, dtype=dtype) + .5) / size
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    return torch.stack((x, y), dim=-1).reshape(-1, 2)


def target_density(raters, valid, size, sigma_grid):
    grid = spatial_grid(size, raters.device, raters.dtype)
    distance = ((raters[..., None, :] - grid) * size / sigma_grid).square().sum(-1)
    kernels = (-.5 * distance).softmax(-1)
    kernels = kernels * valid[..., None, None]
    return kernels.sum(1) / valid.sum(1)[:, None, None]


def density_loss(logits, raters, valid, sigma_grid, image_weights=None):
    target = target_density(raters, valid, logits.shape[-1], sigma_grid)
    log_prob = logits.float().flatten(2).log_softmax(-1)
    # Every image and every landmark has equal weight; raters share image weight.
    values = -(target * log_prob).sum(-1).mean(-1)
    return (values * image_weights).mean() if image_weights is not None else values.mean()


def moments(logits):
    prob = logits.float().flatten(2).softmax(-1)
    grid = spatial_grid(logits.shape[-1], prob.device, prob.dtype)
    mean = torch.einsum('bkp,pd->bkd', prob, grid)
    centered = grid[None, None] - mean[:, :, None]
    cov = torch.einsum('bkp,bkpd,bkpe->bkde', prob, centered, centered)
    return mean, cov, prob


def batch_to(batch, device):
    def move(v):
        if isinstance(v, torch.Tensor): return v.to(device, non_blocking=True)
        if isinstance(v, (list, tuple)) and v and isinstance(v[0], torch.Tensor): return [move(x) for x in v]
        return v
    return {k: move(v) for k, v in batch.items()}


def loader(dataset, batch_size, workers, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=True, drop_last=False, persistent_workers=workers > 0)


def validate(model, batches, device, sigma, row_weighting=False):
    model.eval()
    losses, distances = [], []
    with torch.no_grad():
        for batch in batches:
            b = batch_to(batch, device)
            logits = model(b['image'])
            weights = b['valid'].sum(1) / batches.dataset.mean_raters if row_weighting else None
            loss = density_loss(logits, b['raters_norm'], b['valid'], sigma, weights)
            mu, _, _ = moments(logits)
            error = torch.linalg.vector_norm(mu.double() * b['wh'][:, None] - b['gt_px'], dim=-1)
            losses.append((float(loss), len(mu)))
            distances.extend(error.mean(-1).cpu().tolist())
    return {'loss': sum(v * n for v, n in losses) / sum(n for _, n in losses),
            'mre_px': float(np.mean(distances))}


def make_model(args, initialize=True):
    model = SpatialHeatmap(args.image_size, args.heatmap_size, args.hidden, args.freeze_encoder)
    if initialize:
        weights, label = feature_state(args.backbone_checkpoint)
        model.encoder.load_state_dict(weights, strict=True)
        return model, label
    return model, None


def run_phase(args, out, names, validation_names, epochs, phase):
    seed_all(args.seed)
    model, initialization = make_model(args)
    device = torch.device(args.device)
    model.to(device)
    train_data = dataset(args, names, augment=args.augment)
    train_loader = loader(train_data, args.batch, args.workers, shuffle=True)
    val_loader = (loader(dataset(args, validation_names),
                         args.batch, args.workers) if validation_names else None)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.lr, weight_decay=args.weight_decay)
    # Preserve the selection-phase learning-rate trajectory during a shorter refit.
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr * .05)
    best_loss, best_epoch, stale = math.inf, 0, 0
    log_path = out / f'{phase}_log.jsonl'
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for step, batch in enumerate(train_loader, 1):
            b = batch_to(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(b['image'])
            weights = b['valid'].sum(1) / train_data.mean_raters if args.annotation_row_weighting else None
            loss = density_loss(logits, b['raters_norm'], b['valid'], args.target_sigma_grid, weights)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.)
            optimizer.step()
            losses.append((float(loss.detach()), len(logits)))
            if step == 1 or step % 20 == 0 or step == len(train_loader):
                print(json.dumps({'phase': phase, 'epoch': epoch, 'epochs': epochs, 'step': step,
                    'steps': len(train_loader), 'loss': losses[-1][0],
                    'elapsed_s': round(time.time() - start, 1)}), flush=True)
        metric = validate(model, val_loader, device, args.target_sigma_grid, args.annotation_row_weighting) if val_loader else None
        train_loss = sum(v * n for v, n in losses) / sum(n for _, n in losses)
        improved = metric is not None and metric['loss'] < best_loss
        if improved:
            best_loss, best_epoch, stale = metric['loss'], epoch, 0
        else:
            stale += 1
        record = {'phase': phase, 'epoch': epoch, 'train_loss': train_loss,
                  'training_only_validation': metric, 'best_epoch': best_epoch,
                  'lr': optimizer.param_groups[0]['lr'], 'elapsed_s': time.time() - start}
        with log_path.open('a') as f:
            f.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        payload = {'state_dict': model.state_dict(), 'config': vars(args), 'phase': phase,
                   'epoch': epoch, 'best_epoch': best_epoch, 'validation': metric,
                   'encoder_initialization': initialization, 'optimizer': optimizer.state_dict(),
                   'scheduler': schedule.state_dict(), 'source_backbone_sha256': digest(args.backbone_checkpoint)}
        if improved:
            torch.save(payload, out / 'selection_best.pt')
        # Save progress; the output directory is exclusive to this new experiment.
        torch.save(payload, out / f'{phase}_last.pt')
        schedule.step()
        if metric is not None and stale >= args.patience:
            break
    return best_epoch if val_loader else epoch


def train(args):
    if args.device != 'cuda' and not args.allow_non_cuda:
        raise ValueError('Real-data training requires CUDA; explicit --allow-non-cuda overrides this guard')
    root, out = Path(args.data_root).resolve(), Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a fresh output directory; existing results are protected')
    out.mkdir(parents=True, exist_ok=True)
    args.data_root = str(root)
    args.backbone_checkpoint = str(Path(args.backbone_checkpoint).resolve())
    if args.feature_cache:
        args.feature_cache = str(Path(args.feature_cache).resolve())
        if not args.freeze_encoder:
            raise ValueError('Frozen feature cache cannot support encoder fine-tuning')
        args.augment = False
        cache_manifest_hash = verify_feature_cache(args.feature_cache, digest(args.backbone_checkpoint))
    else:
        cache_manifest_hash = None
    names, fit, internal_val = training_partition(root, args.internal_validation, args.seed)
    inputs = [root / n for n in ['multi_image_list.txt', 'multi_train_list.txt', 'multi_val_list.txt']
              + [f'landmarks_{r}.txt' for r in 'abcde']]
    provenance = {'config': vars(args), 'source_hashes': {str(p): digest(p) for p in inputs},
                  'backbone_sha256': digest(args.backbone_checkpoint),
                  'selection_rule': 'minimum training-only held-out spatial cross-entropy',
                  'new_heatmap_head_checkpoint_selection_uses_training_partition_only': True,
                  'historical_source_encoder_selection_is_unchanged': True,
                  'fit_names': fit, 'internal_validation_names': internal_val,
                  'selection_fit_n': len(fit), 'selection_validation_n': len(internal_val),
                  'assessment_data_used_for_tuning': False, 'torch': torch.__version__, 'timm': timm.__version__,
                  'target': 'per-image average of per-rater normalized Gaussian heatmaps',
                  'training_image_weight': 'R_i / mean_R (matches annotation-row weighting)' if args.annotation_row_weighting else 'equal images',
                  'cached_features_disable_image_augmentation': bool(args.feature_cache),
                  'shared_cache_manifest_sha256': cache_manifest_hash,
                  'source_grid_wh': [617, 563], 'input_grid_wh': [args.image_size] * 2,
                  'heatmap_grid_centers': '(index+0.5)/heatmap_size; coordinates divide by source W/H',
                  'not_a_matched_token_ablation': True}
    write_json(out / 'protocol.json', provenance)
    with (out / 'training_split.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'partition'])
        writer.writerows((n, 'internal_validation' if n in set(internal_val) else 'fit') for n in names)
    best_epoch = run_phase(args, out, fit, internal_val, args.epochs, 'selection')
    if args.refit_all:
        run_phase(args, out, names, None, best_epoch, 'refit')
        final = out / 'refit_last.pt'
    else:
        final = out / 'selection_best.pt'
    write_json(out / 'training_complete.json', {'checkpoint': str(final), 'sha256': digest(final),
               'selected_epochs': best_epoch, 'final_fit_n': len(names) if args.refit_all else len(fit),
               'new_heatmap_head_evaluation_242_used_for_selection': False})


def evaluate(args):
    if args.device != 'cuda' and not args.allow_non_cuda:
        raise ValueError('Real-image inference requires CUDA; do not use this Mac for image inference')
    checkpoint = Path(args.checkpoint).resolve()
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = argparse.Namespace(**payload['config'])
    cfg.data_root = args.data_root
    if args.feature_cache:
        cfg.feature_cache = args.feature_cache
    cache_manifest_hash = (verify_feature_cache(cfg.feature_cache, payload['source_backbone_sha256'])
                           if getattr(cfg, 'feature_cache', None) else None)
    model, _ = make_model(cfg, initialize=False)
    model.load_state_dict(payload['state_dict'], strict=True)
    model.to(args.device).eval()
    root = Path(args.data_root).resolve()
    names = (root / 'multi_val_list.txt').read_text().splitlines()
    if len(names) != 242 or set(names) & set((root / 'multi_train_list.txt').read_text().splitlines()):
        raise ValueError('Evaluation must use exactly the original 242-image partition')
    batches = loader(dataset(cfg, names), args.batch, args.workers)
    values = {k: [] for k in ('pred_px', 'landmark_cov_px2', 'gt_px', 'rater_px', 'rater_valid', 'image_wh')}
    maps = []
    seen = []
    with torch.no_grad():
        for batch in batches:
            b = batch_to(batch, args.device)
            logits = model(b['image'])
            mu, cov, prob = moments(logits)
            wh = b['wh']
            values['pred_px'].append((mu.double() * wh[:, None]).cpu().numpy())
            values['landmark_cov_px2'].append((cov.double() * wh[:, None, :, None] * wh[:, None, None, :]).cpu().numpy())
            values['gt_px'].append(batch['gt_px'].numpy())
            values['rater_px'].append(batch['raters_px'].numpy())
            values['rater_valid'].append(batch['valid'].numpy())
            values['image_wh'].append(batch['wh'].numpy())
            if len(seen) < 8:
                maps.append(prob[:8-len(seen)].cpu().numpy().reshape(-1, 5, cfg.heatmap_size, cfg.heatmap_size))
            seen.extend(batch['name'])
            print(json.dumps({'evaluation_images_done': len(seen), 'total': 242}), flush=True)
    a = {k: np.concatenate(v) for k, v in values.items()}
    n = len(seen)
    full_cov = np.zeros((n, 10, 10), dtype=np.float64)
    samples, quantiles = [], []
    levels = np.array([.5, .6826894921370859, .8, .9, .95])
    for i, name in enumerate(seen):
        for k in range(5):
            full_cov[i, 2*k:2*k+2, 2*k:2*k+2] = a['landmark_cov_px2'][i, k]
        eig, vec = np.linalg.eigh(a['landmark_cov_px2'][i])
        root_cov = vec * np.sqrt(np.maximum(eig, 1.e-8))[:, None, :]
        seed = int(hashlib.sha256(f'heatmap-gaussian-draws-v1|{args.seed}|{name}'.encode()).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((args.draws, 5, 2))
        points = a['pred_px'][i] + np.einsum('skd,ked->ske', z, root_cov)
        angles = angle(points)
        samples.append(angles)
        quantiles.append(np.stack([np.quantile(angles, [(1-q)/2, (1+q)/2], axis=0) for q in levels]))
    samples, quantiles = np.stack(samples), np.stack(quantiles)
    pred_angles, gt_angles = angle(a['pred_px']), angle(a['gt_px'])
    err = np.linalg.norm(a['pred_px'] - a['gt_px'], axis=-1).mean(axis=-1)
    raw = quantiles[:, 3]
    rater_angles = angle(a['rater_px'])
    in_interval = ((rater_angles >= raw[:, None, 0]) & (rater_angles <= raw[:, None, 1]))
    rater_coverage = (in_interval * a['rater_valid'][..., None]).sum(1) / a['rater_valid'].sum(1)[:, None]
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a fresh evaluation output directory')
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'predictions.npz', img_names=np.array(seen), **a,
        cov_px2=full_cov, sigma_px=np.sqrt(np.diagonal(full_cov, axis1=1, axis2=2)).reshape(n,5,2),
        gt_angles_deg=gt_angles, pred_angles_deg=pred_angles, angle_samples_deg=samples,
        interval_levels=levels, mixture_angle_quantiles=quantiles,
        heatmaps_first8=np.concatenate(maps)[:8], heatmap_image_names=np.array(seen[:8]))
    summary = {'n_images': n, 'MRE_px': float(err.mean()),
               'angle_MAE_deg': np.abs(pred_angles - gt_angles).mean(0).tolist(),
               'raw_90_image_weighted_rater_coverage_pct': (rater_coverage.mean(0)*100).tolist(),
               'raw_90_mean_width_deg': (raw[:,1]-raw[:,0]).mean(0).tolist()}
    write_json(out / 'summary.json', summary)
    write_json(out / 'evaluation_protocol.json', {'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
        'data_root': str(root), 'train_list_sha256': digest(root/'multi_train_list.txt'),
        'shared_cache_manifest_sha256': cache_manifest_hash,
        'evaluation_list_sha256': digest(root/'multi_val_list.txt'), 'n_images': n,
        'sampling': 'independent landmarks, within-landmark anisotropic Gaussian fitted by heatmap moments',
        'cross_landmark_covariance': 'zero; block diagonal, not inferred from raters at test time',
        'coordinate_scale': 'normalized heatmap moments multiplied separately by original width 617 and height 563',
        'gaussian_draws_per_image': args.draws, 'interval_quantile_axis': ['image','level','lower_upper','alpha_beta'],
        'spatial_heatmap_target_sigma_grid': cfg.target_sigma_grid,
        'covariance_includes_fixed_target_kernel_smoothing': True,
        'point_reference': 'mean of available rater coordinates; angles evaluated after coordinate averaging',
        'no_calibration_performed_here': True, 'no_image_exclusions_or_anatomical_clipping': True,
        'not_a_matched_architecture_ablation': True,
        'no_original_242_images_used_by_this_training_for_checkpoint_selection': True})
    print(json.dumps(summary), flush=True)


def smoke(args):
    torch.set_num_threads(2)
    seed_all(20261031)
    size, hm, hidden = (448, 112, 128) if args.full_size else (64, 16, 32)
    model = SpatialHeatmap(size=size, heatmap_size=hm, hidden=hidden)
    x = torch.randn(1, 3, size, size)
    logits = model(x)
    points = torch.tensor([[[[.2,.3],[.3,.5],[.5,.6],[.6,.7],[.8,.5]],
                             [[.21,.29],[.31,.49],[.51,.61],[.61,.71],[.81,.51]]]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    target = target_density(points, valid, hm, 2.)
    assert torch.allclose(target.sum(-1), torch.ones(1,5))
    loss = density_loss(logits, points, valid, 2., valid.sum(1)/2.)
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.head.parameters() if p.grad is not None)
    mu, cov, prob = moments(logits)
    assert mu.shape == (1,5,2) and cov.shape == (1,5,2,2)
    assert (torch.linalg.eigvalsh(cov) >= -1.e-6).all()
    # Verify anisotropic covariance and nonzero xy term are actually preserved.
    grid = spatial_grid(16, torch.device('cpu'), torch.float32)
    delta = grid - torch.tensor([.5,.5])
    precision = torch.tensor([[120.,-70.],[-70.,120.]])
    synthetic_logits = (-.5 * torch.einsum('pd,de,pe->p',delta,precision,delta)).reshape(1,1,16,16).repeat(1,5,1,1)
    _, slanted, _ = moments(synthetic_logits)
    assert slanted[0,0,0,1] > 0
    print(json.dumps({'status':'PASS', 'synthetic_only':True, 'real_images_loaded':0,
         'loss':float(loss.detach()), 'logit_shape':list(logits.shape),
         'slanted_covariance':slanted[0,0].detach().tolist()}), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--data-root', required=True)
    t.add_argument('--backbone-checkpoint', required=True)
    t.add_argument('--feature-cache', help='Shared level0.npy..level3.npy and meta.npz; disables image augmentation')
    t.add_argument('--output', required=True)
    t.add_argument('--device', default='cuda')
    t.add_argument('--allow-non-cuda', action='store_true')
    t.add_argument('--image-size', type=int, default=448)
    t.add_argument('--heatmap-size', type=int, default=112)
    t.add_argument('--hidden', type=int, default=128)
    t.add_argument('--epochs', type=int, default=80)
    t.add_argument('--patience', type=int, default=15)
    t.add_argument('--batch', type=int, default=12)
    t.add_argument('--workers', type=int, default=4)
    t.add_argument('--lr', type=float, default=2e-4)
    t.add_argument('--weight-decay', type=float, default=.01)
    t.add_argument('--target-sigma-grid', type=float, default=2.)
    t.add_argument('--internal-validation', type=float, default=.1)
    t.add_argument('--seed', type=int, default=20261031)
    t.add_argument('--freeze-encoder', action=argparse.BooleanOptionalAction, default=True)
    t.add_argument('--augment', action=argparse.BooleanOptionalAction, default=True)
    t.add_argument('--refit-all', action=argparse.BooleanOptionalAction, default=True)
    t.add_argument('--annotation-row-weighting', action=argparse.BooleanOptionalAction, default=True)
    e = sub.add_parser('eval')
    e.add_argument('--data-root', required=True)
    e.add_argument('--checkpoint', required=True)
    e.add_argument('--feature-cache', help='Override cached feature location recorded in checkpoint')
    e.add_argument('--output', required=True)
    e.add_argument('--device', default='cuda')
    e.add_argument('--allow-non-cuda', action='store_true')
    e.add_argument('--batch', type=int, default=12)
    e.add_argument('--workers', type=int, default=4)
    e.add_argument('--draws', type=int, default=4096)
    e.add_argument('--seed', type=int, default=20261031)
    s = sub.add_parser('smoke')
    s.add_argument('--full-size', action='store_true')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if args.command == 'train': train(args)
    elif args.command == 'eval': evaluate(args)
    else: smoke(args)
