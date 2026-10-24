"""Export and verify on a GitHub-hosted runner, never on the author's computer.

Example (GitHub Actions only):
  python cloud/export_models.py --stage1 inputs/stage1.pt --stage2 inputs/stage2.pt \
    --bn inputs/bn_statistics.json --output public/models
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

STAGE1_SHA = "7c102bdd725accd4edb3b6560537fd64f2aa3d31bd4bcc7cbb50bd686c8c8624"
STAGE2_SHA = "11ce7c399e9dc7d035d1d4975f963d1cfc6833e2b6282af90a87a3057652451a"
BN_SHA = "4a2cdd136cb77d97a860ece706d64dfae62464bb59d0259e3f267dadcf0d0901"


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def assert_cloud():
    if os.environ.get("GITHUB_ACTIONS") != "true" or not os.environ.get("GITHUB_RUN_ID"):
        raise SystemExit("Refused: all model loading, export, and inference must run in GitHub Actions cloud.")


def main():
    assert_cloud()
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", required=True, type=Path)
    parser.add_argument("--stage2", required=True, type=Path)
    parser.add_argument("--bn", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preprocessing-report", type=Path, default=Path("validation/resize_pillow_validation.json"))
    parser.add_argument("--passes", type=int, default=50)
    args = parser.parse_args()
    if args.passes < 50:
        parser.error("Release verification requires at least 50 genuine stochastic passes.")
    for path, expected in ((args.stage1, STAGE1_SHA), (args.stage2, STAGE2_SHA), (args.bn, BN_SHA)):
        actual = checksum(path)
        if actual != expected:
            raise RuntimeError(f"Source checksum mismatch for {path.name}: {actual}")
    preprocessing = json.loads(args.preprocessing_report.read_text())
    resize_implementation = Path(__file__).resolve().parents[1] / "validation/resize_pillow.mjs"
    if (preprocessing.get("status") != "passed" or preprocessing.get("patientDataUsed") is not False
            or preprocessing.get("caseCount", 0) < 13 or not preprocessing.get("exactRGB8Parity")
            or preprocessing.get("maxNormalizationError") != 0
            or preprocessing.get("implementationSHA256") != checksum(resize_implementation)):
        raise RuntimeError("Current Pillow-compatible preprocessing has not passed its exact synthetic proof")

    # These heavyweight imports deliberately occur only after the cloud guard.
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from torch import nn
    import timm
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model.architecture import FrozenBackbone, FrozenDecoder, Stage2Predictor, PortableHead, dropout_mask_specs

    torch.set_num_threads(min(4, os.cpu_count() or 2))
    torch.set_num_interop_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    timm.layers.set_fused_attn(False)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "preprocessing_validation.json", preprocessing)
    torch.manual_seed(20261024)

    print("Loading checksummed checkpoints on GitHub runner CPU", flush=True)
    checkpoint1 = torch.load(args.stage1, map_location="cpu", weights_only=True)
    checkpoint2 = torch.load(args.stage2, map_location="cpu", weights_only=True)
    cfg1, cfg2 = dict(checkpoint1["cfg"]), dict(checkpoint2["cfg"])
    if cfg1["img_size"] != 448 or cfg1["dropout"] != 0.0:
        raise RuntimeError("Unexpected archived backbone/decoder configuration")
    backbone = FrozenBackbone(cfg1)
    backbone_state = {k[len("guide_backbone."):]: v for k, v in checkpoint1["state_dict"].items()
                      if k.startswith("guide_backbone.")}
    backbone.load_state_dict(backbone_state, strict=True)
    decoder = FrozenDecoder(cfg1)
    decoder_state = {
        key: checkpoint1["state_dict"]["tokenizer.codebook.codebook" if key == "embedding" else "tokenizer." + key]
        for key in decoder.state_dict()
    }
    decoder.load_state_dict(decoder_state, strict=True)
    predictor = Stage2Predictor(cfg2, dropout=0.3)
    predictor.load_state_dict(checkpoint2["state_dict"], strict=True)
    bn = json.loads(args.bn.read_text())
    modules = dict(predictor.named_modules())
    actual_bn_names = {name for name, module in modules.items() if isinstance(module, nn.BatchNorm2d)}
    if set(bn) != actual_bn_names:
        raise RuntimeError("BN names differ from evaluated train-only statistics")
    with torch.no_grad():
        for name, values in bn.items():
            module = modules[name]
            module.running_mean.copy_(torch.tensor(values["running_mean"]))
            module.running_var.copy_(torch.tensor(values["running_var"]))
            module.num_batches_tracked.fill_(values["num_batches_tracked"])
            module.eps = values["eps"]
    del checkpoint1, checkpoint2, backbone_state, decoder_state
    backbone.requires_grad_(False).eval()
    predictor.requires_grad_(False).eval()
    decoder.requires_grad_(False).eval()
    head = PortableHead(predictor, decoder, cfg1).requires_grad_(False).eval()
    off, on = torch.tensor(False), torch.tensor(True)
    mask_specs = dropout_mask_specs(cfg2, cfg1["img_size"])
    mask_names = [spec["name"] for spec in mask_specs]
    deterministic_masks = tuple(torch.ones(spec["shape"], dtype=torch.uint8) for spec in mask_specs)
    if len(mask_specs) != 26:
        raise RuntimeError("Expected all 26 source dropout sites")

    # A synthetic color-gradient tensor. No patient image or landmark is loaded.
    grid = torch.linspace(0, 1, 448)
    xx, yy = torch.meshgrid(grid, grid, indexing="xy")
    rgb = torch.stack([xx, yy, (xx + yy) / 2])[None]
    image = (rgb - torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]) / torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    with torch.inference_mode():
        features = backbone(image)
        native_mean, native_logvar = decoder(predictor(features))
        native_variance = torch.exp(native_logvar.clamp(2 * cfg1["log_sigma_min"], 2 * cfg1["log_sigma_max"]))
        portable_mean, portable_variance = head(*features, off, *deterministic_masks)
        mean_error = float((portable_mean - native_mean).abs().max())
        variance_error = float((portable_variance - native_variance).abs().max())
        torch.testing.assert_close(portable_mean, native_mean, rtol=1e-4, atol=2e-5)
        torch.testing.assert_close(portable_variance, native_variance, rtol=1e-4, atol=1e-7)

    feature_names = [f"feature{i}" for i in range(4)]
    print("Exporting frozen backbone", flush=True)
    torch.onnx.export(backbone, (image,), str(args.output / "backbone.onnx"),
        input_names=["image"], output_names=feature_names, opset_version=17,
        do_constant_folding=True, dynamo=False)
    print("Exporting runtime-controlled stochastic head", flush=True)
    torch.onnx.export(head, (*features, off, *deterministic_masks), str(args.output / "stage2_decoder.onnx"),
        input_names=[*feature_names, "stochastic", *mask_names], output_names=["mean", "variance"],
        opset_version=17, do_constant_folding=False, dynamo=False)
    for name in ("backbone.onnx", "stage2_decoder.onnx"):
        onnx.checker.check_model(str(args.output / name), full_check=True)
    graph = onnx.load(args.output / "stage2_decoder.onnx")
    graph_inputs = {value.name for value in graph.graph.input}
    if not set(mask_names).issubset(graph_inputs):
        raise RuntimeError("An explicit dropout-mask input was removed during export")
    if any(node.op_type == "Dropout" for node in graph.graph.node):
        raise RuntimeError("Native ONNX Dropout must not be used for the stochastic law")

    options = ort.SessionOptions()
    options.intra_op_num_threads = min(4, os.cpu_count() or 2)
    options.inter_op_num_threads = 1
    encoder = ort.InferenceSession(str(args.output / "backbone.onnx"), options, providers=["CPUExecutionProvider"])
    stochastic_head = ort.InferenceSession(str(args.output / "stage2_decoder.onnx"), options, providers=["CPUExecutionProvider"])
    ort_features = encoder.run(None, {"image": image.numpy()})
    feature_errors = []
    for expected, actual in zip(features, ort_features):
        np.testing.assert_allclose(actual, expected.numpy(), rtol=2e-3, atol=2e-4)
        feature_errors.append(float(np.max(np.abs(actual - expected.numpy()))))
    feeds = dict(zip(feature_names, ort_features))
    feeds["stochastic"] = np.array(False, dtype=np.bool_)
    feeds.update({spec["name"]: np.ones(spec["shape"], dtype=np.uint8) for spec in mask_specs})
    ort_mean, ort_variance = stochastic_head.run(None, feeds)
    np.testing.assert_allclose(ort_mean, native_mean.numpy(), rtol=2e-3, atol=3e-5)
    np.testing.assert_allclose(ort_variance, native_variance.numpy(), rtol=2e-3, atol=2e-7)
    repeat = stochastic_head.run(None, feeds)
    for one, two in zip((ort_mean, ort_variance), repeat):
        np.testing.assert_array_equal(one, two)

    feeds["stochastic"] = np.array(True, dtype=np.bool_)
    samples, variance_samples = [], []
    rng = np.random.default_rng(20261024)
    fixed_mask_parity = []
    fixed_mask_reference = None
    started = time.monotonic()
    print(f"Checking {args.passes} genuine ORT CPU dropout passes", flush=True)
    for index in range(args.passes):
        for spec in mask_specs:
            feeds[spec["name"]] = (rng.random(spec["shape"]) >= spec["probability"]).astype(np.uint8)
        mu, var = stochastic_head.run(None, feeds)
        if index < 3:
            # Same exact independent masks must agree across PyTorch and ONNX.
            with torch.inference_mode():
                expected_mu, expected_var = head(*features, on,
                    *(torch.from_numpy(feeds[name]) for name in mask_names))
            np.testing.assert_allclose(mu, expected_mu.numpy(), rtol=2e-3, atol=5e-5)
            np.testing.assert_allclose(var, expected_var.numpy(), rtol=2e-3, atol=3e-7)
            fixed_mask_parity.append(float(np.max(np.abs(mu - expected_mu.numpy()))))
            if index == 0:
                np.concatenate([feeds[name].reshape(-1) for name in mask_names]).tofile(args.output / "synthetic_masks.u8")
                fixed_mask_reference = {
                    "mean": expected_mu.numpy().flatten().tolist(),
                    "variance": expected_var.numpy().flatten().tolist(),
                }
        if not np.isfinite(mu).all() or not np.isfinite(var).all() or not (var > 0).all():
            raise RuntimeError("Nonfinite model output or nonpositive conditional variance")
        samples.append(mu)
        variance_samples.append(var)
        if (index + 1) % 10 == 0:
            print(f"Verified CPU stochastic passes {index + 1}/{args.passes}", flush=True)
    means = np.stack(samples)
    spread = np.var(means.astype(np.float64), axis=0)
    if not np.any(spread > 1e-12) or np.unique(means.reshape(args.passes, -1), axis=0).shape[0] < args.passes // 2:
        raise RuntimeError("Dropout was silently disabled or repeated")
    feeds["stochastic"] = np.array(False, dtype=np.bool_)
    after_mc = stochastic_head.run(None, feeds)
    for before, after in zip((ort_mean, ort_variance), after_mc):
        np.testing.assert_array_equal(before, after)
    # Small synthetic fixtures allow independent WASM tests without patient data.
    image.numpy().astype("<f4").tofile(args.output / "synthetic_input.f32")
    write_json(args.output / "synthetic_expected.json", {
        "synthetic": True, "shape": [1, 3, 448, 448],
        "mean": native_mean.numpy().flatten().tolist(),
        "variance": native_variance.numpy().flatten().tolist(),
        "fixedMask": fixed_mask_reference,
        "featureMaxAbsErrors": feature_errors,
    })
    proof = {
        "execution": "GitHub Actions hosted Linux CPU only", "patientDataUsed": False,
        "runId": os.environ["GITHUB_RUN_ID"], "commit": os.environ.get("GITHUB_SHA"),
        "torchVersion": torch.__version__, "timmVersion": timm.__version__, "ortVersion": ort.__version__,
        "sourceCheckpoints": {"stage1": STAGE1_SHA, "stage2": STAGE2_SHA, "trainOnlyBN": BN_SHA},
        "strictWeightLoading": True, "frozenBatchNorm": True, "dropoutSites": len(mask_specs),
        "dropoutMechanism": "explicit independent uint8 Bernoulli mask inputs; inverse keep-probability scaling",
        "deterministicNativeParity": {"meanMaxAbsError": mean_error, "varianceMaxAbsError": variance_error},
        "deterministicORTParity": {"meanMaxAbsError": float(np.max(np.abs(ort_mean - native_mean.numpy()))),
                                   "varianceMaxAbsError": float(np.max(np.abs(ort_variance - native_variance.numpy())))},
        "stochasticCPU": {"passes": args.passes, "finite": True,
            "maxCoordinateVariance": float(spread.max()), "frozenStateAfterMC": True,
            "fixedMaskNativeParityMaxAbsErrors": fixed_mask_parity,
            "maskBytesPerPass": sum(int(np.prod(spec["shape"])) for spec in mask_specs),
            "seconds": time.monotonic() - started},
        "browserWASM": {"status": "pending"},
    }
    write_json(args.output / "cloud_validation.json", proof)
    models = {}
    for key, filename in (("backbone", "backbone.onnx"), ("head", "stage2_decoder.onnx")):
        path = args.output / filename
        models[key] = {"url": "models/" + filename, "sha256": checksum(path), "bytes": path.stat().st_size}
    manifest = {
        "schemaVersion": 1, "status": "pending_browser_verification", "modelName": "DDH five-landmark research model",
        "models": models,
        "input": {"name": "image", "shape": [1, 3, 448, 448], "layout": "NCHW", "dtype": "float32",
            "color": "RGB", "resize": "bilinear stretch to 448x448; preserve original width and height for restoration",
            "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "scale": 255}},
        "features": [{"name": name, "shape": list(value.shape)} for name, value in zip(feature_names, features)],
        "outputs": {"mean": {"name": "mean", "shape": [1, 5, 2], "coordinates": "normalized"},
            "variance": {"name": "variance", "shape": [1, 5, 2], "parameterization": "variance", "coordinates": "normalized_squared"}},
        "dropout": {"stochastic": True, "mechanism": "explicit_masks", "probability": 0.3,
            "control": {"name": "stochastic", "dtype": "bool", "shape": []},
            "inputs": mask_specs, "sites": 26,
            "includes": ["channel-wise modulator dropout", "self-attention probability dropout", "cross-attention probability dropout", "decoder residual and feed-forward dropout"],
            "defaultPasses": 50, "decoderFrozen": True, "backboneFrozen": True},
        "runtime": {"module": "runtime/ort.min.mjs", "provider": "wasm", "threads": 1},
        "preprocessingNote": "Decoded opaque RGB8 is resized with the JavaScript Pillow-compatible bilinear implementation. Thirteen synthetic RGB8 cases and float32 normalization passed exact comparison with Pillow 11.3.0. Browser image decoding, ICC/color profiles, and EXIF orientation are outside that proof.",
        "preprocessingValidation": {"status": "passed", "caseCount": preprocessing["caseCount"],
            "exactRGB8Parity": True, "maxNormalizationError": 0,
            "implementationSHA256": preprocessing["implementationSHA256"],
            "report": "models/preprocessing_validation.json"},
        "researchOnly": True, "clinicalUse": False,
        "limitations": ["Internal validation model, not an approved medical device.",
            "Intervals describe the model distribution; they do not establish diagnostic accuracy for a new image.",
            "Stage2 dropout 0.3 is reconstructed from archived source because checkpoint configuration omitted it.",
            "Automatic image plane and landmark plausibility verification is not established."],
        "sourceCheckpoints": proof["sourceCheckpoints"],
    }
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({"exported": True, "browserVerificationPending": True, "models": models}), flush=True)


if __name__ == "__main__":
    main()
