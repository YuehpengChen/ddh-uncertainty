# Browser model package

The public package contains ONNX inference graphs only, with SHA-256 checksums.
The raw checkpoints, optimizer states, training data, and image annotations stay
in the private build workflow. No model inference runs on the author's Mac.

The frozen Swin backbone runs once per selected image. The second graph uses
the original Stage 2 predictor, learned codebook, and original mean/variance
decoder. Its Boolean `stochastic` input controls genuine ONNX Dropout at all 26
archived sites, including attention probability dropout and channel-wise
modulator dropout. No synthetic landmark generator or substitute decoder is used.

All six BatchNorm modules use the saved training-only recalibrated statistics.
The graph never adapts BatchNorm to a visitor's image. Conditional variances are
clamped and exponentiated exactly as in the evaluated implementation. Results
must be restored to the original image width/height before measuring angles.

Release gates are strict weight loading, native deterministic parity, ONNX
deterministic parity, at least 50 finite and nonidentical stochastic passes, and
an actual ONNX Runtime Web WASM test on a GitHub-hosted runner using only a
synthetic color gradient. `manifest.status` remains pending until that test passes.
