# Technical note: landmark covariance and Graf-angle uncertainty

**Source manuscript derivations (author-review version).** This note preserves the mathematical detail from Sections S14–S15 of the frozen full supplement. It is not a substitute for essential equations and definitions in the submitted main manuscript. Equation, section and figure identifiers below refer to that frozen source; a later compact main manuscript may renumber them.

The export converts native Word mathematics to LaTeX while retaining hats, bars, tildes, subscripts, superscripts, matrix delimiters and square roots. The source Word documents remain the authoritative author record.

## S14. Auxiliary Derivations for Coordinate and Angle Uncertainty

The auxiliary derivations explain how each model stage operates and how coordinate variation enters the angle calculation. Image index $i$ identifies one ultrasound image; rater index $r$ identifies an available annotation, and ${R}_{i}$ is its rater count. Landmark index $k$ runs from 1 to 5, coordinate index $j$ runs from 1 to 10, token position $m$ runs from 1 to $M=12$, and code index $c$ runs from 1 to $V=256$. The ten-coordinate ordering is ${\left({x}_{1},{y}_{1},\ldots ,{x}_{5},{y}_{5}\right)}^{\mathsf{T}}$. A tilde denotes width/height-normalized coordinates, and a hat denotes a prediction or empirical estimate. A bar denotes the explicitly defined average or reference; barred training angles are calculated from rater-mean coordinates. A superscript $T$ denotes transpose; the stand-alone $T=50$ is the number of dropout passes. Identity matrices ${I}_{d}$ preserve a $d$-dimensional vector; $\otimes $ forms a Kronecker product, and $\operatorname{diag}$ constructs a diagonal matrix. The norms $\Vert \cdot {\Vert }_{2}$ and $\Vert \cdot {\Vert }_{2}^{2}$ are Euclidean length and squared length; $\operatorname{tr}$ sums a matrix's diagonal; $1\left[\cdot \right]$ equals one when its condition holds and zero otherwise. Natural $\log $ and $\exp $ are inverse operations, and $\operatorname{clip}\left(z,a,b\right)$ restricts $z$ to the closed range $\left[a,b\right]$. The bracketed operator $E\left[\cdot \right]$ denotes expectation. Symbols reused locally have distinct roles: $E$ is the displacement-reproduction metric in (S1)–(S2), the epoch budget in (S21), or the explicitly defined edge set in Eq. (S15); $A$ is a scalar expected squared residual in (S7) and a direction-difference matrix in the explanation of Fig. 1D.

Main Eq. (1) defines the coordinate system used by all subsequent losses and measurements. The pixel vector ${y}_{ir}\in {\mathbb{R}}^{10}$ contains one rater's five landmark positions. Image width ${W}_{i}$ and height ${H}_{i}$ define ${S}_{i}={I}_{5}\otimes \operatorname{diag}\left({W}_{i},{H}_{i}\right)\in {\mathbb{R}}^{10\times 10}$: its five repeated two-axis blocks scale every $x$ and $y$ coordinate separately. Multiplication by ${S}_{i}^{-1}$ divides pixel coordinates by width and height, yielding dimensionless ${\tilde{y}}_{ir}$. The reference ${\bar{y}}_{i}$ averages the available pixel-coordinate vectors using divisor ${R}_{i}$; it is an arithmetic rater mean, not established anatomical truth. Coordinate normalization is reversed before pixel errors and angle geometry are evaluated. The letters $x$ and $y$ used as coordinate axes are distinct from the image input $x$ in the conditional predictive density.

Main Eq. (2) converts the Stage 2 predictor output to decoder input. For image $i$, token position $m$ and code index $c$, ${h}_{imc}$ is a real-valued logit and ${\pi }_{imc}$ is its softmax probability. Exponentiation makes weights positive, and division by the sum over all $V$ codes makes their probabilities sum to one at each position. The temperature is one. The weighted sum ${\bar{z}}_{im}={\sum }_{c}{\pi }_{imc}{e}_{c}$ combines the fixed $d=64$ code vectors ${e}_{c}$ into a token of the same dimension; it does not select one discrete code. The transferred decoder $\operatorname{Dec}$ maps all $M$ weighted tokens to the normalized mean ${\tilde{\mu }}_{i}\in {\mathbb{R}}^{10}$ and diagonal covariance ${\tilde{D}}_{i}\in {\mathbb{R}}^{10\times 10}$. The code index ${a}_{im}^{\left(r\right)}$ in Eq. (S17) is the per-rater target, ${t}_{imc}^{\left(r\right)}$ is the label-smoothed categorical target and ${\varepsilon }_{LS}$ controls the uniform smoothing mass. Token cross-entropy averages over $M$ positions and sums over $V$ categories; its logarithm scores the probability assigned to the reference code. Weighted-embedding decoding and averaging outputs decoded from individual codes are distinct nonlinear operations (Section S10).

Main Eqs. (3) and (4) define the image-conditioned predictive distribution and its moments. The random vector $Y$ contains all ten landmark coordinates, $N\left( \cdot  ;{\mu }_{t},{D}_{t}\right)$ is the Gaussian distribution from dropout pass $t$, and ${q}_{T}$ is their equally weighted finite mixture. Its mean $\bar{\mu }$ is the average of the pass means. The diagonal matrix ${D}_{t}$ describes coordinate variance predicted within that pass; ${\Sigma }_{A}$ averages these matrices, while ${\Sigma }_{E}$ averages the outer products $\left({\mu }_{t}-\bar{\mu }\right){\left({\mu }_{t}-\bar{\mu }\right)}^{\mathsf{T}}$ across passes. The outer product records which coordinates move together. Their sum $\Sigma $ is the full $10\times 10$ predictive covariance, whose diagonal contains coordinate variances and whose off-diagonal entries contain cross-coordinate covariances. Division by $T$ yields the moments of the retained mixture rather than an unbiased covariance estimator for an unknown population of passes. Both components arise from this prediction process; the model omits within-pass coordinate correlations and parameter variation in fixed modules.

Main Eqs. (5)–(9) map coordinate distributions to angle variation. The vectors $u={P}_{2}-{P}_{1}$, $v={P}_{3}-{P}_{4}$ and $w={P}_{5}-{P}_{4}$ define the measurement lines. For nonzero two-dimensional vectors $a$ and $b$, the cross-product magnitude $|{a}_{x}{b}_{y}-{a}_{y}{b}_{x}|$ and dot product ${a}^{\mathsf{T}}b$ enter $\operatorname{atan2}$ to produce an unsigned angle; $\kappa =180/\pi $ converts radians to degrees. The map $f:{\mathbb{R}}^{10}\to {\mathbb{R}}^{2}$ returns alpha and beta. Its Jacobian $J\in {\mathbb{R}}^{2\times 10}$ records the local change in each angle per pixel change in each coordinate. Main Eq. (7) propagates a covariance by $J\Sigma  {J}^{\mathsf{T}}$; the two diagonal entries are angular variances in squared degrees, and their square roots are SDs in degrees. Main Eq. (8) retains the same $J$, means and diagonal variances while removing cross-coordinate products, so the sign of the change depends on both the Jacobian coefficients and the covariances. Main Eq. (9) instead draws a complete coordinate configuration, using one uniformly selected component index ${t}_{l}$ and one standard-normal vector ${\varepsilon }_{l}\in {\mathbb{R}}^{10}$ for draw $l$. The matrix square root ${D}_{{t}_{l}}^{1/2}$ scales its unit-variance noise to the component's coordinate SDs before applying $f$. The derivative method is local, while the sampled method evaluates the nonlinear map for each draw.

Main Eqs. (11) and (12) evaluate uncertainty at the image level. For angle label $a\in \{\alpha ,\beta \}$ and nominal level $\gamma $, ${I}_{ia,\gamma }$ is the interval for image $i$, with lower and upper endpoints ${L}_{ia,\gamma }$ and ${U}_{ia,\gamma }$. The empirical rater coverage ${\hat{C}}_{a,\gamma }^{rater}$ first averages reference-inclusion indicators over the ${R}_{i}$ raters of each image, then averages over $N$ images. The mean-coordinate coverage ${\hat{C}}_{a,\gamma }^{mean}$ uses one reference angle ${f}_{a}\left({\bar{y}}_{i}\right)$ per image. The width ${\hat{W}}_{a,\gamma }$ averages endpoint differences and is measured in degrees. Coverage is dimensionless and is reported as a percentage; $\gamma $ is nominal probability content, not an observed coverage guarantee. In the landmark-spread score ${s}_{i}^{rad}$, ${\Sigma }_{i,kk}$ is the $2\times 2$ covariance block for landmark $k$. Its trace sums the two axis variances, its square root gives a radius in pixels, and the outer factor $1/5$ averages five such radii. This score summarizes marginal positional dispersion; it does not specify cross-landmark dependence or replace the angle-specific propagation.

### A. Optimal Variance at a Fixed Mean

Eq. (S14) omits the Gaussian factor $1/2$, making it twice the mean per-coordinate Gaussian NLL up to an additive constant and setting its relative loss weight. To isolate this term, let $Y$ be one normalized random target coordinate, hold its predicted mean $\mu $ fixed and define $A=E\left[{\left(Y-\mu \right)}^{2}\right]$. Both $A$ and $v$ have squared normalized-coordinate units. The expected scalar loss as a function of positive variance $v$ is $A/v+\log  v$. For $A>0$, differentiating with respect to $v$ and setting the derivative to zero gives

$$
\frac{\partial }{\partial v}\left(\frac{A}{v}+logv\right)=\frac{v-A}{{v}^{2}},
$$

$$
{v}^{*}=A=Var(Y)+(E[Y]-\mu {)}^{2}.\tag{S7}
$$

For $A>0$, the derivative in (S7) is negative for $v<A$ and positive for $v>A$, so ${v}^{*}=A$ minimizes the isolated scalar loss. The decomposition into $\operatorname{Var}\left(Y\right)$ and ${\left(E\left[Y\right]-\mu \right)}^{2}$ separates target-coordinate variance from squared mean-prediction bias. With the allowed range, the optimum is ${v}_{\operatorname{clip}}^{*}=\operatorname{clip}\left(A,{e}^{-14},{e}^{4}\right)$, including $A=0$. Without clipping, $A=0$ has no positive finite minimizer; the log term decreases without bound as $v\to {0}^{+}$. This is a fixed-mean scalar derivation, excluding the edge penalty and shared-network constraints. Learned variance can therefore reflect both rater differences and mean error; it does not identify physical image-noise variance.

Uniform input sampling gives every input annotation of an image the same possible reconstruction targets. If means and variances are freely optimized using only Eq. (S14), every input has the same optimum: the coordinate-wise rater mean and mean squared deviation from that mean, with variance clipped to the allowed range. This statement applies to the reconstruction objective under the stated target distribution, not to unrestricted behavior of the trained system. Image-feature sampling, quantization, shared parameters and commitment and edge losses also affect the learned outputs. Main Fig. 2c illustrates the input-sampling rule; Section S13 reports its effects on reconstruction and input dependence.

### B. Coordinate Transformations and Angle Derivatives

Equation (S8) restores decoder outputs to the coordinate system used for errors and angles. Each of the $T=50$ dropout passes predicts all five landmarks together after the image is resized to $448\times 448$. The decoder means ${\tilde{\mu }}_{it}\in {\mathbb{R}}^{10}$ and covariances ${\tilde{D}}_{it}\in {\mathbb{R}}^{10\times 10}$ use normalized axes. The scale matrix ${S}_{i}$ from main Eq. (1) transforms the mean by one matrix multiplication and the covariance by multiplication on both sides:

$$
{\mu }_{it}={S}_{i}{\tilde{\mu }}_{it},\quad \quad {D}_{it}={S}_{i}{\tilde{D}}_{it}{S}_{i}^{\mathsf{T}}.
$$

$$
Cov({S}_{i}Y+{b}_{i})={S}_{i}Cov(Y){S}_{i}^{\mathsf{T}}.\tag{S8}
$$

In (S8), $Y$ is a coordinate random vector and ${b}_{i}\in {\mathbb{R}}^{10}$ is a fixed offset, distinct from decoder-path label $b$ and landmark weight ${b}_{k}$. The affine transform changes the mean by ${S}_{i}E\left[Y\right]+{b}_{i}$ and covariance by ${S}_{i}\operatorname{Cov}\left(Y\right){S}_{i}^{\mathsf{T}}$; a fixed offset contributes no variance. Normalized $x$ and $y$ axes scale by ${W}_{i}$ and ${H}_{i}$, so their variances scale by the squared factors and mixed covariance entries by the product of their axis factors. The ratios ${W}_{i}/448$ and ${H}_{i}/448$ instead apply when starting from pixel coordinates on the resized grid. All 242 evaluation images are $617\times 563$ pixels. Errors and angles use these original-image coordinates. Physical pixel spacing (mm/pixel) was not retained in the exported image files, so all distance metrics are reported in pixel units.

Equation (S9) gives the local sensitivity of the unsigned angle map in main Eq. (5). For nonzero segment $a={\left({a}_{x},{a}_{y}\right)}^{\mathsf{T}}$, the row ${g}_{a}=\left(-{a}_{y},{a}_{x}\right)/\Vert a{\Vert }_{2}^{2}$ is the derivative of its polar direction with respect to its two vector components. Here $a$ denotes a segment vector, not the angle label used for interval evaluation. Thus ${g}_{a} da$ is a first-order direction change for a small segment perturbation $da$. The angle differential subtracts the direction change of the first segment from that of the second, multiplies by the branch sign ${s}_{ab}$ and converts radians to degrees with $\kappa =180/\pi $:

$$
d\theta (a,b)=\kappa {s}_{ab}(-{g}_{a}\,da+{g}_{b}\,db),
$$

$$
{s}_{ab}=sgn({a}_{x}{b}_{y}-{a}_{y}{b}_{x}).\tag{S9}
$$

The determinant ${a}_{x}{b}_{y}-{a}_{y}{b}_{x}$ identifies the local orientation of the two vectors, and $\operatorname{sgn}$ returns its sign. Equations (S9) and main (6) apply away from zero-length segments and the 0/180-degree branch changes, where this unsigned-angle derivative is not defined by one smooth branch. The segment perturbations are differences of endpoint perturbations: $du=d{P}_{2}-d{P}_{1}$, $dv=d{P}_{3}-d{P}_{4}$ and $dw=d{P}_{5}-d{P}_{4}$. Substituting them into (S9) gives the five two-column blocks of each Jacobian row in main (6), ordered by ${P}_{1}$ through ${P}_{5}$. The shared point ${P}_{4}$ therefore contributes to both alpha and beta sensitivity. Each entry of the degree-valued $J$ has units of degrees per pixel.

Fig. 1D expresses the same local propagation through segment directions. Define $\phi ={\left({\phi }_{12},{\phi }_{43},{\phi }_{45}\right)}^{\mathsf{T}}$, where ${\phi }_{ij}$ is the polar direction of ${P}_{j}-{P}_{i}$ in the coordinate system of the evaluation map. Its Jacobian $G=\partial \phi /\partial  y\in {\mathbb{R}}^{3\times 10}$ maps coordinate changes in pixels to direction changes in radians. On the branch illustrated in the figure, alpha is ${\phi }_{43}-{\phi }_{12}$ and beta is ${\phi }_{12}-{\phi }_{45}$, so the first row of $A\in {\mathbb{R}}^{2\times 3}$ is $\left(-1,1,0\right)$ and the second is $\left(1,0,-1\right)$. Multiplication by $A$ maps direction changes to angle changes. The figure uses radians and the displayed branch signs; the general unsigned degree-valued Jacobian additionally uses the orientation signs in main (6) and the factor $\kappa $. No training-angle direction convention is substituted into this evaluation calculation.

With coordinate covariance ${\Sigma }_{y}=\Sigma \in {\mathbb{R}}^{10\times 10}$, the direction covariance is locally ${\Sigma }_{\phi }\approx  G{\Sigma }_{y}{G}^{\mathsf{T}}\in {\mathbb{R}}^{3\times 3}$ and the illustrated angle covariance is ${\Sigma }_{\theta }\approx  A{\Sigma }_{\phi } {A}^{\mathsf{T}}\in {\mathbb{R}}^{2\times 2}$. These covariance units are squared pixels, squared radians and squared radians, respectively. The transpose operations combine covariance contributions from all pairs of input coordinates or directions. The beta identity shown in Fig. 1D adds the two direction variances and subtracts twice their covariance because beta is their difference. Multiplication by ${\kappa }^{2}$ converts a radian-squared angle covariance to degree-squared units. Main Eq. (7) performs the equivalent coordinate-to-angle calculation directly with $J$, and main Eq. (8) separates its diagonal and cross-coordinate contributions. The approximation concerns local linearization and does not establish interval coverage.

Main Eqs. (7) and (8) distinguish marginal coordinate variance from coordinate dependence. Setting off-diagonal covariance entries to zero removes the covariance terms between different landmarks and between the $x$ and $y$ axes of each landmark; its effect on angle variance can have either sign. Equation (S10) tests one exact cancellation using shared translation. The matrix $B={1}_{5}\otimes  {I}_{2}\in {\mathbb{R}}^{10\times 2}$ repeats the same displacement at all five points, where ${1}_{5}$ is a five-entry column of ones and ${I}_{2}$ is the two-dimensional identity. The random displacement $\delta \in {\mathbb{R}}^{2}$ has covariance $\Omega \in {\mathbb{R}}^{2\times 2}$; this vector differs from the scalar Smooth-L1 transition $\delta =2$ pixels in (S18):

$$
f(y+B\delta )=f(y),\quad \quad JB=0,
$$

$$
{\Sigma }_{\operatorname{tr}}=B\Omega {B}^{\mathsf{T}}\,\implies \,J{\Sigma }_{\operatorname{tr}}{J}^{\mathsf{T}}=0.\tag{S10}
$$

In (S10), $f\left(y+B\delta \right)=f\left(y\right)$ states exact translation invariance of the nondegenerate angle geometry. Differentiating this invariance gives $JB=0$. Shared displacement produces coordinate covariance ${\Sigma }_{\operatorname{tr}}=B\Omega  {B}^{\mathsf{T}}$, so its locally propagated contribution $J{\Sigma }_{\operatorname{tr}}{J}^{\mathsf{T}}$ is zero. Here $\Omega $ and ${\Sigma }_{\operatorname{tr}}$ have squared-pixel units; the propagated covariance has squared-degree units. This zero contribution concerns uncertainty consisting of a common translation, rather than an assertion that the model has zero total angle variance. Discarding coordinate covariances generally loses the cancellation (Fig. S1). Empirical coverage requires a separate interval evaluation.

### C. Quantile Intervals and Matched Gaussian Sampling

Joint sampling propagates coordinate variation through the nonlinear angle calculation. Main Eq. (9) selects one common dropout component for all ten coordinates and adds its scaled Gaussian noise to sample a complete configuration. Equation (S11), also main Eq. (10), converts the resulting nonlinear angle samples to marginal intervals. Equation (S12) then defines a matched Gaussian comparison that changes coordinate dependence while preserving each coordinate's mean and variance.

For draw $l$, the selected component index ${t}_{l}$ is uniform over the $T$ retained passes and the standard-normal vector has ten entries. One common ${t}_{l}$ supplies the mean and diagonal variance of all five landmarks, while the Gaussian noise entries are independent conditional on that pass. Selecting different passes for different landmarks would change coordinate dependence. After applying $f$, ${\theta }_{a}^{\left(l\right)}$ is the sampled angle in degrees for label $a\in \{\alpha ,\beta \}$. For nominal level $\gamma \in \left(0,1\right)$, (S11) takes equal-tail empirical quantiles:

$$
{I}_{a,\gamma }=[{Q}_{(1-\gamma )/2}(\{{\theta }_{a}^{(l)}\}),\,{Q}_{(1+\gamma )/2}(\{{\theta }_{a}^{(l)}\})].\tag{S11}
$$

In (S11), ${Q}_{p}$ is the linearly interpolated sample quantile at probability $p$. The lower tail probability is $\left(1-\gamma \right)/2$ and the upper endpoint uses $\left(1+\gamma \right)/2$, leaving nominal central content $\gamma $. Each angle receives its own interval; this construction alone provides neither simultaneous alpha/beta coverage nor calibrated clinical probabilities. The endpoints have degree units, and their empirical coverage is evaluated against the separately specified rater references. For (S12), the full and diagonal Gaussian draws use the same standard-normal vector so that the comparison changes their covariance structure without adding unrelated sampling differences:

$$
{y}_{F}^{(l)}=\bar{\mu }+C{\epsilon }_{l},\quad \quad C{C}^{\mathsf{T}}=\Sigma ,
$$

$$
{y}_{D}^{(l)}=\bar{\mu }+diag(\sqrt{{\Sigma }_{11}},\ldots ,\sqrt{{\Sigma }_{10,10}}){\epsilon }_{l}.\tag{S12}
$$

In (S12), $\bar{\mu }\in {\mathbb{R}}^{10}$ is the fixed mixture mean and $\Sigma \in {\mathbb{R}}^{10\times 10}$ its total covariance in squared-pixel units. The Cholesky factor $C\in {\mathbb{R}}^{10\times 10}$ has pixel units and satisfies $C{C}^{\mathsf{T}}=\Sigma $; multiplying $C$ by ${\varepsilon }_{l}\sim N\left(0,{I}_{10}\right)$ produces a correlated Gaussian perturbation. The diagonal draw instead multiplies each noise entry by the square root of its marginal variance, ${\Sigma }_{jj}^{1/2}$, so both Gaussian conditions retain the same coordinate SDs. Their only distributional difference is cross-coordinate covariance. Comparison with ${q}_{T}$ additionally changes distributional shape. Numerical validity requires finite coordinates, segments longer than ${10}^{-6}$ pixels and assessment of unsigned-angle branch proximity; valid arithmetic does not ensure anatomical plausibility. Samples remain anatomically unrestricted. The original intervals and probabilities are unrecalibrated. Section S16 separately reports post-hoc interval rescaling.

### D. Evaluation Details for Main Table I

Main Table I evaluates $n=242$ internal-evaluation images. MRE is the mean Euclidean distance between a predicted landmark and its reference, averaged over the five landmarks and images in original-image pixels, with rater-mean coordinates as the reference. Angle MAE averages the absolute alpha or beta error in degrees, using angles calculated from rater-mean coordinates as references. $PCK@r$ is the percentage of landmarks with Euclidean error within the threshold $r$ in original-image pixels; here the thresholds are 5 and 10 pixels. These thresholds have pixel units and are distinct from the angle thresholds used for failure detection. Means are reported with 95% image-bootstrap CIs from 10,000 resamples. The full-minus-Comparator error contrast uses predictions on the same images; Comparator trains its decoder from scratch. Dashes in main Table I indicate paired PCK/category contrasts that were not reported; paired contrasts between dropout-averaged predictions were not reported. BatchNorm statistics were re-estimated from training images. Main Section III-A describes the fixed inference protocol and model-comparison scope.

The bootstrap sampling unit is an image. A replicate draws images with replacement and retains all five landmarks and all raters associated with each selected image; this preserves their within-image relationships when recalculating an error or coverage statistic. Paired model contrasts use the same replicate image indices for both models. The number of resamples differs by analysis in main Section III-B: errors, coverage and component-selection contrasts use 10,000, while associations, AUROC and the retained-error estimates in main Table III use 2,000. These CIs condition on the fitted models and evaluation settings and describe image-level resampling variation. They do not account for patient-level clustering. A collapsed bootstrap interval at observed complete coverage does not establish complete coverage in a wider population.

## S15. Complete Model Training Objectives

The following equations specify the quantization, reconstruction and Stage 2 objectives accompanying the image-conditioned distribution. Symbol definitions follow Section S14 and the main text. Coordinate and angle references retain the original rater and mean-coordinate distinctions.

$$
{a}_{m}={\operatorname{argmin}}_{1\le c\le V}\Vert {z}_{m}-{e}_{c}{\Vert }_{2}^{2},
$$

$$
{z}_{m}^{ST}={z}_{m}+sg({e}_{{a}_{m}}-{z}_{m}),
$$

$$
{L}_{com}=\frac{1}{Md}\sum_{m=1}^{M}{\Vert }{z}_{m}-sg({e}_{{a}_{m}}){\Vert }_{2}^{2}.\tag{S13}
$$

In (S13), $\operatorname{argmin}$ selects code index ${a}_{m}$ nearest each token ${z}_{m}$ by Euclidean distance $\Vert \cdot {\Vert }_{2}$. Operator $\operatorname{sg}$ preserves values with zero derivative, so ${z}_{m}^{ST}$ forwards ${e}_{{a}_{m}}$ and passes gradients to the encoder with identity derivative. Commitment averages squared token-to-code errors over $Md$ scalar entries; exponential moving averages update codes. A shared three-block Mixer (width 256) decodes continuous ($b=c$) or quantized ($b=q$) tokens into ten normalized means ${\tilde{\mu }}_{j}^{b}$ and raw log variances. Clipping replaces out-of-range values by the nearest bound in $\left[-14,4\right]$; natural exponentiation gives positive variances ${v}_{j}^{b}$ bounded by $\left[\exp \left(-14\right),\exp \left(4\right)\right]$. No gradient-trained codebook loss or variational Kullback–Leibler divergence term is used.

The reconstruction objective in (S14) fits normalized means and variances to each target. For coordinate $j=1,\ldots ,10$, ${\tilde{y}}_{ir,j}$ is the target and ${\tilde{\mu }}_{j}^{b}$ the predicted mean. Squared residuals are divided by variance ${v}_{j}^{b}$; natural $\log  {v}_{j}^{b}$ penalizes excessive variance. The coordinate average equals twice mean per-coordinate Gaussian negative log likelihood (NLL), up to a constant. With positive expected squared residuals, the variance minimizing this reconstruction term at fixed means equals annotation variance plus squared mean bias. Clipping, shared parameters and other losses constrain learning; variance does not identify physical image noise.

$$
{L}_{rec}^{b}=\frac{1}{10}\sum_{j=1}^{10}{\left[\frac{({\tilde{y}}_{ir,j}-{\tilde{\mu }}_{j}^{b}{)}^{2}}{{v}_{j}^{b}}+log{v}_{j}^{b}\right]}.\tag{S14}
$$

Each target draws input uniformly from all same-image annotations, including itself. Equal annotation-row weights give five-rater images 2.5 times the weight of two-rater images. To constrain the reconstructed geometry, (S15) compares quantized reconstruction lengths for landmark endpoint pairs $k,l$ in $E=\{(2,4),(2,5),(3,4)\}$:

$$
{\Delta }_{kl}=\Vert {\tilde{\mu }}_{k}^{q}-{\tilde{\mu }}_{l}^{q}{\Vert }_{2}-\Vert {\tilde{y}}_{ir,k}-{\tilde{y}}_{ir,l}{\Vert }_{2},
$$

$$
{q}_{kl}=\sum_{a\in \{x,y\}}{(}{v}_{k,a}^{q}+{v}_{l,a}^{q})+{10}^{-6},
$$

$$
{L}_{edge}=\frac{1}{2|E|}\sum_{(k,l)\in E}{\left[\frac{{\Delta }_{kl}^{2}}{{q}_{kl}}+log{q}_{kl}\right]}.\tag{S15}
$$

In (S15), ${\Delta }_{kl}$ is predicted minus target normalized length; endpoint vectors ${\tilde{\mu }}_{k}^{q},{\tilde{y}}_{ir,k}\in {\mathbb{R}}^{2}$ have x/y coordinates. The sum ${q}_{kl}$ pools variances ${v}_{k,a}^{q}$ over endpoints $k,l$ and axes $a\in \{x,y\}$; ${10}^{-6}$ stabilizes division. Each squared error is divided by ${q}_{kl}$, combined with $\log  {q}_{kl}$ and scaled by $1/\left(2|E|\right)$ for $|E|$ connections. This is heuristic weighting, not directional length variance. The Stage 1 objective in (S16) combines reconstruction, commitment and connection penalties:

$$
{L}_{1}(e)={\lambda }_{c}(e){L}_{rec}^{c}
$$

$$
\quad +{\lambda }_{q}(e)({L}_{rec}^{q}+{L}_{com})
$$

$$
\quad +\mathbf{1}[{\lambda }_{q}(e)>0]{\lambda }_{E}{L}_{edge}.\tag{S16}
$$

In (S16), $e$ indexes epochs; ${\lambda }_{c}$ weights continuous reconstruction, ${\lambda }_{q}$ quantized reconstruction plus commitment, and ${\lambda }_{E}$ the connection penalty. Indicator $1\left[{\lambda }_{q}\left(e\right)>0\right]$ activates that penalty only with quantization. The pair $\left({\lambda }_{c},{\lambda }_{q}\right)$ is $\left(1,0\right)$ in epochs 1–39, ${\lambda }_{c}=1-\left(e-40\right)/60$ and ${\lambda }_{q}=4\min \{1,\left(e-39\right)/60\}$ in epochs 40–99, then $\left(0,4\right)$ from epoch 100. Independently, ${\lambda }_{E}=1$. Histogram entropy contributes no gradient. Under reconstruction alone, uniform inputs share the same targets and optimal rater means; other constraints affect learned outputs.

In main Eq. (2), logits ${h}_{imc}$ score image $i$, token $m$ and code $c$. Softmax normalizes $\exp \left({h}_{imc}\right)$ over $V$ codes into probabilities ${\pi }_{imc}$; $v$ indexes denominator codes. Weighted vectors ${\bar{z}}_{im}\in {\mathbb{R}}^{d}$ form ${\bar{z}}_{i}$, which fixed $\operatorname{Dec}$ maps to ${\tilde{\mu }}_{i}\in {\mathbb{R}}^{10}$ and diagonal covariance ${\tilde{D}}_{i}\in {\mathbb{R}}^{10\times 10}$. Decoding this weighted representation can differ from mixing separately decoded configurations. Equation (S17) targets the Stage 1 code ${a}_{im}^{\left(r\right)}$ for each rater annotation. Target ${t}_{imc}^{\left(r\right)}$ mixes its one-hot indicator with a uniform distribution over $V$ codes; ${\varepsilon }_{LS}$ is the uniform fraction. Cross-entropy sums over codes and averages $M$ tokens. Coordinate/angle targets remain rater means.

$$
{t}_{imc}^{(r)}=(1-{\varepsilon }_{LS})\mathbf{1}[c={a}_{im}^{(r)}]+\frac{{\varepsilon }_{LS}}{V},
$$

$$
{L}_{tok,i}=-\frac{1}{M}\sum_{m=1}^{M}{\sum_{c=1}^{V}{{t}_{imc}^{(r)}}}\log {\pi }_{imc}.\tag{S17}
$$

The coordinate objective in (S18) fits rater-mean coordinates in original pixels. Prediction ${\hat{y}}_{i}={S}_{i}{\tilde{\mu }}_{i}$ restores pixel scale; ${r}_{ik,a}={\left({\hat{y}}_{ik}-{\bar{y}}_{ik}\right)}_{a}$ is the residual for landmark $k=1,\ldots ,5$ and axis $a\in \{x,y\}$. Smooth-L1 uses a quadratic penalty for $|r|<\delta $ and a linear penalty otherwise; its argument $r$ is a scalar residual. Weights ${b}_{k}={\left(1,1,3,3,3\right)}_{k}$ triple the last three landmark contributions; ${w}_{ik}$ comes from (S19). Division by ten averages the coordinate entries without normalizing by the weight sum:

$$
{\rho }_{\delta }(r)=\left\{\begin{matrix}{r}^{2}/(2\delta ), & |r|<\delta , \\ |r|-\delta /2, & |r|\ge \delta ,\end{matrix}\right.
$$

$$
{L}_{pt,i}=\frac{1}{10}\sum_{k=1}^{5}{\sum_{a\in \{x,y\}}{{b}_{k}}}{w}_{ik}{\rho }_{2}({r}_{ik,a}).\tag{S18}
$$

The source-default transition in (S18) is $\delta =2$ pixels. Image quality and A/B agreement determine the coordinate-error weights in (S19). For 8-bit image ${G}_{i}$, contrast is ${q}_{i}^{con\operatorname{tr}ast}=\operatorname{clip}\left(SD\left({G}_{i}\right)/60,0,1\right)$ and sharpness is ${q}_{i}^{sharp}=\operatorname{clip}\left(\operatorname{Var}\left({\nabla }^{2}{G}_{i}\right)/500,0,1\right)$. Here $SD$ is intensity standard deviation, $\operatorname{Var}$ variance, and ${\nabla }^{2}$ the Laplacian; clipping limits scores to $\left[0,1\right]$. Their mean ${Q}_{i}$ and agreement receive weights 0.4 and 0.6. Agreement decays exponentially with the pixel distance between A/B positions ${P}_{iAk},{P}_{iBk}$, on scale 8. Both rater-count groups use A/B. The clipped loss weight ${w}_{ik}\in \left[0.05,1\right]$ is distinct from decoder variance:

$$
{Q}_{i}=\frac{1}{2}({q}_{i}^{sharp}+{q}_{i}^{contrast}),
$$

$$
{w}_{ik}=clip\left(0.4{Q}_{i}+0.6{e}^{-\Vert {P}_{iAk}-{P}_{iBk}{\Vert }_{2}/8},0.05,1\right).\tag{S19}
$$

The angle objective in (S20) compares directed clockwise training angles in degrees after Cartesian y-axis conversion. Unbarred angles ${\alpha }_{i}^{\operatorname{tr}}:u\to v$ and ${\beta }_{i}^{\operatorname{tr}}:w\to u$ use ${\hat{y}}_{i}$; barred targets use ${\bar{y}}_{i}$. For angles $a,b$, $d$ is their absolute difference modulo 360, and ${d}_{circ}$ chooses the shorter circular distance. Mean weight ${\bar{w}}_{i}=\left(1/5\right){\sum }_{k} {w}_{ik}$ scales the penalty. Alpha and beta errors have weights 2 and 1; division by three normalizes them. Evaluation uses unsigned segment angles in main Eq. (5):

$$
d=|a-b|\,\mod \,360,
$$

$$
{d}_{circ}(a,b)=min\{d,360-d\},
$$

$$
{L}_{ang,i}=\frac{{\bar{w}}_{i}}{3}(2{d}_{circ}({\alpha }_{i}^{\operatorname{tr}},{\bar{\alpha }}_{i}^{\operatorname{tr}})
$$

$$
\quad \quad +{d}_{circ}({\beta }_{i}^{\operatorname{tr}},{\bar{\beta }}_{i}^{\operatorname{tr}})).\tag{S20}
$$

The Stage 2 schedule in (S21) sets the token, coordinate and angle loss weights ${\lambda }_{t},{\lambda }_{p},{\lambda }_{a}$. Here $e$ is the epoch, $E$ its budget, and ${c}_{e}$ cosine progress. Losses average annotation rows; token weight falls while coordinate/angle weights rise. Only the predictor updates; gradients traverse the fixed decoder and weighted-codebook operation. Stage 2 adds no variance likelihood.

$$
{L}_{2}(e)={\lambda }_{t}(e){L}_{tok}+{\lambda }_{p}(e){L}_{pt}+{\lambda }_{a}(e){L}_{ang},
$$

$$
{c}_{e}=\frac{1-cos(\pi (e-1)/E)}{2},
$$

$$
({\lambda }_{t},{\lambda }_{p},{\lambda }_{a})=(1-0.5{c}_{e},\,5(1+{c}_{e}),\,2(1+{c}_{e})).\tag{S21}
$$

## Interpretation and provenance

The within-pass and between-pass components describe the prediction process. Their labels do not separately identify physical image noise, annotation noise or anatomical ambiguity. No new derivation or experiment was performed for this export.

The source discussion of physical pixel spacing is an author-review item. Pixel-valued distance results do not establish a calibrated x/y pixel aspect ratio.
