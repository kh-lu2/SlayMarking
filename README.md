# 🌊 SlayMarking — Deep Ocean Image Watermarking Suite

A digital image watermarking research suite and interactive GUI application styled with a deep ocean aesthetic.

## 🔬 Research Watermarking Methods

| Method | Status | Technique & Domain | Requirements |
|---|---|---|---|
| Method 1 | Implemented | DWT-SVD with Semi-Binary Arrays (Wavelet Domain) | Arbitrary images (RGB / Grayscale) |
| Method 2 | Implemented | Hybrid DWT-HMD-SVD with Arnold Scrambling | Strictly 512×512 Grayscale (`'L'`) |
| Method 3 | Implemented | Fragile 2-Level DWT & Schur Decomposition for Tamper Recovery | 512×512 Grayscale (`'L'`) |
| Method 4 | Placeholder | Ready for Paper #4 implementation | — |
| Method 5 | Implemented | Reversible Visible Watermarking & Lossless Image Recovery | Arbitrary images (RGB / Grayscale) |
| Method 6 | Implemented | Generalized DWT-HMD-SVD & Arnold (Luminance & Subband Choice) | Arbitrary images (RGB / RGBA / Grayscale) |

---

### Method 1: DWT-SVD Semi-Binary
- **Paper**: *"Robust Digital Image Watermarking Using Singular Value Decomposition"* (Morteza Makhloghi, Fardin Akhlaghian, Habibollah Danyali, *IEEE Conference on Digital Image Processing & Security*, 2011, pp. 219-224).
- **Core Pipeline**:
  - **Preprocessing & Color Handling**: Operates directly on grayscale (`'L'`) or isolates the luminance $Y$ channel (via YCbCr conversion) for RGB/RGBA images while preserving chrominance and alpha channels. Pads dimensions with reflection to the nearest multiple of $2^{K+1}$ when needed.
  - **Watermark SVD & Scaling**: Generates a $64 \times 64$ watermark image $W$, computes its Singular Value Decomposition ($W = U_W \Sigma_W V_W^T$), and scales its singular values into $[0, 255]$ via $\alpha = \frac{255}{\sigma_1} \cdot \text{strength}$.
  - **$K$-Level Haar DWT**: Decomposes the host luminance plane using a $K$-level 2D Haar DWT (default $K=5$, yielding $64$ blocks of size $2 \times 2$ for a $512 \times 512$ image) to obtain the four coarsest subbands $\{LL_K, LH_K, HL_K, HH_K\}$.
  - **Block Grouping & Semi-Binary Embedding**: Partitions each of the four subbands into $2 \times 2$ blocks (sorting $LL_K$ blocks in descending order by mean intensity, and keeping raster order for $LH_K, HL_K, HH_K$). Converts both the scaled watermark singular values $\sigma_i$ and the $2 \times 2$ wavelet coefficients into **Semi-Binary Arrays (SBA)** (Algorithm 1), redundantly embedding 8 bits of each $\sigma_i$ across the four block coefficients in all four subbands before mapping them back to real numbers (Algorithm 2).
  - **Reconstruction & Blind Extraction**: Reconstructs the watermarked image via $K$-level Inverse Haar DWT and unpadding. During blind extraction, performs $K$-level DWT, extracts the semi-binary bits from all four subbands, averages the reconstructed singular values across $\{LL_K, LH_K, HL_K, HH_K\}$, and rebuilds the watermark via $W' = U_W \Sigma'_{\text{avg}} V_W^T$, evaluating correlation $\rho$.
- **Key Parameters**: `strength` ($\alpha$ multiplier, default: `1.0`), `dwt_level` ($K \in [3, 6]$, default: `5`), `wavelet` (`haar`).
- **Evaluation Metrics**: PSNR, SSIM, Correlation Coefficient $\rho$ (`correlation_rho`), DWT Level (`dwt_level`), Watermark Size (`watermark_size`)

### Method 2: Hybrid DWT-HMD-SVD & Arnold Scrambling
- **Paper**: *"Enhanced medical image watermarking using hybrid DWT-HMD-SVD and Arnold scrambling"* (Himanshi Chaudhary, Preeti Garg & Virendra P. Vishwakarma, *Scientific Reports*, Nature Portfolio 2025, DOI: [10.1038/s41598-025-94080-4](https://doi.org/10.1038/s41598-025-94080-4)).
- **Core Pipeline**:
  - **Strict Paper Specifications**: Host medical image must be strictly **512×512** pixels and **grayscale** (mode `'L'`). Incompatible images trigger an in-app format requirement card.
  - Generates a 64×64 binary watermark canvas from payload and randomizes pixel positions with chaotic **Arnold Cat Map**.
  - Applies 1-level Haar DWT to obtain the $256 \times 256$ $LL$ subband.
  - Performs Upper Hessenberg Matrix Decomposition (**HMD**, $LL = Q H Q^T$) to stabilize coefficients.
  - Computes SVD of upper Hessenberg matrix $H$ and embeds watermark singular values into the first 64 singular values: $\Sigma_{new}[:64] = \Sigma_H[:64] + \alpha \Sigma_W[:64]$.
  - Reconstructs via Inverse SVD $\to$ Inverse HMD $\to$ Inverse DWT.
  - Extraction reverses the pipeline using reference singular values, adaptive thresholding ($T = (\max + \min)/2$), and inverse Arnold descrambling.
- **Key Parameters**: `gain_factor` ($\alpha$, default: `0.02`), `arnold_iters` (Arnold iterations, default: `10`), `wavelet` (`haar`, `db1`, `bior1.1`).

### Method 3: Fragile 2-Level DWT & Schur Decomposition for Tamper Recovery
- **Paper**: *"A Fragile Watermarking Scheme for Image Tamper Detection and Recovery Using Hybrid Transform"* (Prasanth Vaidya S, N V P S Rajesh Kandala, Chandra Mouli PVSSR, *Journal of Visual Communication and Image Representation*, Elsevier 2023).
- **Core Pipeline**:
  - Applies a 2-level Haar DWT to the host image (operating on luminance $Y$ for color images or grayscale `'L'`, padded to multiples of 4) to obtain the $LL_2$ approximation subband and embeds the watermark via $LL_2^W = \alpha \cdot LL_2 + \beta \cdot W$, followed by 2-level Inverse Haar DWT to produce $I^W$.
  - Divides $I^W$ into $4 \times 4$ blocks and generates **32 recovery bits** per block from the 8-bit mean intensities of its four $2 \times 2$ sub-blocks.
  - Maps each block $B$ to a partner block $B'$ via a pseudo-random permutation seeded by `secret_key` and embeds the 32 recovery bits into LSB bit planes 1 and 2 of $B'$.
  - Clears bit 0 (LSB 1) of each block and computes **Schur Decomposition** ($A = U \lambda U^T$).
  - Maps the Schur matrix traces and eigenvalues to $[0, 262143]$, combining them with `secret_key` and spatial block indices (to prevent Copy-Move attacks) to generate **16 Authentication Block Bits (ABB)** embedded into bit 0 (LSB 1), yielding $I_A^W$.
  - **Extraction, Tamper Localization & Self-Recovery**: Extracts the watermark via 2-level DWT ($W'_S = (LL'_2 - \alpha \cdot LL_2) / \beta$) and evaluates NCC; verifies block integrity by comparing stored LSB 0 bits against recomputed Schur ABB signatures, localizing tampered blocks (zeroing pixels) and reconstructing damaged regions from untampered partner blocks (with neighbor interpolation fallback).
- **Key Parameters**: `alpha` ($\alpha$, default: `0.98`), `beta` ($\beta$, default: `0.04`), `secret_key` (seed, default: `42`), `attack` (`None`, `Copy-Move`, `Center Crop`, `Splice Foreign Object`, `Text Attack`, `Salt & Pepper Noise`), `tamper_size` (default: `64`).
- **Evaluation Metrics**: PSNR, SSIM, Watermark NCC (`watermark_ncc`), Tampered Blocks count & percentage (`tamper_percentage`), Recovered PSNR (`recovered_psnr_db`), Recovered SSIM (`recovered_ssim`).

### Method 5: Reversible Visible Watermarking and Lossless Image Recovery
- **Paper**: *"Reversible Visible Watermarking and Lossless Recovery of Original Images"* (Yongjian Hu and Byeungwoo Jeon, *IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)*, Vol. 16, No. 11, November 2006, pp. 1423-1429).
- **Primary Use**: Medical imagery privacy protection, military / remote-sensing observation tagging, and copyright protection. Enables a visible watermark to serve as a prominent tag or privacy overlay on sensitive region $R$, while secretly embedding lossless reconstruction packets in $I - R$. Authorized recipients with the user key can completely remove the visible watermark and losslessly reconstruct 100% of the original image data (0% distortion, $MSE = 0$, $PSNR = \infty$).
- **Core 9-Step Pipeline**:
  1. *Region Selection & Preprocessing*: Select sensitive region $R$ (dimension matching watermark $W$) and extract bit plane $R_D$ ($N_{R_D} = 7$ for MSB heavy tag or $6$ for 2nd MSB light tag) as bit sequence $D$.
  2. *Preprocessing Compression*: Losslessly compress $D$ into $D_c$ ($|D_c| \ll |D|$).
  3. *Key Bit-Plane Selection*: Test bit planes of non-watermarked area $I - R$ from LSB (0) to MSB (7). Group pixels into $2 \times 4$ blocks (Scheme 2 '::::') to form 1-byte characters. Select lowest key bit plane $I_k$ with sufficient spare space ($\text{Cap} \ge |D_c|$).
  4. *Error-Diffused Hybrid Selection*: Search for largest selection step $S_p \in [1, \min(S_{L_p}, 15)]$ using Hybrid selection (RPS seeded with $N_{I_k}$) to disperse artifacts across $I - R$ and eliminate visual stripe noise. Fall back to Fixed-interval if needed.
  5. *Capacity-Adaptive Sequence Constitution*: Construct sequence $S$ until $|S| - |S_c| = |D_c| + n$, recording skip points $n \ge 0$.
  6. *Payload Construction & Data Hiding*: Concatenate $H = S_c \cup D_c$ (plus padding $n$) and replace sequence $S$ on $I_k$ with $H$.
  7. *80-Bit User Key Construction*: Assemble 80-bit key (20 hex chars) encoding watermark size (16 bits), origin $(r_0, c_0)$ (32 bits), $|D_c|$ (16 bits), $N_{I_k}$ (3 bits), hybrid flag (1 bit), step $S_p$ (4 bits), and $n$ (8 bits).
  8. *Interleaved Visible Embedding*: Embed binary logo $W$ across $R_D$ and lower neighbor $R_D'$ under User-Key-Controlled Embedding (UKCE seeded with user key).
  9. *Watermark Removal & Lossless Reconstruction*: Unpack user key, locate $S$ on $I_k$, decode $S_c$ to restore $I - R$, decode $D_c$ to obtain $D$, and invert UKCE bit placement to reconstruct $R$, restoring the 100% bit-exact original image.
- **Key Parameters**: `bit_plane_rd` ($N_{R_D} \in \{6, 7\}$), `watermark_size` (`64x128`, `128x128`, etc.), `region_origin_y` ($r_0$), `region_origin_x` ($c_0$), `selection_mode` (Hybrid, Fixed, Consecutive), `user_key`.
- **Evaluation Metrics**: PSNR, PSNR without $R$ ($PSNR_{I-R}$), SSIM, NCC, Error diffusion ratios $E_d\%$ and $E_n\%$, exact bit-exact reconstruction verification.

### Method 6: Generalized Hybrid DWT-HMD-SVD & Arnold Scrambling (Color & Subband Choice)
- **Generalization Features**:
  - **Color Luminance Processing**: Isolates the chromatic color components from the brightness channel (converts RGB/RGBA to YCbCr) and performs the DWT-HMD-SVD embedding strictly on the $Y$ (brightness) channel, leaving color and alpha channels intact.
  - **Support for All Existing Asset Images**: Automatically applies symmetric square padding to support arbitrary rectangular or non-512×512 dimensions for Hessenberg matrix decomposition, seamlessly unpadding upon reconstruction.
  - **Selectable Frequency Subband ($LL$, $LH$, $HL$, $HH$)**: Enables the user to choose which frequency band to embed in:
    - **$LL$**: Low-frequency approximation band (high imperceptibility, paper baseline).
    - **$LH$**: Low-High horizontal detail band.
    - **$HL$**: High-Low vertical detail band.
    - **$HH$**: High-High diagonal detail band.
- **Key Parameters**: `gain_factor` ($\alpha$, default: `0.02`), `subband` (`LL`, `LH`, `HL`, `HH`, default: `LL`), `arnold_iters` (Arnold iterations, default: `10`), `wavelet` (`haar`, `db1`, `db2`, `sym2`, `bior1.3`).
- **Evaluation Metrics**: PSNR, SSIM, NC, UACI, NPCR, MSE, MAE, NCC.

---

## 🚀 Running the App

```powershell
python main.py
```

Run test suite:
```powershell
python -m unittest tests/test_method_1.py tests/test_method_2.py tests/test_method_3.py tests/test_method_4.py tests/test_method_5.py tests/test_method_6.py
```
