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
| Method 6 | Placeholder | Ready for Paper #6 implementation | — |

---

### Method 1: DWT-SVD Semi-Binary
- **Paper**: *"Robust Digital Image Watermarking Using Singular Value Decomposition"* (Morteza Makhloghi, Fardin Akhlaghian, Habibollah Danyali, IEEE 2011).
- **Core Pipeline**:
  - Decomposes the host image via 1-level 2D DWT into subbands $\{LL, LH, HL, HH\}$.
  - Groups wavelet coefficients into $2 \times 2$ blocks sorted by significant energy coefficients.
  - Converts watermark singular values and coefficient blocks into semi-binary arrays to redundantly embed payload bits into wavelet subbands.
- **Key Parameters**: `strength` (embedding intensity), `dwt_level` (decomposition level), `wavelet` (wavelet family).

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
### Method 5: Reversible Visible Watermarking and Lossless Image Recovery
- **Paper**: *"Reversible Visible Watermarking and Lossless Recovery of Original Images"* (Yongjian Hu and Byeungwoo Jeon, *IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)*, Vol. 16, No. 11, November 2006, pp. 1423-1429, DOI: [10.1109/TCSVT.2006.884011](https://doi.org/10.1109/TCSVT.2006.884011)).
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

---

## 🚀 Running the App

```powershell
python main.py
```

Run test suite:
```powershell
python -m unittest tests/test_app.py tests/test_method_1.py tests/test_method_2.py
```
