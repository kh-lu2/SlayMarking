# 🌊 SlayMarking — Deep Ocean Image Watermarking Suite

A digital image watermarking research suite and interactive GUI application styled with a deep ocean aesthetic.

## 🔬 Research Watermarking Methods

| Method | Status | Technique & Domain | Requirements |
|---|---|---|---|
| Method 1 | Implemented | DWT-SVD with Semi-Binary Arrays (Wavelet Domain) | Arbitrary images (RGB / Grayscale) |
| Method 2 | Implemented | Hybrid DWT-HMD-SVD with Arnold Scrambling | Strictly 512×512 Grayscale (`'L'`) |
| Method 3 | Placeholder | Ready for Paper #3 implementation | — |
| Method 4 | Placeholder | Ready for Paper #4 implementation | — |
| Method 5 | Placeholder | Ready for Paper #5 implementation | — |
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
- **Evaluation Metrics**: PSNR, SSIM, Normalized Cross-Correlation (NC), NPCR %, UACI %, MSE, MAE.

---

## 🚀 Running the App

```powershell
python main.py
```

Run test suite:
```powershell
python -m unittest tests/test_app.py tests/test_method_1.py tests/test_method_2.py
```
