# Phase 5: Comprehensive Evaluation

**Phase 5** provides comprehensive evaluation tools for learned image compression models with BD-rate calculation, rate-distortion curves, SOTA comparison, and publication-ready visualizations.

## 🎯 Key Features

### 1. **BD-Rate Calculation**
- Bjøntegaard Delta metrics (BD-Rate, BD-PSNR)
- Multiple interpolation methods (cubic, pchip, akima)
- Batch comparison support

### 2. **Rate-Distortion Curves**
- Publication-quality plotting
- Multi-dataset visualization
- Comparison tables with color coding
- Convex hull analysis

### 3. **Multi-Dataset Evaluation**
- Support for Kodak, CLIC, Tecnick
- Parallel evaluation
- Comprehensive metrics (PSNR, MS-SSIM, BPP)

### 4. **SOTA Comparison**
- Compare with VTM, BPG, JPEG2000
- Automatic report generation
- Statistical analysis

### 5. **Visualization Dashboard**
- Interactive dashboards
- Statistical plots
- Distribution analysis
- Publication-ready figures

---

## 🚀 Quick Start

### **1. Evaluate on Kodak**
```bash
bash run_eval_kodak.sh
```

Evaluates Phase 4 model on Kodak dataset at 5 quality levels.

### **2. Evaluate on All Datasets**
```bash
bash run_eval_all.sh
```

Evaluates on Kodak, CLIC, and Tecnick datasets.

### **3. Compare with SOTA**
```bash
bash run_compare_sota.sh
```

Compares results with VTM, BPG, JPEG2000.

### **4. Generate Dashboard**
```bash
bash run_generate_dashboard.sh
```

Creates visualization dashboard with all plots.

---

## 📁 File Structure

```
phase5/
├── evaluate.py                      # Main evaluation script (300+ lines)
├── compare_sota.py                  # SOTA comparison (150+ lines)
├── run_eval_kodak.sh                # Evaluate Kodak
├── run_eval_all.sh                  # Evaluate all datasets
├── run_compare_sota.sh              # Compare SOTA
├── run_generate_dashboard.sh        # Generate dashboard
├── src/
│   └── evaluation/
│       ├── __init__.py              # Package init
│       ├── bd_rate.py               # BD-rate calculation (400+ lines)
│       ├── rd_curve.py              # RD curve plotting (500+ lines)
│       ├── metrics.py               # Metrics calculation (250+ lines)
│       └── comparator.py            # Model comparison (350+ lines)
└── scripts/
    └── generate_dashboard.py        # Dashboard generation (250+ lines)
```

---

## 🔧 Usage Examples

### **Example 1: Evaluate Single Model**

```bash
python evaluate.py \
    --checkpoint ../phase4/outputs/best_model.pth \
    --model hpcm_base \
    --method_name "HPCM-Phase4" \
    --datasets kodak \
    --quality_levels 1 2 3 4 5 \
    --device cuda \
    --output_dir ./outputs/evaluation/kodak
```

**Output:**
- `results_kodak.json`: Full evaluation results
- `rd_curve_kodak.json`: RD curve data

### **Example 2: Compare Multiple Methods**

```python
from src.evaluation import ModelComparator, RDCurve

# Load curves
phase4_curve = RDCurve.load('outputs/rd_curve_kodak.json')
phase3_curve = RDCurve.load('../phase3/outputs/rd_curve_kodak.json')
bpg_curve = RDCurve(
    name='BPG',
    rates=[0.20, 0.35, 0.55, 0.80],
    psnrs=[30.5, 33.2, 35.8, 38.0],
)

# Compare
comparator = ModelComparator()
comparison_df = comparator.compare(
    test_curves={'Phase4': phase4_curve, 'Phase3': phase3_curve},
    reference_curve=bpg_curve,
    dataset_name='Kodak',
)

print(comparison_df)
```

### **Example 3: Calculate BD-Rate**

```python
from src.evaluation import compute_bd_rate, compute_bd_psnr

# Method A
method_a_rates = [0.2, 0.4, 0.6, 0.8]
method_a_psnrs = [32.5, 35.2, 37.1, 38.5]

# Reference
ref_rates = [0.25, 0.5, 0.75, 1.0]
ref_psnrs = [31.8, 34.5, 36.3, 37.8]

# Calculate
bd_rate = compute_bd_rate(method_a_rates, method_a_psnrs, ref_rates, ref_psnrs)
bd_psnr = compute_bd_psnr(method_a_rates, method_a_psnrs, ref_rates, ref_psnrs)

print(f"BD-Rate: {bd_rate:.2f}%")
print(f"BD-PSNR: {bd_psnr:.3f} dB")
```

### **Example 4: Plot RD Curves**

```python
from src.evaluation import RDCurve, RDCurvePlotter

# Create curves
curves = [
    RDCurve('HPCM-Phase4', [0.2, 0.4, 0.6], [32.5, 35.2, 37.1]),
    RDCurve('BPG', [0.25, 0.5, 0.75], [31.8, 34.5, 36.3]),
    RDCurve('JPEG2000', [0.3, 0.6, 0.9], [30.5, 33.2, 35.5]),
]

# Plot
plotter = RDCurvePlotter(figsize=(10, 6), dpi=300)
fig = plotter.plot(
    curves=curves,
    title='Rate-Distortion Comparison',
    save_path='rd_curves.png',
)
```

---

## 📊 Evaluation Workflow

### **Step 1: Prepare Datasets**

```bash
# Download datasets
mkdir -p datasets
cd datasets

# Kodak (24 images)
wget http://r0k.us/graphics/kodak/kodak.zip
unzip kodak.zip -d kodak

# CLIC (download from http://compression.cc/)
# Tecnick (download from https://testimages.org/)
```

### **Step 2: Run Evaluation**

```bash
# Evaluate on all datasets
bash run_eval_all.sh
```

**Expected Output:**
```
Evaluating on KODAK
Quality 1: PSNR=32.50±0.82 dB, BPP=0.2145±0.0312
Quality 2: PSNR=35.20±0.75 dB, BPP=0.4028±0.0456
Quality 3: PSNR=37.10±0.68 dB, BPP=0.6112±0.0589
Quality 4: PSNR=38.50±0.62 dB, BPP=0.8256±0.0712
Quality 5: PSNR=39.80±0.58 dB, BPP=1.0421±0.0845
```

### **Step 3: Compare with SOTA**

```bash
# Compare with VTM, BPG, JPEG2000
bash run_compare_sota.sh
```

**Expected Output:**
```
Comparison Results:
Dataset | Method       | BD-Rate (%) | BD-PSNR (dB)
--------|--------------|-------------|-------------
Kodak   | HPCM-Phase4  | -6.80       | +0.30
Kodak   | VTM          | -8.50       | +0.38
CLIC    | HPCM-Phase4  | -5.90       | +0.28
CLIC    | VTM          | -7.20       | +0.32
```

### **Step 4: Generate Visualizations**

```bash
# Create dashboard
bash run_generate_dashboard.sh
```

**Generated Files:**
- `dashboard.png`: Comprehensive dashboard
- `rd_curves.png`: RD curves plot
- `rd_curves_multi_dataset.png`: Multi-dataset plots

---

## 📈 BD-Rate Calculation

### **What is BD-Rate?**

BD-Rate (Bjøntegaard Delta Rate) measures the average percentage bitrate difference between two RD curves at the same quality.

**Formula:**
```
BD-Rate = 100 × (exp(avg_log_rate1 - avg_log_rate2) - 1)
```

**Interpretation:**
- **Negative**: Method 1 is more efficient (better)
- **Positive**: Method 1 is less efficient (worse)
- **Example**: BD-Rate = -6.8% means 6.8% bitrate savings

### **Calculation Details**

```python
from src.evaluation import BDRateCalculator

calc = BDRateCalculator(
    interpolation_type='cubic',    # or 'pchip', 'akima'
    integration_samples=1000,      # Number of samples for integration
)

bd_rate = calc.bd_rate(
    rate1=[0.2, 0.4, 0.6, 0.8],    # Method 1 rates
    psnr1=[32.5, 35.2, 37.1, 38.5], # Method 1 PSNRs
    rate2=[0.25, 0.5, 0.75, 1.0],   # Reference rates
    psnr2=[31.8, 34.5, 36.3, 37.8], # Reference PSNRs
)

print(f"BD-Rate: {bd_rate:.2f}%")
```

**Requirements:**
1. At least 3 points per curve
2. Overlapping PSNR/rate ranges
3. Monotonically increasing PSNR with rate

---

## 📉 Rate-Distortion Curves

### **Plotting Features**

1. **Simple Plot**
   ```python
   plotter.plot(curves, title='RD Curves', save_path='plot.png')
   ```

2. **Comparison Table**
   ```python
   plotter.plot_comparison_table(test_curves, reference_curve, save_path='table.png')
   ```
   - Includes BD-rate values
   - Color-coded cells (green=good, red=bad)

3. **Convex Hull**
   ```python
   plotter.plot_with_convex_hull(curves, save_path='hull.png')
   ```
   - Highlights best performance envelope

4. **Multi-Dataset**
   ```python
   plotter.plot_multi_dataset(curves_by_dataset, save_path='multi.png')
   ```
   - Subplots for each dataset

### **Customization**

```python
plotter = RDCurvePlotter(
    figsize=(12, 8),           # Figure size
    dpi=300,                   # Resolution
    style='seaborn-paper',     # Matplotlib style
)

fig = plotter.plot(
    curves=curves,
    metric='ms_ssim',          # or 'psnr'
    xlim=(0, 1.0),            # Rate range
    ylim=(30, 40),            # PSNR range
    legend_loc='lower right',
    grid=True,
)
```

---

## 🏆 SOTA Comparison

### **Pre-defined SOTA Results**

Phase 5 includes pre-defined results for:

| Method | Kodak | CLIC | Tecnick |
|--------|-------|------|---------|
| **VTM** | ✅ | ✅ | ❌ |
| **BPG** | ✅ | ✅ | ❌ |
| **JPEG2000** | ✅ | ❌ | ❌ |

### **Adding Custom SOTA Data**

Edit [src/evaluation/comparator.py](src/evaluation/comparator.py):

```python
SOTA_RESULTS = {
    'Kodak': {
        'Your_Method': {
            'rates': [0.2, 0.4, 0.6, 0.8],
            'psnrs': [32.0, 34.5, 36.8, 38.5],
        },
    },
}
```

### **Comparison Output**

```
Dataset | Method       | BD-Rate (%) | BD-PSNR (dB) | Points
--------|--------------|-------------|--------------|-------
Kodak   | HPCM-Phase4  | -6.80       | +0.30        | 5
Kodak   | VTM          | -8.50       | +0.38        | 4
Kodak   | BPG          | 0.00        | 0.00         | 4 (ref)
CLIC    | HPCM-Phase4  | -5.90       | +0.28        | 5
Average | HPCM-Phase4  | -6.35       | +0.29        | -
```

---

## 📊 Metrics

### **Supported Metrics**

1. **PSNR (Peak Signal-to-Noise Ratio)**
   - Range: 20-50 dB
   - Higher is better
   - Most common metric

2. **MS-SSIM (Multi-Scale Structural Similarity)**
   - Range: 0-1
   - Higher is better
   - Perceptually aligned

3. **BPP (Bits Per Pixel)**
   - Compression rate
   - Lower is better
   - Actual file size / pixels

### **Calculation Example**

```python
from src.evaluation import MetricsCalculator
import torch

calc = MetricsCalculator(device='cuda')

original = torch.rand(1, 3, 256, 256) * 255.0
reconstructed = original + torch.randn_like(original) * 5.0

psnr = calc.psnr(original, reconstructed)
ms_ssim = calc.ms_ssim(original, reconstructed)

print(f"PSNR: {psnr:.2f} dB")
print(f"MS-SSIM: {ms_ssim:.4f}")
```

---

## 🎨 Visualization Dashboard

### **Dashboard Components**

1. **RD Curves** (top-left, 2×2)
   - All datasets overlaid
   - Legends and markers

2. **PSNR Bar Chart** (top-right)
   - Average PSNR per dataset
   - Error bars for std

3. **BPP Bar Chart** (middle-right)
   - Average BPP per dataset

4. **PSNR vs BPP Scatter** (bottom-left)
   - All data points
   - Colored by dataset

5. **Rate Distribution** (bottom-middle)
   - Histogram of BPP values

6. **PSNR Distribution** (bottom-right)
   - Histogram of PSNR values

### **Example Dashboard**

![Dashboard](assets/dashboard_example.png)

---

## 📝 Output Files

### **Evaluation Output**

```
outputs/evaluation/
├── results_kodak.json              # Full results (JSON)
├── results_clic.json
├── results_tecnick.json
├── results_all.json                # Combined results
├── rd_curve_kodak.json             # RD curve data
├── rd_curve_clic.json
└── rd_curve_tecnick.json
```

### **Comparison Output**

```
outputs/comparison/
├── sota_comparison.csv             # CSV table
├── sota_comparison.txt             # Formatted text
├── sota_comparison.json            # JSON data
├── sota_comparison_summary.txt     # Summary statistics
├── rd_comparison_Kodak.png         # RD curves
├── rd_comparison_CLIC.png
├── rd_table_Kodak.png              # Comparison tables
└── rd_table_CLIC.png
```

### **Visualization Output**

```
outputs/visualization/
├── dashboard.png                   # Main dashboard
├── rd_curves.png                   # Simple RD plot
└── rd_curves_multi_dataset.png     # Multi-dataset plot
```

---

## 🔬 Advanced Usage

### **Batch Evaluation**

```bash
# Evaluate multiple checkpoints
for checkpoint in ../phase*/outputs/best_model.pth; do
    python evaluate.py \
        --checkpoint $checkpoint \
        --method_name "$(basename $(dirname $(dirname $checkpoint)))" \
        --datasets kodak clic \
        --output_dir ./outputs/evaluation/batch
done
```

### **Custom Metrics**

```python
from src.evaluation import MetricsCalculator

class CustomMetrics(MetricsCalculator):
    def lpips(self, img1, img2):
        # Add LPIPS calculation
        pass
    
    def compute_all(self, original, reconstructed, bpp):
        metrics = super().compute_all(original, reconstructed, bpp)
        metrics['lpips'] = self.lpips(original, reconstructed)
        return metrics
```

### **Parallel Evaluation**

```python
from multiprocessing import Pool

def evaluate_image(args):
    model, img_path, quality = args
    # Evaluate single image
    return metrics

with Pool(8) as pool:
    results = pool.map(evaluate_image, tasks)
```

---

## 📚 References

1. **BD-Rate**: G. Bjøntegaard, "Calculation of average PSNR differences between RD-curves", VCEG-M33, 2001.
2. **MS-SSIM**: Z. Wang et al., "Multi-scale structural similarity for image quality assessment", Asilomar 2003.
3. **VTM**: H.266/VVC reference software
4. **BPG**: F. Bellard, Better Portable Graphics

---

## 🛠️ Troubleshooting

### **Issue: Interpolation failed**
**Solution:** Ensure at least 3 points per curve and overlapping ranges.

### **Issue: No SOTA results**
**Solution:** Add custom SOTA data in `comparator.py` or use your own reference curves.

### **Issue: Dashboard generation fails**
**Solution:** Install matplotlib: `pip install matplotlib pandas`

---

## 🗺️ Next Steps

- Publish results in paper/report
- Compare with more SOTA methods
- Add LPIPS and other perceptual metrics
- Generate LaTeX tables for publication

---

## 📧 Citation

```bibtex
@inproceedings{balanced2025,
  title={Balanced Rate-Distortion Optimization in Learned Image Compression},
  booktitle={CVPR},
  year={2025}
}
```

---

## 📚 Related Phases

- [Phase 1](../phase1/README.md): Basic Balanced R-D
- [Phase 2](../phase2/README.md): Adaptive optimization
- [Phase 3](../phase3/README.md): Hierarchical 5-task optimization
- [Phase 4](../phase4/README.md): Context-aware fine-tuning
- **Phase 5**: Comprehensive evaluation (this phase)
