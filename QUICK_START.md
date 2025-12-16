# Quick Start Guide - Product Recommendation System

## Prerequisites

1. **Python 3.8+** installed
2. **Dataset files** in the `data/` folder:
   - `data/AMAZON_FASHION_5.json.gz`
   - `data/meta_AMAZON_FASHION.json.gz`

## Installation

```bash
# Navigate to project directory
cd "/Users/DK19/Downloads/Product Recommendation"

# Install dependencies
pip install -r requirements.txt
```

Or if using Python 3 specifically:
```bash
python3 -m pip install -r requirements.txt
```

## Running the Complete Pipeline

### Option 1: Run All Steps Sequentially

```bash
# Step 1: Explore and understand the data
python3 01_data_exploration.py

# Step 2: Prepare and split the data
python3 02_data_preparation.py

# Step 3: Train baseline models
python3 03_baseline_models.py

# Step 4: Train text-based model
python3 04_text_based_model.py

# Step 5: Extract image features
python3 05_image_encoder.py

# Step 6: Train multimodal model
python3 06_multimodal_model.py

# Step 7: Evaluate all models
python3 07_evaluation.py

# Step 8: Generate documentation
python3 08_documentation.py
```

### Option 2: Run All Steps at Once (Bash Script)

Create a file `run_all.sh`:
```bash
#!/bin/bash
python3 01_data_exploration.py
python3 02_data_preparation.py
python3 03_baseline_models.py
python3 04_text_based_model.py
python3 05_image_encoder.py
python3 06_multimodal_model.py
python3 07_evaluation.py
python3 08_documentation.py
```

Then run:
```bash
chmod +x run_all.sh
./run_all.sh
```

### Option 3: Python Script to Run All

Create `run_pipeline.py`:
```python
import subprocess
import sys

scripts = [
    "01_data_exploration.py",
    "02_data_preparation.py",
    "03_baseline_models.py",
    "04_text_based_model.py",
    "05_image_encoder.py",
    "06_multimodal_model.py",
    "07_evaluation.py",
    "08_documentation.py"
]

for script in scripts:
    print(f"\n{'='*60}")
    print(f"Running {script}...")
    print('='*60)
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"Error running {script}")
        sys.exit(1)
    print(f"✓ {script} completed successfully")

print("\n" + "="*60)
print("🎉 All steps completed successfully!")
print("="*60)
```

Run with:
```bash
python3 run_pipeline.py
```

## Expected Output Files

After running the complete pipeline, you should have:

### Generated Data Files:
- `prepared_data.pkl` - Preprocessed data (from step 2)
- `baseline_results.pkl` - Baseline model results (from step 3)
- `text_encoder_results.pkl` - Text model results (from step 4)
- `image_encoder_data.pkl` - Image embeddings (from step 5)
- `multimodal_results.pkl` - Multimodal model results (from step 6)

### Results Tables:
- `model_comparison.csv` - Full comparison table
- `test_results_summary.csv` - Test set summary

### Documentation:
- `PROJECT_REPORT.md` - Comprehensive report
- `ARCHITECTURE.md` - System architecture
- `VIVA_PREPARATION.md` - Defense guide
- `RESUME_DESCRIPTION.txt` - Resume description

## Time Estimates

- **Step 1** (Data Exploration): ~30 seconds
- **Step 2** (Data Preparation): ~10 seconds
- **Step 3** (Baseline Models): ~1-2 minutes
- **Step 4** (Text Model): ~5-10 minutes (downloads DistilBERT first time)
- **Step 5** (Image Encoder): ~1-2 minutes (downloads ResNet18 first time)
- **Step 6** (Multimodal Model): ~5-10 minutes
- **Step 7** (Evaluation): ~30 seconds
- **Step 8** (Documentation): ~5 seconds

**Total time: ~15-25 minutes** (first run, includes model downloads)

## Troubleshooting

### Issue: ModuleNotFoundError
**Solution:** Install missing packages
```bash
pip install pandas numpy torch transformers torchvision tqdm scikit-learn Pillow requests
```

### Issue: Out of Memory
**Solution:** The project is designed to be memory-efficient. If issues persist:
- Close other applications
- Reduce batch sizes in training scripts
- Use CPU instead of GPU (default)

### Issue: Image Download Fails
**Solution:** Some image URLs may be broken. The script handles this gracefully and continues.

### Issue: Pickle File Errors
**Solution:** Delete old pickle files and re-run from step 2:
```bash
rm *.pkl
python3 02_data_preparation.py
```

## Getting Recommendations

After training, you can use the models for recommendations:

1. **Load the trained model** from saved pickle files
2. **Encode new product text/images** using the encoders
3. **Get user embeddings** from the trained model
4. **Score items** using the fusion model
5. **Rank and return top-K** recommendations

See the evaluation script (`07_evaluation.py`) for examples of how models are used for recommendations.

## Next Steps

1. Review `PROJECT_REPORT.md` for detailed methodology
2. Check `ARCHITECTURE.md` for system design
3. Use `RESUME_DESCRIPTION.txt` for your resume
4. Prepare for defense using `VIVA_PREPARATION.md`

