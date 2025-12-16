#!/usr/bin/env python3
"""
Run Complete Pipeline
=====================
Executes all steps of the product recommendation pipeline in sequence.
"""

import subprocess
import sys
import os

def run_script(script_name):
    """Run a Python script and return success status."""
    if not os.path.exists(script_name):
        print(f"❌ Error: {script_name} not found!")
        return False
    
    print(f"\n{'='*60}")
    print(f"Running {script_name}...")
    print('='*60)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False
        )
        print(f"✓ {script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {script_name}: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted by user. Stopping pipeline.")
        return False


def main():
    """Main function to run all pipeline steps."""
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
    
    print("="*60)
    print("PRODUCT RECOMMENDATION PIPELINE")
    print("="*60)
    print("\nThis will run all 8 steps of the pipeline.")
    print("Estimated time: 15-25 minutes (first run)")
    print("\nPress Ctrl+C to stop at any time.\n")
    
    for i, script in enumerate(scripts, 1):
        print(f"\n[{i}/{len(scripts)}] {script}")
        success = run_script(script)
        
        if not success:
            print(f"\n❌ Pipeline stopped at {script}")
            print("Please fix the error and re-run from this step.")
            sys.exit(1)
    
    print("\n" + "="*60)
    print("🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nGenerated files:")
    print("  - prepared_data.pkl")
    print("  - baseline_results.pkl")
    print("  - text_encoder_results.pkl")
    print("  - image_encoder_data.pkl")
    print("  - multimodal_results.pkl")
    print("  - model_comparison.csv")
    print("  - PROJECT_REPORT.md")
    print("\nCheck PROJECT_REPORT.md for results and analysis.")


if __name__ == "__main__":
    main()

