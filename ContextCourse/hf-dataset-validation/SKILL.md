---
name: "hf-dataset-validation"
description: "Validate Hugging Face datasets for schema, format, and data quality issues. Use when checking datasets before publishing, training, or sharing."
license: "MIT"
compatibility: "Python 3.8+, requires pandas and datasets"
metadata:
  author: "your-username"
  version: "1.0.0"
  created: "2026-04-13"
---

# Dataset Validation Skill

## Overview

This skill teaches agents how to validate Hugging Face datasets for:
- **Schema validation**: Correct columns and data types
- **Data quality**: Missing values, duplicates, outliers
- **Format compliance**: CSV, Parquet, Arrow, JSON formats
- **Size checks**: File sizes, record counts, memory footprint

Use this skill before publishing datasets or using them for training.

## Prerequisites Checklist

- [ ] Python 3.8 or higher installed
- [ ] pandas library installed (`pip install pandas`)
- [ ] datasets library installed (`pip install datasets`)
- [ ] Dataset file accessible locally

## Step-by-Step Guide

### Step 1: Install Dependencies

```bash
pip install pandas datasets numpy
```

### Step 2: Check Dataset Format

Identify what format your dataset is in:

```python
from pathlib import Path

file_path = "data/my_dataset.csv"

# Check file extension
if file_path.endswith('.csv'):
    print("Format: CSV")
elif file_path.endswith('.parquet'):
    print("Format: Parquet")
elif file_path.endswith('.json'):
    print("Format: JSON")
else:
    print("Unknown format")
```

### Step 3: Load and Inspect

Load your dataset and check basic properties:

```python
import pandas as pd

# Load CSV dataset
df = pd.read_csv("data/my_dataset.csv")

# Check shape and columns
print(f"Shape: {df.shape}")  # (rows, columns)
print(f"Columns: {list(df.columns)}")
print(f"Data types:\\n{df.dtypes}")
```

### Step 4: Validate Schema

Check that your dataset has expected columns and correct data types:

```python
# Define expected schema
expected_columns = {'text', 'label', 'split'}
actual_columns = set(df.columns)

# Check all required columns exist
missing = expected_columns - actual_columns
if missing:
    print(f"ERROR: Missing columns: {missing}")

# Check for unexpected columns
extra = actual_columns - expected_columns
if extra:
    print(f"WARNING: Extra columns: {extra}")

# Verify data types
if df['label'].dtype not in ['int64', 'object']:
    print("WARNING: label column should be integer or string")
```

### Step 5: Check Data Quality

Identify common data quality issues:

```python
# Check for missing values
missing_count = df.isna().sum()
if missing_count.any():
    print("Missing values found:")
    print(missing_count[missing_count > 0])

# Check for duplicates
duplicates = df.duplicated().sum()
if duplicates > 0:
    print(f"WARNING: {duplicates} duplicate rows found")

# Check for empty strings
for col in df.select_dtypes(include='object').columns:
    empty = (df[col].str.strip() == '').sum()
    if empty > 0:
        print(f"WARNING: Column '{col}' has {empty} empty strings")
```

### Step 6: Generate Validation Report

Use the provided validation script (see references below):

```bash
python scripts/validate_dataset.py data/my_dataset.csv
```

The report includes:
- Dataset summary (rows, columns, size)
- Data quality metrics
- Issues found and recommendations
- Remediation suggestions

## Common Issues and Solutions

### Issue: Encoding Error Reading CSV

**Problem**: `UnicodeDecodeError` when loading file

**Solution**: Try different character encodings
```python
encodings = ['utf-8', 'latin-1', 'iso-8859-1']
for encoding in encodings:
    try:
        df = pd.read_csv("file.csv", encoding=encoding)
        print(f"Success with {encoding}")
        break
    except:
        continue
```

### Issue: Memory Error with Large Files

**Problem**: `MemoryError` when loading large CSV

**Solution**: Read in chunks
```python
chunks = pd.read_csv("large_file.csv", chunksize=10000)
for i, chunk in enumerate(chunks):
    print(f"Processing chunk {i}...")
    # Process each chunk
```

### Issue: Inconsistent Column Names

**Problem**: Column names have mixed capitalization or spaces

**Solution**: Standardize column names
```python
df.columns = df.columns.str.lower().str.replace(' ', '_')
print(df.columns)
```

## Helper Scripts

### scripts/validate_dataset.py

```python
#!/usr/bin/env python3
"""Validate Hugging Face datasets."""

import json
import sys
import pandas as pd
from pathlib import Path

def validate_csv(filepath):
    """Validate a CSV dataset file."""
    
    errors = []
    warnings = []
    report = {
        "filepath": filepath,
        "format": "csv",
        "errors": errors,
        "warnings": warnings,
        "metadata": {}
    }
    
    # Check file exists
    if not Path(filepath).exists():
        errors.append(f"File not found: {filepath}")
        return report
    
    try:
        # Load file
        df = pd.read_csv(filepath)
        report["metadata"]["rows"] = len(df)
        report["metadata"]["columns"] = list(df.columns)
        report["metadata"]["dtypes"] = {k: str(v) for k, v in df.dtypes.items()}
        
        # Check for missing values
        missing = df.isna().sum()
        if missing.any():
            report["metadata"]["missing_values"] = missing.to_dict()
        
        # Check for duplicates
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            warnings.append(f"Found {dup_count} duplicate rows")
        
        # Check for empty strings
        for col in df.select_dtypes(include='object').columns:
            empty = (df[col].str.strip() == '').sum()
            if empty > 0:
                warnings.append(f"Column '{col}' has {empty} empty strings")
        
    except Exception as e:
        errors.append(f"Error reading file: {str(e)}")
    
    return report

def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_dataset.py <filepath>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    report = validate_csv(filepath)
    print(json.dumps(report, indent=2))
    
    if report["errors"]:
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### scripts/generate_report.py

```python
#!/usr/bin/env python3
"""Generate a human-readable validation report."""

import sys
import pandas as pd
from pathlib import Path

def generate_report(filepath):
    """Generate a text report for a dataset."""
    
    report = []
    report.append("=" * 60)
    report.append("DATASET VALIDATION REPORT")
    report.append("=" * 60)
    report.append(f"\nFile: {filepath}\n")
    
    if not Path(filepath).exists():
        report.append(f"ERROR: File not found: {filepath}")
        return "\n".join(report)
    
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        report.append(f"ERROR: Could not read file: {e}")
        return "\n".join(report)
    
    # Basic stats
    report.append("BASIC STATISTICS")
    report.append(f"  Rows: {len(df)}")
    report.append(f"  Columns: {len(df.columns)}")
    report.append(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Column info
    report.append("\nCOLUMN INFORMATION")
    for col in df.columns:
        dtype = df[col].dtype
        non_null = df[col].notna().sum()
        report.append(f"  {col}: {dtype} ({non_null}/{len(df)} non-null)")
    
    # Data quality
    report.append("\nDATA QUALITY")
    missing = df.isna().sum().sum()
    report.append(f"  Missing values: {missing}")
    duplicates = df.duplicated().sum()
    report.append(f"  Duplicate rows: {duplicates}")
    
    # Recommendations
    report.append("\nRECOMMENDATIONS")
    if missing > 0:
        report.append("  - Handle missing values (drop or impute)")
    if duplicates > 0:
        report.append("  - Remove duplicate rows")
    report.append("  - Document dataset and preprocessing steps")
    report.append("  - Add LICENSE and README.md files")
    
    report.append("\n" + "=" * 60)
    return "\n".join(report)

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_report.py <filepath>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    print(generate_report(filepath))

if __name__ == "__main__":
    main()
```