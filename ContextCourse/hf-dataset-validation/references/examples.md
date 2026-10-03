# Usage Examples

## Validate a CSV file
```bash
python scripts/validate_dataset.py data/my_dataset.csv
```

Output:
```json
{
  "filepath": "data/my_dataset.csv",
  "format": "csv",
  "errors": [],
  "warnings": ["Found 2 duplicate rows"],
  "metadata": {
    "rows": 1000,
    "columns": ["text", "label", "split"],
    "missing_values": {"text": 0}
  }
}
```

## Validate in Python
```python
from pathlib import Path
import sys
sys.path.insert(0, "scripts")
from validate_dataset import validate_csv

report = validate_csv("data/my_dataset.csv")
print(report)
```