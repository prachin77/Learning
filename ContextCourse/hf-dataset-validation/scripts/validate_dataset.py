Set-Content scripts/validate_dataset.py @'
import csv
import sys

required_columns = ["text", "label", "split"]

if len(sys.argv) != 2:
    print("Usage: py scripts/validate_dataset.py <csv_file>")
    sys.exit(1)

file_path = sys.argv[1]

try:
    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames != required_columns:
            print(f"ERROR: Expected columns: {required_columns}")
            print(f"Found: {reader.fieldnames}")
            sys.exit(1)

        rows = list(reader)

        if not rows:
            print("ERROR: Dataset is empty.")
            sys.exit(1)

        for i, row in enumerate(rows, start=2):
            if not row["text"].strip():
                print(f"ERROR: Empty text at row {i}")
                sys.exit(1)

            if row["label"] not in {"0", "1"}:
                print(f"ERROR: Invalid label at row {i}: {row["label"]}")
                sys.exit(1)

            if row["split"] not in {"train", "test"}:
                print(f"ERROR: Invalid split at row {i}: {row["split"]}")
                sys.exit(1)

        print(f"Dataset valid: {len(rows)} rows")

except FileNotFoundError:
    print(f"ERROR: File not found: {file_path}")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
'@
