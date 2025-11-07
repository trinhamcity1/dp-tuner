from dp_tuner import main
from pathlib import Path
import pandas as pd

if __name__ == "__main__":

    # The transformer only takes 1 single file under input_data folder
    # Checking the file availability in the working folder
    folder = Path("input_data")
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found or not a directory: {folder}")
    # Case-insensitive .csv, ignore hidden/temp files
    csvs = [
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".csv"
        and not p.name.startswith(".")
        and not p.name.endswith("~")
    ]

    if len(csvs) == 0:
        raise FileNotFoundError(f"No CSV files found in: {folder}")
    if len(csvs) >= 2:
        raise ValueError(f"Expected exactly 1 CSV, found {len(csvs)} in: {folder}")

        # Read only the header row and selected the output column
    try:
        cols = pd.read_csv(csvs[0], nrows=0).columns
    except UnicodeDecodeError:
        # Fallback encodings if needed
        for enc in ("utf-8-sig", "latin-1", "utf-16"):
            try:
                cols = pd.read_csv(csvs[0], nrows=0, encoding=enc).columns
                break
            except Exception:
                continue
        else:
            raise

    if len(cols) == 0:
        raise ValueError(f"No columns found in: {csvs[0]}")

    output_column = cols[-1]

    main(
        b=512,
        c=1.6,
        epochs=150,
        delta=None,
        sigma_grid=[1.2, 1.4, 1.6, 1.8, 2.0],
        seeds=[0, 1],
        gen_kind="dp_ctgan",
        external_data_source=True,
        external_data_path=csvs[0],
        label_col = output_column
        )