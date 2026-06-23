import os
import sys
import pandas as pd

def main():
    submission_path = "submission.csv"
    
    print("==================================================")
    print("      SautiEdge Submission Format Validator       ")
    print("==================================================")
    
    # 1. Check if the submission file exists
    if not os.path.exists(submission_path):
        print(f"ERROR: Submission file '{submission_path}' does not exist!")
        print("Please run the ASR inference pipeline to generate it.")
        sys.exit(1)
        
    try:
        # Load the CSV
        df = pd.read_csv(submission_path)
    except Exception as e:
        print(f"ERROR: Failed to read '{submission_path}'. Error: {e}")
        sys.exit(1)
        
    errors = []
    
    # 2. Check column count and exact names
    expected_cols = ["id", "language", "transcription"]
    actual_cols = list(df.columns)
    
    if len(actual_cols) != 3:
        errors.append(f"Invalid column count. Expected 3 columns ({expected_cols}), found {len(actual_cols)} columns: {actual_cols}")
    else:
        for idx, col in enumerate(expected_cols):
            if actual_cols[idx] != col:
                errors.append(f"Column index {idx} mismatch. Expected '{col}', found '{actual_cols[idx]}'.")
                
    # 3. Check for empty submission file
    if len(df) == 0:
        errors.append("The submission file is empty (contains 0 records).")
        
    # If basic structure is incorrect, exit early before checking content
    if errors:
        print("\nStructure Validation Failed!")
        for error in errors:
            print(f" - {error}")
        sys.exit(1)
        
    # 4. Check Language Codes (must be strict 3-letter ISO)
    allowed_langs = {"swa", "kik", "luo", "som", "mas", "kln"}
    unique_langs = set(df["language"].dropna().unique())
    invalid_langs = unique_langs - allowed_langs
    
    if invalid_langs:
        errors.append(f"Invalid ISO language codes detected: {invalid_langs}. Allowed codes are strictly: {allowed_langs}")
        
    # 5. Check for null or empty values in transcription
    # We check if there are empty values or NaN
    null_trans_count = df["transcription"].isna().sum()
    if null_trans_count > 0:
        errors.append(f"Detected {null_trans_count} null/missing transcriptions. All audio items must have a transcription.")
        
    # Also check for empty string values (whitespace-only)
    if "transcription" in df.columns:
        empty_str_mask = df["transcription"].astype(str).str.strip() == ""
        empty_str_count = empty_str_mask.sum() - null_trans_count  # avoid double counting
        if empty_str_count > 0:
            errors.append(f"Detected {empty_str_count} empty (blank) text transcriptions.")
            
    # 6. Check for duplicate IDs
    duplicate_count = df["id"].duplicated().sum()
    if duplicate_count > 0:
        errors.append(f"Detected {duplicate_count} duplicate audio ID entries. Each ID must be unique.")
        
    # 7. Print Final Status
    if errors:
        print("\nContent Validation Failed! The following errors were found:")
        for idx, error in enumerate(errors, 1):
            print(f"{idx}. {error}")
        print("\nPlease fix the pipeline issues and regenerate the submission.")
        sys.exit(1)
    else:
        print("\nStructure: OK")
        print("Column names: OK")
        print(f"Row count: {len(df)}")
        print(f"Unique language codes: {unique_langs}")
        print("Duplicate check: OK")
        print("Null/Empty checks: OK")
        print("\n==================================================")
        print(" SUCCESS: SUBMISSION IS VALID AND KAGGLE-READY!   ")
        print("==================================================")
        sys.exit(0)

if __name__ == "__main__":
    main()
