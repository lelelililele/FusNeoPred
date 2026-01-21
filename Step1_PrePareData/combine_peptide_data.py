#!/usr/bin/env python3
"""
Integrate S1_coverage, S3_Cppred, S4_HLA, and S5_MHCScore files
Output complete information for each S5 peptide
"""

import re
import pandas as pd
from collections import defaultdict

def parse_s3_cppred(filepath):
    """Parse S3_Cppred file"""
    data = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    header = lines[0].strip().replace('#', '').split('\t')
    
    for line in lines[1:]:
        if line.strip():
            parts = line.strip().split('\t')
            if len(parts) > 0:
                s3_id = parts[0]
                data[s3_id] = parts[1:]  # Store all columns except ID
    
    print(f"S3_Cppred: Loaded {len(data)} records")
    return data, header[1:]

def parse_s1_coverage(filepath):
    """Parse S1_coverage file"""
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    s1_id = parts[0]
                    data[s1_id] = parts[1:3]  # Take only first two columns
    
    print(f"S1_coverage: Loaded {len(data)} records")
    return data

def parse_s4_hla(filepath):
    """Parse S4_HLA file"""
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) >= 7:  # ID + 6 HLA values
                    hla_id = parts[0]  # T1, T2, etc.
                    data[hla_id] = parts[1:7]  # Take 6 HLA values
    
    print(f"S4_HLA: Loaded {len(data)} HLA types")
    return data

def parse_s5_mhcscore(filepath):
    """Parse S5_MHCScore file, extract peptides and ORF IDs"""
    peptides = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    hla_type = parts[0]  # First column: HLA type
                    peptide_name = parts[1]  # Second column: peptide name
                    other_cols = parts[2:]  # Remaining columns
                    
                    # Extract numeric ID from peptide name
                    # Look for patterns like "11_" or "_11_"
                    orf_id = None
                    
                    # Try multiple pattern matches
                    patterns = [
                        r'_(\d+)_',  # Numbers surrounded by underscores
                        r'^(\d+)_',  # Numbers at the beginning
                        r'_(\d+)$',  # Numbers at the end
                        r'^(\d+)',   # Pure numbers at the beginning
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, peptide_name)
                        if match:
                            orf_id = match.group(1)
                            break
                    
                    # If no match found, try to find numbers in the whole string
                    if orf_id is None:
                        all_numbers = re.findall(r'\d+', peptide_name)
                        if all_numbers:
                            orf_id = all_numbers[0]  # Take the first number
                        else:
                            orf_id = "unknown"
                    
                    peptides.append({
                        'hla_type': hla_type,
                        'peptide_name': peptide_name,
                        'orf_id': orf_id,
                        'other_columns': other_cols
                    })
    
    print(f"S5_MHCScore: Parsed {len(peptides)} peptides")
    return peptides

def combine_data():
    """Integrate all data"""
    print("Starting data integration...")
    
    # Load all files
    s1_data = parse_s1_coverage("S1_coverage")
    s3_data, s3_columns = parse_s3_cppred("S3_Cppred")
    s4_data = parse_s4_hla("S4_HLA")
    s5_peptides = parse_s5_mhcscore("S5_MHCScore")
    
    # Statistics
    matched_s1 = 0
    matched_s3 = 0
    matched_s4 = 0
    fully_matched = 0
    
    # Prepare output lines
    output_lines = []
    
    # Create header
    header_parts = ["Peptide_ID", "Peptide_Name"]
    
    # S1's two columns
    header_parts.extend(["S1_Coverage1", "S1_Coverage2"])
    
    # S3's all columns
    header_parts.extend([f"S3_{col}" for col in s3_columns])
    
    # S4's 6 columns
    header_parts.extend([f"S4_HLA{i+1}" for i in range(6)])
    
    # S5's other columns
    if s5_peptides:
        num_s5_cols = len(s5_peptides[0]['other_columns'])
        header_parts.extend([f"S5_Col{i+1}" for i in range(num_s5_cols)])
    
    output_lines.append("\t".join(header_parts))
    
    # Process each peptide
    for idx, peptide in enumerate(s5_peptides, 1):
        hla_type = peptide['hla_type']
        peptide_name = peptide['peptide_name']
        orf_id = peptide['orf_id']
        other_cols = peptide['other_columns']
        
        # Get S1 data
        s1_info = s1_data.get(orf_id, ["NA", "NA"])
        if s1_info != ["NA", "NA"]:
            matched_s1 += 1
        
        # Get S3 data
        s3_info = s3_data.get(orf_id, ["NA"] * len(s3_columns))
        if s3_info != ["NA"] * len(s3_columns):
            matched_s3 += 1
        
        # Get S4 data
        s4_info = s4_data.get(hla_type, ["NA"] * 6)
        if s4_info != ["NA"] * 6:
            matched_s4 += 1
        
        # Check for full match
        if (s1_info != ["NA", "NA"] and s3_info != ["NA"] * len(s3_columns) and 
            s4_info != ["NA"] * 6):
            fully_matched += 1
        
        # Build output row
        row_parts = [hla_type, peptide_name]
        row_parts.extend(s1_info)
        row_parts.extend(s3_info)
        row_parts.extend(s4_info)
        row_parts.extend(other_cols)
        
        # Ensure all values are strings
        row_parts = [str(x) for x in row_parts]
        output_lines.append("\t".join(row_parts))
    
    # Save to file
    output_file = "S5_combined_complete.tsv"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(output_lines))
    
    # Print statistics
    print("\n" + "="*50)
    print("Data integration complete!")
    print(f"Output file: {output_file}")
    print(f"Total peptides: {len(s5_peptides)}")
    print(f"Peptides matching S1: {matched_s1}/{len(s5_peptides)}")
    print(f"Peptides matching S3: {matched_s3}/{len(s5_peptides)}")
    print(f"Peptides matching S4: {matched_s4}/{len(s5_peptides)}")
    print(f"Fully matched peptides: {fully_matched}/{len(s5_peptides)}")
    
    # Show some examples
    if s5_peptides:
        print("\nORF ID matching for first 5 peptides:")
        for i in range(min(5, len(s5_peptides))):
            peptide = s5_peptides[i]
            print(f"  {i+1}. {peptide['peptide_name']} -> ORF ID: {peptide['orf_id']}")

def main():
    """Main function"""
    print("="*50)
    print("Peptide Information Integration Program")
    print("="*50)
    
    # Check if files exist
    required_files = ["S1_coverage", "S3_Cppred", "S4_HLA", "S5_MHCScore"]
    missing_files = []
    
    for filename in required_files:
        try:
            with open(filename, 'r'):
                pass
        except FileNotFoundError:
            missing_files.append(filename)
    
    if missing_files:
        print("Error: The following files are missing:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease ensure all files are in the current directory:")
        print("  S1_coverage, S3_Cppred, S4_HLA, S5_MHCScore")
        return
    
    # Execute integration
    combine_data()

if __name__ == "__main__":
    main()