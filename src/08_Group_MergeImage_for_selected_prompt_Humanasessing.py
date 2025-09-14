from pathlib import Path
import shutil
import os

# Define folders
base_input_folder = Path(r'report/jpg')
reference_folder = base_input_folder / 'ABC001'
os.makedirs(reference_folder, exist_ok=True)
output_folder = base_input_folder / 'merge'

# Create output folder if it doesn't exist
output_folder.mkdir(parents=True, exist_ok=True)

# Check if reference folder exists
if not reference_folder.exists():
    print(f"Error: Reference folder does not exist: {reference_folder}")
    print("Please verify that the ABC001 folder exists in report\\jpg\\")
else:
    # Get all reference file names from ABC001
    reference_files = [f.name for f in reference_folder.iterdir() 
                      if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg']]
    
    if not reference_files:
        print(f"No JPG files found in reference folder: {reference_folder}")
    else:
        print(f"Processing {len(reference_files)} reference files from ABC001...")
        
        # For each reference file, create a folder and search for matching files
        for ref_file in reference_files:
            # Create folder name using reference file name without extension
            ref_name_no_extension = Path(ref_file).stem  # e.g., "row31" from "row31.jpg"
            ref_folder = output_folder / ref_name_no_extension
            ref_folder.mkdir(exist_ok=True)
            
            print(f"\nSearching for files starting with '{ref_name_no_extension}'...")
            
            # Search all subfolders for files that START with this reference name
            for root, dirs, files in os.walk(base_input_folder):
                current_folder = Path(root)
                current_folder_name = current_folder.name
                
                # Skip reference folder and output folder
                if current_folder == reference_folder or current_folder == output_folder:
                    continue
                
                # Look for files that START with the reference filename (without extension)
                for filename in files:
                    if (filename.lower().endswith(('.jpg', '.jpeg')) and 
                        filename.startswith(ref_name_no_extension)):
                        
                        source_file = current_folder / filename
                        
                        # Create new filename: "SourceFolder original_filename"
                        new_filename = f"{current_folder_name} {filename}"
                        destination_file = ref_folder / new_filename
                        
                        try:
                            if not destination_file.exists():
                                shutil.copy2(source_file, destination_file)
                                print(f"  ✓ Copied: {filename} -> {new_filename}")
                            else:
                                print(f"  - Already exists: {new_filename}")
                        except Exception as e:
                            print(f"  ✗ Error copying {filename}: {e}")
        
        print(f"\nAll operations completed! Check output folder: {output_folder}")
