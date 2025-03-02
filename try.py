import os
import sys

def rename_json_files(base_dir, new_name):
    """
    Rename all .json files in the given directory and its subdirectories to a new name.
    
    Parameters:
    - base_dir: The base directory to start searching from
    - new_name: The new name to replace the filename (all files will have the same name)
    """
    count = 0
    
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(base_dir):
        for filename in files:
            # Check if the file has a .json extension
            if filename.endswith('.json'):
                # Get the file path
                file_path = os.path.join(root, filename)
                
                # Use the new name directly, ensuring it has the .json extension
                new_filename = f"{new_name}.json"
                
                # Create the new file path
                new_file_path = os.path.join(root, new_filename)
                
                # Check if the new file path already exists and it's not the same file
                if os.path.exists(new_file_path) and file_path != new_file_path:
                    # Remove the existing file
                    os.remove(new_file_path)
                    print(f"Removed existing file: {new_file_path}")
                
                # Rename the file
                os.rename(file_path, new_file_path)
                print(f"Renamed: {file_path} -> {new_file_path}")
                count += 1
    
    print(f"Total files renamed: {count}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python rename_json_files.py <base_directory> <new_name>")
        sys.exit(1)
    
    base_dir = sys.argv[1]
    new_name = sys.argv[2]
    
    rename_json_files(base_dir, new_name)