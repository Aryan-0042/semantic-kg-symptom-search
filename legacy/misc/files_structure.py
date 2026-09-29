import os

def print_directory_tree(start_path=".", ignore_dirs=None):
    """
    Recursively prints directory structure while skipping specified directories.
    """
    if ignore_dirs is None:
        ignore_dirs = {'venv', '.venv', '__pycache__', '.git'}

    for root, dirs, files in os.walk(start_path):
        # Modify dirs in-place so os.walk doesn't descend into ignored folders
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        # Calculate indentation level relative to start_path
        rel_path = os.path.relpath(root, start_path)
        level = 0 if rel_path == '.' else rel_path.count(os.sep) + 1
        indent = ' ' * 4 * level
        
        # Print current folder name
        folder_name = os.path.basename(os.path.abspath(root))
        print(f"{indent}📂 {folder_name}/")
        
        # Print files inside current folder
        sub_indent = ' ' * 4 * (level + 1)
        for file in files:
            print(f"{sub_indent}📜 {file}")

if __name__ == "__main__":
    print_directory_tree(".")