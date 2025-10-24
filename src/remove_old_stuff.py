import os
import re
import shutil
from pathlib import Path


def remove_global_step_files(root_path):
    """
    Cycles through a given path and removes global_stepX files from checkpoint-X folders.
    Only searches at depth 1 (immediate subdirectories).

    Args:
        root_path (str): The root directory path to search through
    """
    root_path = Path(root_path)

    if not root_path.exists():
        print(f"Error: Path '{root_path}' does not exist")
        return

    # Pattern to match checkpoint-X folders where X is an integer
    checkpoint_pattern = re.compile(r'^checkpoint-(\d+)$')

    # Iterate through immediate subdirectories only (depth 1)
    for major_item in root_path.iterdir():

        if major_item.is_dir():
            for item in major_item.iterdir():
                #print(item)
                folder_name = item.name

                # Check if folder matches checkpoint-X pattern
                match = checkpoint_pattern.match(folder_name)
                if match:
                    step_number = match.group(1)
                    print(f"Found checkpoint folder: {item}")

                    # Look for global_stepX files in this checkpoint folder
                    global_step_pattern = f"global_step{step_number}"

                    # Check all files in the checkpoint folder
                    for checkpoint_item in item.iterdir():
                        if checkpoint_item.name == global_step_pattern:
                            try:
                                if checkpoint_item.is_file():
                                    checkpoint_item.unlink()
                                    print(f"  Deleted file: {checkpoint_item}")
                                elif checkpoint_item.is_dir():
                                    shutil.rmtree(checkpoint_item)
                                    print(f"  Deleted directory: {checkpoint_item}")
                            except Exception as e:
                                print(f"  Error deleting {checkpoint_item}: {e}")


def main():
    """
    Main function to run the cleanup script.
    """
    # Get the path from user input
    #path = input("Enter the root path to search through: ").strip()
    path = "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/focusreason/runs"

    if not path:
        print("No path provided. Exiting.")
        return

    print(f"Searching for checkpoint folders in: {path}")
    print("Looking for global_stepX files to delete...")
    print("-" * 50)

    remove_global_step_files(path)

    print("-" * 50)
    print("Cleanup completed.")


if __name__ == "__main__":
    main()