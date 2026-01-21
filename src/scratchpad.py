import os
import shutil


def delete_folders_with_checkpoint_10(base_path="."):
    """
    Delete all first-level subfolders that contain a subfolder named 'checkpoint-10'

    Args:
        base_path (str): The base directory to search in (default: current directory)
    """
    try:
        # Get all items in the base directory
        items = os.listdir(base_path)

        # Filter to get only directories (first-level subfolders)
        first_level_dirs = [item for item in items
                            if os.path.isdir(os.path.join(base_path, item))]

        deleted_folders = []

        for folder in first_level_dirs:
            folder_path = os.path.join(base_path, folder)
            checkpoint_path = os.path.join(folder_path, "checkpoint-10")

            # Check if the checkpoint-10 subfolder exists
            if os.path.exists(checkpoint_path) and os.path.isdir(checkpoint_path):
                try:
                    # Delete the entire folder
                    shutil.rmtree(folder_path)
                    deleted_folders.append(folder)
                    print(f"Deleted folder: {folder}")
                except Exception as e:
                    print(f"Error deleting folder {folder}: {e}")

        if deleted_folders:
            print(f"\nTotal folders deleted: {len(deleted_folders)}")
            print("Deleted folders:", deleted_folders)
        else:
            print("No folders containing 'checkpoint-10' subfolder were found.")

    except Exception as e:
        print(f"Error accessing directory {base_path}: {e}")


if __name__ == "__main__":
    # You can specify a different base path here if needed
    base_directory = "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/focusreason/runs"  # Current directory

    # Ask for confirmation before proceeding
    response = input(f"This will delete all first-level subfolders in '{os.path.abspath(base_directory)}' "
                     f"that contain a 'checkpoint-10' subfolder. Continue? (y/N): ")

    if response.lower() in ['y', 'yes']:
        delete_folders_with_checkpoint_10(base_directory)
    else:
        print("Operation cancelled.")