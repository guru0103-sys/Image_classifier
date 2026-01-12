import os

# ================= CONFIGURATION =================
# PASTE THE FULL PATH TO THE FOLDER YOU WANT TO RENAME HERE
# Example: r"C:\Users\DELL\Desktop\Guru\Cdes\My_Dataset\Animals\Cats"
TARGET_FOLDER = r"Users\DELL\Desktop\Guru\Cdes\my_dataset\catE\Ai"

# What text do you want to add?
SUFFIX = "_ai" 
# =================================================

def bulk_rename():
    # Verify path exists
    if not os.path.exists(TARGET_FOLDER):
        print(" Error: That folder does not exist.")
        return

    count = 0
    print(f"Scanning: {TARGET_FOLDER}")

    for filename in os.listdir(TARGET_FOLDER):
        # Get full path
        old_path = os.path.join(TARGET_FOLDER, filename)

        # Skip if it's a folder (we only want files)
        if os.path.isdir(old_path):
            continue

        # Check file extension (images only)
        name, ext = os.path.splitext(filename)
        ext = ext.lower()
        
        if ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
            continue

        # Skip if already renamed (prevents cat_ai_ai.jpg)
        if name.endswith(SUFFIX):
            print(f"Skipping {filename} (already has {SUFFIX})")
            continue

        # Create new name
        new_name = f"{name}{SUFFIX}{ext}"
        new_path = os.path.join(TARGET_FOLDER, new_name)

        # Rename
        try:
            os.rename(old_path, new_path)
            print(f" Renamed: {filename} -> {new_name}")
            count += 1
        except Exception as e:
            print(f" Failed to rename {filename}: {e}")

    print("------------------------------------------------")
    print(f" Done! Renamed {count} files.")

if __name__ == "__main__":
    # DIRECTLY RUN THE FUNCTION (No input needed)
    print(" Starting bulk rename immediately...")
    bulk_rename()