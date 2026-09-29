import zipfile
import os
import shutil

src_zip = r"C:\Users\JAYLO\Downloads\oracle_phase3.zip"
temp_dir = "temp_ppo_extract"
dest_zip = "oracle_model.zip"

if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)
os.makedirs(temp_dir)

# Extract original
with zipfile.ZipFile(src_zip, 'r') as zf:
    zf.extractall(temp_dir)

# The original has a folder "oracle_phase3" inside
inner_dir = os.path.join(temp_dir, "oracle_phase3")
if not os.path.exists(inner_dir):
    inner_dir = temp_dir # maybe it wasn't nested

# Rezip correctly
with zipfile.ZipFile(dest_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(inner_dir):
        for file in files:
            file_path = os.path.join(root, file)
            # Add to zip without the parent directory path
            arcname = os.path.relpath(file_path, inner_dir)
            zf.write(file_path, arcname)

shutil.rmtree(temp_dir)
print("Repackaged PPO model successfully using Python zipfile!")
