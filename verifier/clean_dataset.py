import os
import cv2

BASE_DIR = "data"
classes = ["normal", "pothole"]

for cls in classes:
    folder = os.path.join(BASE_DIR, cls)
    print(f"\nChecking {folder}")

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        try:
            # Use absolute path for robustness
            abs_path = os.path.abspath(path)
            # Prepend \\?\ on Windows for long path support
            if os.name == 'nt' and not abs_path.startswith('\\\\?\\'):
                long_path = '\\\\?\\' + abs_path
            else:
                long_path = abs_path
            
            # Check for long specific filename and rename if necessary (to fix train_cnn.py issues)
            if len(file) > 64: # If filename is extremely long
                 import uuid
                 ext = os.path.splitext(file)[1]
                 new_name = f"{uuid.uuid4().hex[:8]}{ext}"
                 new_path = os.path.join(folder, new_name)
                 new_long_path = os.path.abspath(new_path)
                 if os.name == 'nt' and not new_long_path.startswith('\\\\?\\'):
                     new_long_path = '\\\\?\\' + new_long_path
                 
                 print(f"🔄 Renaming long file: {file[:20]}... -> {new_name}")
                 os.rename(long_path, new_long_path)
                 
                 # Update path variables for verification step
                 path = new_path
                 abs_path = new_long_path
                 # We don't need to re-prepend \\?\ as new_long_path already has it or is short enough

            # Now verify the image (using the potentially new path)
            img = cv2.imread(abs_path)
            if img is None:
                print(f"❌ Removing unreadable: {path}")
                os.remove(abs_path)
                
        except Exception as e:
            print(f"❌ Error processing: {file} | {e}")
            try:
                if 'long_path' in locals() and os.path.exists(long_path):
                     os.remove(long_path)
                     print("   Forced removal of broken file.")
            except:
                pass

print("\n✅ Dataset cleaned successfully")