import os

from PIL import Image
from tqdm import tqdm


def resize_images_in_place(folder_path, size=(512, 512), is_mask=False):
    if not os.path.exists(folder_path):
        print(f"[WARNING] Skipping non-existent folder: {folder_path}")
        return

    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(
        f"\n[INFO] Resizing {len(image_files)} images in-place in: {folder_path}")

    for filename in tqdm(image_files, desc=f"Resizing {os.path.basename(folder_path)}"):
        image_path = os.path.join(folder_path, filename)

        try:
            with Image.open(image_path) as img:
                if img.size == size:
                    continue

                img = img.convert("RGB") if not is_mask else img.convert("L")
                interpolation = Image.NEAREST if is_mask else Image.LANCZOS
                img = img.resize(size, interpolation)
                img.save(
                    image_path, format='JPEG' if not is_mask else 'PNG', quality=95)
        except Exception as e:
            print(f"[ERROR] Failed to process {filename}: {e}")


def main():
    resize_images_in_place('./data/ISIC_2020/train')

    resize_images_in_place('./data/ISIC_2019/train')

    resize_images_in_place('./data/ISIC_2018/train')

    print("\nAll images (and masks) have been resized in-place to 512x512.")


if __name__ == '__main__':
    main()
