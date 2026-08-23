from pathlib import Path

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener


# Enable HEIC/HEIF support in Pillow
register_heif_opener()


# Folder containing the original HEIC images
SOURCE_FOLDER = Path("heic_images")

# JPG images will be saved here
OUTPUT_FOLDER = Path("jpg_images")

SUPPORTED_EXTENSIONS = {".heic", ".heif"}

converted_count = 0
skipped_count = 0
failed_count = 0


def convert_images():
    global converted_count, skipped_count, failed_count

    if not SOURCE_FOLDER.exists():
        print(f"Source folder not found: {SOURCE_FOLDER.resolve()}")
        return

    heic_files = [
        file
        for file in SOURCE_FOLDER.rglob("*")
        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    print("=" * 60)
    print("HEIC TO JPG BATCH CONVERTER")
    print("=" * 60)
    print(f"Images found: {len(heic_files)}")
    print(f"Source folder: {SOURCE_FOLDER.resolve()}")
    print(f"Output folder: {OUTPUT_FOLDER.resolve()}")
    print("-" * 60)

    for index, source_path in enumerate(heic_files, start=1):
        try:
            # Preserve category subfolders such as phone, cup and bottle
            relative_path = source_path.relative_to(SOURCE_FOLDER)
            output_path = OUTPUT_FOLDER / relative_path.with_suffix(".jpg")

            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_path.exists():
                skipped_count += 1
                print(
                    f"[{index}/{len(heic_files)}] "
                    f"Skipped: {output_path.name}"
                )
                continue

            with Image.open(source_path) as image:
                # Correct image rotation using EXIF information
                image = ImageOps.exif_transpose(image)

                # JPEG requires RGB instead of HEIC transparency modes
                image = image.convert("RGB")

                image.save(
                    output_path,
                    format="JPEG",
                    quality=95,
                    optimize=True,
                )

            converted_count += 1

            print(
                f"[{index}/{len(heic_files)}] "
                f"Converted: {relative_path}"
            )

        except Exception as error:
            failed_count += 1
            print(
                f"[{index}/{len(heic_files)}] "
                f"FAILED: {source_path.name}"
            )
            print(f"Reason: {error}")

    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"Converted : {converted_count}")
    print(f"Skipped   : {skipped_count}")
    print(f"Failed    : {failed_count}")
    print(f"JPG folder: {OUTPUT_FOLDER.resolve()}")


if __name__ == "__main__":
    convert_images()