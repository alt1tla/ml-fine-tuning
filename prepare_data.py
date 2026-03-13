from pathlib import Path
from PIL import Image

# пути
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# параметры
IMAGE_SIZE = (224, 224)
SUPPORTED_FORMATS = (".jpeg")


def process_image(input_path, output_path):
    """
    Открывает изображение, изменяет размер и сохраняет
    """
    with Image.open(input_path) as img:
        img = img.convert("RGB")
        img = img.resize(IMAGE_SIZE)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)


def prepare_dataset():
    classes = [p.name for p in RAW_DIR.iterdir() if p.is_dir()]

    print("Классы:", classes)

    for cls in classes:
        class_dir = RAW_DIR / cls
        output_class_dir = PROCESSED_DIR / cls

        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in SUPPORTED_FORMATS:

                output_path = output_class_dir / img_path.name

                process_image(img_path, output_path)

                print(f"Processed {img_path}")


if __name__ == "__main__":
    prepare_dataset()
