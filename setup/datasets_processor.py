from pathlib import Path
from typing import Dict
import albumentations as A
import cv2


class VideoProcessor:
    """Handles video processing operations."""

    @staticmethod
    def exctract_bbox_text(
        video_path: str,
        prefix: str,
        output_dir: str,
        interval_sec: int,
        num_images: int,
    ) -> None:
        """Extract bounding box text from video at specified intervals."""
        # not yet created..

    @staticmethod
    def extract_frames(
        video_path: str,
        prefix: str,
        output_dir: str,
        interval_sec: int,
        num_images: int,
    ) -> None:
        """Extract frames from video at specified intervals."""
        video = cv2.VideoCapture(video_path)
        fps = int(video.get(cv2.CAP_PROP_FPS))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        frame_positions = [
            int((start_time + i * (interval_sec / num_images)) * fps)
            for start_time in range(0, int(duration), interval_sec)
            for i in range(num_images)
        ]

        extracted_count = 0
        current_frame = 0

        while True:
            ret, frame = video.read()
            if not ret:
                break

            if current_frame in frame_positions:
                frame_path = output_dir / f"{prefix}_frame_{extracted_count:04d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                print(f"Saved frame: {frame_path}")
                extracted_count += 1

            current_frame += 1
        video.release()
        print("Extraction complete.")


class ImageAugmenter:
    """Handles image augmentation operations using albumentations library."""

    def __init__(self):
        self.transformations = {
            "blur": A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            "noise": A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            "blur_noise": A.Compose(
                [
                    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                ]
            ),
        }

    def apply_augmentations(self, image: cv2.Mat) -> Dict[str, cv2.Mat]:
        """Apply all defined augmentations to an image."""
        return {
            aug_type: transform(image=image)["image"]
            for aug_type, transform in self.transformations.items()
        }

    def process_dataset(self, datasets_path: str) -> None:
        """Process and augment images in a dataset."""
        image_path = Path(datasets_path) / "images"
        label_path = Path(datasets_path) / "labels"

        for img_file in image_path.glob("*.[jp][pn][g]"):
            image = cv2.imread(str(img_file))
            if image is None:
                print(f"Error reading image: {img_file}")
                continue

            augmented_images = self.apply_augmentations(image)

            # Save augmented images and copy corresponding labels
            for aug_type, aug_img in augmented_images.items():
                aug_filename = f"{img_file.stem}_{aug_type}.jpg"
                aug_img_path = image_path / aug_filename
                cv2.imwrite(str(aug_img_path), aug_img)

                # Copy label file if it exists
                label_file = label_path / f"{img_file.stem}.txt"
                if label_file.exists():
                    aug_label_path = (
                        label_path / f"{aug_filename.replace('.jpg', '.txt')}"
                    )
                    aug_label_path.write_text(label_file.read_text())
                    print(f"Copied label: {aug_label_path}")


def main():
    try:
        image_augment = ImageAugmenter()
        extractor = VideoProcessor()

        for dataset_type in ["train", "valid", "test"]:
            image_augment.process_dataset(f"datasets/notation_detection/{dataset_type}")
        extractor.extract_frames(
            video_path="datasets/fight_replay/combo1.mp4",
            prefix="test",
            output_dir="datasets/fight_replay/sample",
            interval_sec=1,
            num_images=60,
        )
    except Exception as e:
        print(f"Execution error: {e}")


if __name__ == "__main__":
    main()
