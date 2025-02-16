from game_metric import GameMetrics
from ultralytics import YOLO
import cv2


class FightAnalyzer:
    """Main class for analyzing fighting game footage."""

    def __init__(self, model_path: str = None):
        self.model = YOLO(model_path) if model_path else None
        self.metric = GameMetrics()
        

    def analyze_video(self, video_path: str) -> None:
        """Analyze video using YOLO model."""
        if not self.model:
            raise ValueError(
                "Model not initialized. Please provide model path during initialization."
            )

        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame.shape[:2] != (720, 1280):
                frame = cv2.resize(frame, (1280, 720))
            frame = self.metric.draw_bounding_boxes(frame)
            results = self.model(frame)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    label = result.names[int(box.cls[0])]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv2.putText(
                        frame,
                        f"{label} {confidence:.2f}",
                        (x1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (0, 255, 0),
                        1,
                    )

            cv2.imshow("YOLO Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    try:
        analyzer = FightAnalyzer("./datasets/runs/detect/train2/weights/best.pt")
        analyzer.analyze_video("./datasets/fight_replay/combo1.mp4")

    except Exception as e:
        print(f"Execution error: {e}")


if __name__ == "__main__":
    main()
