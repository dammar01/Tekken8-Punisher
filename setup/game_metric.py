from dataclasses import dataclass
from typing import Tuple
import cv2

# from paddleocr import PaddleOCR


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @classmethod
    def from_tuple(cls, coords: Tuple[int, int, int, int]):
        return cls(coords[0], coords[1], coords[2], coords[3])


@dataclass
class PlayerMetrics:
    input_frame: BoundingBox
    total_damage: BoundingBox
    max_combo_damage: BoundingBox
    hit_properties: BoundingBox
    damage: BoundingBox
    recoverable_damage: BoundingBox
    event: BoundingBox
    attack_startup_frames: BoundingBox
    frame_advantage: BoundingBox
    status: BoundingBox
    distance: BoundingBox
    hit_combo: BoundingBox
    damage_combo: BoundingBox
    timestamp: BoundingBox
    health: BoundingBox


class GameMetrics:
    """Handles visualization of game metrics coordinates."""

    def __init__(self, frame_width: int = 1280, frame_height: int = 720):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.notation_location_p1 = 0
        self.notation_location_p2 = 0
        self.player1 = self._create_player1_metrics()
        self.player2 = self._create_player2_metrics()
        self.colors = {
            "input_frame": (255, 0, 0),  # Red
            "total_damage": (0, 255, 0),  # Green
            "max_combo_damage": (0, 0, 255),  # Blue
            "hit_properties": (255, 255, 0),  # Yellow
            "damage": (255, 0, 255),  # Magenta
            "recoverable_damage": (0, 255, 255),  # Cyan
            "event": (128, 0, 0),  # Dark red
            "attack_startup_frames": (0, 128, 0),  # Dark green
            "frame_advantage": (0, 0, 128),  # Dark blue
            "status": (128, 128, 0),  # Olive
            "distance": (128, 0, 128),  # Purple
            "hit_combo": (0, 128, 255),
            "damage_combo": (0, 128, 255),
            "timestamp": (128, 128, 128),
            "health": (128, 0, 0),
        }
        # self.reader = PaddleOCR(use_angle_cls=False, lang='en', rec_char_type="en", drop_score=0)

    def update_notation_location(self, notation_location: int, player: int = 1) -> None:
        if player == 1:
            self.notation_location_p1 = notation_location
        else:
            self.notation_location_p2 = notation_location

    def _create_player1_metrics(self) -> PlayerMetrics:
        """Creates PlayerMetrics instance for player 1 with predefined coordinates."""
        input_frame_y = round(157 + (22.75 * self.notation_location_p1))
        return PlayerMetrics(
            input_frame=BoundingBox(135, input_frame_y, 155, input_frame_y + 18),
            total_damage=BoundingBox(433, 442, 465, 460),
            max_combo_damage=BoundingBox(433, 462, 465, 480),
            hit_properties=BoundingBox(403, 482, 465, 502),
            damage=BoundingBox(403, 505, 465, 523),
            recoverable_damage=BoundingBox(403, 527, 465, 545),
            event=BoundingBox(178, 548, 326, 566),
            attack_startup_frames=BoundingBox(383, 579, 465, 597),
            frame_advantage=BoundingBox(383, 601, 465, 619),
            status=BoundingBox(383, 623, 465, 641),
            distance=BoundingBox(423, 643, 465, 661),
            hit_combo=BoundingBox(170, 354, 330, 426),
            damage_combo=BoundingBox(170, 354, 330, 426),
            timestamp=BoundingBox(594, 20, 686, 90),
            health=BoundingBox(390, 28, 443, 46),
        )

    def _create_player2_metrics(self) -> PlayerMetrics:
        """Creates PlayerMetrics instance for player 2 with predefined coordinates."""
        input_frame_y = round(157 + (22.75 * self.notation_location_p2))
        return PlayerMetrics(
            input_frame=BoundingBox(1227, input_frame_y, 1247, input_frame_y + 18),
            total_damage=BoundingBox(1073, 442, 1105, 460),
            max_combo_damage=BoundingBox(1073, 462, 1105, 480),
            hit_properties=BoundingBox(1043, 482, 1105, 502),
            damage=BoundingBox(1043, 505, 1105, 523),
            recoverable_damage=BoundingBox(1043, 527, 1105, 545),
            event=BoundingBox(818, 548, 966, 566),
            attack_startup_frames=BoundingBox(1023, 579, 1105, 597),
            frame_advantage=BoundingBox(1023, 601, 1105, 619),
            status=BoundingBox(1023, 623, 1105, 641),
            distance=BoundingBox(1063, 643, 1105, 661),
            hit_combo=BoundingBox(962, 351, 1122, 425),
            damage_combo=BoundingBox(962, 351, 1122, 425),
            timestamp=BoundingBox(594, 20, 686, 90),
            health=BoundingBox(878, 28, 931, 46),
        )

    def draw_bounding_boxes(self, frame: cv2.Mat) -> cv2.Mat:
        """Draw all bounding boxes on the frame with different colors for each metric."""
        visualized_frame = frame.copy()

        for player_num, player_metrics in enumerate([self.player1, self.player2], 1):
            for metric_name, bbox in player_metrics.__dict__.items():
                color = self.colors[metric_name]
                # Draw rectangle
                cv2.rectangle(
                    visualized_frame,
                    (bbox.x2, bbox.y1),  # top-left
                    (bbox.x1, bbox.y2),  # bottom-right
                    color,
                    1,
                )  # thickness

                # Add label
                label = f"P{player_num}-{metric_name}"
                cv2.putText(
                    visualized_frame,
                    label,
                    (bbox.x2, bbox.y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,  # Font scale
                    color,
                    1,
                )  # Thickness
        return visualized_frame

    def check_bounding_boxes(self, path: str) -> None:
        """Checking Game Metrics bounding box to the selected image"""
        image = cv2.imread(path)
        image = cv2.resize(image, (1280, 720))
        image = self.draw_bounding_boxes(image)
        cv2.imshow("Game metrics bounding box", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_cord_img(self, path: str) -> None:
        """Getting coordinate of location which the user clicks"""
        image = cv2.imread(path)

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print(f"Koordinat: ({x}, {y})")
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow("Image", image)

        image = cv2.resize(image, (1280, 720))
        cv2.imshow("Image", image)
        cv2.setMouseCallback("Image", click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    metric = GameMetrics()
    path_image = "./datasets/fight_replay/sample/tes.png"
    # metric.get_cord_img(path_image)
    metric.check_bounding_boxes(path_image)


if __name__ == "__main__":
    main()
