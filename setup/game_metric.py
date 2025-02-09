from dataclasses import dataclass
from typing import Tuple
import cv2


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
    combo: BoundingBox


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
            "combo": (0, 128, 255),
        }

    def update_notation_location(self, notation_location: int, player: int = 1) -> None:
        if player == 1:
            self.notation_location_p1 = notation_location
        else:
            self.notation_location_p2 = notation_location

    def _create_player1_metrics(self) -> PlayerMetrics:
        """Creates PlayerMetrics instance for player 1 with predefined coordinates."""
        input_frame_y = round(157 + (22.75 * self.notation_location_p1))
        return PlayerMetrics(
            input_frame=BoundingBox(155, input_frame_y, 135, input_frame_y + 18),
            total_damage=BoundingBox(465, 442, 433, 460),
            max_combo_damage=BoundingBox(465, 462, 433, 480),
            hit_properties=BoundingBox(465, 482, 403, 502),
            damage=BoundingBox(465, 505, 403, 523),
            recoverable_damage=BoundingBox(465, 527, 403, 545),
            event=BoundingBox(178, 548, 326, 566),
            attack_startup_frames=BoundingBox(465, 579, 383, 597),
            frame_advantage=BoundingBox(465, 601, 383, 619),
            status=BoundingBox(465, 623, 383, 641),
            distance=BoundingBox(465, 643, 423, 661),
            combo=BoundingBox(170, 354, 330, 426),
        )

    def _create_player2_metrics(self) -> PlayerMetrics:
        """Creates PlayerMetrics instance for player 2 with predefined coordinates."""
        input_frame_y = round(157 + (22.75 * self.notation_location_p2))
        return PlayerMetrics(
            input_frame=BoundingBox(1247, input_frame_y, 1227, input_frame_y + 18),
            total_damage=BoundingBox(1105, 442, 1073, 460),
            max_combo_damage=BoundingBox(1105, 462, 1073, 480),
            hit_properties=BoundingBox(1105, 482, 1043, 502),
            damage=BoundingBox(1105, 505, 1043, 523),
            recoverable_damage=BoundingBox(1105, 527, 1043, 545),
            event=BoundingBox(818, 548, 966, 566),
            attack_startup_frames=BoundingBox(1105, 579, 1023, 597),
            frame_advantage=BoundingBox(1105, 601, 1023, 619),
            status=BoundingBox(1105, 623, 1023, 641),
            distance=BoundingBox(1105, 643, 1063, 661),
            combo=BoundingBox(1122, 425, 962, 351),
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
    try:
        metric = GameMetrics()
        path_image = "./datasets/fight_replay/sample3/tes3_frame_0505.jpg"
        # metric.get_cord_img(path_image)
        metric.check_bounding_boxes(path_image)

    except Exception as e:
        print(f"Execution error: {e}")


if __name__ == "__main__":
    main()
