"""File description: Shared constants for MABe mouse-triplets experiments."""

FRAME_WIDTH = 850
FRAME_HEIGHT = 850
NUM_MICE = 3
NUM_KEYPOINTS = 12
COORDINATES = 2

DEFAULT_SEQUENCE_LENGTH = 20
DEFAULT_OBSERVATION_LENGTH = 8
DEFAULT_PREDICTION_LENGTH = 12
DEFAULT_WINDOW_STRIDE = 1
DEFAULT_VALIDATION_FRACTION = 0.2
DEFAULT_SPLIT_SEED = 42

KEYPOINT_NAMES = (
    "nose",
    "left_ear",
    "right_ear",
    "neck",
    "left_forepaw",
    "right_forepaw",
    "center_back",
    "left_hindpaw",
    "right_hindpaw",
    "tail_base",
    "tail_middle",
    "tail_tip",
)
