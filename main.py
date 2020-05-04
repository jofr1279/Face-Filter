""" main.py

The main frontend for our CSCI 4622 - Machine Learning final project.

By Joshua Franklin and Tiffany Phan

Usage:
    Run with `python main.py` to start normally.
    Run with `python main.py --mode debug` to start in debug mode, as explained in a comment below.
    Run with `python main.py --mode datagen` to start in datagen mode, which is a tool for generated data as explained
    in the notebook.
"""

import sys

import pygame
import pygame.camera
import numpy as np

from argparse import ArgumentParser
from keras import Model
from keras.models import load_model
from enum import Enum

from typing import Tuple

IMAGE_SIZE = 120
FILTER_SIZE = 100
FILTER_RESOLUTION = 1000

DATA_GEN_FACES_FILE = 'data_gen_faces'
DATA_GEN_POINTS_FILE = 'data_gen_points'


class RunMode(Enum):
    NORMAL = 0
    DEBUG = 1
    DATA_GEN = 2


def process_image(image: np.ndarray) -> np.ndarray:
    """ Take a raw image from the camera and turn it into something the machine learning model can work with. """

    return np.rot90(np.mean(image.copy(), axis=2), 3) / 255


def get_nose(model: Model, image: np.ndarray) -> Tuple[float, float]:
    """ Use the machine learning model to predict where the nose (center of face) is located in it. """

    image_copy = process_image(image)
    image_copy = image_copy.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 1))
    prediction = model.predict(image_copy)[0]

    # The image from the camera is mirrored, so IMAGE_SIZE - prediction[0] can counter that.
    return IMAGE_SIZE - prediction[0], prediction[1]


def apply_filter(image: np.ndarray, nose: Tuple[int, int]):
    """ Take an image and apply the bulge filter to it at the given nose coordinates. """

    # Copy the image around the center of the nose so modifications can be made to the original image while retaining
    # the original data.
    image_copy = image[nose[0] - FILTER_SIZE:nose[0] + FILTER_SIZE, nose[1] - FILTER_SIZE:nose[1] + FILTER_SIZE].copy()

    for row in range(-FILTER_SIZE, FILTER_SIZE):
        for col in range(-FILTER_SIZE, FILTER_SIZE):
            # Euclidean distance to center of nose.
            dist = int(np.sqrt(row**2 + col**2) * (FILTER_RESOLUTION / FILTER_SIZE))

            # Cutting off the radius creates a circle.
            if dist > FILTER_RESOLUTION:
                continue

            try:
                # Copy over the pixel that is located along at the same angle as the target pixel, but at a radius
                # proportional to the distance away from the nose.
                image[nose[0] + row, nose[1] + col] = image_copy[
                    int(row * dist / FILTER_RESOLUTION) + FILTER_SIZE,
                    int(col * dist / FILTER_RESOLUTION) + FILTER_SIZE,
                ]
            except IndexError:
                # If we end up out of bounds, just don't process that pixel.
                pass


def get_mode() -> RunMode:
    """ Gets the run mode specified in the command line arguments. Defaults to RunMode.NORMAL is no run mode is
        specified. """

    parser = ArgumentParser()
    parser.add_argument('--mode', help='specify the mode to run in')
    args = parser.parse_args()

    if args.mode is None or args.mode == 'normal':
        return RunMode.NORMAL
    elif args.mode == 'debug':
        return RunMode.DEBUG
    elif args.mode == 'datagen':
        return RunMode.DATA_GEN
    else:
        raise ValueError('Invalid mode.')


def main():
    mode = get_mode()

    pygame.init()
    pygame.camera.init()
    pygame.font.init()

    font = pygame.font.SysFont('Ariel', 30)

    cameras = pygame.camera.list_cameras()
    if not cameras:
        raise Exception('Camera not detected.')

    camera = pygame.camera.Camera(cameras[0])
    camera.start()

    pygame.display.set_caption('Face Filter')
    screen = pygame.display.set_mode((640, 480))

    crop_surface = pygame.Surface((480, 480))
    input_surface = pygame.Surface((IMAGE_SIZE, IMAGE_SIZE))

    saved_faces = []
    saved_points = []

    # There is no need to load the model if we are in datagen mode.
    model = load_model('model.h5') if mode != RunMode.DATA_GEN else None

    while True:
        camera.get_image(screen)
        crop_surface.blit(screen, (0, 0), (80, 0, 480, 480))
        pygame.transform.scale(crop_surface, (IMAGE_SIZE, IMAGE_SIZE), input_surface)

        for event in pygame.event.get():
            # This checks if the user exits out of the window.
            if event.type == pygame.QUIT:
                if mode == RunMode.DATA_GEN:
                    # Save the generated data to a file.
                    np.save(DATA_GEN_FACES_FILE, np.array(saved_faces))
                    np.save(DATA_GEN_POINTS_FILE, np.array(saved_points))

                    print('Data gen results saved to {}.npy and {}.npy.'.format(
                        DATA_GEN_FACES_FILE,
                        DATA_GEN_POINTS_FILE,
                    ))

                pygame.quit()
                sys.exit()

            # This checks if the user clicked their mouse and are in data gen mode.
            if mode == RunMode.DATA_GEN and event.type == pygame.MOUSEBUTTONDOWN:
                # Capture an image at the current mouse coordinates.
                mouse_pos = pygame.mouse.get_pos()

                if mouse_pos[0] < 80 or mouse_pos[0] > 560:
                    print('Coordinates out of bounds.')
                else:
                    pixel_x = IMAGE_SIZE - int((mouse_pos[0] - 80) * (IMAGE_SIZE / 480))
                    pixel_y = int(mouse_pos[1] * (IMAGE_SIZE / 480))

                    saved_faces.append(process_image(pygame.surfarray.pixels3d(input_surface)))
                    saved_points.append([pixel_x, pixel_y])

                    print('Image captured with face center at {}, {}.'.format(pixel_x, pixel_y))

        # There is no need to predict where the nose is if we are in datagen mode.
        raw_nose = None
        if mode != RunMode.DATA_GEN:
            raw_nose = get_nose(model, pygame.surfarray.pixels3d(input_surface))
            nose = int(raw_nose[0] * (640 / IMAGE_SIZE)), int(raw_nose[1] * (480 / IMAGE_SIZE))
            apply_filter(pygame.surfarray.pixels3d(screen), nose)

        # Debug mode makes it easier to see what the model is predicting by displaying a smaller copy of your image at
        # the top left, putting a red dot over where the model predicts the center of your face is, and displays the
        # pixel coordinates of the prediction underneath it.
        if mode == RunMode.DEBUG:
            nose_int = tuple(map(int, raw_nose))
            pygame.draw.circle(input_surface, (255, 0, 0), nose_int, 5)
            nose_text = font.render(str(nose_int), False, (255, 255, 255))
            screen.blit(input_surface, (0, 0))
            screen.blit(nose_text, (5, IMAGE_SIZE + 5))

        pygame.display.flip()


if __name__ == '__main__':
    main()
