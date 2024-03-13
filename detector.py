import argparse                                         # parse command-line arguments
import pickle                                           # used for serialization and deserialization of Python objects
import os
import cv2
import math
from collections import Counter                         # Counter is used for counting hashable objects
from pathlib import Path                                # Path provides an object-oriented interface for working with file system path

from flask import Flask, render_template, request, send_file
import pickle

import face_recognition                                 # face recognition tasks
from PIL import Image, ImageDraw, ImageFont             # used for manipulating images

from object_detection_image import object_detection


BOUNDING_BOX_COLOR = "blue"                             # box color
TEXT_COLOR = "white"                                    # text color

# Get the absolute path of the directory where this script is located
current_directory = os.path.dirname(os.path.abspath(__file__))

encoding_directory = os.path.join(current_directory, 'encoding')

output_directory = os.path.join(current_directory, 'static', 'output')

# Create encoding directory if it doesn't already exist
os.makedirs(encoding_directory, exist_ok=True)

DEFAULT_ENCODINGS_PATH = os.path.join(encoding_directory, 'encodings.pkl')   # the path to the default encodings file

# Create directories if they don't already exist
# Path("training").mkdir(exist_ok=True)
# Path("encoding").mkdir(exist_ok=True)
# Path("validation").mkdir(exist_ok=True)


parser = argparse.ArgumentParser(description="Recognize faces in an image")             # create an argument parser object named parser with a description

parser.add_argument("--train", action="store_true", help="Train the model on input data -> training folder")         # adds an argument to the parser. The argument is --train, which means that if you 
                                                                                        # include --train when running the program, it will set a variable args.train to True
                                                                                        
parser.add_argument(                                                                    # adds validate argument
    "--validate", action="store_true", help="Validate trained model using pictures -> validation folder"
)

parser.add_argument(                                                                    # adds test argument
    "--test", action="store_true", help="Test the model with any image -> use it with -f"
)

parser.add_argument(                                                                    # it expects a value to be provided. The value will be stored in args.m. It also has a default value of "hog". 
    "-m",                                                                               # the argument only allows two choices: "hog" and "cnn".
    action="store",
    default="hog",
    choices=["hog", "cnn"],
    help="Choose model for training: hog (CPU), cnn (GPU)",
)

parser.add_argument(                                                                    # it expects a value to be provided, which will be stored in args.f. It doesn't have a default value
    "-f", action="store", help="Path to an image"
)

args = parser.parse_args()                                                              # this line parses the command-line arguments and stores the values in the args variable


def encode_known_faces(
    model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH
) -> None:
    """
    Loads images in the training directory and builds a dictionary of their
    names and encodings.
    """
    names = []
    encodings = []

    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)


def recognize_faces(
    image_location: str,
    model: str = "hog",
    encodings_location: str = DEFAULT_ENCODINGS_PATH,
) -> None:
    """
    Given an unknown image, get the locations and encodings of any faces and
    compares them against the known encodings to find potential matches.
    """
    target_size = (800, 600)
    
    image_name = os.path.basename(image_location)
    
    # Convert the encodings_location to a Path object
    encodings_location = Path(encodings_location)
    
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    input_image = face_recognition.load_image_file(image_location)
    
    
    input_face_locations = face_recognition.face_locations(
        input_image, model=model
    )
    input_face_encodings = face_recognition.face_encodings(
        input_image, input_face_locations
    )
    
    pillow_image = Image.fromarray(input_image)
    
    draw = ImageDraw.Draw(pillow_image)

    for bounding_box, unknown_encoding in zip(
        input_face_locations, input_face_encodings
    ):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "Unknown"
        _display_face(draw, bounding_box, name)

    del draw
    
    output_path = os.path.join(output_directory, 'face_' + image_name)
    pillow_image.save(output_path)


def _recognize_face(unknown_encoding, loaded_encodings):
    """
    Given an unknown encoding and all known encodings, find the known
    encoding with the most matches.
    """
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding
    )
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )
    if votes:
        return votes.most_common(1)[0][0]



def _display_face(draw, bounding_box, name):
    """
    Draws styled bounding boxes around faces and adds stylized text captions.
    """
    top, right, bottom, left = bounding_box

    # Define the style parameters
    box_width = 3
    box_color = (255, 0, 0)  # Red color for the bounding box
    # text_font = ImageFont.truetype("OpenSans-SemiboldItalic.ttf", 28)
    text_font = ImageFont.truetype("calibri.ttf", 28) 
    text_color = (255, 255, 255)  # White color for the text

    # Draw the bounding box
    draw.rectangle(((left, top), (right, bottom)), width=box_width, outline=box_color)

    # Calculate the position of the text caption
    text_width, text_height = draw.textsize(name, font=text_font)
    text_left = left + int((right - left - text_width) / 2)
    text_top = bottom + 5  # Position the text just below the bounding box

    # Draw a filled rectangle as the background for the text caption
    text_box_left = text_left
    text_box_right = text_left + text_width
    text_box_top = text_top
    text_box_bottom = text_top + text_height
    draw.rectangle(
        ((text_box_left - 5, text_box_top), (text_box_right + 5, text_box_bottom + 5)),
        fill=box_color,
    )

    # Draw the text caption
    draw.text(
        (text_left, text_top),
        name,
        font=text_font,
        fill=text_color,
    )


def validate(model: str = "hog"):
    """
    Runs recognize_faces on a set of images with known faces to validate
    known encodings.
    """
    for filepath in Path("validation").rglob("*"):
        if filepath.is_file():
            recognize_faces(
                image_location=str(filepath.absolute()), model=model
            )


if __name__ == "__main__":
    if args.train:
        encode_known_faces(model=args.m)                                # Trains the model with the data provided in the training folder
    if args.validate:
        validate(model=args.m)                                          # Validates the trained data with images in validation folder
    if args.test:
        object_detection(args.f)                                        # Detect the objects in the image first
        recognize_faces(image_location=args.f, model=args.m)            # Then detect the faces in the image
