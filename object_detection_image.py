import cv2
import os
from PIL import ImageFont

# Get the absolute path of the directory where this script is located
current_directory = os.path.dirname(os.path.abspath(__file__))

output_directory = os.path.join(current_directory, 'static', 'output')

def object_detection (input_image) -> None:
    
    image_name = os.path.basename(input_image)
    
    try:
        image = cv2.imread(input_image)
        #image = cv2.resize(image, (800, 600))
        h = image.shape[0]
        w = image.shape[1]

    except Exception as e:
        print(str(e))
        

    # path to the weights and model files
    # pre-trained models 
    weights = os.path.join(current_directory, 'ssd_mobilenet', 'frozen_inference_graph.pb')
    model = os.path.join(current_directory, 'ssd_mobilenet', 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
    # load the MobileNet SSD model trained  on the COCO dataset
    net = cv2.dnn.readNetFromTensorflow(weights, model)

    # load the class labels the model was trained on
    class_names = []
    with open(os.path.join(current_directory, 'ssd_mobilenet', 'coco_names.txt'), "r") as f:
        class_names = f.read().strip().split("\n")

    # create a blob from the image
    blob = cv2.dnn.blobFromImage(
        image, 1.0/127.5, (320, 320), [127.5, 127.5, 127.5])
    # pass the blog through our network and get the output predictions
    net.setInput(blob)
    output = net.forward()  # shape: (1, 1, 100, 7)

    # Define colors for bounding boxes and text
    box_color = (255, 0, 0)  # Blue color for bounding boxes
    text_color = (255, 255, 255)  # White color for text
    
    # loop over the number of detected objects
    for detection in output[0, 0, :, :]:  # output[0, 0, :, :] has a shape of: (100, 7)
        # the confidence of the model regarding the detected object
        probability = detection[2]

        # if the confidence of the model is lower than 50%,
        # we do nothing (continue looping)
        if probability < 0.5:
            continue

        # perform element-wise multiplication to get
        # the (x, y) coordinates of the bounding box
        box = [int(a * b) for a, b in zip(detection[3:7], [w, h, w, h])]
        box = tuple(box)
        # draw the bounding box of the object
        cv2.rectangle(image, box[:2], box[2:], box_color, thickness=2)

        # extract the ID of the detected object to get its name
        class_id = int(detection[1])
            
        # draw the name of the predicted object along with the probability
        label = f"{class_names[class_id - 1].upper()} {probability * 100:.2f}%"
        
        # Determine the size of the text
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Set the background coordinates
        background_coords = ((box[0], box[1] + 18), (box[0] + text_width, box[1] + 15 - text_height))

        # Draw a black filled rectangle as the background
        cv2.rectangle(image, background_coords[0], background_coords[1], (30, 30, 30), -1)

        # Draw text with a black outline
        outline_color = (0, 0, 0)  # Black color for outline
        outline_thickness = 2
        cv2.putText(image, label, (box[0], box[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, outline_color, outline_thickness)

        # Draw the actual text on top of the outline
        cv2.putText(image, label, (box[0], box[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    
    # Save the image in the specified folder
    output_path = os.path.join(output_directory, 'object_' + image_name)
    cv2.imwrite(output_path, image)

    # Destroy the window after saving the image
    cv2.destroyAllWindows()
    