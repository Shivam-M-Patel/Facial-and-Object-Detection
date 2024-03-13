import os
from flask import Flask, render_template, request, send_file, redirect, url_for
from PIL import Image
from detector import recognize_faces
from object_detection_image import object_detection

# Get the absolute path of the directory where this script is located
current_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute paths for the 'uploads' and 'output' directories
uploads_directory = os.path.join(current_directory, 'uploads')
output_directory = os.path.join(current_directory, 'static', 'output')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/object-info')
def object_info():
    return render_template('object-info.html')

@app.route('/facial-info')
def facial_info():
    return render_template('facial-info.html')

@app.route('/about-me')
def about_me():
    return render_template('about-me.html')

@app.route('/upload', methods=['POST'])
def upload():
    
    file = request.files['image']
    
    # Check if an image file was uploaded
    if file.filename == '':
        return 'ERROR! No image file uploaded.'
    
    # Check if the file has a valid extension
    if not allowed_file(file.filename):
        return 'ERROR! Invalid file extension. (Please upload a ".jpg" or ".jpeg" file)'

    # Save the uploaded file
    input_image_path = os.path.join(uploads_directory, file.filename)
    file.save(input_image_path)
    
    # Check the selected detection option
    detection_option = request.form.get('detection-option')
    
    # Perform object or face detection based on the selected option
    if detection_option == 'facial-detection':
        recognize_faces(input_image_path)
        # Save the modified image
        #output_path = os.path.join('output', 'face_' + file.filename)
        output_path = 'output/' + 'face_' + file.filename
    
    elif detection_option == 'object-detection':
        object_detection(input_image_path)
        # Save the modified image
        #output_path = os.path.join(output_directory, 'object_' + file.filename)
        output_path = 'output/' + 'object_' + file.filename
    
    else:
        return 'ERROR! Invalid detection option selected.'

    # Provide the output image path to the template for display
    print("OUTPUT" + output_path)
    return render_template('result.html', image_path=output_path)


def allowed_file(filename):
    allowed_extensions = {'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


if __name__ == '__main__':
    # Create the 'uploads' and 'output' directories if they don't exist
    os.makedirs(uploads_directory, exist_ok=True)
    os.makedirs(output_directory, exist_ok=True)
    
    app.run(debug=True)
