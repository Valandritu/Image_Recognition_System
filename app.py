import os
import uuid
import cv2
import numpy as np
from flask import Flask, request, render_template, url_for
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg','webp'}

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load pre-trained model
model = ResNet50(weights='imagenet')


# Function to check file type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



def sepia_effect(img):
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    sepia_img = cv2.transform(img, sepia_filter)
    sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
    return sepia_img

# Image Processing Functions
def apply_effect(filepath, effect):
    img = cv2.imread(filepath)

    if effect.startswith("bw"):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if effect == "bw_low":
            alpha = 0.4
        elif effect == "bw_medium":
            alpha = 0.6
        elif effect == "bw_high":
            alpha = 1.0

        processed = cv2.addWeighted(gray_bgr, alpha, img, 1 - alpha, 0)

    elif effect.startswith("increase_brightness"):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        if effect == "increase_brightness_low":
            value = 20
        elif effect == "increase_brightness_medium":
            value = 40
        elif effect == "increase_brightness_high":
            value = 60

        hsv[:, :, 2] = cv2.add(hsv[:, :, 2], value)
        processed = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


    elif effect.startswith("decrease_brightness"):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        if effect == "decrease_brightness_low":
            value = 30
        elif effect == "decrease_brightness_medium":
            value = 60
        elif effect == "decrease_brightness_high":
            value = 90

    # Subtract brightness safely, ensuring no negative values
        hsv[:, :, 2] = cv2.subtract(hsv[:, :, 2], value)

        processed = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


    elif effect.startswith("blur_background"):

        # Read original image
        img = cv2.imread(filepath)
        height, width = img.shape[:2]

        # Downscale image for faster processing
        scale = 0.10
        img_small = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        mask_small = np.zeros(img_small.shape[:2], np.uint8)

        # Define smaller rectangle for grabCut
        h_s, w_s = img_small.shape[:2]
        rect_small = (10, 10, w_s - 20, h_s - 20)

        # Apply GrabCut on downscaled image
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(img_small, mask_small, rect_small, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)

        # Convert result to binary mask and resize to original
        mask2_small = np.where((mask_small == 2) | (mask_small == 0), 0, 1).astype("uint8")
        mask2 = cv2.resize(mask2_small, (width, height), interpolation=cv2.INTER_LINEAR)

        # Determine blur level
        blur_strength = {
            "blur_background_low": (9, 9),
            "blur_background_medium": (21, 21),
            "blur_background_high": (35, 35),
        }
        ksize = blur_strength.get(effect, (15, 15))
        blurred = cv2.GaussianBlur(img, ksize, 0)

        # Create the output image
        mask2_3ch = mask2[:, :, np.newaxis]
        foreground = img * mask2_3ch
        background = blurred * (1 - mask2_3ch)
        processed = cv2.add(foreground, background)


    elif effect.startswith("filters_"):
        filter_type = effect.split("_")[1]

        if filter_type == "fresh":
        # Boost brightness slightly and reduce saturation
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = cv2.subtract(hsv[:, :, 1], 30)  # reduce saturation
            hsv[:, :, 2] = cv2.add(hsv[:, :, 2], 30)       # increase brightness
            processed = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        elif filter_type == "cool":
        # Add a blueish tint
            b, g, r = cv2.split(img)
            b = cv2.add(b, 30)
            r = cv2.subtract(r, 10)
            processed = cv2.merge((b, g, r))

        elif filter_type == "warm":
        # Add a reddish/yellowish tint
            b, g, r = cv2.split(img)
            r = cv2.add(r, 30)
            b = cv2.subtract(b, 10)
            processed = cv2.merge((b, g, r))

        elif filter_type == "sunset":
        # Add a warm orange glow
            b, g, r = cv2.split(img)
            r = cv2.add(r, 40)
            g = cv2.add(g, 20)
            b = cv2.subtract(b, 30)
            processed = cv2.merge((b, g, r))

        elif filter_type == "fade":
        # Lower contrast for a faded look
            fade = cv2.addWeighted(img, 0.5, np.full_like(img, 128), 0.5, 0)
            processed = fade

    elif effect.startswith("vignette"):
        level = effect.split("_")[1]  # Extract the level (low, medium, high)
    
        # Get the image dimensions
        rows, cols = img.shape[:2]
    
        # Create meshgrid for each pixel in the image
        X_result, Y_result = np.meshgrid(np.arange(cols), np.arange(rows))
    
        # Get the center of the image
        centerX, centerY = cols / 2, rows / 2
    
        # Calculate distance from the center for each pixel
        distance = np.sqrt((X_result - centerX)**2 + (Y_result - centerY)**2)
    
        # Define vignette intensity based on distance and effect level
        if level == "low":
            max_distance = np.max(distance) * 1.2 # Full effect for high level
        # Less effect for low level
        elif level == "medium":
            max_distance = np.max(distance)   # Medium effect for medium level
        else:
            max_distance = np.max(distance) *0.9

        # Normalize the distance and create a mask for the vignette
        mask = np.clip(distance / max_distance, 0, 1)
    
        # Apply the mask to each color channel in the image
        vignette_img = np.zeros_like(img, dtype=np.float32)
        for i in range(3):  # Iterate through RGB color channels
            vignette_img[:, :, i] = img[:, :, i] * (1 - mask)
    
        # Convert the result back to uint8
        processed = np.clip(vignette_img, 0, 255).astype(np.uint8)

    elif effect.startswith("enhance"):
        if effect == "enhance_low":
            sigma_s = 150
            sigma_r = 0.7          
        elif effect == "enhance_medium":
            sigma_s = 80
            sigma_r = 0.4
        elif effect == "enhance_high":
            sigma_s = 30
            sigma_r = 0.2
        else:
            sigma_s = 50
            sigma_r = 0.3
        processed = cv2.detailEnhance(img, sigma_s=sigma_s, sigma_r=sigma_r)

    elif effect.startswith("sketch"):
        if effect == "sketch_low":
            ksize = 5
        elif effect == "sketch_medium":
            ksize = 7
        elif effect == "sketch_high":
            ksize = 9
        else:
            ksize = 5

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inverted = 255 - gray
        blurred = cv2.GaussianBlur(inverted, (ksize, ksize), 0)
        sketch = cv2.divide(gray, 255 - blurred, scale=256)
        processed = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    elif effect.startswith("contrast"):
        if effect == "contrast_low":
            alpha = 1.2  # Slight increase
        elif effect == "contrast_medium":
            alpha = 1.5  # Medium increase
        elif effect == "contrast_high":
            alpha = 2.0  # Strong increase
        else:
            alpha = 1.3  # Default

        processed = cv2.convertScaleAbs(img, alpha=alpha, beta=0)

    elif effect.startswith("rotate"):
        angle_map = {
            "rotate_0": 0,
            "rotate_90": 90,
            "rotate_180": 180,
            "rotate_270": 270
        }
        angle = angle_map.get(effect, 0)

        if angle == 0:
            processed = img
        else:
            (h, w) = img.shape[:2]
            center = (w / 2, h / 2)

            # Compute the rotation matrix
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

            # Calculate new bounding dimensions
            cos = abs(matrix[0, 0])
            sin = abs(matrix[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))

            # Adjust the rotation matrix to account for translation
            matrix[0, 2] += (new_w / 2) - center[0]
            matrix[1, 2] += (new_h / 2) - center[1]

            # Perform the actual rotation
            processed = cv2.warpAffine(img, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


    elif effect == 'sepia':
        processed = sepia_effect(img)

    elif effect == "flip":
        processed = cv2.flip(img, 1)

    elif effect == "adjust":
        # Convert to LAB color space for better brightness/contrast control
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        # Merge channels and convert back to BGR
        limg = cv2.merge((cl, a, b))
        adjusted = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # Slight sharpening
        kernel = np.array([[0, -1, 0],
                           [-1, 5,-1],
                           [0, -1, 0]])
        sharpened = cv2.filter2D(adjusted, -1, kernel)

        # Mild boost to saturation
        hsv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2HSV)
        hsv[...,1] = cv2.add(hsv[...,1], 15)
        processed = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    else:
        processed = img

    processed_filename = str(uuid.uuid4()) + ".jpg"
    processed_filepath = os.path.join(PROCESSED_FOLDER, processed_filename)
    cv2.imwrite(processed_filepath, processed)

    return processed_filename

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return "Invalid file type. Please upload a PNG or JPG or WEBp image."

    filename = str(uuid.uuid4()) + "_" + file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)
    decoded_preds = decode_predictions(preds, top=3)[0]
    labels = [f"{p[1]} ({p[2]*100:.2f}%)" for p in decoded_preds]
    label_text = ", ".join(labels)

    return render_template('result.html', filename=filename, label=label_text)

@app.route('/process/<effect>/<filename>')
def process_image(effect, filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        return "File not found"

    processed_filename = apply_effect(filepath, effect)
    return render_template('result.html', filename=filename, processed_filename=processed_filename, label="")



@app.route('/add_text/<filename>', methods=['POST'])
def add_text(filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        return "File not found"

    text = request.form.get("text")
    if not text:
        return "No text entered"

    img = cv2.imread(filepath)

    # Add text to the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    color = (0, 0, 255)
    thickness = 2
    position = (50, 50)  # top-left corner

    cv2.putText(img, text, position, font, font_scale, color, thickness)

    processed_filename = str(uuid.uuid4()) + ".jpg"
    processed_filepath = os.path.join(PROCESSED_FOLDER, processed_filename)
    cv2.imwrite(processed_filepath, img)

    return render_template("result.html", filename=filename, processed_filename=processed_filename, label="")

@app.route('/clear_text/<filename>', methods=['POST'])
def clear_text(filename):
    # Just return the original uploaded image without modification
    return render_template("result.html", filename=filename, label="")

@app.route('/crop_tool/<filename>', methods=['POST'])
def crop_tool(filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        return "File not found"

    crop_mode = request.form.get("crop_mode", "free")
    img = cv2.imread(filepath)
    h, w, _ = img.shape

    def get_crop_area(mode):
        if mode == "1:1":
            size = min(w, h)
            x = (w - size) // 2
            y = (h - size) // 2
            return x, y, x + size, y + size

        elif mode == "3:4":
            target_h = h
            target_w = int(h * 3 / 4)
            if target_w > w:
                target_w = w
                target_h = int(w * 4 / 3)
            x = (w - target_w) // 2
            y = (h - target_h) // 2
            return x, y, x + target_w, y + target_h

        elif mode == "9:16":
            target_h = h
            target_w = int(h * 9 / 16)
            if target_w > w:
                target_w = w
                target_h = int(w * 16 / 9)
            x = (w - target_w) // 2
            y = (h - target_h) // 2
            return x, y, x + target_w, y + target_h

        elif mode == "full":
            return 0, 0, w, h

        else:  # Free (same as full by default)
            return 0, 0, w, h

    x1, y1, x2, y2 = get_crop_area(crop_mode)
    cropped = img[y1:y2, x1:x2]

    processed_filename = str(uuid.uuid4()) + ".jpg"
    processed_filepath = os.path.join(PROCESSED_FOLDER, processed_filename)
    cv2.imwrite(processed_filepath, cropped)

    return render_template('result.html', filename=filename, processed_filename=processed_filename, label="")


@app.route('/border/<style>/<filename>')
def apply_border(style, filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        return "File not found"

    img = cv2.imread(filepath)
    h, w = img.shape[:2]

    if style == "simple":
        bordered = cv2.copyMakeBorder(img, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    elif style == "double":
        bordered = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        bordered = cv2.copyMakeBorder(bordered, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    elif style == "shadow":
        shadow = np.full((h + 40, w + 40, 3), 50, dtype=np.uint8)
        shadow[20:20+h, 20:20+w] = img
        bordered = shadow

    elif style == "polaroid":
        bordered = cv2.copyMakeBorder(img, 20, 60, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    elif style == "circle":
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        radius = min(center[0], center[1])
        cv2.circle(mask, center, radius, 255, -1)
        result = cv2.bitwise_and(img, img, mask=mask)
        bordered = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
        bordered[:, :, 3] = mask  # Set alpha channel

    else:
        bordered = img  # fallback

    ext = ".png" if style == "circle" else ".jpg"
    processed_filename = str(uuid.uuid4()) + ext
    processed_filepath = os.path.join(PROCESSED_FOLDER, processed_filename)

    if style == "circle":
        cv2.imwrite(processed_filepath, bordered, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    else:
        cv2.imwrite(processed_filepath, bordered)

    return render_template("result.html", filename=filename, processed_filename=processed_filename, label="")

if __name__ == '__main__':
    app.run(debug=True, threaded=True)