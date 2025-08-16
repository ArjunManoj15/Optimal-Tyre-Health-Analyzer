import cv2
import numpy as np
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="C:/Users/ARJUN/Downloads/converted_tflite (1)/model_unquant.tflite")
interpreter.allocate_tensors()

# Get model input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get input shape
input_shape = input_details[0]['shape']
height, width = input_shape[1], input_shape[2]

# Define class labels
class_names = ['Bad Tyre', 'Good Tyre']

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and normalize the image
    input_image = cv2.resize(frame, (width, height))
    input_image = input_image.astype(np.float32) / 255.0
    input_image = np.expand_dims(input_image, axis=0)

    # Set model input
    interpreter.set_tensor(input_details[0]['index'], input_image)

    # Run inference
    interpreter.invoke()

    # Get output
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Interpret result
    prediction = np.argmax(output_data)
    label = class_names[prediction]
    confidence = np.max(output_data)

    # Choose color based on prediction
    if label == 'Good Tyre':
        color = (0, 255, 0)  # Green
    else:
        color = (0, 0, 255)  # Red

    # Print result in console
    print(f"Prediction: {label} ({confidence:.2f})")

    # Show result on webcam feed with color
    cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Tyre Classification", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
