from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import time
import cv2
from ultralytics import YOLO

app = Flask(__name__)
UPLOAD_FOLDER = 'static/results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load YOLO model once
model = YOLO("best.pt")

# Ensure result folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        files = request.files.getlist("file")
        results = []

        start_time = time.time()

        for file in files:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # YOLO detection
            process_start = time.time()
            detection_results = model(filepath, conf=0.5)
            for det in detection_results:
                annotated = det.plot()
                cv2.imwrite(filepath, annotated)
            process_end = time.time()

            results.append({
                "filename": filename,
                "processing_time": round(process_end - process_start, 2)
            })

        total_time = round(time.time() - start_time, 2)
        return render_template("result.html", results=results, folder=UPLOAD_FOLDER, processing_time=total_time)

    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True)
