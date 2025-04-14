from flask import Flask, render_template, request, send_file
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            filepath = os.path.join("static", file.filename)
            file.save(filepath)

            image_arr = mpimg.imread(filepath)
            (h, w, c) = image_arr.shape
            image_2d = image_arr.reshape(h * w, c)

            model = KMeans(n_clusters=6)
            labels = model.fit_predict(image_2d)
            rgb_codes = model.cluster_centers_.round(0).astype(int)

            quantized_image = np.reshape(rgb_codes[labels], (h, w, c))

            plt.imsave("static/quantized_out.jpeg", quantized_image.astype(np.uint8))

            return render_template("index.html",
                                   original=file.filename,
                                   quantized="quantized_out.jpeg")
    return render_template("index.html")

if __name__ == "__main__":
    #app.run(debug=True)
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT not set
    app.run(host="0.0.0.0", port=port)
