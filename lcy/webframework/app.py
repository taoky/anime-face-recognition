#!/usr/bin/env python3
from flask import *

app = Flask(__name__)


# fd == facedetect
@app.route("/fd", methods=['GET', 'POST'])
def fd():
    return render_template("js_face_detection.html")


if __name__ == '__main__':
    app.run("0.0.0.0", port=5000, debug=True)
