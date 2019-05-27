#!/usr/bin/env python3
from flask import *
import os
from datetime import datetime
import base64
from random import randint
from classification import web_api

IDENT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/ident/'
app = Flask(__name__)


# fd == facedetect
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/uploadForIdent", methods=['POST'])
def submitPic():
    magic_str = "data:image/png;base64,"
    pic = request.form['data']
    # print(pic)
    if pic is None or pic == "" or pic[:len(magic_str)] != magic_str:
        return "An error occurred when receiving your images."
    else:
        pic = pic[len(magic_str):]
    file_name = IDENT_DIR + str(datetime.now())[:10] + str(randint(0,100)) + ".png"
    with open(file_name, "wb") as f:
        f.write(base64.urlsafe_b64decode(
            pic.encode("utf-8")
            ))
        f.close()
        return web_api(file_name)

if __name__ == '__main__':
    app.run("0.0.0.0", port=5000, debug=True)
