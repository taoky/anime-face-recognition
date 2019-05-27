function facedet() {
    let src = cv.imread('canvasInput');
    let src_canvas = document.getElementById('canvasInput')
    let gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
    let faces = new cv.RectVector();
    let faceCascade = new cv.CascadeClassifier();
    console.log("Face Detection Initialization");
    faceCascade.load('lbpcascade_animeface.xml');
    console.log("lbpcascade_animeface.xml loaded");
    // detect faces
    let msize = new cv.Size(30, 30);
    faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0, msize);
    console.log("face detected");
    for (let i = 0; i < faces.size(); ++i) {
        let face = faces.get(i);
        let width = face.width;
        let height = face.height;
        console.log(faces.get(i));

        addToList(face.x, face.y, width, height, src_canvas)
        
    }
    src.delete(); gray.delete(); faceCascade.delete();
    faces.delete();
}

function addToList(x, y, width, height, src_canvas) {
    let result = $("<div class=\"container-fluid\"><div class='row'><div class='col-md-4'><canvas class='result_pic' width='" + width + "' height='" + height + 
                    "'></canvas></div><div class='col-md-4 align-self-center'><div class='btn-group' role='group' aria-label='Basic example'><button class='delete_button btn btn-danger'>删除此项</button><button class='report_button btn btn-info'>结果不对？</button></div></div><p class='result col-md-4 align-self-center'>等待上传……</p></div></div>");
    $('#mylist').append(result);
    result = result.children().children();

    console.log(result);
    (result[0].children)[0].getContext('2d').drawImage(src_canvas,
        x, y, width, height,
        0, 0, width, height
    );
    (result[1].children)[0].children[0].addEventListener('click', () => {
        result.remove()
    });
    (result[1].children)[0].children[1].addEventListener('click', () => {
        let img_canvas = result[0].children[0];
        let name = window.prompt("识别为了错误的角色？请输入他/她正确的名称，将上传图片和您输入的标签信息。");
        if (name != null) {
            $.post("/report", {
                data: img_canvas.toDataURL('image/png'),
                real_name: name
            }, (data, status) => {
                if (status == "success") {
                    alert("Success!")
                } else {
                    alert("Error!")
                }
            });
        }
    })
}