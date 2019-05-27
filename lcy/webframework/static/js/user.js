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
        // let roiGray = gray.roi(face);
        // let roiSrc = src.roi(face);
        // let point1 = new cv.Point(face.x, face.y);
        // let point2 = new cv.Point(face.x + width, face.y + height);
        console.log(faces.get(i));
        // cv.rectangle(src, point1, point2, [255, 0, 0, 255]);

        var $result = $("<canvas class='result_pic' width='" + width + "' height='" + height + "'></canvas><button class='delete_button'>删除此项</button>");
		$('body').append($result);
		$result[0].getContext('2d').drawImage(src_canvas,
			face.x, face.y, width, height,
			0, 0, width, height
        );
        
        // roiGray.delete(); roiSrc.delete();
    }
    // cv.imshow('canvasOutput', src);
    src.delete(); gray.delete(); faceCascade.delete();
    faces.delete();
}