# 动漫人物分类识别
Python &amp; Deep Learning 101 Assignment #2

## 分工情况

[TBD]

## 任务说明

### 来源

一开始定题的时候大家都很纠结，想了几天都没有很好的想法。之后有一天我看到有人在某个群里发了一张很好玩的 GIF 表情。

![1](report_imgs/1.gif)

*就是这张，原图没有找到，自己花了点时间重新做了一份。*

然后呢，我就隐隐约约有一种「好像很熟悉但是忘了是在哪儿看到的了」这种感觉，~~都是京都脸害的，~~ 之后就想——要不就搞一个学习分类 ACG 人物的项目吧，告诉我们图片中出现的角色到底是谁。图像分类应该是比较成熟的了，对于我们这四条以前从来没玩过深度学习的咸鱼来说（看起来）还是比较合适的。

于是最后大家就一致同意（？）了，在考完数理方程之后就开搞了。

### 实际问题

正如上文提到的那样，有时候你看到了一张可爱或者搞笑的表情包、一张动画截图，或者一副 ACG 人物的插画，但很多时候你就是不知道在图片里面他们/她们是谁。

Google 搜图可以帮助解决一部分的问题，但是——

- 不是所有人都有畅通的网络。

- 有时候第一眼的结果会让你比较失望。

  ![2](report_imgs/2.png)

  *你说得很对，但是一点用都没有。*

同时，网络中也有一些专门用于搜索 ACG 图片等的服务，例如 [trace.moe](https://github.com/soruly/trace.moe)，使用了 [MPEG 7 Color Layout Descriptor](https://en.wikipedia.org/wiki/Color_layout_descriptor)，一种高效的分块、概括颜色的算法。其对动画的每帧都做了这样的索引。但也同时由于这个特性，在图像被裁减之后，由于全图颜色分布发生了比较大的变化，识别准确率会大幅下降。

所以，如果我们有<u>足够的数据</u>的话，最后用深度学习来做这个效果可能会比这几种方案都好得多。

### 难点

- 数据很可能会不够：收集**足够多**的、**高质量**的角色面部数据不是一件简单的事情。对于神经网络来说，可能几千张还嫌少。
- 「长得差不多」：有好几位同学都跟我说「感觉这些角色长得都差不多啊」，~~不知道我们训练出来的网络会不会也这么想。~~  最后能不能有效区分各个角色也是一个问题。

## 数据处理

既然这个题目是我提的，而且看起来我们四个人中就只有我~~这个死宅~~看动画比较多，所以数据的收集与处理就只能我来做了。结果就是：爆肝了近一周才勉强把数据弄好。

### 收集

#### 现成的数据集？

第一步当然是去找有没有现成的数据集了。结果在 Kaggle 上有一个， 其中一个压缩包连标签都打好了，看起来很 promising。

![3](report_imgs/3.png)

*[Tagged Anime Illustrations](https://www.kaggle.com/mylesoneill/tagged-anime-illustrations/home)*

我花（浪费）了两天时间整理了一下 `moeimouto-faces.zip` 中的内容，发现了一些残酷的现实：

- ~~有很多角色我都不认识。~~
- 图片的质量说实话，不是很好，有一些分类错误、难以辨认的，甚至还有一些比较恶心的图片……

最后我们没有使用这个数据集。如果你感到好奇的话，我发布了我整理过的版本（大小：311.37MB）：

```
分享地址：http://rec.ustc.edu.cn/share/052db520-7ba4-11e9-9c5f-2172227f4743
分享密码：948a
```

#### 视频、`ffmpeg` 与面部识别方案

之后，我就打算自己来搞数据了。首先想到的是从现有的动画的视频中截取一些帧出来再做处理。

- 如何下载需要的动画视频？对于大多数在线的流媒体服务来说，上面的视频可以用 [`you-get`](https://github.com/soimort/you-get), [`youtube-dl`](https://github.com/ytdl-org/youtube-dl), [`annie`](https://github.com/iawia002/annie) 等工具下载到。至于其它来源……方法总比困难多嘛，这里就不赘述了。

- 如何从视频中抽取帧？我的第一反应就是 `ffmpeg`（一个开源的多媒体处理程序），[果不其然](https://trac.ffmpeg.org/wiki/Create%20a%20thumbnail%20image%20every%20X%20seconds%20of%20the%20video)。以下的命令可以每 5 秒取一张图保存：

  ```
  ffmpeg -i 1.flv -vf fps=1/5 test%d.png
  ```

  为了 SSD 空间和后续处理方便考虑，需要把导出的图片弄小一点，加一个参数就能解决。

  ```
  ffmpeg -i 1.flv -s 960x540 -vf fps=1/5 test%d.png
  ```

  之后导出的图片大小都是 960x540 的了。

- 把一整张图打上标签就可以了吗？当然不行！

  - 一张图中可能会有多个角色。
  - 一整张图中有很多部分对我们识别人物没有价值，去掉它们可以加快训练的速度，减小对内存等资源的要求。
  - 所以要做面部识别！

##### OpenCV 方案：`lbpcascade_animeface.xml`

传统上来讲，OpenCV 使用预训练的 Haar Cascades 分类器检测物体（比如说人脸、人眼、猫咪，下面以人脸为例）。其预先拿一堆有人脸（positive）和没人脸（negative）的图片训练 cascade function。训练时使用了哈尔特征（Haar-like features），计算窗口中不同位置像素和的差，以差值做分类。此外，还使用了一些其他的算法，例如整合不同分类器的 Adaboost。[^opencv]

对于人脸来说，预置的分类器大部分时候没啥问题。

![4](report_imgs/4.png)

*Lenna 图官方样例。*

但根据以上的描述，我们可以猜到，这个分类器用在 ACG 人物上会发生什么。

![5](report_imgs/5.png)

*看起来不是很靠谱。*

所以怎么办？OpenCV 文档里有一篇[训练自己的 Haar Cascades 分类器的指南](https://docs.opencv.org/3.4/dc/d88/tutorial_traincascade.html)。但这不代表我们要自己来，因为已经有人干过这样的事情了。2011 年，[nagadomi](https://github.com/nagadomi)（他/她也是著名的动画风格图片超分辨率重建程序 [waifu2x](https://github.com/nagadomi/waifu2x) 的作者！）训练了一个动漫人物分类器：[lbpcascade_animeface](https://github.com/nagadomi/lbpcascade_animeface)。直接拿过来用就行。

![6](report_imgs/6.png)

*识别出来了。*

##### `animeface-2009`

这是另一个动画面部识别的方案（和上面那个还是同一个作者），链接在[此](https://github.com/nagadomi/animeface-2009)。它与 `lbpcascade_animeface.xml` 相比准确性更高，但同时也需要更多的内存资源。如果内存不足，对于过大的图片可能会出现错误。

![7](report_imgs/7.png)

*左：`animeface-2009` 结果；右：`lbpcascade_animeface.xml` 结果。*

我构建了一个 Dockerfile 文件，方便此程序的部署。

最终的流程是：

- `ffmpeg` 在视频中抽取图像并缩放。
- `animeface-2009` 识别每张图中的脸部，生成「数据集」。
  - 我们使用的参数为：`--threshold 0.7 --margin 0.3`
- 那 `lbpcascade_animeface.xml` 呢？我们之后也会用到，后面会提到。

#### 无聊的日常

很遗憾，在这套流程走完之后，接下来就是机械性的打标签流程了。

![8](report_imgs/8.png)

*把左边的图片中的 true positive 的部分正确地拖动到右边的文件夹里就行了。根据视频采样频率的不同，可能处理每一集的数据会有几十次到上百次的拖动。*

这花费了我绝大多数的时间。实话讲我不想这么搞，但是没有找到很好的办法。拜托别人来做也很困难，因为这一项分类需要后验知识（你在不了解某部作品的情况下，是不知道谁是谁的）。

#### 数据能不能再多一点？

我之前考虑过从一些插画网站上用爬虫爬取一些图片来做处理。最终我尝试使用 [PixivUtil2](https://github.com/Nandaka/PixivUtil2) 来根据标签爬取 Pixiv 上对应的图片，然后扔给 `animeface-2009` 来处理。

听起来是个不错的主意，但是：

- 我没有会员，搜索的时候没有办法按照热门度排序，只能按照时间来排。而这样的后果就是，在搜索列表里我看到了一些……像是小学生绘画风格的……作品。
- 就算在爬取的时候设置了收藏数的门槛，但是最后获得的结果依然画风各异，不太稳定。在 `animeface-2009` 处理完之后，有些我根本认不出来是谁。

最后向收集的数据集中加入了少量这类的图片。

#### 数据集总结

最终的数据集合计 31 个分类，来自 6 部动画作品 + 2 个小表情包，总计 7106 张图片。因为配角~~戏份~~数据不够，只给主角做了分类。

## 算法原理

[TBD]

## 实验细节

### 关于深度学习

[TBD]

### 关于「网站」的前后端设计

我们最终呈现给用户的是一个网站。这个网站允许用户选择图片中出现的动漫人物脸部图片并上传，服务器计算后返回结果。与此同时，用户可以向我们反馈识别错误的结果，以允许我们进一步优化。

#### 前端

为了方便用户选择脸部特征，我们决定把识别任务放在前端来做。这里就使用到了 OpenCV。通过近年来大热的 WebAssembly 技术——在现代浏览器中执行类似汇编的二进制代码，达到接近原生的性能——，OpenCV 也可以在浏览器端执行。这就是 OpenCV.js。

##### OpenCV.js

为了得到 OpenCV.js 文件，我们需要安装 Emscripten，它可以将 C/C++ 代码编译为 JavaScript 代码。

获取 OpenCV 源码后，就可以编译了。
```
git clone https://github.com/opencv/opencv.git
<opencv_src_dir>/platforms/js/build_js.py <build_dir>
```
需要注意的是这个脚本依赖 cmake。
![](report_imgs/9.png)

编译完成后，我们可以使用官方样例测试一下。

```html
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Hello OpenCV.js</title>
</head>
<body>
<h2>Hello OpenCV.js</h2>
<p id="status">OpenCV.js is loading...</p>
<div>
  <div class="inputoutput">
    <img id="imageSrc" alt="No Image" />
    <div class="caption">imageSrc <input type="file" id="fileInput" name="file" /></div>
  </div>
  <div class="inputoutput">
    <canvas id="canvasOutput" ></canvas>
    <div class="caption">canvasOutput</div>
  </div>
</div>
<script type="text/javascript">
let imgElement = document.getElementById('imageSrc');
let inputElement = document.getElementById('fileInput');
inputElement.addEventListener('change', (e) => {
  imgElement.src = URL.createObjectURL(e.target.files[0]);
}, false);
imgElement.onload = function() {
  let mat = cv.imread(imgElement);
  cv.imshow('canvasOutput', mat);
  mat.delete();
};
function onOpenCvReady() {
  document.getElementById('status').innerHTML = 'OpenCV.js is ready.';
}
</script>
<script async src="opencv.js" onload="onOpenCvReady();" type="text/javascript"></script>
</body>
</html>
```

![](report_imgs/10.png)

结果显示 `OpenCV.js is ready.`，没有问题。

##### 其它逻辑部分

这里的主要工作是写 HTML, CSS 和 Javascript。最起始的代码是 lcy 写的，使用了 OpenCV.js 和[一个交互式剪裁图片的前端库](https://github.com/zimv/zmCanvasCrop)。但是由于 lcy 误解了 tky 的需求，并且代码质量也不是很高，就由我来对前端代码进行修改，并且添加需要的功能。

主要的修改有：

- 由于 lcy 以前没有写过 JS，代码出现了一些非常怪异的做法，比如说在某个标签中加入代码，然后 JS 部分获取标签中的内容再 `eval()`。我修改了一部分代码，将一些功能包装成了函数，以使结构看起来更加清晰。
- 删除了一些根本没有用到的函数。
- 原先 lcy 选择的前端库中一剪裁结束就会调用上传函数，不太符合我们的需求。对这个前端库的代码做了一些修改。
- 使 OpenCV.js 脸部识别和用户手动剪裁的结果能够共同塞入一个 `<div>` 中，形成列表。加入了删除某项的功能、反馈错误结果的功能，最终上传时会一起上传到服务器上。

同时，我也把后端接收图片的功能单独包装成了函数。

使用 Bootstrap 对网页进行美化的工作由 tky 完成。

#### 后端

后端使用 Flask 框架，配合 TensorFlow，使用已经训练好的模型提供分类结果。Python 代码主要分成两个文件：`app.py` 和 `classification.py`。

##### `app.py`

这个文件为整个 Flask web app 的核心文件，设置了三个路由：

- `@app.route("/")`: 返回 `index.html` 的内容。
- `@app.route("/uploadForIdent", methods=['POST'])`: 接收用户上传识别的图片，并调用 `classification` 中的 `web_api` 函数。
- `@app.route("/report", methods=['POST'])`: 接收用户的错误反馈，并保存。

其中 `receive_img()` 是从前端接收图片（的 base64）并保存的辅助函数。

##### `classification.py`

此文件作为神经网络的 API 接口，我们通过此文件实现后端和神经网络的交互。

其它需要的 CSS, JS 与数据文件在 `static/` 目录下，主页的 HTML 在 `templates/` 目录下。

## 实验总结

[TBD]