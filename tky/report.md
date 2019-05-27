## 任务说明

### 来源

一开始定题的时候大家都很纠结，想了几天都没有很好的想法。之后有一天我看到有人在某个群里发了一张很好玩的 GIF 表情。

![1](../report_imgs/1.gif)

*就是这张，原图没有找到，自己花了点时间重新做了一份。*

然后呢，我就隐隐约约有一种「好像很熟悉但是忘了是在哪儿看到的了」这种感觉，~~都是京都脸害的，~~ 之后就想——要不就搞一个学习分类 ACG 人物的项目吧，告诉我们图片中出现的角色到底是谁。图像分类应该是比较成熟的了，对于我们这四条以前从来没玩过深度学习的咸鱼来说（看起来）还是比较合适的。

于是最后大家就一致同意（？）了，在考完数理方程之后就开搞了。

### 实际问题

正如上文提到的那样，有时候你看到了一张可爱或者搞笑的表情包、一张动画截图，或者一副 ACG 人物的插画，但很多时候你就是不知道在图片里面他们/她们是谁。

Google 搜图可以帮助解决一部分的问题，但是——

- 不是所有人都有畅通的网络。

- 有时候第一眼的结果会让你比较失望。

  ![2](../report_imgs/2.png)

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

![3](../report_imgs/3.png)

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

对于人脸来说，这样做大部分时候没啥问题。

![4](../report_imgs/4.png)

*Lenna 图官方样例。*

但根据以上的原理，我们可以猜到，这个分类器用在 ACG 人物上会发生什么。

![5](../report_imgs/5.png)

*看起来不是很靠谱。*

所以怎么办？OpenCV 文档里有一篇[训练自己的 Haar Cascades 分类器的指南](https://docs.opencv.org/3.4/dc/d88/tutorial_traincascade.html)。但这不代表我们要自己来，因为已经有人干过这样的事情了。2011 年，有人训练过了一个动漫人物分类器：[lbpcascade_animeface](https://github.com/nagadomi/lbpcascade_animeface)。直接拿过来用就行。

![6](../report_imgs/6.png)

*识别出来了。*



---

[^opencv]: https://docs.opencv.org/3.4/d2/d99/tutorial_js_face_detection.html