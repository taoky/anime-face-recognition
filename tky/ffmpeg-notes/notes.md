每 5 秒取一张图，缩放为 960x540：

`ffmpeg -i 1.flv -s 960x540 -vf fps=1/5 test%d.png`
