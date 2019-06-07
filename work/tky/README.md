<!-- coding: utf-8 -->

# ACG 人物脸部数据集

*仅用于学习神经网络训练等教育用途，版权归属原作者！*

下载地址：

```
分享地址：http://rec.ustc.edu.cn/share/7497ddc0-7ee8-11e9-a0f8-3ba73db94c06
分享密码：0984
```

## 简单介绍

由 @taoky 连续爆肝接近一周完成的数据集，包含以下人物角色（按文件夹名前编号排序）：

*名称相关信息根据 Wikipedia 等来源整理*

- K-ON! | 《轻音少女》 | けいおん!
    - Tsumugi Kotobuki | 琴吹䌷 | 琴吹 紬
    - Yui Hirasawa | 平泽唯 | 平沢 唯
    - Ritsu Tainaka | 田井中律 | 田井中 律
    - Azusa Nakano | 中野梓 | 中野 梓
    - Mio Akiyama | 秋山澪 | 秋山 澪
- CLANNAD | 《CLANNAD》 | クラナド
    - Tomoya Okazaki | 冈崎朋也 | 岡崎 朋也
    - Nagisa Furukawa | 古河渚 | 古河 渚
    - Kyou Fujibayashi | 藤林杏 | 藤林 杏
    - Kotomi Ichinose | 一之濑琴美 | 一ノ瀬 ことみ
    - Tomoyo Sakagami | 坂上智代 | 坂上 智代
    - Fuko Ibuki | 伊吹风子 | 伊吹 風子
    - Ushio Okazaki | 冈崎汐 | 岡崎 汐
- Hyouka | 《冰菓》 | 氷菓
    - Eru Chitanda | 千反田爱瑠 | 千反田 える
    - Hotaro Oreki | 折木奉太郎 | 折木 奉太郎
    - Satoshi Fukube | 福部里志 | 福部 里志
    - Mayaka Ibara | 伊原摩耶花 | 伊原 摩耶花
- Little Busters! | 《Little Busters!》 | リトルバスターズ!
    - Riki Naoe | 直枝理树 | 直枝 理樹
    - Rin Natsume | 枣铃 | 棗 鈴
    - Komari Kamikita | 神北小球 | 神北 小毬
    - Haruka Saigusa | 三枝叶留佳 | 三枝 葉留佳
    - Kudryavka Noumi | 能美·库特莉亚芙卡 | 能美 クドリャフカ
    - Yuiko Kurugaya | 来谷唯湖 | 来ヶ谷 唯湖
    - Mio Nishizono | 西园美鱼 | 西園 美魚
- Planetarian: The Reverie of a Little Planet | 《星之梦》 | planetarian ～ちいさなほしのゆめ～
    - Yumemi Hoshino | 星野梦美 | ほしのゆめみ
- Is the Order a Rabbit? | 《请问您今天要来点兔子吗？》 | ご注文はうさぎですか?
    - Cocoa Hoto | 保登心爱 | 保登 心愛
    - Chino Kafu | 香风智乃 | 香風 智乃
    - Rize Tedeza | 天天座理世 | 天々座 理世
    - Chiya Ujimatsu | 宇治松千夜 | 宇治松 千夜
    - Syaro Kirima | 桐间纱路 | 桐間 紗路
- 其它 | Others
    - Doudou | 豆豆
    - Menhera | Menhera 酱 | メンヘラちゃん

合计 31 个分类，总计 7106 张图片。

## 命名规则

### 文件夹

命名规则为：`NUMBER-GENDER(F/M)-Main/Secondary character/Unknown-English Name`

注意编号可能不连续。

### 图像文件

文件名开头的：

- `epX` 与 `asepX` 代表对应集数。
- `pixiv` 代表是从 Pixiv 上爬取的。
- `game` 代表来源于原游戏的 CG。
- `stickers` 代表来源于表情包。

## 数量

```
(tf_env) ➜  mtrain for i in ./*; do; echo $i; find $i -type f -print | wc -l; done;
./1-F-M-KON-Tsumugi Kotobuki
     352
./10-F-M-CLANNAD-Nagisa Furukawa
     367
./11-F-M-CLANNAD-Kyou Fujibayashi
     158
./12-F-M-CLANNAD-Kotomi Ichinose
     170
./13-F-M-CLANNAD-Tomoyo Sakagami
     139
./14-F-M-CLANNAD-Fuko Ibuki
      93
./15-F-M-CLANNAD-Ushio Okazaki
      82
./2-F-M-KON-Yui Hirasawa
     577
./24-F-M-Hyouka-Eru Chitanda
     192
./25-M-M-Hyouka-Hotaro Oreki
     188
./26-M-M-Hyouka-Satoshi Fukube
      95
./27-F-M-Hyouka-Mayaka Ibara
     107
./28-M-M-Little Busters-Riki Naoe
     549
./29-F-M-Little Busters-Rin Natsume
     334
./33-F-M-Little Busters-Komari Kamikita
     284
./34-F-M-Little Busters-Haruka Saigusa
     187
./35-F-M-Little Busters-Kudryavka Noumi
     243
./36-F-M-Little Busters-Yuiko Kurugaya
     128
./37-F-M-Little Busters-Mio Nishizono
      98
./38-F-M-Planetarian-Yumemi Hoshino
     121
./39-F-M-Is the Order a Rabbit-Cocoa Hoto
     287
./4-F-M-KON-Ritsu Tainaka
     402
./40-F-M-Is the Order a Rabbit-Chino Kafu
     259
./41-F-M-Is the Order a Rabbit-Rize Tedeza
     181
./42-F-M-Is the Order a Rabbit-Chiya Ujimatsu
     121
./43-F-M-Is the Order a Rabbit-Syaro Kirima
     116
./44-M-U-stickers-Doudou
      21
./45-F-U-stickers-Menhera
      83
./7-F-M-KON-Azusa Nakano
     211
./8-F-M-KON-Mio Akiyama
     461
./9-M-M-CLANNAD-Tomoya Okazaki
     500
./README.md
       1
```
