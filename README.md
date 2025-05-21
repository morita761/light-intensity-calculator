## 環境構築

- pip install opencv-python
- pip install opencv-contrib-python

## ロジック

1. 画像からメダラを抽出->緑の領域と画像のトリミング
2. メダラをV-D方向に分ける->`cv2.inRange`による緑マスク化から中央値を計算し，左右分割
3. 馬蹄形を認識->着手中，gray scale,コントラスト補正，
4. 輝度測定


## TODO
- 作成したコードの調整，馬蹄形の認識をまずやる

## 馬蹄形をどう認識するか
- 一つの馬蹄形でも明度の差があるため輪郭抽出がむずかしい
- 複数の馬蹄形が密集している

2025/05/21 21:51の段階
gray scaleからコントラスト補正をかけた画像は，だいぶ馬蹄形が強調されている！  
ここから，フィルターをかけて輪郭抽出を考えている。  
まずは馬蹄形で輪郭抽出ができている画像を得たい。どうしようか。線がつながってないんだよなぁ  
モルフォロジー処理がうまくいっていない。やめるか？

コントラスト補正->適応閾値（輪郭抽出の第一段階，点が離れている）->モルフォロジー処理

馬蹄形を自分で書いて，それとマッチするものを選ぶ方式をためした  
pic/にpptとペイントで作成した馬蹄形を入れた  
輪郭がわかれていないものがまだあったので，3つほどしか認識できていない。

`とりあえず，コントラスト補正->Canny法->輪郭認識が一番綺麗な馬蹄形とれる，ただ，運がよくないとだめ，すくない`  
`コントラスト補正->適応閾値->モルフォロジー処理->輪郭認識->馬蹄形選びが，上下の比較が無いし，形が微妙`  
輝度を測定もとりあえずで実装したが，offsetがよくない。そもそも測定領域が枠内にはいってない　 

今後，設定値をファイル形式で入力・出力ができるようにしたい。また，保存する関数も別に設定して，そのときの画像も取得したい。


## OpenCV

### 二値化処理

https://www.codevace.com/py-opencv-threshold/  
cv2.THRESH_BINARY  
cv2.THRESH_TOZERO  



### 適応的しきい値

明暗差の影響を軽減し、局所的な境界を強調  
blockSizeを調整して局所的な領域を調整できる，
https://www.codevace.com/py-opencv-adaptivethreshold/  

```
adaptive = cv2.adaptiveThreshold(
    equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY_INV, blockSize=11, C=2)
```

### 輪郭抽出
findContours  
https://www.codevace.com/py-opencv-findcontours/

### 輪郭近似
今回は違うか
cv2.approxPolyDP  
https://www.codevace.com/py-opencv-approxpolydp/  
```
approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
```

### 図形でフィルターをかける
楕円フィッティングの処理  
https://qiita.com/kakuteki/items/fd42a591efd2567d05fd  
```
for contour in contours:
    # 輪郭が5点以上の場合のみ楕円フィッティングが可能
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        cv2.ellipse(image, ellipse, (0, 255, 0), 2)  # 楕円を描画
```

### matchShape
cv2.matchShapes  
https://pystyle.info/opencv-match-shape/  
https://www.codevace.com/py-opencv-humoments/  




### ヒストグラム平坦化でコントラスト補正

ヒストグラム平坦化でコントラスト補正（均一化）
```
equalized = cv2.equalizeHist(gray)
```
CLAHE（局所適応ヒストグラム平坦化），こっちを採用
```
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
equalized = clahe.apply(gray)
```

### モルフォロジー処理

膨張収縮で輪郭をつなげる
```
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
closed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=2)
```

https://qiita.com/sitar-harmonics/items/ae8088d43d5645d671e6  
https://www.codevace.com/py-opencv-morphologyex/  

### グレースケールとガウシアン（ぼかし）
```
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
```


### Canny法
エッジ検出のためのアルゴリズム，つまり二値化する  

https://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_canny/py_canny.html  

```
edges = cv2.Canny(blurred, threshold1=30, threshold2=100)
```
自動でしきい値を調整する方法  
webで見つけたCanny法の自動値算出方法，境界がシッカリしないとだめ
```
med_val = np.median(blurred)
sigma = 0.33  # 0.33
min_val = int(max(0, (1.0 - sigma) * med_val))
max_val = int(max(255, (1.0 + sigma) * med_val))
edges = cv2.Canny(blurred, threshold1=min_val, threshold2=max_val)
```
### DoG
Difference of Gaussian， シグマの値が異なる2つのガウシアンフィルタ画像の差分です。
https://python.joho.info/opencv/opencv-dog-filter-py/
```
blur1 = cv2.GaussianBlur(gray, (3, 3), 0.5)
blur2 = cv2.GaussianBlur(gray, (7, 7), 1.5)
dog = cv2.subtract(blur1, blur2)
```

### 輪郭抽出＋フィルタ
```
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
horseshoe_mask = np.zeros_like(gray)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if 300 < area < 1000:  # 馬蹄形のサイズ範囲を想定（要調整）
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        # ここで形状が「開いた円弧」か「U字形」かをチェックするロジックも組める
        cv2.drawContours(horseshoe_mask, [cnt], -1, 255, -1)
```

### 重なり分離には「Watershed法」も有効
特に 馬蹄形が密集・接触していて分離不能な場合：  
Watershed segmentation（分水嶺アルゴリズム）
1. 二値化画像に距離変換（cv2.distanceTransform）
2. ローカルマキシマムを種としてWatershedで分離
3. ラベリングされた馬蹄形を個別に処理可能