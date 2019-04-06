# Mask-RCNN

###### tags: `Computer Vision`
# Mask R-CNN

![](https://i.imgur.com/IxgFXtS.png)

&emsp;&emsp;我們會先介紹這篇論文的摘要和研究背景，還有物體分割這個領域，然後簡單介紹Mask R-CNN的整體架構，最後說明實驗結果和結論。

![](https://i.imgur.com/AO1SgAe.png)

&emsp;&emsp;首先，Mask R-CNN是物體分割(Instance Segmentation)一個簡單、有彈性、一般性的框架。這個方法它能夠偵測出一個影像中的物件同時對每個物件產生高品質的mask。

&emsp;&emsp;Mask R-CNN是Faster R-CNN的延伸，它除了有現有的用來識別邊界框(bounding box)的branch之外，它增加一個branch去同時間預測物體的mask。他相較Faster R-CNN來說只增加了很小的overhead。

&emsp;&emsp;Mask R-CNN在物體分割、邊界框物件偵測和人物的關鍵點偵測上都有很好的表現。

![](https://i.imgur.com/h8V0uaY.png)

&emsp;&emsp;在電腦視覺的領域中，物件偵測(Object detection)和語義分割(semantic segmentation)都在短時間內有快速的發展。這兩個部分的進步分別是因為Fast/Faster RCNN和Fully Convolutional Network (FCN)的出現。

&emsp;&emsp;Segmentation的問題其實就是物件偵測(Object detection)和語義分割(semantic segmentation)這兩個問題的結合。也就是先對影像中的個別物件做分類並產生bounding box，初步定出每個物件的位置(物件偵測)再根據bounding box範圍內的每個pixel去分類，訂出物件的mask(語義分割)。

&emsp;&emsp;語義分割（semantic segmentation）：是指以像素為單位標示出物體, 也就是說每個像素都會有分類的結果。以下圖為例，藍色輪廓內的每個像素都被分類為DOG, 而我們通常把這樣的輸出叫做遮罩Mask。

&emsp;&emsp;實例分割(Instance segmentation): 和語義分割的差別在於實例分割可以區別圖像中不同的物體, 比如標示圖像中有兩隻CAT, 他們會各自有不同的mask，而在語義分割只會有一個mask。

![](https://i.imgur.com/GmfFaCg.png)




![](https://i.imgur.com/0YKp6qP.png)

&emsp;&emsp;物件偵測的問題用Fast或Faster R-CNN可以很好解決；語義分割用Fully Convolutional Network可以很好解決；Mask R-CNN的目標也就是要產生像他們一樣是meta algorithm、有好的速度、準確率、直觀又好用的framework。


![](https://i.imgur.com/HDAfCY8.png)



&emsp;&emsp;在物體分割的領域中，根據過去幾年所提出的方法可以分成兩種導向，一種就是以R-CNN為導向，先語義分割每個物體再做分類，另一種是以FCN為導向，先整塊分割再去cut每個物體。


![](https://i.imgur.com/JIiWU2l.png)

&emsp;&emsp;Mask R-CNN其實就是結合了這兩項技術，他是Faster R-CNN與run在每個RoI特徵區域上的FCN的結合。


![](https://i.imgur.com/mQSzO0S.png)

&emsp;&emsp;現在我們要來看Mask R-CNN的整體架構，我們會從以上這幾個層面來說明。

![](https://i.imgur.com/uETL8Qz.png)

&emsp;&emsp;首先我們來看它的整體架構，作者將網絡分成兩個部分：用於提取特徵的“backbone”，和用於classification、regression、mask prediction的“head”。


![](https://i.imgur.com/UrZpE6z.png)

&emsp;&emsp;"backbone"方面，作者用了兩種不同深度的殘差網絡(50層的ResNet和101層的ResNeXt)。ResNet提出了一種減輕網絡訓練負擔的殘差學習框架，這種網絡比以前使用過的網絡本質上層次更深。


![](https://i.imgur.com/zBu5uiv.png)

&emsp;&emsp;另外，作者還使用了Feature Pyramid Network（FPN）。 FPN根據特徵的規模大小，從不同級別層次的特徵中提取 RoI features 。FPN結構中包括自下而上，自上而下和橫向連接三個部分，如左圖所示。這種結構可以將各個層級的特徵進行融合，使其同時具有強語義訊息和強空間訊息，FPN實際上是一種通用架構，可以結合各種骨架網絡使用，比如VGG，ResNet等。 

&emsp;&emsp;Mask RCNN文章中使用了ResNet-FPN網絡結構。如右圖。那因為這篇論文裡面沒有去解釋backbone的部分，所以這裡暫不介紹，有興趣的話大家可以去看ResNet和FPN的論文。在網絡主幹上，作者的結論是：在Mask R-CNN，用ResNet-FPN的骨幹做特徵提取會得到良好的準確度和速度。


![](https://i.imgur.com/Iu074ii.png)

&emsp;&emsp;接下來Feature maps 會經過RPN然後產生Proposals這個RPN的方法是延自Faster RCNN，Faster RCNN拋棄了傳統的滑動窗口和selective search方法，直接使用RPN生成檢測框，能極大提升檢測框的生成速度。可以看到RPN網絡實際分為2條線，上面一條通過softmax分類anchors獲得foreground和background（檢測目標是foreground），下面一條用於計算對於anchors的bounding box regression偏移量，以獲得精確的proposal。而最後的Proposal層則負責綜合foreground anchors和bounding box regression偏移量獲取proposals，同時剔除太小和超出邊界的proposals。其實整個網絡到了Proposal Layer這裡，就完成了相當於目標定位的功能。


![](https://i.imgur.com/JEWaAOU.png)

&emsp;&emsp;這邊在更詳細說明一下，RPN依靠一個在共享特徵圖上滑動的窗口，為每個位置生成9種預先設置好長寬比與面積的目標框(叫做anchor)。這9種初始anchor包含三種面積(128×128，256×256，512×512)，每種面積又包含三種長寬比(1:1，1:2，2:1)。如上方右圖所示。

&emsp;&emsp;對於生成的anchor，RPN要做的事情有兩個，第一個是判斷anchor到底是前景還是背景，意思就是判斷這個anchor到底有沒有覆蓋目標，第二個是為屬於前景的anchor進行第一次坐標修正。


![](https://i.imgur.com/Zfb6EhA.png)

&emsp;&emsp;再來，我們來介紹RoIAlign的運作原理。


![](https://i.imgur.com/bko9qBN.png)

&emsp;&emsp;RoIAlign主要的目的是讓目標定位更準確。而讓定位更準確的主要原因有二個:第一個是 No quantization，第二個是 Bilinear interpolation。

&emsp;&emsp;No quantization 就是計算過程中不會強制把小數點變成整數。

![](https://i.imgur.com/U6fdUL4.png)

&emsp;&emsp;Bilinear interpolation 就是在x軸和y軸方向上，分別做一次線性插值。


![](https://i.imgur.com/PihikM8.png)

&emsp;&emsp;RoIAlign包括Bilinear interpolation和Max pooling，計算過程皆不會去除小數點。


* RoIAlign要解決的問題（即RoI Pooling存在缺陷）：
&emsp;&emsp;在計算RoI Pooling時，會進行兩次量化（在這裡指的就是去掉小數部分，只保留整數），在進行量化時，特徵圖對應的原始數據會有誤差，影響模型整體精度。

* 解決方案：
&emsp;&emsp;保留所有特徵圖所在浮點數位置坐標，使用雙線性插值獲取特徵圖上所有點的像素值。

&emsp;&emsp;圖中藍點就是當前特徵圖中各點位置（位置坐標不一定是整數）。圖中藍線相交點位置就是原始特徵圖中各點的實際位置（位置坐標都是整數）。

* RoIAlign步驟:

	1. 將RoI區域分割成k x k個bin，每個bin的邊界也不做量化。(bin即圖中的圓點)。

	2. 對四個與當前bin最近的實際特徵點使用雙線性插值，來計算當前特徵點的像素值然後進行Max pooling。



![](https://i.imgur.com/QSF7iZB.png)

&emsp;&emsp;假設我們要把17x17的圖max pooling成7x7，那我們就必須知道他的stride，stride就是filter每次移動的間隔，stride算法很簡單，把input的邊長除以output的邊長就可以得到，由於RoIAlign是No quantization ，所以會保留小數點，這樣可以避免誤差的產生。






![](https://i.imgur.com/VElwlE7.png)

&emsp;&emsp;通過RoIAlign會得到固定大小的feature map，接著這個feature map會傳到mask branch做mask prediction 也會傳給另外一個branch做box regression跟classification，而這三個任務是同時進行的，我們先來介紹mask branch。


![](https://i.imgur.com/KKLa02H.png)

&emsp;&emsp;Mask branch是作用於每個RoI的全卷積神經網路(FCN)，用pixel-to-pixel的方式來預測segmentation mask。從他的輸出可以得到一個binary mask(如上圖所示)，我們可以透過設計一個threshold，讓這個binary mask變得更完整(只有黑和白)。


![](https://i.imgur.com/6gxfhUb.png)

&emsp;&emsp;接下來，產生出來的binary mask會貼回去圖上，讓圖中的目標有像這樣的一個mask


![](https://i.imgur.com/7gTqgX8.png)

&emsp;&emsp;最後，我們來介紹box regression和classification的運作原理。

![](https://i.imgur.com/c0YISOp.png)

&emsp;&emsp;Box Regression簡單來說就是平移+尺度縮放，藉由這樣的方法讓預測結果更接近目標的真實位置。

&emsp;&emsp;在訓練階段中，有二個Input，一個是Region Proposal，另一個則是Ground Truth，由上面這二個元素，我們可以計算出每個proposal的位置偏移量(需要平移、縮放的量)，藉此來訓練模型，希望預測結果(bbox_pred)和真實位置(Ground Truth)誤差越小越好。


![](https://i.imgur.com/iB5UL5o.png)

&emsp;&emsp;Classification這個階段是用來分類各個物體的類別，cls_prob即是物體屬於那個類別的機率。


![](https://i.imgur.com/hJeqTQp.png)

&emsp;&emsp;對於不相連的物體Mask R-CNN也可以正確的分割


![](https://i.imgur.com/TTSBsuu.png)

&emsp;&emsp;對於小物體Mask R-CNN也可以分割出來


![](https://i.imgur.com/8TaDkE9.png)

&emsp;&emsp;比較各種backbone架構的準確度，網路越深效果越好。

&emsp;&emsp;傳統上要提高模型的準確率，都是加深或加寬網路，但是隨著超參數數量的增加，網絡設計的難度和計算開銷也會增加。ResNeXt 結構可以在不增加參數複雜度的前提下提高準確率，同時還減少了超參數的數量。

&emsp;&emsp;AP，一種衡量準確率的東西。AP at IoU=.50，會計算IoU為0.50時的準確率。AP at IoU=0.50:0.05:0.95 ，會計算IoU=0.5、0.05、0.95等等的準確率然後取平均。

&emsp;&emsp;(IOU:模型產生的目標窗口和原來標記窗口的交疊率)

![](https://i.imgur.com/vDxAeCN.png)




![](https://i.imgur.com/ueEKuGz.png)

&emsp;&emsp;Apbb : bounding box的averge precise。


![](https://i.imgur.com/bYP3HBI.png)

&emsp;&emsp;作者比較了用MLP和FCN來做mask prediction的效果。


![](https://i.imgur.com/XXaMT7t.png)

&emsp;&emsp;各種方法的比較，Mask R-CNN在小物體分割上相對其他方法有較好的結果。

&emsp;&emsp;AP Across Scales:
&emsp;&emsp; - APsmall，AP for small objects: area < 32^2
&emsp;&emsp; - APmedium，AP for medium objects: 32^2 < area < 96^2
&emsp;&emsp; - APlarge，AP for large objects: area > 96^2


![](https://i.imgur.com/8EhJsja.png)

&emsp;&emsp;Instance Segmentation包含了Objection Detection & Semantic Segmentation 。因為Mask R-CNN的設計讓他在instance segmentation有很好的效果。

&emsp;&emsp;Mask R-CNN是Faster R-CNN的擴展，他增加了FCN並且用RolAlign代替RoIpool，提高了目標定位的準確率。








