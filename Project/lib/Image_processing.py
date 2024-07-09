import cv2
import numpy as np
import matplotlib.pyplot as plt

class Image():
    def __init__(self):
        self.path = None
        self.original_img = None
        self.resize_img = None
        self.dumy_img = None
        self.cliping_img = None
        self.blur_img = None
        self.binary_img = None
        self.canny_img = None
        self.processed_img = None
        self.basewidth = None
        self.rate = 2
        self.img_size = 8

    def RoadImage(self, path):
        """
        画像読み込み
        Args:
            path : 画像のファイルパス
        """
        self.path = path
        self.original_img = cv2.imread(self.path)

    def Initial_process(self):
        """
        読み込んだ画像の初期化処理
        """
        # 画像サイズの変更
        self.resize_img = cv2.resize(self.original_img, (self.original_img.shape[0]//self.img_size, self.original_img.shape[1]//self.img_size))
        # 画像のグレースケール化
        self.resize_img = cv2.cvtColor(self.resize_img, cv2.COLOR_BGR2GRAY)
        # 20nmの基準値取得
        self.basewidth = self.BaseWidth(self.resize_img, threshhold=200)
        # 輝度の反転
        self.resize_img = cv2.bitwise_not(self.resize_img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.resize_img = clahe.apply(self.resize_img)

    def Binary(self, thresh):
        """
        画像の2値化処理
        Args:
            thresh : 2値化の閾値
        """
        self.dumy_img = self.blur_img.copy()
        _, self.processed_img = cv2.threshold(self.dumy_img, thresh, 255, cv2.THRESH_BINARY)
        return self.processed_img
    
    def GaussianBlor(self, filter):
        """
        ガウシアンフィルタ処理
        Args:
            filter : ガウシアンフィルタにつかうカーネルサイズ
        """
        self.dumy_img = self.cliping_img.copy()
        sigma = 0
        self.dumy_img = cv2.GaussianBlur(self.dumy_img, (filter, filter), sigma)
        return self.dumy_img
    
    def ShowRectangle(self, x, y):
        """
        切り取り場所を描画
        Args:
            x : x座標
            y : y座標
        """
        self.dumy_img = self.resize_img.copy()
        self.processed_img = cv2.rectangle(self.dumy_img, (x, y), (x+self.basewidth, y+self.basewidth), (255, 255, 255), thickness=1)
        return self.processed_img
    
    def CutRectangle(self, x, y):
        """
        画像切り取り
        Args:
            x : x座標
            y : y座標
        """
        self.dumy_img = self.resize_img.copy()
        self.processed_img = self.dumy_img[y:y+self.basewidth, x:x+self.basewidth]
        # self.processed_img = cv2.resize(self.processed_img, (self.original_img.shape[0]//self.img_size, self.original_img.shape[1]//self.img_size))
        return self.processed_img
    
    def EqulizeHist(self):
        self.dumy_img = self.blur_img.copy()
        
        return self.processed_img
    
    def BaseWidth(self, img, threshhold=200):
        """
        20nmの基準を検出
        img : 2値化された画像データ (20nmの文字が白い場合のみ)
        threshhold : 長方形のみを検出するための閾値
                     長方形の長さが閾値より小さいと見つけられない
        """
        # 2値化
        _, img = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY)
        # 輪郭の数が一つのみになるまでthreshholdを上げる
        while(True):
            # 輪郭を検出
            contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            width_list = []
            for i in range(0, len(contours)):
                if len(contours[i] > 0):
                    # 閾値より小さいものを排除
                    if cv2.contourArea(contours[i]) < threshhold:
                        continue
                    contour = contours[i].reshape(-1, 2)
                    # 幅の計算
                    width = np.int64((contour[:, 0].max() - contour[:, 0].min()) / self.rate)
                    width_list.append(width)

            if len(width_list) != 1:
                threshhold += 10
            else:
                break
        return width_list[-1]


    def Object_Detection(self):
        """
        切り取った画像から物体の長さを検出
        """
        self.dumy_img = self.canny_img.copy()
        # 輪郭を検出
        contours, hierarchy = cv2.findContours(self.dumy_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.dumy_img = cv2.cvtColor(self.dumy_img, cv2.COLOR_GRAY2BGR)
        self.object_length = []
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int64(box)
            perimeter = cv2.arcLength(contour, True)
            # 周囲長さを10nmの基準長さで割る
            perimeter = perimeter / self.basewidth
            self.object_length.append(perimeter)


    def Canny(self, threshold_upper, threshold_bottom):
        """
        エッジ検出
        """
        self.dumy_img = self.binary_img.copy()
        self.processed_img = cv2.Canny(self.dumy_img, threshold1=threshold_bottom, threshold2=threshold_upper)
        return self.processed_img


    def Object_analytics(self):
        """
        検出したオブジェクトをヒストグラムで表示
        """
        self.object_length = np.array(self.object_length)
        # グラフ描画
        # plt.xlim(1, 10)
        plt.xlabel("nm", fontsize=20)
        plt.grid(True)
        plt.tick_params(labelsize=12)
        plt.hist(self.object_length*10, bins=100, range=(0.5, 10.5), histtype='bar', align='mid', rwidth=1.0)
        plt.show()
        
        plt.show()

    def Reset(self):
        self.path = None
        self.original_img = None
        self.resize_img = None
        self.dumy_img = None
        self.cliping_img = None
        self.blur_img = None
        self.binary_img = None
        self.canny_img = None
        self.processed_img = None
        self.basewidth = None