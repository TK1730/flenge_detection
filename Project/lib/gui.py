import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import numpy as np

class ImageProcessor:
    def __init__(self, root, image):
        self.root = root
        self.image = image
        self.root.title("Interactive Image Processor")
        self.image_path = None
        self.processed_img = None

        self.current_stage = 0  # 0 for threshold, 1 for Gaussian filter

        # 画像の表示
        self.image_label = ttk.Label(self.root)
        self.image_label.pack()
        self.image_label.bind("<Button-1>", self.StartCrop)
        self.image_label.bind("<Button-3>", self.EndCrop)
        
        # step 1
        self.load_button = ttk.Button(self.root, text="Load Image", command=self.LoadImage)
        self.load_button.pack()        

        # step 2
        self.clip_text = ttk.Label(self.root, text="左クリック : 切り取り範囲描画\n右クリック : 切り取り")

        # step 3
        self.filter_size_frame = ttk.Frame(self.root)
        self.filter_size_label = ttk.Label(self.filter_size_frame, text="GaussianBlur処理のフィルタサイズを決めてください")
        self.filter_size_slider = ttk.Scale(self.filter_size_frame, from_=1, to=21, orient=tk.HORIZONTAL, command=self.update_filter_size, value=3)
        self.filter_size_entry = ttk.Entry(self.filter_size_frame, width=5)
        self.filter_size_label.pack(side='top')
        self.filter_size_slider.pack(side='left')
        self.filter_size_entry.pack(side='left')
        self.filter_size_entry.bind('<Return>', self.update_filter_size_entry)

        # step 4
        self.binary_threshold_text = ttk.Label(self.root, text="二値化処理")
        self.binary_threshold_frame = ttk.Frame(self.root)
        self.binary_threshold_label = ttk.Label(self.binary_threshold_frame, text="閾値を決めてください")
        self.binary_threshold_slider = ttk.Scale(self.binary_threshold_frame, from_=0, to=255, orient=tk.HORIZONTAL, command=self.update_binary_threshold, value=100)
        self.binary_threshold_entry = ttk.Entry(self.binary_threshold_frame, width=5)
        self.binary_threshold_button = ttk.Button(self.binary_threshold_frame, text='OTHU', command=self.BinaryOtsu)
        
        self.binary_threshold_label.pack(side='top')
        self.binary_threshold_slider.pack(side='left')
        self.binary_threshold_entry.pack(side='left')
        self.binary_threshold_button.pack(side='left')
        self.binary_threshold_entry.bind('<Return>', self.update_binary_threshold_entry)

        # step 5
        self.edge_threshold_text = ttk.Label(self.root, text='エッジ検出処理')
        self.edge_threshold_frame = ttk.Frame(self.root)
        self.edge_threshold_label = ttk.Label(self.edge_threshold_frame, text="閾値")
        self.edge_threshold_upper_slider = ttk.Scale(self.edge_threshold_frame, from_=0, to=255, value=30, orient=tk.HORIZONTAL, command=self.update_canny_threshold)
        self.edge_threshold_upper_entry = ttk.Entry(self.edge_threshold_frame, width=5)
        self.edge_threshold_bottom_slider = ttk.Scale(self.edge_threshold_frame, from_=0, to=255, value=30, orient=tk.HORIZONTAL, command=self.update_canny_threshold)
        self.edge_threshold_bottom_entry = ttk.Entry(self.edge_threshold_frame, width=5)

        self.edge_threshold_label.grid(row=0)
        self.edge_threshold_upper_slider.grid(row=1, column=0)
        self.edge_threshold_upper_entry.grid(row=1, column=1)
        self.edge_threshold_bottom_slider.grid(row=2, column=0)
        self.edge_threshold_bottom_entry.grid(row=2, column=1)
        self.edge_threshold_upper_entry.bind('<Return>', self.update_canny_threshold_entry)
        self.edge_threshold_bottom_entry.bind('<Return>', self.update_canny_threshold_entry)

        # step 6
        self.object_label = ttk.Label(self.root, text='フレンジの大きさ')
        self.object_frame = ttk.Frame(self.root)
        self.object_upper_button = ttk.Button(self.object_frame, text="フレンジの周囲長さ取得", command=self.ObjetDrow)
        self.object_bottom_button = ttk.Button(self.object_frame, text="Restart", command=self.Restart)

        self.object_upper_button.pack()
        self.object_bottom_button.pack()

        # nextボタン
        self.next_button = ttk.Button(self.root, text="Next", command=self.next_stage)
        self.next_button.pack(side='right')

        # backボタン
        self.back_button = ttk.Button(self.root, text="Back", command=self.back_stage)
        self.back_button.pack(side='left')
        self.back_button.config(state='disabled')


    def LoadImage(self):
        self.image_path = filedialog.askopenfilename()
        if self.image_path and self.current_stage == 0:
            self.image.RoadImage(self.image_path)
            # 画像の前処理
            self.image.Initial_process()
            self.display_image(self.image.resize_img)

        
    def StartCrop(self, event):
        """
        左クリックで切り取り範囲の描画
        """
        if self.current_stage == 1:
            self.x, self.y = event.x, event.y
            print(self.x, self.y)
            self.processed_img = self.image.ShowRectangle(self.x, self.y)
            self.display_image(self.processed_img)


    def EndCrop(self, event):
        """
        右クリックで切り取り
        """
        if self.current_stage == 1:
            print(self.x, self.y)
            self.processed_img = self.image.CutRectangle(self.x, self.y)
            self.image.cliping_img = self.processed_img
            self.display_image(self.processed_img)


    def BinaryOtsu(self):
        """
        大津の2値化で自動的に閾値を計算
        """
        ret, _ = cv2.threshold(self.image.blur_img.copy(), 0, 255, cv2.THRESH_OTSU)
        self.binary_threshold_slider.set(ret)
        self.update_binary_threshold()


    def update_binary_threshold(self, event=None):
        """
        2値化の閾値 スライドバー処理
        """
        if self.image.processed_img is not None and self.current_stage == 3:
            threshold = int(self.binary_threshold_slider.get())
            self.binary_threshold_entry.delete(0, tk.END)
            self.binary_threshold_entry.insert(0, str(threshold))
            self.processed_img = self.image.Binary(threshold)
            self.image.binary_img = self.processed_img
            self.display_image(self.processed_img)


    def update_binary_threshold_entry(self, event=None):
        """
        2値化の閾値 数値入力処理
        """
        try:
            threshold = int(self.binary_threshold_entry.get())
            if 0 <= threshold <= 255:
                self.binary_threshold_slider.set(threshold)
                self.update_binary_threshold()
        except ValueError:
            pass


    def update_canny_threshold(self, event=None):
        if self.image.binary_img is not None and self.current_stage == 4:
            # upperの処理
            threshold_upper = int(self.edge_threshold_upper_slider.get())
            self.edge_threshold_upper_entry.delete(0, tk.END)
            self.edge_threshold_upper_entry.insert(0, str(threshold_upper))
            # bottomの処理
            threshold_bottom = int(self.edge_threshold_bottom_slider.get())
            self.edge_threshold_bottom_entry.delete(0, tk.END)
            self.edge_threshold_bottom_entry.insert(0, str(threshold_bottom))
            # 画像の処理
            self.processed_img = self.image.Canny(threshold_upper, threshold_bottom)
            self.image.canny_img = self.processed_img
            self.display_image(self.processed_img)


    def update_canny_threshold_entry(self, event=None):
        try:
            threshold_bottom = int(self.edge_threshold_bottom_entry.get())
            threshold_upper = int(self.edge_threshold_upper_entry.get())
            if 0 <= threshold_bottom <= threshold_upper and threshold_upper <= 255:
                self.edge_threshold_upper_slider.set(threshold_upper)
                self.edge_threshold_bottom_slider.set(threshold_bottom)
                self.update_canny_threshold()
        except ValueError:
            pass

    def update_filter_size(self, event=None):
        """
        MedianBlur スライドバー処理
        """
        if self.image.processed_img is not None and self.current_stage == 2:
            filter_size = int(self.filter_size_slider.get())
            if filter_size % 2 == 0:
                filter_size += 1
            self.filter_size_entry.delete(0, tk.END)
            self.filter_size_entry.insert(0, str(filter_size))
            self.processed_img = self.image.GaussianBlor(filter=filter_size)
            self.image.blur_img = self.processed_img
            self.display_image(self.processed_img)

    def update_filter_size_entry(self, event=None):
        """
        MedianBlur 数値入力処理
        """
        try:
            filter_size = int(self.filter_size_entry.get())
            if filter_size % 2 == 0:
                filter_size += 1
            if 1 <= filter_size <= 21:
                self.filter_size_slider.set(filter_size)
                self.update_filter_size()
        except ValueError:
            pass

    def ObjetDrow(self):
        if self.image.canny_img is not None and self.current_stage == 5:
            self.image.Object_Detection()
            self.image.Object_analytics()
    
    def Restart(self):
        if self.current_stage == 5:
            self.image.Reset()
            self.current_stage = 0
            # 前の表示を消す
            self.object_label.pack_forget()
            self.object_frame.pack_forget()
            self.image_label.pack_forget()
            # 次の表示
            self.image_label.pack()
            self.load_button.pack()
            self.back_button.config(state='disabled')
            self.next_button.config(state='normal')


    def display_image(self, image):
        image = Image.fromarray(image)
        image_tk = ImageTk.PhotoImage(image)
        self.image_label.configure(image=image_tk)
        self.image_label.image = image_tk


    def next_stage(self):
        # 画像切り取り
        if self.current_stage == 0:
            self.current_stage = 1
            # 前の表示を消す
            self.load_button.pack_forget()
            # 次の表示            
            self.clip_text.pack()
            
        
        # 画像のぼかし
        elif self.current_stage == 1:
            self.current_stage = 2
            # 前の表示を消す
            self.clip_text.pack_forget()
            # 次の表示
            self.filter_size_frame.pack()
            self.update_filter_size()
            self.back_button.config(state='normal')

        elif self.current_stage == 2:
            self.current_stage = 3
            # 前の表示を消す
            self.filter_size_frame.pack_forget()
            # 次の表示
            self.binary_threshold_text.pack()
            self.binary_threshold_frame.pack()
            self.update_binary_threshold()

        elif self.current_stage == 3:
            self.current_stage = 4
            # 前の表示を消す
            self.binary_threshold_text.pack_forget()
            self.binary_threshold_frame.pack_forget()
            # 次の表示
            self.edge_threshold_text.pack()
            self.edge_threshold_frame.pack()
            self.update_canny_threshold()

        elif self.current_stage == 4:
            self.current_stage = 5
            # 前の表示を消す
            self.edge_threshold_text.pack_forget()
            self.edge_threshold_frame.pack_forget()
            # 次の表示
            self.object_label.pack()
            self.object_frame.pack()
            self.next_button.config(state='disabled')

    def back_stage(self):
        if self.current_stage == 5:
            self.current_stage = 4
            # 前の表示を消す
            self.object_label.pack_forget()
            self.object_frame.pack_forget()
            # 次の表示
            self.edge_threshold_text.pack()
            self.edge_threshold_frame.pack()
            self.display_image(self.image.canny_img)
            self.next_button.config(state='normal')
        
        elif self.current_stage == 4:
            self.current_stage = 3
            # 前の表示を消す
            self.edge_threshold_text.pack_forget()
            self.edge_threshold_frame.pack_forget()
            # 次の表示
            self.binary_threshold_label.pack()
            self.binary_threshold_frame.pack()
            self.display_image(self.image.binary_img)

        elif self.current_stage == 3:
            self.current_stage = 2
            # 前の表示を消す
            self.binary_threshold_label.pack_forget()
            self.binary_threshold_frame.pack_forget()
            # 次の表示
            self.filter_size_frame.pack()
            self.display_image(self.image.blur_img)
        
        elif self.current_stage == 2:
            self.current_stage = 1
            # 前の表示を消す
            self.filter_size_frame.pack_forget()
            # 次の表示
            self.clip_text.pack()
            self.back_button.config(state='disabled')
            self.display_image(self.image.resize_img)
