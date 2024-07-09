import lib
import tkinter as tk

if __name__ == '__main__':
    image = lib.Image()
    root = tk.Tk()
    root.title("PMのフレンジ検出")
    root.geometry("500x500")
    app = lib.ImageProcessor(root, image)
    root.mainloop()