import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.filename = filedialog.askopenfilename(initialdir="test_data/",
                                            title="Select An Image Of A European Wonder",
                                            filetypes=(("JPG files", "*.jpg"),("All Files", "*.*")))


imgPath = root.filename
root.destroy()