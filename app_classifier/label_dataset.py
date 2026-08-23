import pandas as pd
import tkinter as tk
from tkinter import messagebox

# =====================================================
# Path
# =====================================================
CSV_FILE = "app_classifier/dataset/app_titles.csv"

# =====================================================
# Load Dataset
# =====================================================
df = pd.read_csv(CSV_FILE, encoding="latin1")
df["label"] = df["label"].fillna("")

# =====================================================
# Find next unlabeled index
# =====================================================
def get_next_index():
    for i in range(len(df)):
        if str(df.loc[i, "label"]).strip() == "":
            return i
    return None

current_index = get_next_index()

# =====================================================
# Save function
# =====================================================
def save_data():
    df.to_csv(CSV_FILE, index=False, encoding="utf-8")

# =====================================================
# UI Functions
# =====================================================
def label_data(label_name):
    global current_index

    if current_index is None:
        messagebox.showinfo("Done", "No more data to label!")
        root.quit()
        return

    df.loc[current_index, "label"] = label_name
    save_data()

    next_row()

def skip():
    global current_index

    if current_index is None:
        root.quit()
        return

    next_row()

def next_row():
    global current_index

    current_index = get_next_index()

    if current_index is None:
        messagebox.showinfo("Done", "Labeling completed!")
        root.quit()
        return

    update_ui()

def update_ui():
    if current_index is None:
        return

    process_var.set(df.loc[current_index, "process_name"])
    title_var.set(df.loc[current_index, "window_title"])
    index_var.set(f"Index: {current_index}")

# =====================================================
# GUI Setup
# =====================================================
root = tk.Tk()
root.title("Dataset Labeling Tool")
root.geometry("800x300")

index_var = tk.StringVar()
process_var = tk.StringVar()
title_var = tk.StringVar()

# Labels
tk.Label(root, textvariable=index_var, font=("Arial", 12)).pack(pady=5)
tk.Label(root, textvariable=process_var, font=("Arial", 14, "bold")).pack(pady=5)
tk.Label(root, textvariable=title_var, font=("Arial", 12), wraplength=750).pack(pady=5)

# Buttons
frame = tk.Frame(root)
frame.pack(pady=20)

tk.Button(frame, text="Focused", width=15, bg="green", fg="white",
          command=lambda: label_data("Focused")).grid(row=0, column=0, padx=5)

tk.Button(frame, text="Distracted", width=15, bg="red", fg="white",
          command=lambda: label_data("Distracted")).grid(row=0, column=1, padx=5)

tk.Button(frame, text="Neutral", width=15, bg="gray", fg="white",
          command=lambda: label_data("Neutral")).grid(row=0, column=2, padx=5)

tk.Button(frame, text="Skip", width=15,
          command=skip).grid(row=1, column=0, pady=10)

tk.Button(frame, text="Quit", width=15,
          command=root.quit).grid(row=1, column=2, pady=10)

# Load first row
update_ui()

# Run app
root.mainloop()