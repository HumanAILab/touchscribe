import tkinter as tk
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db


if not firebase_admin._apps:
    cred = credentials.Certificate('soundcaption-a6e7d-firebase-adminsdk-mwgfx-7e8cba13f0.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://soundcaption-a6e7d-default-rtdb.firebaseio.com/'})

class DynamicTextDisplay:

    def __init__(self, root):
        self.root = root
        self.root.title("Dynamic Text Display")
        

        # Create a label with an initial font size
        self.font_size = 64
        self.label = tk.Label(root, text="Hello, World!", font=("Helvetica", self.font_size), wraplength=300, fg="black")
        self.label.pack(fill=tk.BOTH, expand=True)

        self.ref_caption = db.reference('caption')
        self.ref_caption.listen(self.listen_caption)

        # Bind the window resizing event
        root.bind("<Configure>", self.resize_text)
        
    # Method to change the text dynamically
    def change_text(self, new_text):
        self.label.config(text=new_text)

    # Method to dynamically adjust font size based on window size
    def resize_text(self, event):
        # Dynamically adjust the font size based on window width
        new_font_size = max(10, event.width // 20)  # Adjust the divisor to control scaling
        self.label.config(font=("Helvetica", new_font_size))
        # Dynamically adjust wrap length based on window width
        self.label.config(wraplength=event.width - 20)  # Subtracting some padding

    def listen_caption(self,event):
        print(event.data["text"])
        caption = str(event.data["text"]) 
        self.change_text(caption)
        return

# Main application code
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1200x800")  # Set initial window size

    app = DynamicTextDisplay(root)

    # Example of changing the text after 2 seconds
    root.after(2000, lambda: app.change_text("Descriptions."))

    root.mainloop()
