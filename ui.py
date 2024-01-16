import tkinter as tk
from tkinter import messagebox
from FakeNews.DetectNews import DetectNews
 # Import the function from your other file

class SimpleUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fake News Detection")

        self.root.geometry("850x500")

        self.center_window()

        self.NameEntry = tk.Entry(self.root)
        self.NameEntry.place(relx=0.35, rely=0.15, relwidth=0.3, relheight=0.1)

        FindName = tk.Button(self.root, text="Detect Fake News", cursor="hand2", command=self.detect_fake_news)
        FindName.place(relx=0.35, rely=0.4, relwidth=0.3, relheight=0.1)

        exit_button = self.create_button("Exit", self.root.destroy)
        exit_button.place(relx=0.35, rely=0.65, relwidth=0.3, relheight=0.1)

    def center_window(self):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        window_width = 850
        window_height = 500

        x_position = (screen_width - window_width) // 2
        y_position = (screen_height - window_height) // 2

        self.root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

    def create_button(self, text, command):
        button = tk.Button(self.root, text=text, command=command, font=("Arial", 14), bg="#8370ee", activebackground="#6f61a8")
        return button

    def detect_fake_news(self):
        # Get the input from the entry widget
        input_text = self.NameEntry.get()

        # Ensure the input is not empty
        if not input_text:
            messagebox.showwarning("Empty Input", "Please enter a news article.")
            return

        # Call the DetectNews function from your other file
        result = DetectNews(input_text)

        # Display the result using a message box
        if result == 1:
            messagebox.showinfo("Fake News Detection", "The news is classified as FAKE.")
        else:
            messagebox.showinfo("Fake News Detection", "The news is classified as REAL.")

if __name__ == "__main__":
    print("Starting Tkinter UI.")
    root = tk.Tk()
    app = SimpleUI(root)
    root.mainloop()
    print("Tkinter UI closed.")
