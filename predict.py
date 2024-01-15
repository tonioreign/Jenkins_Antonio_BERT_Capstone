import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QDialog,
    QProgressBar, 
    QMessageBox,
)

from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from transformers import BertTokenizer
import torch
from bert_train import BertToxicModel
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import threading
import time
import webbrowser

# Set TensorFlow logging to avoid unnecessary warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Initialize the Bert model
model_path = "bert_toxic_model_best.pth"
model = BertToxicModel()
# Load the state dictionary, ignoring missing keys
state_dict = torch.load(model_path, map_location=torch.device("cpu"))
try:
    model.load_state_dict(state_dict)
except RuntimeError as e:
    print(f"RuntimeError: {e}")
model.eval()

class BarChartWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Toxic Classification Counts")
        self.setGeometry(300, 300, 600, 400)  # Set the window size
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.canvas = FigureCanvas(Figure())
        layout.addWidget(self.canvas)

    def draw_chart(self, sums):
        ax = self.canvas.figure.subplots()
        ax.clear()
        sns.barplot(x=sums.index, y=sums.values, ax=ax)
        ax.set_title('Toxic Classification Counts')
        ax.set_ylabel('Counts')
        ax.set_xlabel('Categories')
        self.canvas.draw()


# Define the predict function
def predict(model, text, max_len=128):
    model.eval()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenized_text = tokenizer(
        text,
        add_special_tokens=True,
        padding="max_length",
        max_length=max_len,
        truncation=True,
        return_tensors="pt",
    )
    ids = tokenized_text["input_ids"]
    mask = tokenized_text["attention_mask"]
    token_type_ids = tokenized_text["token_type_ids"]

    with torch.no_grad():
        logits = model(ids, token_type_ids, mask)
        predictions = torch.sigmoid(logits).cpu().numpy()

    return predictions


# Main Application Window
class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = "Toxic Comment Prediction with Metrics"
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(100, 100, 800, 600)
        self.setMinimumSize(600, 400)  # Ensure a minimum window size for proper layout

        # Main widget and layout
        self.widget = QWidget(self)
        self.setCentralWidget(self.widget)
        self.layout = QVBoxLayout()
        self.widget.setLayout(self.layout)

        # Inside the initUI method of the App class
        self.tensorboard_button = QPushButton("Show TensorBoard Metrics", self)
        self.tensorboard_button.clicked.connect(self.show_tensorboard)
        self.layout.addWidget(self.tensorboard_button)

        # Text input with placeholder and monospace font
        self.text_input = QTextEdit(self)
        self.text_input.setFont(QFont("Consolas", 10))  # Easier to read font
        self.text_input.setPlaceholderText("Enter text here...")
        self.layout.addWidget(self.text_input)

        # Predict button with style
        self.predict_button = QPushButton("Predict", self)
        self.predict_button.setFont(QFont("Arial", 11, QFont.Bold))
        self.predict_button.setStyleSheet("background-color: #4CAF50; color: white;")
        self.predict_button.clicked.connect(self.on_predict)
        self.layout.addWidget(self.predict_button)

        # Bar Chart button
        self.bar_chart_button = QPushButton("Bar Chart", self)
        self.bar_chart_button.clicked.connect(self.show_bar_chart)
        self.layout.addWidget(self.bar_chart_button)

        # Matplotlib graph
        self.canvas = FigureCanvas(Figure())
        self.layout.addWidget(self.canvas)
        self.add_plot()

    def show_tensorboard(self):
        # Define the TensorBoard URL, usually it runs on localhost at port 6006
        tensorboard_url = 'http://localhost:6006/'
        # Launch TensorBoard in a separate thread to avoid blocking the PyQt5 app
        current_dir = os.path.dirname(os.path.abspath(__file__))
        threading.Thread(target=lambda: os.system(f'tensorboard --logdir="{current_dir}"')).start()
        # Inform the user that TensorBoard is loading and may take a few seconds
        QMessageBox.information(self, "TensorBoard Loading", "Please wait for 5 seconds for the metrics to load in your broswer.")
        # Give TensorBoard a few seconds to start
        time.sleep(5)
        # Open the TensorBoard URL in the default web browser
        webbrowser.open(tensorboard_url)

    def show_bar_chart(self):
        df = pd.read_csv('data/train.csv')
        sums = df.iloc[:, 2:].sum()
        self.bar_chart_window = BarChartWindow(self)  # Create a new window instance
        self.bar_chart_window.draw_chart(sums)  # Draw the chart
        self.bar_chart_window.show()

    def add_plot(self, percentages=None):
        ax = self.canvas.figure.subplots()
        ax.clear()  # Clear the previous plot
        if percentages is not None:
            categories = [
                "Toxic",
                "Severe Toxic",
                "Obscene",
                "Threat",
                "Insult",
                "Identity Hate",
            ]
            y_pos = np.arange(len(categories))
            bars = ax.bar(
                y_pos, percentages, align="center", alpha=0.7, color="#007ACC"
            )

            # Add labels and title with increased font sizes
            ax.set_xticks(y_pos)
            ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=10)
            ax.set_ylabel("Percentages", fontsize=12)
            ax.set_title("Toxic Comment Prediction Percentages", fontsize=14)

            # Set y-axis range to 0-100%
            ax.set_ylim(0, 100)

            # Add the percentage above the bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.2f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            # Add a soft grid
            ax.grid(True, color="grey", linestyle="--", linewidth=0.5, alpha=0.7)
            ax.set_axisbelow(True)

            # Set background color for the axes and the figure
            ax.set_facecolor("#F8F8F8")
            self.canvas.figure.set_facecolor("#F8F8F8")

            # Manually adjust the margins and improve layout
            self.canvas.figure.subplots_adjust(
                bottom=0.2, top=0.9, left=0.1, right=0.95
            )

            self.canvas.draw()

    def on_predict(self):
        input_text = self.text_input.toPlainText()
        prediction = predict(model, input_text)
        percentages = prediction[0] * 100
        self.add_plot(percentages)  # Update the plot with new percentages

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())