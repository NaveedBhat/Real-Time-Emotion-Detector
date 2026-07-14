# Local Setup Guide

Follow these steps to run the Real-Time Emotion Detector locally on your machine. Because this is an Edge AI application, all processing happens locally on your computer with zero cloud dependencies.

## 1. Prerequisites

Before you begin, ensure you have the following installed:
- **Python 3.9 - 3.11** (TensorFlow is best supported on these versions)
- **A working webcam** connected to your computer
- **Git**

## 2. Clone the Repository

First, download the code to your local machine:

```bash
# Clone the project from GitHub
git clone https://github.com/NaveedBhat/Real-Time-Emotion-Detector.git

# Navigate into the project directory
cd Real-Time-Emotion-Detector
```

## 3. Install Dependencies

The project includes a handy `Makefile` to automate the setup process. This command will automatically create an isolated Python virtual environment (`venv`) and install all required heavy AI libraries (TensorFlow, MediaPipe, OpenCV, FastAPI).

```bash
# Create virtual environment and install dependencies
make install
```
*Note: This might take a few minutes as it downloads TensorFlow and other large machine learning models.*

## 4. Verify Installation

Before starting the server, it's highly recommended to run the test suite to ensure the environment was built correctly and the internal math logic is working.

```bash
# Run the automated tests
make test
```
*You should see output indicating that all tests passed successfully.*

## 5. Start the Application

You can run the application in two different modes depending on your preference.

### Option A: Web Dashboard (Recommended)
This runs the application as a FastAPI server and streams the video and real-time analytics to a beautiful, modern web dashboard.

```bash
# Start the web server
make serve
```
Once you see `Application startup complete` in your terminal:
1. Open your web browser (Chrome/Safari recommended).
2. Go to: [http://127.0.0.1:8080](http://127.0.0.1:8080)

### Option B: Desktop Window
This skips the web browser entirely and opens a native OpenCV window on your desktop with an overlaid heads-up display (HUD).

```bash
# Start the desktop window mode
make run
```

## 6. Stopping the Application

When you are done, simply go to your terminal and press:
`Ctrl + C`

This will cleanly shut down the server, release the webcam hardware, and safely save your session CSV file to the `sessions/` directory.

---

## 🛠 Troubleshooting

**Error: Cannot open camera / blank screen**
- **macOS:** You must grant your Terminal (or VS Code) permission to access the camera. Go to `System Settings -> Privacy & Security -> Camera` and toggle the switch on for your terminal app.
- **Multiple Cameras:** If you have multiple webcams (e.g., a laptop camera and a USB camera) and it's grabbing the wrong one, open `config.yaml` and change `index: 0` to `index: 1`.

**Error: ModuleNotFoundError**
Ensure you are using the virtual environment. The `make` commands handle this automatically, but if you are running the python scripts directly, you must activate the environment first:
```bash
source venv/bin/activate
```
