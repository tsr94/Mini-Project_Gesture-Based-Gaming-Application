# Gesture-Controlled Gaming Project

A hand gesture and face tracking system for controlling Unity games using computer vision. This project enables hands-free gameplay through MediaPipe-based gesture recognition and ZeroMQ communication between Python and Unity.

## 🎮 Features

- **Hand Gesture Recognition**: Control game actions using intuitive hand gestures
- **Face Tracking**: Camera/aim control using eye and head movements
- **Real-time Communication**: Low-latency ZeroMQ messaging between Python and Unity
- **Multi-Game Support**: Works with both 2D platformer and FPS games
- **Dual-Hand Controls**: Separate left and right hand gesture mappings

## 🏗️ Project Structure

```
tilak/
├── gesture_sender_v1.py      # Main gesture recognition script (latest version)
├── gesture_sender.py          # Alternative implementations
├── gesture_sender1.py
├── gesture_sender2.py
├── plat_gs.py                 # Platformer-specific gesture scripts
├── plat_gs2.py
├── plat_gs3.py
├── gs3.py                     # Other gesture script variations
├── gs4.py
├── gs5.py
├── movements.md               # Gesture mapping reference
├── 2dGame/                    # Unity 2D Platformer project
│   ├── Assets/
│   ├── ProjectSettings/
│   └── Packages/
└── MyFPS/                     # Unity FPS project
    ├── Assets/
    ├── ProjectSettings/
    └── Packages/
```

## 🎯 Gesture Controls

### Left Hand (Movement)
| Gesture | Action | Description |
|---------|--------|-------------|
| ✋ Palm open (5 fingers) | Move Forward | Clear "forward" cue |
| ✊ Fist closed | Move Backward | Opposite of forward |
| 👉 Index finger pointing | Move Left | Single finger direction |

### Right Hand (Actions)
| Gesture | Action | Description |
|---------|--------|-------------|
| ✊ Fist | Shoot | Simple trigger gesture |
| ✋ Palm open | Jump/Stop | Universal "up" gesture |
| 👍 Thumb up | Next Weapon | Quick weapon switch |

See [movements.md](movements.md) for complete gesture mapping details.

## 🛠️ Prerequisites

### Python Environment
- Python 3.8+
- OpenCV (`cv2`)
- MediaPipe
- PyZMQ
- NumPy

### Unity Projects
- Unity 2020.3 or later
- Universal Render Pipeline (URP)
- ZeroMQ Unity package/plugin

## 📦 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd tilak
```

### 2. Set Up Python Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install opencv-python mediapipe pyzmq numpy
```

### 3. Open Unity Projects
- Open `2dGame/` in Unity for the platformer game
- Open `MyFPS/` in Unity for the FPS game

## 🚀 Usage

### 1. Start the Gesture Recognition Script
```bash
python gesture_sender_v1.py
```

This will:
- Open your webcam
- Start detecting hand gestures and face tracking
- Publish gesture data via ZeroMQ on `tcp://*:5555`

### 2. Run Unity Game
- Open one of the Unity projects
- Ensure the game is configured to connect to `tcp://localhost:5555`
- Press Play in Unity Editor

### 3. Controls
- Position your hands in front of the camera
- Use the gestures from the table above
- Face tracking will control camera/aim movement
- Press `ESC` or `Q` to quit the Python script

## ⚙️ Configuration

### Gesture Sensitivity (in `gesture_sender_v1.py`)
```python
DEBUG = False                    # Enable console debug output
LOOK_SENSITIVITY = 0.8          # Camera movement sensitivity
LOOK_SMOOTHING = 0.88           # Look smoothing (0-1, higher = smoother)
LOOK_DEADZONE = 0.01            # Deadzone for jitter reduction
```

### ZeroMQ Connection
Default: `tcp://*:5555`

To change port or address, modify the socket binding in the Python script:
```python
socket.bind("tcp://*:5555")
```

## 🎓 Unity Projects

### 2D Platformer (`2dGame/`)
- Classic side-scrolling platformer microgame
- Gesture-controlled movement and jumping
- Touch-friendly level design

### FPS Game (`MyFPS/`)
- First-person shooter microgame
- Gesture-controlled movement, shooting, and weapon switching
- Face tracking for aiming

## 🐛 Troubleshooting

### Camera not detected
- Ensure your webcam is not being used by another application
- Check camera permissions in your OS settings

### Unity not receiving gestures
- Verify ZeroMQ connection (check IP address and port)
- Ensure firewall is not blocking port 5555
- Check Unity console for connection errors

### Poor gesture recognition
- Ensure good lighting conditions
- Keep hands visible and within camera frame
- Adjust `min_detection_confidence` in the script
- Try adjusting `LOOK_SENSITIVITY` and `LOOK_DEADZONE` values

### Video Link
[![Watch Video](https://drive.google.com/uc?export=view&id=1WzbicTkuy9QwffDyRUMIelGxERuqFVhC)](https://drive.google.com/file/d/1WzbicTkuy9QwffDyRUMIelGxERuqFVhC/view)


## 📝 License

This project uses:
- Unity Microgame templates
- MediaPipe (Apache License 2.0)
- OpenCV (Apache License 2.0)

See individual component licenses for details.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new gestures or features
- Improve gesture recognition accuracy
- Add support for more games

## 📧 Contact

For questions or suggestions, please open an issue in the repository.

---

**Note**: This is a research/educational project. Performance may vary based on hardware, lighting conditions, and individual hand characteristics.
