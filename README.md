# Personal Training Assistant

## Deadlift Posture Tracker

## Overview
This project utilizes **OpenCV** and **MediaPipe** to track body posture in real-time and assess **deadlift form** using key pose landmarks. The system calculates **back and knee angles** to determine whether the user's posture aligns with proper deadlifting mechanics.

## Features
- **Real-time posture analysis** using webcam input
- **Angle calculation** for back and knee positioning
- **On-screen feedback** to indicate proper or improper posture

## Installation
Install required dependencies:
```bash
pip install opencv-python mediapipe numpy
```

## Usage
Run the script to start real-time posture tracking:
```bash
python deadlift_tracker.py
```
Press **'q'** to exit.

## Future Enhancements
- **Squat and Bench Press Tracking**: Upcoming updates will include posture analysis for squats and bench press.
- **LLM Implementation:** Include a local LLM to give feedback on workouts.
