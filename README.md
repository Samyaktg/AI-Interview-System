# AI Interview System with Body Language Analysis

## Overview
This AI Interview System helps users practice for job interviews by combining video recording with real-time body language analysis, speech recognition, and AI feedback. The system generates job-specific interview questions, analyzes both verbal responses and non-verbal cues, and provides comprehensive feedback to help improve interview performance.

## Features
- **Job-specific question generation**: Generate tailored interview questions based on the target job role
- **Real-time body language analysis**: Detect and analyze non-verbal cues during interview responses
- **Video recording with live feedback**: See body language detection in real-time while recording
- **Speech transcription**: Convert your spoken answers to text using AI speech recognition
- **Comprehensive AI feedback**: Get detailed feedback on content quality, verbal communication, and body language
- **Visual analysis**: View charts showing your body language patterns throughout the interview
- **Easy-to-use interface**: Simple tabbed interface with setup, interview, and feedback sections

## Technical Components
- **Frontend**: Built with Python and Tkinter for a cross-platform desktop experience
- **Computer Vision**: Uses YOLO (You Only Look Once) object detection for real-time body language analysis
- **Speech Recognition**: Implements OpenAI's Whisper model for accurate speech-to-text conversion
- **Natural Language Processing**: Leverages Mistral 7B for generating questions and feedback
- **Data Visualization**: Matplotlib for visualizing body language analysis results

## Requirements
- Python 3.8+
- OpenCV
- PyAudio
- Matplotlib
- Pillow
- TensorFlow/PyTorch (for YOLO)
- OpenAI Whisper
- Ultralytics YOLO
- Requests
- NumPy

## Installation
Clone this repository:
```sh
git clone https://github.com/your-username/ai-interview-system.git
cd ai-interview-system
```

Install required packages:
```sh
pip install -r requirements.txt
```

Download the pre-trained models:
- The body language detection model (`body_language_model.pt`) should be placed in the project root directory
- Whisper will download automatically on first run

## Usage
Run the application:
```sh
python main.py
```

1. Enter your target job position on the setup screen.
2. Click **"Start Interview"** to generate a relevant interview question.
3. Record your response using the webcam and microphone, or upload an audio file.
4. Review the AI-generated feedback on your performance.

## Customization
- **Custom models**: Replace the body language model with your own trained YOLO model.
- **API keys**: Update the Hugging Face API token in the code to use your own account.
- **Question templates**: Modify the prompt templates in the code to generate different question styles.

## Privacy Note
All processing happens locally on your device, except for:
- **Question generation via Hugging Face API**
- **Feedback generation via Hugging Face API**

No video or audio recordings are sent to external servers.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- **Ultralytics** for the YOLO implementation
- **OpenAI** for the Whisper speech recognition model
- **Hugging Face** for hosting the Mistral 7B model

**Note**: This project is designed for interview practice and self-improvement. The AI feedback should be considered as suggestions rather than definitive evaluations.
