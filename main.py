import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import tempfile
import os
import numpy as np
import time
import threading
import pyaudio
import wave
import subprocess
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import requests
import json
import whisper
from ultralytics import YOLO
import logging
import soundfile as sf
import sys

# Set up basic logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('interview_app')

# Set Hugging Face API Token
HF_API_TOKEN = "" 

class InterviewApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Interview System")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)  # Handle window close

        # App state variables
        self.current_stage = "setup"  # setup, question, recording, processing, feedback
        self.recording = False
        self.job_role = "Software Developer"
        self.interview_question = None
        self.video_path = None
        self.audio_path = None
        self.transcript = None
        self.feedback = None
        self.body_language_analysis = None
        self.cap = None
        self.out = None
        self.audio_thread = None
        self.stop_audio_event = None
        self.model = None
        self.whisper_model = None
        self.body_language_detections = []
        
        # Create main frames
        self.setup_ui()
        
        # Start the app with the setup screen
        self.show_setup_screen()

    def setup_ui(self):
        """Create the main UI components with scrolling support"""
        # Create a notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tab frames first (direct children of notebook)
        self.setup_tab_frame = ttk.Frame(self.notebook)
        self.question_tab_frame = ttk.Frame(self.notebook)
        self.feedback_tab_frame = ttk.Frame(self.notebook)
        
        # Add the tab frames to the notebook
        self.notebook.add(self.setup_tab_frame, text="Setup")
        self.notebook.add(self.question_tab_frame, text="Interview")
        self.notebook.add(self.feedback_tab_frame, text="Feedback")
        
        # Create scrollable content inside each tab frame
        self.setup_tab = self.create_scrollable_frame(self.setup_tab_frame)
        self.question_tab = self.create_scrollable_frame(self.question_tab_frame)
        self.feedback_tab = self.create_scrollable_frame(self.feedback_tab_frame)
        
        # Setup the individual tab contents
        self.setup_setup_tab()
        self.setup_question_tab()
        self.setup_feedback_tab()
        
        # Status bar
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(fill='x', side='bottom', padx=10, pady=5)
        
        self.status_label = ttk.Label(self.status_frame, text="Ready")
        self.status_label.pack(side='left')
        
        # Timer/progress
        self.timer_label = ttk.Label(self.status_frame, text="00:00")
        self.timer_label.pack(side='right')

    def create_scrollable_frame(self, parent):
        """Create a scrollable frame with both vertical and horizontal scrollbars"""
        # Create a Canvas for scrolling
        canvas = tk.Canvas(parent)
        canvas.pack(side='left', fill='both', expand=True)
        
        # Add vertical scrollbar
        v_scrollbar = ttk.Scrollbar(
            parent, orient="vertical", command=canvas.yview
        )
        v_scrollbar.pack(side='right', fill='y')
        
        # Add horizontal scrollbar
        h_scrollbar = ttk.Scrollbar(
            parent, orient="horizontal", command=canvas.xview
        )
        h_scrollbar.pack(side='bottom', fill='x')
        
        # Configure canvas scrolling
        canvas.configure(
            yscrollcommand=v_scrollbar.set,
            xscrollcommand=h_scrollbar.set
        )
        
        # Create the inner frame for content
        inner_frame = ttk.Frame(canvas)
        window_id = canvas.create_window((0, 0), window=inner_frame, anchor="nw")
        
        # Configure canvas to resize with window
        def configure_canvas(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            # Update the width of the inner frame to match the canvas
            canvas.itemconfig(window_id, width=canvas.winfo_width())
        
        inner_frame.bind("<Configure>", configure_canvas)
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(window_id, width=canvas.winfo_width()))
        
        # Bind mouse wheel to scrolling
        def _bound_to_mousewheel(event):
            canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
            canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
            canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))
        
        def _unbound_to_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")
        
        inner_frame.bind("<Enter>", _bound_to_mousewheel)
        inner_frame.bind("<Leave>", _unbound_to_mousewheel)
        
        return inner_frame
    
    def setup_setup_tab(self):
        """Setup the job role selection and interview start screen"""
        # The setup_tab is already the inner frame
        container = self.setup_tab
        
        # Title
        title_label = ttk.Label(container, text="AI Interview System", font=("Arial", 24, "bold"))
        title_label.pack(pady=20)
        
        subtitle_label = ttk.Label(container, 
                                text="Practice interviews with real-time body language analysis and AI feedback",
                                font=("Arial", 12))
        subtitle_label.pack(pady=10)
        
        # Job role selection
        job_frame = ttk.Frame(container)
        job_frame.pack(pady=30)
        
        job_label = ttk.Label(job_frame, text="Enter the job position you're applying for:", font=("Arial", 12))
        job_label.pack(side='left', padx=5)
        
        self.job_entry = ttk.Entry(job_frame, width=30, font=("Arial", 12))
        self.job_entry.pack(side='left', padx=5)
        self.job_entry.insert(0, self.job_role)
        
        # Start button
        self.start_button = ttk.Button(container, text="Start Interview", command=self.start_interview)
        self.start_button.pack(pady=20)
        
        # Add test audio button
        audio_test_button = ttk.Button(container, text="Test Audio Devices", 
                                    command=self.test_audio_devices)
        audio_test_button.pack(pady=10)
        
        # Help info
        help_frame = ttk.LabelFrame(container, text="About this app", padding=10)
        help_frame.pack(fill='x', padx=20, pady=20)
        
        help_text = """
        This AI Interview System helps you practice for job interviews by:
        
        1. Generating relevant interview questions for your target role
        2. Recording your video response with real-time body language analysis
        3. Analyzing your verbal response and body language
        4. Providing personalized AI feedback on your performance
        
        Tips for best results:
        - Ensure good lighting
        - Position yourself clearly in the frame
        - Speak clearly and at a normal pace
        - Recording will stop after 5 minutes
        """
        
        help_label = ttk.Label(help_frame, text=help_text, wraplength=600, justify='left')
        help_label.pack(padx=10, pady=10)
    
    def setup_question_tab(self):
        """Setup the interview question and recording screen with scrolling"""
        # The question_tab is already the inner frame, not an object with a frame attribute
        container = self.question_tab
        
        # Question display
        self.question_frame = ttk.LabelFrame(container, text="Interview Question", padding=10)
        self.question_frame.pack(fill='x', padx=20, pady=20)
        
        self.question_label = ttk.Label(self.question_frame, text="Your question will appear here", 
                                    wraplength=800, font=("Arial", 12))
        self.question_label.pack(padx=10, pady=10)
        
        # Video frame
        self.video_frame = ttk.LabelFrame(container, text="Video Feed", padding=10)
        self.video_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill='both', expand=True)
        
        # Control buttons
        self.control_frame = ttk.Frame(container)
        self.control_frame.pack(fill='x', padx=20, pady=20)
        
        self.start_rec_button = ttk.Button(self.control_frame, text="Start Recording", 
                                        command=self.start_recording)
        self.start_rec_button.pack(side='left', padx=10)
        
        self.stop_rec_button = ttk.Button(self.control_frame, text="Stop Recording", 
                                        command=self.stop_recording, state='disabled')
        self.stop_rec_button.pack(side='left', padx=10)
        
        # Audio options
        self.audio_frame = ttk.LabelFrame(container, text="Audio Options", padding=10)
        self.audio_frame.pack(fill='x', padx=20, pady=10)
        
        self.audio_upload_button = ttk.Button(self.audio_frame, text="Upload Audio File", 
                                            command=self.upload_audio)
        self.audio_upload_button.pack(side='left', padx=10, pady=10)
        
        self.audio_status_label = ttk.Label(self.audio_frame, text="No audio uploaded")
        self.audio_status_label.pack(side='left', padx=10)

        # Audio status visualization
        self.audio_status_indicator = ttk.Label(self.control_frame, text="◯ Audio", foreground="gray")
        self.audio_status_indicator.pack(side='right', padx=10)
        
        # Add a test audio button
        self.test_audio_button = ttk.Button(self.audio_frame, text="Test Audio Devices", 
                                        command=self.test_audio_devices)
        self.test_audio_button.pack(side='right', padx=10, pady=10)
    
    def setup_feedback_tab(self):
        """Setup the feedback display screen"""
        # The feedback_tab is already the inner frame
        container = self.feedback_tab
        
        # Create a notebook for feedback sections
        self.feedback_notebook = ttk.Notebook(container)
        self.feedback_notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create the feedback subtabs
        self.feedback_text_tab = ttk.Frame(self.feedback_notebook)
        self.video_playback_tab = ttk.Frame(self.feedback_notebook)
        self.transcript_tab = ttk.Frame(self.feedback_notebook)
        self.body_language_tab = ttk.Frame(self.feedback_notebook)
        
        # Add the tabs to the notebook
        self.feedback_notebook.add(self.feedback_text_tab, text="Feedback")
        self.feedback_notebook.add(self.video_playback_tab, text="Video")
        self.feedback_notebook.add(self.transcript_tab, text="Transcript")
        self.feedback_notebook.add(self.body_language_tab, text="Body Language")
        
        # Feedback text tab
        self.feedback_text = tk.Text(self.feedback_text_tab, wrap='word', font=("Arial", 11))
        self.feedback_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Video playback tab (will be populated dynamically)
        self.video_frame_fb = ttk.Frame(self.video_playback_tab)
        self.video_frame_fb.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.play_button = ttk.Button(self.video_frame_fb, text="Play Video", command=self.play_video)
        self.play_button.pack(pady=10)
        
        # Transcript tab
        self.transcript_text = tk.Text(self.transcript_tab, wrap='word', font=("Arial", 11))
        self.transcript_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Body language tab - will contain charts
        self.body_language_frame = ttk.Frame(self.body_language_tab)
        self.body_language_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Practice again button
        self.practice_again_button = ttk.Button(container, text="Practice Another Question", 
                                            command=self.reset_interview)
        self.practice_again_button.pack(pady=20)
    
    # Navigation and UI state management
    def show_setup_screen(self):
        """Switch to the setup screen"""
        self.notebook.select(0)  # Select setup tab
        self.current_stage = "setup"
    
    def show_question_screen(self):
        """Switch to the question screen"""
        self.notebook.select(1)  # Select question tab
        self.current_stage = "question"
    
    def show_feedback_screen(self):
        """Switch to the feedback screen"""
        self.notebook.select(2)  # Select feedback tab
        self.current_stage = "feedback"
    
    def update_status(self, message):
        """Update the status bar message"""
        self.status_label.config(text=message)
        self.root.update_idletasks()
    
    def update_timer(self, seconds):
        """Update the timer display"""
        minutes = seconds // 60
        seconds = seconds % 60
        self.timer_label.config(text=f"{minutes:02d}:{seconds:02d}")
        self.root.update_idletasks()
    
    def update_audio_indicator(self, is_recording=True):
        """Update the audio status indicator"""
        if is_recording:
            self.audio_status_indicator.config(text="⬤ Audio", foreground="green")
        else:
            self.audio_status_indicator.config(text="◯ Audio", foreground="red")

    # Interview process functions
    def start_interview(self):
        """Start the interview process"""
        self.job_role = self.job_entry.get()
        self.update_status("Generating interview question...")
        
        # Disable start button while processing
        self.start_button.config(state='disabled')
        
        # Generate question in a separate thread to avoid UI freeze
        threading.Thread(target=self._generate_question_thread).start()
    
    def _generate_question_thread(self):
        """Generate question in a background thread"""
        try:
            # Generate question
            self.interview_question = self.generate_interview_question(self.job_role)
            
            # Update UI on main thread
            self.root.after(0, self._show_question)
        except Exception as e:
            logger.error(f"Error generating question: {e}")
            self.root.after(0, lambda: self.show_error(f"Failed to generate question: {str(e)}"))
            self.root.after(0, lambda: self.start_button.config(state='normal'))
    
    def _show_question(self):
        """Update UI with the generated question"""
        self.question_label.config(text=self.interview_question)
        self.start_button.config(state='normal')
        self.show_question_screen()
        self.update_status("Ready to record your answer")
    
    def start_recording(self):
        """Start recording video and audio"""
        self.update_status("Initializing recording...")
        self.recording = True
        
        # Update UI
        self.start_rec_button.config(state='disabled')
        self.stop_rec_button.config(state='normal')
        self.audio_upload_button.config(state='disabled')
        
        # Start recording in separate thread
        threading.Thread(target=self._recording_thread).start()
    def test_audio_devices(self):
        """Test audio devices and provide feedback to user"""
        self.update_status("Testing audio devices...")
        
        try:
            p = pyaudio.PyAudio()
            
            # Get all devices
            device_list = []
            default_device = None
            
            try:
                default_info = p.get_default_input_device_info()
                default_device = default_info['index']
            except Exception:
                default_device = None
            
            # List all devices with input channels
            for i in range(p.get_device_count()):
                try:
                    dev_info = p.get_device_info_by_index(i)
                    if dev_info['maxInputChannels'] > 0:
                        is_default = i == default_device
                        device_list.append((i, dev_info['name'], is_default))
                except Exception:
                    pass
                    
            p.terminate()
            
            # Show results to user
            if not device_list:
                messagebox.showwarning("Audio Devices", 
                    "No audio input devices found! You will need to upload an audio file.")
                return
                
            # Format message
            msg = "Available audio input devices:\n\n"
            for idx, name, is_default in device_list:
                prefix = "* " if is_default else "  "
                msg += f"{prefix}{idx}: {name}\n"
                
            msg += "\nIf recording doesn't work, try uploading an audio file instead."
            
            messagebox.showinfo("Audio Devices", msg)
            self.update_status("Ready")
            
        except Exception as e:
            messagebox.showerror("Audio Test Error", f"Error testing audio: {str(e)}")

    def _recording_thread(self):
        """Handle the recording process in a background thread"""
        try:
            # Load model if not already loaded
            if self.model is None:
                self.update_status("Loading body language model...")
                self.model = self.load_model()
                if self.model is None:
                    raise Exception("Failed to load body language model")
            
            # Initialize webcam
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Could not open webcam")
            
            # Set camera resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Create temp files
            temp_video_path = tempfile.mktemp(suffix=".mp4")
            temp_audio_path = tempfile.mktemp(suffix=".wav")
            
            # Create video writer
            self.out = self.initialize_video_writer(temp_video_path, width=640, height=480, fps=20.0)
            
            # Initialize variables
            start_time = time.time()
            frame_count = 0
            self.body_language_detections = []
            fps = 20  # Assumed FPS
            MAX_DURATION = 300  # 5 minutes max
            
            # Start audio recording
            self.stop_audio_event = threading.Event()
            self.audio_thread = threading.Thread(
                target=self.record_audio_direct,
                args=(temp_audio_path, self.stop_audio_event)
            )
            self.audio_thread.start()
            
            # Update the audio indicator
            self.root.after(0, lambda: self.update_audio_indicator(True))
            self.update_status("Recording video and audio")
            
            # For pulsating audio indicator
            is_pulse_on = False
            
            # Main recording loop
            while self.recording:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    raise Exception("Could not read frame from webcam")
                
                # Calculate elapsed time
                elapsed = int(time.time() - start_time)
                self.root.after(0, lambda t=elapsed: self.update_timer(t))
                
                # Pulsate audio indicator (toggle every second)
                if elapsed % 2 == 0 and not is_pulse_on:
                    self.root.after(0, lambda: self.update_audio_indicator(True))
                    is_pulse_on = True
                elif elapsed % 2 == 1 and is_pulse_on:
                    self.root.after(0, lambda: self.audio_status_indicator.config(foreground="#007700"))
                    is_pulse_on = False
                
                # Check if max duration reached
                if elapsed >= MAX_DURATION:
                    self.update_status("Maximum recording time reached")
                    self.recording = False
                    break
                
                # Run YOLO detection
                results = self.model(frame)
                
                # Get annotated frame for display
                display_frame = results[0].plot(
                    line_width=3,
                    font_size=1.2,
                    boxes=True
                )
                
                # Add recording indicators
                display_frame = self.add_recording_indicators(display_frame, elapsed)
                
                # Extract body language results
                detections = []
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    for i, box in enumerate(results[0].boxes):
                        # Extract detection data
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        
                        # Get body language class name
                        body_language = results[0].names[class_id]
                        
                        # Store detection (every 5th frame to prevent data overload)
                        if frame_count % 5 == 0:
                            detections.append({
                                "timestamp": elapsed,
                                "frame_number": frame_count,
                                "body_language": body_language,
                                "confidence": confidence,
                                "bbox": [x1, y1, x2, y2]
                            })
                
                # Add detections to the main list
                if frame_count % 5 == 0 and detections:
                    self.body_language_detections.extend(detections)
                
                # Update display
                self._update_video_display(display_frame)
                
                # Write the original frame to video file (without overlay)
                self.out.write(frame)
                
                frame_count += 1
            
            # Recording complete - cleanup
            self.stop_audio_event.set()
            self.audio_thread.join(timeout=5.0)
            
            # Release video resources
            self.cap.release()
            self.out.release()
            self.cap = None
            self.out = None
            
            # Save paths
            self.video_path = temp_video_path
            self.audio_path = temp_audio_path
            
            # Reset audio indicator
            self.root.after(0, lambda: self.update_audio_indicator(False))
            
            # Process results
            self.update_status("Processing recording...")
            self.process_recording(temp_video_path, temp_audio_path, frame_count, time.time() - start_time, fps)
            
        except Exception as e:
            logger.error(f"Recording error: {e}")
            self.root.after(0, lambda: self.show_error(f"Recording error: {str(e)}"))
            self.recording = False
            
            # Clean up resources
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            if self.out is not None:
                self.out.release()
                self.out = None
            if self.stop_audio_event is not None:
                self.stop_audio_event.set()
            
            # Reset UI
            self.root.after(0, lambda: self.start_rec_button.config(state='normal'))
            self.root.after(0, lambda: self.stop_rec_button.config(state='disabled'))
            self.root.after(0, lambda: self.audio_upload_button.config(state='normal'))
            self.root.after(0, lambda: self.update_audio_indicator(False))
    
    def stop_recording(self):
        """Stop the recording process"""
        self.recording = False
        self.update_status("Stopping recording...")
        
        # Disable both buttons during processing
        self.start_rec_button.config(state='disabled')
        self.stop_rec_button.config(state='disabled')
    
    def add_recording_indicators(self, frame, elapsed_time):
        """Add recording indicators to the frame"""
        # Create a copy to avoid modifying the original
        display_frame = frame.copy()
        
        # Add recording indicator
        cv2.circle(display_frame, (30, 30), 10, (0, 0, 255), -1)  # Red filled circle
        
        # Add timer
        minutes = elapsed_time // 60
        seconds = elapsed_time % 60
        time_text = f"Recording: {minutes:02d}:{seconds:02d}"
        
        # Background rectangle for better visibility
        cv2.rectangle(display_frame, (5, 5), (200, 40), (0, 0, 0), -1)
        cv2.putText(display_frame, time_text, (50, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return display_frame
    
    def _update_video_display(self, frame):
        """Update the video display with the current frame"""
        # Convert the OpenCV frame to a Tkinter-compatible photo
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Update the label with the new image
        self.video_label.imgtk = imgtk  # Keep a reference!
        self.video_label.config(image=imgtk)
        self.root.update_idletasks()
    
    def upload_audio(self):
        """Upload an audio file instead of recording"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Audio Files", "*.wav;*.mp3")],
            title="Select Audio File"
        )
        
        if file_path:
            self.audio_path = file_path
            self.audio_status_label.config(text=f"Audio uploaded: {os.path.basename(file_path)}")
            messagebox.showinfo("Success", "Audio file uploaded successfully!")
    
    def process_recording(self, video_path, audio_path, frame_count, duration, fps):
        """Process the recorded video and audio"""
        try:
            self.update_status("Analyzing body language...")
            
            # Process body language data
            self.body_language_analysis = self.process_collected_data(
                self.body_language_detections, 
                frame_count,
                duration,
                fps
            )
            
            # Use uploaded audio if available
            final_audio_path = audio_path
            if hasattr(self, 'audio_path') and self.audio_path and os.path.exists(self.audio_path) and self.audio_path != audio_path:
                final_audio_path = self.audio_path
            
            # Transcribe audio
            self.update_status("Transcribing your speech...")
            self.transcript = self.transcribe_audio(final_audio_path)
            
            # Generate feedback
            self.update_status("Generating AI feedback...")
            self.feedback = self.analyze_interview_response(
                self.transcript,
                self.body_language_analysis,
                self.interview_question
            )
            
            # Enable the audio upload button again
            self.root.after(0, lambda: self.audio_upload_button.config(state='normal'))
            
            # Update the UI with results
            self.root.after(0, self.display_results)
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            self.root.after(0, lambda: self.show_error(f"Processing error: {str(e)}"))
            # Reset UI
            self.root.after(0, lambda: self.start_rec_button.config(state='normal'))
            self.root.after(0, lambda: self.stop_rec_button.config(state='disabled'))
            self.root.after(0, lambda: self.audio_upload_button.config(state='normal'))
    
    def display_results(self):
        """Display the processed results in the feedback tab"""
        # Update status
        self.update_status("Results ready")
        
        # Reset recording buttons
        self.start_rec_button.config(state='normal')
        self.stop_rec_button.config(state='disabled')
        
        # Update feedback text
        self.feedback_text.delete("1.0", tk.END)
        self.feedback_text.insert(tk.END, self.feedback)
        
        # Update transcript
        self.transcript_text.delete("1.0", tk.END)
        self.transcript_text.insert(tk.END, self.transcript)
        
        # Create body language chart
        self.create_body_language_chart()
        
        # Switch to feedback tab
        self.show_feedback_screen()
    
    def create_body_language_chart(self):
        """Create body language distribution chart"""
        # Clear previous chart
        for widget in self.body_language_frame.winfo_children():
            widget.destroy()
        
        # Check if we have results
        if not self.body_language_analysis or 'summary' not in self.body_language_analysis:
            ttk.Label(self.body_language_frame, text="No body language data available").pack()
            return
        
        summary = self.body_language_analysis.get("summary", {})
        
        # Create summary text
        summary_frame = ttk.LabelFrame(self.body_language_frame, text="Summary", padding=10)
        summary_frame.pack(fill='x', pady=10)
        
        duration = summary.get("video_duration", 0)
        dominant_bl = summary.get("dominant_body_language", "Unknown")
        
        summary_text = f"Response Duration: {duration:.1f} seconds\n"
        summary_text += f"Dominant Body Language: {dominant_bl}\n"
        
        ttk.Label(summary_frame, text=summary_text).pack()
        
        # Display body language distribution chart if available
        if "body_language_percentages" in summary:
            chart_frame = ttk.LabelFrame(self.body_language_frame, text="Body Language Distribution", padding=10)
            chart_frame.pack(fill='both', expand=True, pady=10)
            
            # Create a bar chart
            labels = list(summary["body_language_percentages"].keys())
            values = list(summary["body_language_percentages"].values())
            
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.bar(labels, values, color='skyblue')
            ax.set_ylabel('Percentage (%)')
            ax.set_title('Body Language Distribution')
            
            # Add data labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Embed in Tkinter
            canvas = FigureCanvasTkAgg(fig, master=chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def play_video(self):
        """Play the recorded video using system default player"""
        if self.video_path and os.path.exists(self.video_path):
            # Use the system's default video player
            if os.name == 'nt':  # Windows
                os.startfile(self.video_path)
            else:  # Mac/Linux
                opener = 'open' if sys.platform == 'darwin' else 'xdg-open'
                subprocess.call([opener, self.video_path])
        else:
            messagebox.showerror("Error", "Video file not found")
    
    def reset_interview(self):
        """Reset for another interview question"""
        # Clear previous results
        self.interview_question = None
        self.video_path = None
        self.audio_path = None
        self.transcript = None
        self.feedback = None
        self.body_language_analysis = None
        self.body_language_detections = []
        
        # Reset UI
        self.question_label.config(text="Your question will appear here")
        self.audio_status_label.config(text="No audio uploaded")
        self.video_label.config(image='')
        self.timer_label.config(text="00:00")
        self.feedback_text.delete("1.0", tk.END)
        self.transcript_text.delete("1.0", tk.END)
        self.update_audio_indicator(False)
        
        # Go back to setup screen
        self.show_setup_screen()
    
    def show_error(self, message):
        """Show an error message box"""
        messagebox.showerror("Error", message)
        self.update_status("Error occurred")
    
    def on_closing(self):
        """Handle window closing"""
        # Clean up resources
        if self.cap is not None:
            self.cap.release()
        
        if self.recording:
            self.recording = False
            if self.stop_audio_event is not None:
                self.stop_audio_event.set()
        
        self.root.destroy()
    
    # Core functionality methods
    def load_model(self):
        """Load the YOLOv8 model"""
        try:
            logging.info("Loading YOLOv8 model...")
            model = YOLO('body_language_model.pt')  # Load the trained model
            logging.info(f"Model loaded successfully. Model type: {type(model)}")
            return model
        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
            # Fall back to general YOLOv8 model if custom model not found
            try:
                logging.info("Attempting to load default YOLOv8n model...")
                model = YOLO('yolov8n.pt')
                logging.info("Default model loaded successfully")
                return model
            except Exception as e2:
                logging.error(f"Failed to load default model: {str(e2)}")
                return None
    
    def load_whisper_model(self):
        """Load the Whisper ASR model with English optimization"""
        try:
            if self.whisper_model is None:
                logging.info("Loading Whisper model...")
                # Use "base.en" for English-optimized model instead of just "base"
                self.whisper_model = whisper.load_model("base.en")  
                logging.info("English-optimized Whisper model loaded successfully")
            return self.whisper_model
        except Exception as e:
            logging.error(f"Failed to load Whisper model: {str(e)}")
            return None
    
    def initialize_video_writer(self, path, width, height, fps):
        """Initialize OpenCV video writer object"""
        try:
            # Create the video writer object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
            out = cv2.VideoWriter(
                path,
                fourcc,
                fps,
                (int(width), int(height))
            )
            
            # Check if video writer was created successfully
            if not out.isOpened():
                logging.error(f"Failed to create video writer for {path}")
                return None
                
            logging.info(f"Video writer initialized for {path} ({width}x{height} @ {fps}fps)")
            return out
        except Exception as e:
            logging.error(f"Error initializing video writer: {str(e)}")
            return None

    def record_audio_direct(self, audio_file_path, stop_event, sample_rate=44100):
        """Record audio directly to a WAV file"""
        try:
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RATE = sample_rate
            CHUNK = 1024
            
            p = pyaudio.PyAudio()
            
            # List available devices for debugging
            logging.info("\n--- Available Audio Input Devices ---")
            for i in range(p.get_device_count()):
                dev_info = p.get_device_info_by_index(i)
                logging.info(f"Device {i}: {dev_info['name']} (Channels: {dev_info['maxInputChannels']})")
            
            # Find working input device
            working_device = None
            
            # Try default device first
            try:
                default_device_info = p.get_default_input_device_info()
                device_id = default_device_info['index']
                logging.info(f"Trying default input device {device_id}: {default_device_info['name']}")
                
                # Test device
                test_stream = p.open(format=FORMAT,
                                    channels=CHANNELS,
                                    rate=RATE,
                                    input=True,
                                    input_device_index=device_id,
                                    frames_per_buffer=CHUNK)
                test_stream.stop_stream()
                test_stream.close()
                working_device = device_id
                logging.info(f"Default input device {device_id} works!")
            except Exception as e:
                logging.error(f"Default device failed: {e}")
            
            # If default failed, try others
            if working_device is None:
                for i in range(p.get_device_count()):
                    dev_info = p.get_device_info_by_index(i)
                    if dev_info['maxInputChannels'] > 0:  # Has microphone input
                        try:
                            logging.info(f"Trying device {i}: {dev_info['name']}")
                            # Test if device works
                            test_stream = p.open(format=FORMAT,
                                                channels=CHANNELS,
                                                rate=RATE,
                                                input=True,
                                                input_device_index=i,
                                                frames_per_buffer=CHUNK)
                            test_data = test_stream.read(CHUNK)
                            test_stream.stop_stream()
                            test_stream.close()
                            working_device = i
                            logging.info(f"Found working input device: {i}")
                            break
                        except Exception as e:
                            logging.error(f"Device {i} failed: {e}")
            
            if working_device is None:
                logging.error("No working audio input devices found!")
                return False
                
            # Open the WAV file for writing
            wf = wave.open(audio_file_path, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            
            # Start recording stream
            stream = p.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            input_device_index=working_device,
                            frames_per_buffer=CHUNK)
                            
            logging.info(f"Recording audio to {audio_file_path}...")
            
            # Record until stop event is set
            while not stop_event.is_set():
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    wf.writeframes(data)
                except Exception as e:
                    logging.error(f"Error reading audio: {e}")
                    time.sleep(0.1)  # Prevent CPU hogging
                    
            # Clean up resources
            stream.stop_stream()
            stream.close()
            p.terminate()
            wf.close()
            
            # Verify the recording
            if os.path.exists(audio_file_path):
                size = os.path.getsize(audio_file_path)
                logging.info(f"Audio recording complete: {size} bytes")
                return size > 0
            else:
                logging.error("Error: Audio file not created")
                return False
                
        except Exception as e:
            logging.error(f"Error during audio recording: {e}")
            return False
    
    def transcribe_audio(self, audio_path):
        """Transcribe audio using Whisper ASR (English only)"""
        try:
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                return "No audio data available for transcription."
            
            model = self.load_whisper_model()
            if model is None:
                return "Error: Failed to load speech recognition model"
            
            # Force English language transcription
            result = model.transcribe(
                audio_path,
                language="en",     # Force English language
                task="transcribe", # Explicitly set task to transcription
                fp16=False         # Use fp32 for better accuracy on CPU
            )
            
            logging.info("Transcription completed successfully in English")
            return result["text"]
        except Exception as e:
            logging.error(f"Failed to transcribe audio: {str(e)}")
            return f"Error transcribing audio: {str(e)}"
    
    def process_collected_data(self, body_language_detections, frame_count, duration, fps):
        """Process body language detection data"""
        logging.info(f"Processing collected data: {len(body_language_detections)} detections over {duration:.1f} seconds")
        
        # Generate summary
        summary = {}
        if body_language_detections:
            # Count occurrences of each body language class
            body_language_counts = {}
            for detection in body_language_detections:
                bl = detection["body_language"]
                if bl not in body_language_counts:
                    body_language_counts[bl] = 0
                body_language_counts[bl] += 1
            
            logging.info(f"Body language counts: {body_language_counts}")
            
            # Calculate percentages
            total_detections = len(body_language_detections)
            body_language_percentages = {
                bl: (count / total_detections) * 100
                for bl, count in body_language_counts.items()
            }
            
            # Find dominant body language
            dominant_body_language = max(body_language_counts.items(), key=lambda x: x[1])[0]
            logging.info(f"Dominant body language: {dominant_body_language}")
            
            # Group detections by timestamp
            timestamp_groups = {}
            for detection in body_language_detections:
                # Round timestamp to nearest second for grouping
                rounded_time = round(detection["timestamp"])
                if rounded_time not in timestamp_groups:
                    timestamp_groups[rounded_time] = []
                timestamp_groups[rounded_time].append(detection)
            
            # Find key moments (changes in body language)
            key_moments = []
            prev_bl = None
            for ts, detections in sorted(timestamp_groups.items()):
                # Get the highest confidence detection for this timestamp
                best_detection = max(detections, key=lambda x: x["confidence"])
                current_bl = best_detection["body_language"]
                
                if prev_bl is None or current_bl != prev_bl:
                    key_moments.append({
                        "timestamp": best_detection["timestamp"],
                        "frame_number": best_detection["frame_number"],
                        "body_language": current_bl,
                        "confidence": best_detection["confidence"]
                    })
                    logging.info(f"Key moment at {best_detection['timestamp']:.1f}s: {current_bl}")
                
                prev_bl = current_bl
            
            # Create summary
            frames_analyzed = frame_count // 5  # Since we only store detections every 5th frame
            summary = {
                "video_duration": duration,
                "frames_analyzed": frames_analyzed,
                "total_detections": total_detections,
                "body_language_counts": body_language_counts,
                "body_language_percentages": body_language_percentages,
                "dominant_body_language": dominant_body_language,
                "key_moments": key_moments
            }
        
        logging.info(f"Analysis completed: {len(body_language_detections)} detections")
        return {
            "status": "success" if body_language_detections else "no_detections",
            "video_info": {
                "duration": duration,
                "fps": fps,
                "frame_count": frame_count
            },
            "body_language_detections": body_language_detections,
            "key_frames": [],  # Interview mode doesn't save key frames
            "summary": summary
        }
    
    def combine_with_ffmpeg(self, video_path, audio_path, output_path):
        """Combine video and audio using FFmpeg"""
        try:
            logging.info(f"Combining video and audio using FFmpeg...")
            logging.info(f"- Video: {video_path}")
            logging.info(f"- Audio: {audio_path}")
            logging.info(f"- Output: {output_path}")
            
            # Ensure the files exist
            if not os.path.exists(video_path):
                logging.error(f"Error: Video file not found: {video_path}")
                return None
            if not os.path.exists(audio_path):
                logging.error(f"Error: Audio file not found: {audio_path}")
                return None
                
            # Build FFmpeg command
            cmd = [
                "ffmpeg",
                "-y",                # Overwrite output files
                "-i", video_path,    # Input video
                "-i", audio_path,    # Input audio
                "-c:v", "copy",      # Copy video stream without re-encoding
                "-c:a", "aac",       # Convert audio to AAC
                "-strict", "experimental",
                "-map", "0:v:0",     # Use first video stream from first input
                "-map", "1:a:0",     # Use first audio stream from second input
                "-shortest",         # Cut the longest stream to match the shortest
                output_path
            ]
            
            # Execute FFmpeg
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            
            if result.returncode == 0:
                logging.info("FFmpeg successfully combined video and audio")
                if os.path.exists(output_path):
                    logging.info(f"Output file created: {output_path} ({os.path.getsize(output_path)} bytes)")
                    return output_path
                else:
                    logging.error(f"Error: Output file not created")
                    return None
            else:
                logging.error(f"FFmpeg error: {result.stderr}")
                return None
                
        except Exception as e:
            logging.error(f"Error combining video and audio: {e}")
            return None
    
    def generate_interview_question(self, job_role="General"):
        """Generate an interview question using Mistral API"""
        logging.info(f"Generating interview question for: {job_role}")
        
        # API settings
        API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"  
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        
        # Create a prompt for the LLM
        prompt = f"""
        Task: Generate a professional interview question for a {job_role} position.
        
        The question should:
        1. Be challenging but fair
        2. Assess both technical skills and soft skills
        3. Allow the candidate to demonstrate their expertise and thought process
        4. Be open-ended, requiring more than a yes/no answer
        
        Provide only the question without additional context or explanation.
        """
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 200,
                "temperature": 0.7,
                "top_p": 0.95,
                "do_sample": True,
                "return_full_text": False
            }
        }
        
        try:
            logging.info("Sending request to Mistral API for interview question...")
            response = requests.post(API_URL, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    question = result[0].get('generated_text', '')
                    return question.strip()
            
            # Fallback questions if API fails
            fallback_questions = [
                "Can you tell us about a challenging project you worked on and how you overcame obstacles?",
                "How do you prioritize tasks when facing multiple deadlines?",
                "Describe a situation where you had to adapt quickly to changing requirements.",
                "How do you approach collaboration with team members who have different working styles?",
                "What methods do you use to stay current in your field?"
            ]
            import random
            return random.choice(fallback_questions)
        
        except Exception as e:
            logging.error(f"Exception during API call for question - {str(e)}")
            return "Tell me about a challenging problem you solved recently."
    
    def analyze_interview_response(self, transcript, body_language_analysis, question):
        """Generate feedback on interview response"""
        logging.info("Generating comprehensive interview feedback...")
        
        # API settings
        API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"  
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        
        # Extract body language insights
        summary = body_language_analysis.get("summary", {})
        dominant_bl = summary.get("dominant_body_language", "unknown")
        bl_percentages = summary.get("body_language_percentages", {})
        
        # Format body language data
        formatted_bl = []
        for bl, pct in bl_percentages.items():
            formatted_bl.append(f"{bl}: {pct:.1f}%")
        
        # Create detailed prompt for analysis
        prompt = f"""
        Task: Analyze this interview response and provide constructive feedback.
        
        Interview question: {question}
        
        Verbal response transcript: 
        {transcript}
        
        Body language during response:
        Dominant body language: {dominant_bl}
        Body language breakdown: {', '.join(formatted_bl)}
        
        Provide comprehensive feedback on:
        1. Content quality: How well did they answer the specific question? Was it structured and relevant?
        2. Verbal communication: Analyze clarity, conciseness, and persuasiveness
        3. Body language interpretation: How their body language affected impression (confidence, nervousness, etc.)
        4. Areas of improvement: Specific suggestions for both verbal and non-verbal aspects
        5. Overall impression: Summarize the strengths and areas for improvement
        
        Format your response as constructive, actionable feedback to help the candidate improve.
        """
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 1000,
                "temperature": 0.7,
                "top_p": 0.95,
                "do_sample": True,
                "return_full_text": False
            }
        }
        
        try:
            logging.info("Sending request to Mistral API for interview feedback...")
            response = requests.post(API_URL, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    feedback = result[0].get('generated_text', '')
                    return feedback
            
            return "Error: Unable to generate interview feedback. Please try again."
        
        except Exception as e:
            logging.error(f"Exception during API call for feedback - {str(e)}")
            return f"Error generating interview feedback: {str(e)}"


# Main entry point
def main():
    root = tk.Tk()
    app = InterviewApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
