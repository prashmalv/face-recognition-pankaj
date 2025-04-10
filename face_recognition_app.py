import face_recognition
import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
from datetime import datetime
import shutil
import time
import queue

class FaceRecognitionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Missing Person Face Recognition System")
        self.master.geometry("1200x800")
        self.master.configure(bg="#f0f0f0")
        
        # Set app icon
        # self.master.iconbitmap("app_icon.ico")  # Uncomment and add icon file if available
        
        # Initialize variables
        self.target_image_path = None
        self.target_face_encoding = None
        self.target_image = None
        self.input_paths = []
        self.confidence_threshold = 0.6
        self.results = []
        self.processing_queue = queue.Queue()
        self.results_queue = queue.Queue()
        self.stop_processing = False
        
        # For webcam
        self.webcam = None
        self.webcam_active = False
        self.capture_mode = "target"  # or "search"
        
        # Create temp directory for processing
        self.temp_dir = "temp_processing"
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Create output directory for results
        self.output_dir = "results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create the UI
        self.create_ui()
        
        # Start worker thread for processing
        self.worker_thread = threading.Thread(target=self.processing_worker, daemon=True)
        self.worker_thread.start()
        
        # Start the update loop for the UI
        self.update_ui()
    
    def create_ui(self):
        """Create the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Split into two parts: left panel and right panel
        left_panel = ttk.Frame(main_frame, width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)
        
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # ====== LEFT PANEL ======
        # Target Person Section
        target_frame = ttk.LabelFrame(left_panel, text="Target Person")
        target_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Target image display
        self.target_canvas = tk.Canvas(target_frame, width=380, height=300, bg="white")
        self.target_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Target buttons frame
        target_btn_frame = ttk.Frame(target_frame)
        target_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(target_btn_frame, text="Select Image", command=lambda: self.select_target_image()).pack(side=tk.LEFT, padx=5)
        ttk.Button(target_btn_frame, text="Capture from Webcam", command=lambda: self.toggle_webcam("target")).pack(side=tk.LEFT, padx=5)
        
        # Search Section
        search_frame = ttk.LabelFrame(left_panel, text="Search Input")
        search_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Search mode selection
        search_mode_frame = ttk.Frame(search_frame)
        search_mode_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(search_mode_frame, text="Search From:").pack(side=tk.LEFT, padx=5)
        
        self.search_mode = tk.StringVar(value="files")
        ttk.Radiobutton(search_mode_frame, text="Files", variable=self.search_mode, value="files").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(search_mode_frame, text="Webcam", variable=self.search_mode, value="webcam").pack(side=tk.LEFT, padx=5)
        
        # Search buttons
        search_btn_frame = ttk.Frame(search_frame)
        search_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(search_btn_frame, text="Select Images/Videos", command=self.select_search_input).pack(side=tk.LEFT, padx=5)
        ttk.Button(search_btn_frame, text="Select Folder", command=self.select_search_folder).pack(side=tk.LEFT, padx=5)
        self.webcam_search_btn = ttk.Button(search_btn_frame, text="Start Webcam Search", command=lambda: self.toggle_webcam("search"))
        self.webcam_search_btn.pack(side=tk.LEFT, padx=5)
        
        # Selected files listbox
        ttk.Label(search_frame, text="Selected Input Files:").pack(anchor=tk.W, padx=5)
        
        self.files_listbox_frame = ttk.Frame(search_frame)
        self.files_listbox_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.files_listbox = tk.Listbox(self.files_listbox_frame)
        self.files_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        files_scrollbar = ttk.Scrollbar(self.files_listbox_frame, orient=tk.VERTICAL, command=self.files_listbox.yview)
        files_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.files_listbox.config(yscrollcommand=files_scrollbar.set)
        
        # Control Section
        control_frame = ttk.LabelFrame(left_panel, text="Controls")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Confidence threshold
        confidence_frame = ttk.Frame(control_frame)
        confidence_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(confidence_frame, text="Confidence Threshold:").pack(side=tk.LEFT, padx=5)
        
        self.confidence_var = tk.DoubleVar(value=0.6)
        confidence_scale = ttk.Scale(confidence_frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL, 
                                    variable=self.confidence_var, length=200)
        confidence_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.confidence_label = ttk.Label(confidence_frame, text="0.60")
        self.confidence_label.pack(side=tk.LEFT, padx=5)
        
        # Update confidence label when scale changes
        confidence_scale.bind("<Motion>", self.update_confidence_label)
        
        # Process button
        self.process_btn = ttk.Button(control_frame, text="Start Processing", command=self.start_processing)
        self.process_btn.pack(fill=tk.X, padx=5, pady=5)
        
        self.stop_btn = ttk.Button(control_frame, text="Stop Processing", command=self.stop_processing_cmd, state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(left_panel, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, padx=5, pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(left_panel, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        # ====== RIGHT PANEL ======
        # Webcam preview (when active)
        self.webcam_frame = ttk.LabelFrame(right_panel, text="Webcam Preview")
        self.webcam_canvas = tk.Canvas(self.webcam_frame, bg="black", width=640, height=480)
        self.webcam_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        webcam_btn_frame = ttk.Frame(self.webcam_frame)
        webcam_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.capture_btn = ttk.Button(webcam_btn_frame, text="Capture", command=self.capture_from_webcam)
        self.capture_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(webcam_btn_frame, text="Close Webcam", command=self.close_webcam).pack(side=tk.LEFT, padx=5)
        
        # Results Section
        self.results_frame = ttk.LabelFrame(right_panel, text="Results")
        self.results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Results notebook with tabs
        self.results_notebook = ttk.Notebook(self.results_frame)
        self.results_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Grid tab
        self.grid_tab = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.grid_tab, text="Grid View")
        
        # Create scrollable canvas for grid view
        grid_canvas_frame = ttk.Frame(self.grid_tab)
        grid_canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.grid_canvas = tk.Canvas(grid_canvas_frame, bg="#ffffff")
        self.grid_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        grid_scrollbar = ttk.Scrollbar(grid_canvas_frame, orient=tk.VERTICAL, command=self.grid_canvas.yview)
        grid_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.grid_canvas.configure(yscrollcommand=grid_scrollbar.set)
        
        self.grid_frame = ttk.Frame(self.grid_canvas)
        self.grid_canvas.create_window((0, 0), window=self.grid_frame, anchor="nw")
        
        self.grid_frame.bind("<Configure>", lambda e: self.grid_canvas.configure(scrollregion=self.grid_canvas.bbox("all")))
        
        # Table tab
        self.table_tab = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.table_tab, text="Table View")
        
        # Create table
        table_columns = ("id", "source", "timestamp", "confidence")
        self.results_table = ttk.Treeview(self.table_tab, columns=table_columns, show="headings")
        
        # Define headings
        self.results_table.heading("id", text="#")
        self.results_table.heading("source", text="Source")
        self.results_table.heading("timestamp", text="Timestamp")
        self.results_table.heading("confidence", text="Confidence")
        
        # Define columns
        self.results_table.column("id", width=50, anchor=tk.CENTER)
        self.results_table.column("source", width=300)
        self.results_table.column("timestamp", width=150)
        self.results_table.column("confidence", width=100, anchor=tk.CENTER)
        
        # Add table to frame
        self.results_table.pack(fill=tk.BOTH, expand=True)
        
        # Add table scrollbar
        table_scrollbar = ttk.Scrollbar(self.table_tab, orient=tk.VERTICAL, command=self.results_table.yview)
        table_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_table.configure(yscrollcommand=table_scrollbar.set)
        
        # Bind double-click to show full image
        self.results_table.bind("<Double-1>", self.show_full_image)
        
        # Export results button
        export_frame = ttk.Frame(self.results_frame)
        export_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(export_frame, text="Export Results", command=self.export_results).pack(side=tk.RIGHT, padx=5)
        ttk.Button(export_frame, text="Clear Results", command=self.clear_results).pack(side=tk.RIGHT, padx=5)
        
        # Initially hide webcam frame - show it only when needed
        self.webcam_frame.pack_forget()
    
    def update_confidence_label(self, event=None):
        """Update the confidence threshold label"""
        value = round(self.confidence_var.get(), 2)
        self.confidence_label.config(text=f"{value:.2f}")
    
    def select_target_image(self):
        """Select target person image from file"""
        filetypes = [("Image files", "*.jpg *.jpeg *.png *.bmp")]
        filepath = filedialog.askopenfilename(title="Select Target Person Image", filetypes=filetypes)
        
        if filepath:
            try:
                # Load the target image
                self.load_target_image(filepath)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load target image: {e}")
    
    def load_target_image(self, filepath):
        """Load and process target person image"""
        # Load image
        image = face_recognition.load_image_file(filepath)
        face_locations = face_recognition.face_locations(image)
        
        if not face_locations:
            messagebox.showwarning("Warning", "No face detected in the image. Please select a clear face image.")
            return
        
        # Get face encoding
        self.target_face_encoding = face_recognition.face_encodings(image, face_locations)[0]
        
        # Save the path
        self.target_image_path = filepath
        
        # Create a copy of the image with face rectangle
        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Draw rectangle around the face
        top, right, bottom, left = face_locations[0]
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Save the processed image
        self.target_image = img
        
        # Display the image
        self.display_image_on_canvas(self.target_canvas, img, max_size=(380, 300))
        
        # Update status
        self.status_var.set(f"Target person loaded from {os.path.basename(filepath)}")
    
    def select_search_input(self):
        """Select images or videos for searching"""
        filetypes = [
            ("All supported files", "*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov *.mkv *.wmv"),
            ("Image files", "*.jpg *.jpeg *.png *.bmp"),
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv")
        ]
        
        filepaths = filedialog.askopenfilenames(title="Select Images or Videos", filetypes=filetypes)
        
        if filepaths:
            for filepath in filepaths:
                if filepath not in self.input_paths:
                    self.input_paths.append(filepath)
                    self.files_listbox.insert(tk.END, os.path.basename(filepath))
            
            self.status_var.set(f"Added {len(filepaths)} file(s) to search list")
    
    def select_search_folder(self):
        """Select a folder of images and videos for searching"""
        folder_path = filedialog.askdirectory(title="Select Folder with Images or Videos")
        
        if folder_path:
            count = 0
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.mp4', '.avi', '.mov', '.mkv', '.wmv')):
                    filepath = os.path.join(folder_path, filename)
                    if filepath not in self.input_paths:
                        self.input_paths.append(filepath)
                        self.files_listbox.insert(tk.END, filename)
                        count += 1
            
            self.status_var.set(f"Added {count} file(s) from folder to search list")
    
    def toggle_webcam(self, mode):
        """Toggle webcam for target capture or search"""
        self.capture_mode = mode
        
        if not self.webcam_active:
            # Show webcam frame
            self.results_frame.pack_forget()
            self.webcam_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Update webcam frame title
            if mode == "target":
                self.webcam_frame.config(text="Webcam - Capture Target Person")
            else:
                self.webcam_frame.config(text="Webcam - Search for Target Person")
            
            # Start webcam
            self.webcam = cv2.VideoCapture(0)
            if not self.webcam.isOpened():
                messagebox.showerror("Error", "Could not open webcam")
                return
            
            self.webcam_active = True
            self.update_webcam()
        else:
            # Already active, just change mode
            if mode == "target":
                self.webcam_frame.config(text="Webcam - Capture Target Person")
            else:
                self.webcam_frame.config(text="Webcam - Search for Target Person")
    
    def update_webcam(self):
        """Update webcam preview"""
        if self.webcam_active and self.webcam is not None:
            ret, frame = self.webcam.read()
            if ret:
                # Convert to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # If in search mode and target is set, perform face detection
                if self.capture_mode == "search" and self.target_face_encoding is not None:
                    # Find faces in current frame
                    face_locations = face_recognition.face_locations(frame_rgb)
                    
                    if face_locations:
                        face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)
                        
                        # Check each face
                        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                            # Calculate face distance
                            face_distance = face_recognition.face_distance([self.target_face_encoding], face_encoding)[0]
                            similarity_score = (1 - face_distance) * 100
                            
                            # Draw rectangle with color based on similarity
                            color = (0, 255, 0) if similarity_score >= (self.confidence_var.get() * 100) else (255, 0, 0)
                            cv2.rectangle(frame_rgb, (left, top), (right, bottom), color, 2)
                            
                            # Display confidence score
                            cv2.putText(frame_rgb, f"{similarity_score:.2f}%", 
                                        (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Display the frame
                self.display_image_on_canvas(self.webcam_canvas, frame_rgb, max_size=(640, 480))
                
                # Schedule next update
                self.master.after(10, self.update_webcam)
            else:
                self.close_webcam()
    
    def capture_from_webcam(self):
        """Capture current webcam frame"""
        if self.webcam_active and self.webcam is not None:
            ret, frame = self.webcam.read()
            if ret:
                # Create timestamp and filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                if self.capture_mode == "target":
                    # Save as target image
                    filename = f"target_{timestamp}.jpg"
                    filepath = os.path.join(self.temp_dir, filename)
                    cv2.imwrite(filepath, frame)
                    
                    # Load as target
                    self.load_target_image(filepath)
                    
                    # Close webcam after capture
                    self.close_webcam()
                else:
                    # Save as search image
                    filename = f"search_{timestamp}.jpg"
                    filepath = os.path.join(self.temp_dir, filename)
                    cv2.imwrite(filepath, frame)
                    
                    # Add to search list
                    if filepath not in self.input_paths:
                        self.input_paths.append(filepath)
                        self.files_listbox.insert(tk.END, filename)
                    
                    self.status_var.set(f"Captured frame added to search list")
    
    def close_webcam(self):
        """Close the webcam and hide webcam frame"""
        if self.webcam is not None:
            self.webcam.release()
            self.webcam = None
        
        self.webcam_active = False
        self.webcam_frame.pack_forget()
        self.results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def display_image_on_canvas(self, canvas, image, max_size=None):
        """Display an image on a canvas with optional resizing"""
        h, w = image.shape[:2]
        
        # Resize if needed
        if max_size:
            max_w, max_h = max_size
            scale = min(max_w / w, max_h / h)
            
            if scale < 1:
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                h, w = new_h, new_w
        
        # Convert to PIL format
        pil_img = Image.fromarray(image)
        
        # Convert to PhotoImage format for tkinter
        tk_img = ImageTk.PhotoImage(image=pil_img)
        
        # Update canvas
        canvas.config(width=w, height=h)
        canvas.create_image(w//2, h//2, image=tk_img)
        
        # Keep a reference to prevent garbage collection
        canvas.image = tk_img
    
    def process_image(self, image_path):
        """Process a single image to find target person"""
        result = {
            "path": image_path,
            "matches": [],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            # Load and process the image
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)
            
            if not face_locations:
                return result
                
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            # Convert image for OpenCV processing
            processed_image = cv2.imread(image_path)
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            
            # Check each face in the image
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Calculate face distance (lower = more similar)
                face_distance = face_recognition.face_distance([self.target_face_encoding], face_encoding)[0]
                
                # Convert to similarity score (higher = more similar)
                similarity_score = round((1 - face_distance) * 100, 2)
                confidence_threshold = self.confidence_var.get() * 100
                
                # If score is high enough, it's a match
                if similarity_score >= confidence_threshold:
                    # Draw rectangle and confidence score
                    cv2.rectangle(processed_image, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(processed_image, f"{similarity_score}%", 
                                (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    result["matches"].append({
                        "bbox": (top, right, bottom, left),
                        "confidence": similarity_score
                    })
            
            # If there are matches, save the processed image
            if result["matches"]:
                base_filename = os.path.basename(image_path)
                output_filename = f"match_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{base_filename}"
                output_path = os.path.join(self.output_dir, output_filename)
                
                # Save image
                cv2.imwrite(output_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
                result["output_path"] = output_path
                result["processed_image"] = processed_image
        
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
        
        return result
    
    def process_video(self, video_path, sample_rate=1):
        """Process video to find target person in frames"""
        results = []
        
        try:
            # Open the video file
            video = cv2.VideoCapture(video_path)
            if not video.isOpened():
                return results
            
            fps = video.get(cv2.CAP_PROP_FPS)
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frame_count = 0
            
            while True:
                # Check if processing should be stopped
                if self.stop_processing:
                    break
                
                ret, frame = video.read()
                if not ret:
                    break
                
                # Process every nth frame based on sample_rate
                if frame_count % (int(fps) * sample_rate) == 0:
                    # Create a temporary file path for the frame
                    frame_path = os.path.join(self.temp_dir, f"frame_{frame_count}.jpg")
                    cv2.imwrite(frame_path, frame)
                    
                    # Process the frame
                    result = self.process_image(frame_path)
                    
                    # If matches found, add to results
                    if result["matches"]:
                        results.append(result)
                    
                    # Remove temporary frame file
                    if os.path.exists(frame_path):
                        os.remove(frame_path)
                    
                    # Update progress
                    progress = (frame_count / total_frames) * 100
                    self.results_queue.put(("progress", progress))
                
                frame_count += 1
            
            video.release()
        
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
        
        return results
    
    def start_processing(self):
        """Start processing selected inputs"""
        # Check if target person is set
        if self.target_face_encoding is None:
            messagebox.showwarning("Warning", "Please select or capture a target person first.")
            return
        
        # Check if inputs are selected
        if not self.input_paths:
            messagebox.showwarning("Warning", "Please select images or videos to search in.")
            return
        
        # Update UI state
        self.process_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.progress_var.set(0)
        self.status_var.set("Processing...")
        
        # Reset stop flag
        self.stop_processing = False
        
        # Clear current results
        self.clear_results(ask_confirmation=False)
        
        # Add tasks to queue
        for path in self.input_paths:
            self.processing_queue.put(path)
    
    def processing_worker(self):
        """Worker thread for processing images and videos"""
        while True:
            try:
                # Check if there are items to process
                if not self.processing_queue.empty() and not self.stop_processing:
                    # Get next item
                    input_path = self.processing_queue.get()
                    
                    # Update status
                    self.results_queue.put(("status", f"Processing {os.path.basename(input_path)}..."))
                    
                    # Process based on file type
                    if input_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        result = self.process_image(input_path)
                        if result["matches"]:
                            self.results_queue.put(("result", result))
                    
                    elif input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
                        results = self.process_video(input_path)
                        for result in results:
                            self.results_queue.put(("result", result))
                    
                    # Mark task as done
                    self.processing_queue.task_done()
                    
                    # Update progress
                    completed = len(self.input_paths) - self.processing_queue.qsize()
                    progress = (completed / len(self.input_paths)) * 100
                    self.results_queue.put(("progress", progress))
                    
                    # Check if all done
                    if self.processing_queue.empty():
                        self.results_queue.put(("done", None))
                
                # Sleep a bit to reduce CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error in processing worker: {e}")
    
    def update_ui(self):
        """Update UI with new results"""
        try:
            # Process all items in results queue
            while not self.results_queue.empty():
                msg_type, msg_data = self.results_queue.get()
                
                if msg_type == "status":
                    self.status_var.set(msg_data)
                
                elif msg_type == "progress":
                    self.progress_var.set(msg_data)
                
                elif msg_type == "result":
                    # Add result to results list
                    self.results.append(msg_data)
                    
                    # Update grid view
                    self.add_result_to_grid(msg_data)
                    
                    # Update table view
                    self.add_result_to_table(msg_data)
                
                elif msg_type == "done":
                    # All processing complete
                    self.status_var.set(f"Processing complete. Found {len(self.results)} matches.")
                    self.process_btn.config(state=tk.NORMAL)
                    self.stop_btn.config(state=tk.DISABLED)
        
        except Exception as e:
            print(f"Error updating UI: {e}")
        
        # Schedule next update
        self.master.after(100, self.update_ui)
    
    def add_result_to_grid(self, result):
        """Add a result to the grid view"""
        try:
            # Create a frame for this result
            result_frame = ttk.Frame(self.grid_frame, borderwidth=2, relief=tk.GROOVE)
            
            # Get row and column based on current number of results
            num_results = len(self.results)
            row = (num_results - 1) // 3
            col = (num_results - 1) % 3
            
            # Place in grid
            result_frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            
            # Display the image
            img = result["processed_image"]
            h, w = img.shape[:2]
            
            # Resize for thumbnail
            scale = 200 / max(h, w)
            thumb_w, thumb_h = int(w * scale), int(h * scale)
            img_thumb = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
            
            # Convert to PIL and then to PhotoImage
            pil_img = Image.fromarray(img_thumb)
            tk_img = ImageTk.PhotoImage(image=pil_img)
            
            # Create image label
            img_label = ttk.Label(result_frame, image=tk_img)
            img_label.image = tk_img  # Keep reference
            img_label.pack(padx=5, pady=5)
            
            # Add source info
            source_label = ttk.Label(result_frame, text=os.path.basename(result["path"]))
            source_label.pack(padx=5)
            
            # Find highest confidence score
            max_conf = max([match["confidence"] for match in result["matches"]])
            
            # Add confidence info with appropriate color
            if max_conf >= 90:
                confidence_color = "#00cc00"  # Green for high confidence
            elif max_conf >= 75:
                confidence_color = "#ffcc00"  # Yellow for medium confidence
            else:
                confidence_color = "#ff6600"  # Orange for lower confidence
            
            conf_label = ttk.Label(result_frame, text=f"Confidence: {max_conf:.2f}%")
            conf_label.pack(padx=5, pady=5)
            
            # Make the thumbnail clickable to show full image
            img_label.bind("<Button-1>", lambda e, img=result["processed_image"], 
                        path=result["path"]: self.show_full_result(img, path))
            
        except Exception as e:
            print(f"Error adding result to grid: {e}")
    
    def add_result_to_table(self, result):
        """Add a result to the table view"""
        try:
            # Find highest confidence score
            max_conf = max([match["confidence"] for match in result["matches"]])
            
            # Get item ID
            item_id = len(self.results)
            
            # Insert into table
            self.results_table.insert("", "end", values=(
                item_id,
                os.path.basename(result["path"]),
                result["timestamp"],
                f"{max_conf:.2f}%"
            ))
            
        except Exception as e:
            print(f"Error adding result to table: {e}")
    
    def show_full_image(self, event):
        """Show full image when double-clicked in table"""
        try:
            # Get selected item
            selected_item = self.results_table.selection()[0]
            item_id = int(self.results_table.item(selected_item)["values"][0])
            
            # Get corresponding result
            if 0 <= item_id < len(self.results):
                result = self.results[item_id]
                self.show_full_result(result["processed_image"], result["path"])
                
        except (IndexError, ValueError):
            pass
    
    def show_full_result(self, img, path):
        """Show full-size result image in a new window"""
        try:
            # Create new window
            result_window = tk.Toplevel(self.master)
            result_window.title(f"Match - {os.path.basename(path)}")
            
            # Calculate window size (maintain aspect ratio)
            h, w = img.shape[:2]
            max_size = 800  # Maximum dimension
            
            scale = min(1.0, max_size / max(h, w))
            new_w, new_h = int(w * scale), int(h * scale)
            
            # Set window size
            result_window.geometry(f"{new_w}x{new_h}")
            
            # Resize image if needed
            if scale < 1.0:
                display_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                display_img = img.copy()
            
            # Convert to PIL and then to PhotoImage
            pil_img = Image.fromarray(display_img)
            tk_img = ImageTk.PhotoImage(image=pil_img)
            
            # Create canvas to display image
            canvas = tk.Canvas(result_window, width=new_w, height=new_h)
            canvas.pack(fill=tk.BOTH, expand=True)
            
            # Display image
            canvas.create_image(new_w//2, new_h//2, image=tk_img)
            canvas.image = tk_img  # Keep reference
            
        except Exception as e:
            print(f"Error showing full result: {e}")
    
    def export_results(self):
        """Export results to a directory"""
        if not self.results:
            messagebox.showinfo("Info", "No results to export.")
            return
        
        # Ask for export directory
        export_dir = filedialog.askdirectory(title="Select Export Directory")
        
        if export_dir:
            try:
                # Create export directory if it doesn't exist
                os.makedirs(export_dir, exist_ok=True)
                
                # Copy result images
                for i, result in enumerate(self.results):
                    if "output_path" in result:
                        # Source path
                        src_path = result["output_path"]
                        
                        # Destination path
                        dst_filename = f"match_{i+1}_{os.path.basename(result['path'])}"
                        dst_path = os.path.join(export_dir, dst_filename)
                        
                        # Copy file
                        shutil.copy2(src_path, dst_path)
                
                # Create summary file
                with open(os.path.join(export_dir, "results_summary.txt"), "w") as f:
                    f.write(f"Face Recognition Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Target Person: {os.path.basename(self.target_image_path)}\n")
                    f.write(f"Confidence Threshold: {self.confidence_var.get():.2f}\n")
                    f.write(f"Total Matches: {len(self.results)}\n\n")
                    
                    for i, result in enumerate(self.results):
                        f.write(f"Match #{i+1}:\n")
                        f.write(f"  Source: {result['path']}\n")
                        f.write(f"  Timestamp: {result['timestamp']}\n")
                        
                        # Find highest confidence score
                        max_conf = max([match["confidence"] for match in result["matches"]])
                        f.write(f"  Confidence: {max_conf:.2f}%\n\n")
                
                messagebox.showinfo("Success", f"Results exported to {export_dir}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results: {e}")
    
    def clear_results(self, ask_confirmation=True):
        """Clear all results"""
        if ask_confirmation and self.results:
            if not messagebox.askyesno("Confirm", "Are you sure you want to clear all results?"):
                return
        
        # Clear results list
        self.results = []
        
        # Clear grid view
        for widget in self.grid_frame.winfo_children():
            widget.destroy()
        
        # Clear table view
        for item in self.results_table.get_children():
            self.results_table.delete(item)
        
        # Update status
        if ask_confirmation:
            self.status_var.set("Results cleared")
    
    def stop_processing_cmd(self):
        """Command to stop processing"""
        self.stop_processing = True
        self.status_var.set("Processing stopped")
        self.process_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
    
    def __del__(self):
        """Clean up on exit"""
        # Close webcam if open
        if self.webcam is not None:
            self.webcam.release()
        
        # Clean up temporary files
        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except:
                pass

def main():
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
