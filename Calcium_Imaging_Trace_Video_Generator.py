# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 16:16:07 2025

@author: bsmsa18b
"""

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    TKINTER_AVAILABLE = True
except ImportError:
    print("tkinter not available. Will use command-line version.")
    TKINTER_AVAILABLE = False

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    
    import re
 
    import queue
    import threading
    
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.patches import Circle
    from PIL import Image, ImageTk
    CORE_LIBRARIES_AVAILABLE = True
except ImportError as e:
    print(f"Core libraries not available: {e}")
    print("Please install: pip install pandas numpy matplotlib pillow")
    CORE_LIBRARIES_AVAILABLE = False

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    print("OpenCV not available. Video generation will use matplotlib instead.")
    OPENCV_AVAILABLE = False

import threading
import os


try:
    from read_roi import read_roi_zip, read_roi_file
    READROI_AVAILABLE = True
except ImportError:
    print("read-roi not available. Install with: pip install read-roi")
    READROI_AVAILABLE = False

# Alternative import if read-roi doesn't work
try:
    import roifile
    ROIFILE_AVAILABLE = True
except ImportError:
    ROIFILE_AVAILABLE = False




class CalciumImagingGUI:
    
    def __init__(self, root):
            self.root = root
            self.root.title("Calcium Imaging Trace Video Generator")
            self.root.geometry("1400x900")
            # Data storage
            self.trace_data = None
            self.imaging_data = None
            self.roi_names = []
            self.roi_coordinates = []
            self.colors = None
            self.current_animation = None
            # Video properties
            self.original_fps = 30.0  # Default FPS
            self.frame_times = None  # Time array for each frame
            # Threading control
            self.video_generation_queue = queue.Queue()
            self.video_generation_complete = threading.Event()
            self.is_generating_video = False
            self.video_thread = None  # To store the thread object
            # Create GUI
            self.setup_gui()
            # Handle window close
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_gui(self):
        """Setup the main GUI components with all improvements"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Calcium Imaging Trace Video Generator", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Left panel for controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # File loading section
        file_frame = ttk.LabelFrame(control_frame, text="Load Data", padding="5")
        file_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Button(file_frame, text="Load ROI Data (CSV/Excel)", 
                  command=self.load_roi_data).pack(fill="x", pady=2)
        ttk.Button(file_frame, text="Load Original Video (AVI/MP4)", 
                  command=self.load_video_data).pack(fill="x", pady=2)
        ttk.Button(file_frame, text="Load ImageJ ROIs", 
                  command=self.load_imagej_rois).pack(fill="x", pady=2)
        ttk.Button(file_frame, text="Load ROI Coordinates (CSV)", 
                  command=self.load_roi_coordinates).pack(fill="x", pady=2)
        
        # Sampling rate section
        sampling_frame = ttk.LabelFrame(control_frame, text="Data Settings", padding="5")
        sampling_frame.pack(fill="x", pady=(0, 10))
        
        # Sampling rate input
        rate_frame = ttk.Frame(sampling_frame)
        rate_frame.pack(fill="x", pady=2)
        ttk.Label(rate_frame, text="Sampling Rate (Hz):").pack(side="left")
        self.sampling_rate_var = tk.StringVar(value="5.0")
        sampling_entry = ttk.Entry(rate_frame, textvariable=self.sampling_rate_var, width=8)
        sampling_entry.pack(side="right")
        
        # Add tooltip/help text
        help_label = ttk.Label(sampling_frame, text="Converts frame data to time (e.g., 5 Hz = frame 5 becomes 1 second)", 
                              font=("Arial", 8), foreground="gray")
        help_label.pack(anchor="w", pady=(0, 5))
        
        # Video info section
        video_info_frame = ttk.LabelFrame(control_frame, text="Video Information", padding="5")
        video_info_frame.pack(fill="x", pady=(0, 10))
        
        self.video_info_text = tk.Text(video_info_frame, height=4, width=40, wrap=tk.WORD)
        video_scrollbar = ttk.Scrollbar(video_info_frame, orient="vertical", command=self.video_info_text.yview)
        self.video_info_text.configure(yscrollcommand=video_scrollbar.set)
        self.video_info_text.pack(side="left", fill="both", expand=True)
        video_scrollbar.pack(side="right", fill="y")
        
        # Data info section
        info_frame = ttk.LabelFrame(control_frame, text="Trace Data Information", padding="5")
        info_frame.pack(fill="x", pady=(0, 10))
        
        self.info_text = tk.Text(info_frame, height=6, width=40, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(info_frame, orient="vertical", command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=scrollbar.set)
        self.info_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # ROI selection section
        roi_frame = ttk.LabelFrame(control_frame, text="ROI Selection", padding="5")
        roi_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(roi_frame, text="Select ROIs to include:").pack(anchor="w")
        
        # ROI listbox with checkboxes
        self.roi_listbox_frame = ttk.Frame(roi_frame)
        self.roi_listbox_frame.pack(fill="x", pady=5)
        
        self.roi_vars = {}  # Will store checkbox variables
        
        # Video settings section
        settings_frame = ttk.LabelFrame(control_frame, text="Video Settings", padding="5")
        settings_frame.pack(fill="x", pady=(0, 10))
        
        # Output frame rate settings
        fps_frame = ttk.Frame(settings_frame)
        fps_frame.pack(fill="x", pady=2)
        ttk.Label(fps_frame, text="Output Video FPS:").pack(side="left")
        self.original_fps_var = tk.StringVar(value="30.0")
        fps_entry = ttk.Entry(fps_frame, textvariable=self.original_fps_var, width=8)
        fps_entry.pack(side="right")
        
        # Help text for FPS
        fps_help = ttk.Label(settings_frame, text="Controls video playback speed (separate from data sampling rate)", 
                            font=("Arial", 8), foreground="gray")
        fps_help.pack(anchor="w", pady=(0, 5))
        
        # Playback speed settings
        speed_frame = ttk.Frame(settings_frame)
        speed_frame.pack(fill="x", pady=2)
        ttk.Label(speed_frame, text="Playback Speed:").pack(side="left")
        self.speed_var = tk.StringVar(value="1.0")
        speed_combo = ttk.Combobox(speed_frame, textvariable=self.speed_var, 
                                   values=["0.1", "0.25", "0.5", "1.0", "2.0", "4.0", "8.0"], width=8)
        speed_combo.pack(side="right")
        
        # Trace spacing settings
        spacing_frame = ttk.Frame(settings_frame)
        spacing_frame.pack(fill="x", pady=2)
        ttk.Label(spacing_frame, text="Trace Spacing:").pack(side="left")
        self.spacing_var = tk.StringVar(value="auto")
        spacing_entry = ttk.Entry(spacing_frame, textvariable=self.spacing_var, width=10)
        spacing_entry.pack(side="right")
        
        # Output format settings
        format_frame = ttk.Frame(settings_frame)
        format_frame.pack(fill="x", pady=2)
        ttk.Label(format_frame, text="Output Format:").pack(side="left")
        self.format_var = tk.StringVar(value="mp4")
        format_combo = ttk.Combobox(format_frame, textvariable=self.format_var, 
                                   values=["mp4", "gif", "both"], width=10)
        format_combo.pack(side="right")
        
        # Generate video section with animation controls
        generate_frame = ttk.LabelFrame(control_frame, text="Generate Video", padding="5")
        generate_frame.pack(fill="x", pady=(0, 10))
        
        # Animation control buttons in horizontal layout
        animation_button_frame = ttk.Frame(generate_frame)
        animation_button_frame.pack(fill="x", pady=2)
        
        self.preview_button = ttk.Button(animation_button_frame, text="Preview Animation", 
                  command=self.preview_animation)
        self.preview_button.pack(side="left", padx=(0, 5))
        
        self.stop_button = ttk.Button(animation_button_frame, text="Stop Preview", 
                  command=self.stop_animation, state="disabled")
        self.stop_button.pack(side="left", padx=(0, 5))
        
        # Generate video button
        ttk.Button(generate_frame, text="Generate Video", 
                  command=self.generate_video).pack(fill="x", pady=(5, 2))
        
        # Progress display
        self.progress_var = tk.StringVar(value="Ready")
        progress_label = ttk.Label(generate_frame, textvariable=self.progress_var)
        progress_label.pack(anchor="w", pady=2)
        
        # Right panel for preview
        self.preview_frame = ttk.LabelFrame(main_frame, text="Preview", padding="10")
        self.preview_frame.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Canvas for matplotlib
        self.canvas_frame = ttk.Frame(self.preview_frame)
        self.canvas_frame.pack(fill="both", expand=True)
        
        # Add status bar at bottom
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.status_var = tk.StringVar(value="Ready to load data")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, font=("Arial", 9))
        status_label.pack(side="left")
        
        # Add version info
        version_label = ttk.Label(status_frame, text="Calcium Imaging GUI v3.0", 
                                 font=("Arial", 8), foreground="gray")
        version_label.pack(side="right")

    def load_roi_coordinates(self):
        """Load ROI coordinates from a CSV file"""
        file_path = filedialog.askopenfilename(
            title="Select ROI Coordinates File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            # Load coordinates file - expect columns: ROI_Name, X, Y
            coord_df = pd.read_csv(file_path)
            
            # Check required columns
            required_cols = ['ROI_Name', 'X', 'Y']
            if not all(col in coord_df.columns for col in required_cols):
                messagebox.showerror("Error", f"CSV must contain columns: {', '.join(required_cols)}")
                return
            
            # Create coordinate mapping
            coord_map = {}
            for _, row in coord_df.iterrows():
                coord_map[row['ROI_Name']] = (row['X'], row['Y'])
            
            # Update ROI coordinates if we have trace data
            if hasattr(self, 'roi_names') and self.roi_names:
                updated_coords = []
                for roi_name in self.roi_names:
                    if roi_name in coord_map:
                        updated_coords.append(coord_map[roi_name])
                    else:
                        # Use default position if not found
                        default_coords = self.generate_default_coordinates(1)
                        updated_coords.append(default_coords[0])
                        print(f"Warning: No coordinates found for ROI {roi_name}, using default")
                
                self.roi_coordinates = updated_coords
                messagebox.showinfo("Success", f"Loaded coordinates for {len(coord_map)} ROIs")
            else:
                messagebox.showwarning("Warning", "Load ROI data first, then load coordinates")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load coordinates: {str(e)}")
        
    def convert_frames_to_time(self, n_frames):
        """Convert frame indices to time based on sampling rate"""
        try:
            sampling_rate = float(self.sampling_rate_var.get())
            if sampling_rate > 0:
                # Create time array: time = frame_index / sampling_rate
                return np.arange(n_frames) / sampling_rate
            else:
                # If no valid sampling rate, return frame indices
                return np.arange(n_frames)
        except ValueError:
            # If invalid sampling rate, return frame indices
            return np.arange(n_frames)
      
    def stop_animation(self):
        """Stop the current animation"""
        if hasattr(self, 'current_animation') and self.current_animation is not None:
            self.current_animation.event_source.stop()
            self.current_animation = None
            self.stop_button.config(state="disabled")
            self.preview_button.config(state="normal")
            self.progress_var.set("Animation stopped")
 
    def load_video_data(self):
        """Load video data from AVI/MP4 file"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.avi *.mp4 *.mov"), ("AVI files", "*.avi"), ("MP4 files", "*.mp4"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            if not OPENCV_AVAILABLE:
                messagebox.showerror("Error", "OpenCV is required to load video files. Please install opencv-python.")
                return
            
            # Open video file
            cap = cv2.VideoCapture(file_path)
            
            if not cap.isOpened():
                messagebox.showerror("Error", "Could not open video file.")
                return
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.original_fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Update FPS display
            self.original_fps_var.set(f"{self.original_fps:.2f}")
            
            # Read all frames
            frames = []
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                frame_count += 1
                
                # Show progress
                if frame_count % 100 == 0:
                    self.progress_var.set(f"Loading video... {frame_count}/{total_frames}")
                    self.root.update()
            
            cap.release()
            
            # Convert to numpy array
            self.imaging_data = np.array(frames)
            
            # Create time array based on FPS
            self.frame_times = np.arange(total_frames) / self.original_fps
            
            # Update video info display
            self.update_video_info_display(file_path, total_frames, width, height)
            
            # Update general info display
            self.update_info_display()
            
            messagebox.showinfo("Success", f"Loaded video: {total_frames} frames at {self.original_fps:.2f} FPS")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load video: {str(e)}")
        finally:
            self.progress_var.set("Ready")
    
    def update_video_info_display(self, file_path, total_frames, width, height):
        """Update the video information display"""
        self.video_info_text.delete(1.0, tk.END)
        
        duration = total_frames / self.original_fps
        info_text = f"Video File: {os.path.basename(file_path)}\n"
        info_text += f"Frames: {total_frames}\n"
        info_text += f"Resolution: {width}x{height}\n"
        info_text += f"FPS: {self.original_fps:.2f}\n"
        info_text += f"Duration: {duration:.2f} seconds"
        
        self.video_info_text.insert(1.0, info_text)
    
    def sort_roi_names(self, roi_names):
        """Sort ROI names in the desired order: t1, t2, t3, a1, a2, etc."""
        def extract_roi_info(name):
            # Extract segment type and side from names like "Mean(t1l)" or "Mean(a3r)"
            import re
            match = re.search(r'([ta])(\d+)([lr])', name.lower())
            if match:
                segment_type = match.group(1)  # 't' or 'a'
                segment_num = int(match.group(2))  # 1, 2, 3, etc.
                side = match.group(3)  # 'l' or 'r'
                
                # Sort priority: t segments first, then a segments
                type_priority = 0 if segment_type == 't' else 1
                return (type_priority, segment_num, side, name)
            else:
                # If name doesn't match pattern, put at end
                return (2, 999, 'z', name)
        
        # Sort ROI names and get the indices
        sorted_items = sorted(enumerate(roi_names), key=lambda x: extract_roi_info(x[1]))
        sorted_indices = [item[0] for item in sorted_items]
        sorted_names = [item[1] for item in sorted_items]
        
        return sorted_indices, sorted_names

    def generate_roi_colors(self, roi_names):
        """Generate red/blue colors for left/right ROIs"""
        colors = []
        for name in roi_names:
            if 'l' in name.lower():
                colors.append('red')  # Left side = red
            elif 'r' in name.lower():
                colors.append('blue')  # Right side = blue
            else:
                colors.append('gray')  # Unknown side = gray
        return colors

    def update_roi_selection(self):
        """Update ROI selection checkboxes"""
        # Clear existing checkboxes
        for widget in self.roi_listbox_frame.winfo_children():
            widget.destroy()
        
        self.roi_vars = {}
        
        if not self.roi_names:
            return
        
        # Create scrollable frame
        canvas = tk.Canvas(self.roi_listbox_frame, height=120)
        scrollbar = ttk.Scrollbar(self.roi_listbox_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Add checkboxes
        for i, roi_name in enumerate(self.roi_names):
            var = tk.BooleanVar(value=True)  # All selected by default
            self.roi_vars[roi_name] = var
            
            cb = ttk.Checkbutton(scrollable_frame, text=roi_name, variable=var)
            cb.grid(row=i, column=0, sticky="w", padx=5, pady=1)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Add select all/none buttons
        button_frame = ttk.Frame(self.roi_listbox_frame)
        button_frame.pack(fill="x", pady=5)
        
        ttk.Button(button_frame, text="Select All", 
                  command=self.select_all_rois).pack(side="left", padx=2)
        ttk.Button(button_frame, text="Select None", 
                  command=self.select_no_rois).pack(side="left", padx=2)
    
    def select_all_rois(self):
        """Select all ROIs"""
        for var in self.roi_vars.values():
            var.set(True)
    
    def select_no_rois(self):
        """Deselect all ROIs"""
        for var in self.roi_vars.values():
            var.set(False)

    def get_selected_data(self):
        if self.trace_data is None:
            return None, None, None, None
        
        selected_indices = []
        selected_names = []
        selected_coords = []
        selected_colors = []
        
        for i, (roi_name, var) in enumerate(self.roi_vars.items()):
            if var.get():
                selected_indices.append(i)
                selected_names.append(roi_name)
                selected_coords.append(self.roi_coordinates[i])
                selected_colors.append(self.colors[i])
        
        if not selected_indices:
            messagebox.showwarning("Warning", "No ROIs selected!")
            return None, None, None, None
        
        selected_traces = self.trace_data[selected_indices, :]
        return selected_traces, selected_names, selected_coords, selected_colors

    def preview_animation(self):
        """Fixed preview animation with proper updates"""
        selected_traces, selected_names, selected_coords, selected_colors = self.get_selected_data()
        
        if selected_traces is None:
            return
        
        try:
            # Enable stop button, disable preview button
            self.stop_button.config(state="normal")
            self.preview_button.config(state="disabled")
            
            # Clear previous preview
            for widget in self.canvas_frame.winfo_children():
                widget.destroy()
            
            # Generate or use existing imaging data
            n_frames = selected_traces.shape[1]
            if self.imaging_data is None:
                imaging_data = self.generate_synthetic_imaging(n_frames, selected_coords)
            else:
                imaging_data = self.imaging_data[:n_frames]
            
            # Apply fluorescence normalization for better display
            normalized_traces = self.apply_fluorescence_normalization(selected_traces)
            
            # Create matplotlib figure
            self.preview_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            self.preview_fig.patch.set_facecolor('white')
            self.preview_fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.1, wspace=0.15)
            
            # Setup imaging plot (left side)
            ax1.set_title('Calcium Imaging with ROIs', fontsize=14, pad=15)
            im = ax1.imshow(imaging_data[0], cmap='hot', animated=True)
            ax1.set_xlabel('X (pixels)', fontsize=12)
            ax1.set_ylabel('Y (pixels)', fontsize=12)
            
            # Add ROI circles - FIXED: Make sure they're visible
            roi_circles = []
            for i, (x, y) in enumerate(selected_coords):
                color = selected_colors[i] if i < len(selected_colors) else 'gray'
                circle = Circle((x, y), radius=16, fill=False, 
                                color=color, linewidth=2.0, alpha=1.0)
                ax1.add_patch(circle)
                roi_circles.append(circle)
            
            # Setup trace plot (right side)
            ax2.set_title('Fluorescence Traces', fontsize=14, pad=15)
            ax2.set_facecolor('#FFFFFF')
            ax2.set_xlabel('Time (s)', fontsize=14)
            ax2.set_ylabel('ΔF/F₀ (%)', fontsize=14)
            
            # Get signal pairs for organization
            pairs = self.get_ordered_signal_pairs(selected_names)
            
            # Calculate time values
            if self.frame_times is not None:
                time_values_full = self.frame_times
            else:
                time_values_full = np.arange(n_frames) / float(self.original_fps_var.get())
            
            # Calculate spacing and range
            base_spacing = 70 * 0.7
            total_signals = len([p for p in pairs if p[0] is not None or p[1] is not None])
            total_range = base_spacing * total_signals
            
            # Set axis limits
            ax2.set_xlim(-time_values_full[-1] * 0.1, time_values_full[-1] * 1.02)
            ax2.set_ylim(-base_spacing * 0.2, total_range + base_spacing * 0.2)
            
            # Configure styling
            ax2.grid(True, axis='both', linestyle='--', alpha=0.3, linewidth=0.8)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.tick_params(axis='both', which='major', length=6, width=1.2, labelsize=12)
            
            # Embed in tkinter
            canvas = FigureCanvasTkAgg(self.preview_fig, self.canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
            # FIXED: Animation function - No blitting, complete redraw each frame
            def animate(frame):
                # Update imaging data
                im.set_array(imaging_data[frame])
                
                # Update title
                current_time = time_values_full[frame] if frame < len(time_values_full) else frame
                ax1.set_title(f'Calcium Imaging - Frame {frame+1}/{n_frames} (t={current_time:.2f}s)', fontsize=14)
                
                # Clear and redraw traces
                ax2.clear()
                ax2.set_facecolor('#FFFFFF')
                ax2.set_title('Fluorescence Traces', fontsize=14, pad=15)
                ax2.set_xlabel('Time (s)', fontsize=14)
                ax2.set_ylabel('ΔF/F₀ (%)', fontsize=14)
                
                # Get time values up to current frame
                current_time_values = time_values_full[:frame+1]
                
                # Plot traces up to current frame
                signals_plotted = 0
                
                for left_signal, right_signal in reversed(pairs):
                    if left_signal is None and right_signal is None:
                        continue
                        
                    base_offset = signals_plotted * base_spacing
                    signals_plotted += 1
                    
                    # Get base name for labeling
                    if left_signal is not None:
                        # Extract base name like "T1" from "Mean(t1l)"
                        match = re.search(r'mean\(([ta]\d+)[lr]\)', left_signal.lower())
                        if match:
                            base_name = match.group(1).upper()
                        else:
                            base_name = left_signal[:6]
                    elif right_signal is not None:
                        match = re.search(r'mean\(([ta]\d+)[lr]\)', right_signal.lower())
                        if match:
                            base_name = match.group(1).upper()
                        else:
                            base_name = right_signal[:6]
                    else:
                        base_name = "ROI"
                    
                    # Plot left signal
                    if left_signal is not None:
                        left_idx = selected_names.index(left_signal)
                        y_left = normalized_traces[left_idx, :frame+1] + base_offset
                        ax2.plot(current_time_values, y_left, 'red', alpha=0.8, linewidth=1.5)
                    
                    # Plot right signal
                    if right_signal is not None:
                        right_idx = selected_names.index(right_signal)
                        y_right = normalized_traces[right_idx, :frame+1] + base_offset
                        ax2.plot(current_time_values, y_right, 'blue', alpha=0.8, linewidth=1.5)
                    
                    # Add label
                    if len(current_time_values) > 0:
                        x_label_pos = -time_values_full[-1] * 0.08
                        
                        # Calculate label position
                        if left_signal is not None:
                            left_idx = selected_names.index(left_signal)
                            first_points_avg = np.mean(normalized_traces[left_idx, :min(5, frame+1)])
                        else:
                            right_idx = selected_names.index(right_signal)
                            first_points_avg = np.mean(normalized_traces[right_idx, :min(5, frame+1)])
                        
                        y_label_pos = base_offset + first_points_avg
                        
                        ax2.text(x_label_pos, y_label_pos, base_name,
                                verticalalignment='center', horizontalalignment='right',
                                color='black', fontsize=14, fontweight='bold')
                
                # Restore axis formatting
                ax2.set_xlim(-time_values_full[-1] * 0.1, time_values_full[-1] * 1.02)
                ax2.set_ylim(-base_spacing * 0.2, total_range + base_spacing * 0.2)
                ax2.grid(True, axis='both', linestyle='--', alpha=0.3, linewidth=0.8)
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                ax2.tick_params(axis='both', which='major', length=6, width=1.2, labelsize=12)
                
                # Return empty list since we're not using blitting
                return []
            
            # Create animation - FIXED: Use blit=False for complete redraw
            playback_speed = float(self.speed_var.get())
            effective_fps = float(self.original_fps_var.get()) * playback_speed
            
            self.current_animation = animation.FuncAnimation(
                self.preview_fig, animate, frames=min(n_frames, 200),
                interval=1000/effective_fps, blit=False, repeat=True
            )
            
            self.progress_var.set(f"Preview running at {playback_speed}x speed - {len(selected_names)} ROIs")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create preview: {str(e)}")
            # Re-enable preview button on error
            self.stop_button.config(state="disabled")
            self.preview_button.config(state="normal")
            import traceback
            traceback.print_exc()

    def update_info_display(self):
        """Updated info display with time information"""
        self.info_text.delete(1.0, tk.END)
        
        info_text = "Trace Data Information:\n" + "="*25 + "\n\n"
        
        if self.trace_data is not None:
            n_rois, n_timepoints = self.trace_data.shape
            sampling_rate = float(self.sampling_rate_var.get()) if self.sampling_rate_var.get() else 0
            
            info_text += f"ROI Traces: {n_rois} ROIs, {n_timepoints} time points\n"
            info_text += f"ROI Names: {', '.join(self.roi_names[:3])}"
            if len(self.roi_names) > 3:
                info_text += f"... and {len(self.roi_names)-3} more\n"
            else:
                info_text += "\n"
            info_text += f"Data range: {np.min(self.trace_data):.2f} to {np.max(self.trace_data):.2f}\n"
            info_text += f"Sampling rate: {sampling_rate} Hz\n"
            
            if self.frame_times is not None:
                duration = self.frame_times[-1]
                info_text += f"Duration: {duration:.2f} seconds\n"
        
        if self.imaging_data is not None:
            info_text += f"\nImaging Data: {self.imaging_data.shape}\n"
        else:
            info_text += "\nImaging Data: Not loaded (will generate synthetic background)\n"
        
        # Add ROI coordinate info
        if hasattr(self, 'roi_coordinates') and self.roi_coordinates:
            info_text += f"ROI Coordinates: {len(self.roi_coordinates)} positions loaded\n"
        
        self.info_text.insert(1.0, info_text)

    def generate_video(self):
            """Generate the final video file"""
            selected_traces, selected_names, selected_coords, selected_colors = self.get_selected_data()
            
            if selected_traces is None:
                return
            
            # Get output file path
            file_path = filedialog.asksaveasfilename(
                title="Save Video As",
                defaultextension=".avi",  # Default to AVI
                filetypes=[("AVI files", "*.avi"), ("MP4 files", "*.mp4"), ("GIF files", "*.gif"), ("All files", "*.*")]
            )
            
            if not file_path:
                return
            
            # Start video generation in separate thread
            self.is_generating_video = True
            self.progress_var.set("Generating video... 0%")
            self.file_path = file_path  # Store for use in main thread
            self.video_thread = threading.Thread(
                target=self._generate_video_thread,
                args=(selected_traces, selected_names, selected_coords, selected_colors, file_path),
                daemon=False  # Non-daemon thread to ensure cleanup
            )
            self.video_thread.start()
            
            # Schedule periodic checks for queue updates
            self._check_queue()

    def _check_queue(self):
            """Check the queue for updates from the video generation thread"""
            try:
                while not self.video_generation_queue.empty():
                    message = self.video_generation_queue.get_nowait()
                    if message.startswith("PROGRESS:"):
                        self.progress_var.set(message.replace("PROGRESS:", ""))
                    elif message.startswith("ERROR:"):
                        messagebox.showerror("Error", message.replace("ERROR:", ""))
                        self.progress_var.set("Error occurred during video generation")
                        self.is_generating_video = False
                        self.video_generation_complete.set()
                    elif message == "COMPLETE":
                        self.progress_var.set("Video saved successfully!")
                        messagebox.showinfo("Success", f"Video generated successfully!\n{self.file_path}")
                        self.is_generating_video = False
                        self.video_generation_complete.set()
            except queue.Empty:
                pass
            
            if self.is_generating_video:
                self.root.after(100, self._check_queue)  # Check every 100ms

    def _generate_video_thread(self, selected_traces, selected_names, selected_coords, selected_colors, file_path):
            """Generate video in separate thread to avoid GUI freezing"""
            try:
                n_frames = selected_traces.shape[1]
                playback_speed = float(self.speed_var.get())
                output_fps = float(self.original_fps_var.get()) * playback_speed
                
                # Generate or use existing imaging data
                if self.imaging_data is None:
                    imaging_data = self.generate_synthetic_imaging(n_frames, selected_coords)
                else:
                    imaging_data = self.imaging_data[:n_frames]
                
                # Ensure we have frame times
                if self.frame_times is None:
                    original_fps = float(self.original_fps_var.get())
                    self.frame_times = np.arange(n_frames) / original_fps
                
                # Calculate spacing
                spacing_val = self.spacing_var.get()
                if spacing_val == "auto":
                    spacing = (np.max(selected_traces) - np.min(selected_traces)) * 1.2
                else:
                    spacing = float(spacing_val)
                
                if OPENCV_AVAILABLE:
                    # Use OpenCV method (better quality)
                    self._generate_with_opencv(imaging_data, selected_traces, selected_names, 
                                             selected_coords, selected_colors, file_path, output_fps, spacing)
                else:
                    # Use matplotlib method (fallback)
                    self._generate_with_matplotlib(imaging_data, selected_traces, selected_names, 
                                                 selected_coords, selected_colors, file_path, output_fps, spacing)
                
                self.video_generation_queue.put("COMPLETE")
                
            except Exception as e:
                error_message = f"Failed to generate video: {str(e)}"
                self.video_generation_queue.put(f"ERROR:{error_message}")

    def _generate_with_opencv(self, imaging_data, selected_traces, selected_names, 
                             selected_coords, selected_colors, file_path, output_fps, spacing):
        """Updated video generation ensuring both imaging and traces update"""
        n_frames = imaging_data.shape[0]
        frame_width, frame_height = 1600, 800
        
        # Apply fluorescence normalization
        normalized_traces = self.apply_fluorescence_normalization(selected_traces)
        
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter(file_path, fourcc, output_fps, (frame_width, frame_height))
        
        max_time = self.frame_times[-1] if len(self.frame_times) > 0 else n_frames / float(self.original_fps_var.get())
        
        for frame_idx in range(n_frames):
            progress = int((frame_idx / n_frames) * 100)
            self.video_generation_queue.put(f"PROGRESS:Generating video... {progress}%")
            
            # Create new figure for each frame
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            fig.patch.set_facecolor('white')
            fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.1, wspace=0.15)
            
            # FIXED: Plot imaging data for CURRENT frame
            ax1.imshow(imaging_data[frame_idx], cmap='hot')  # Use current frame data
            current_time = self.frame_times[frame_idx] if frame_idx < len(self.frame_times) else frame_idx / float(self.original_fps_var.get())
            ax1.set_title(f'Calcium Imaging - Frame {frame_idx+1}/{n_frames} (t={current_time:.2f}s)', fontsize=14)
            ax1.set_xlabel('X (pixels)', fontsize=12)
            ax1.set_ylabel('Y (pixels)', fontsize=12)
            
            # Add ROI circles
            for i, (x, y) in enumerate(selected_coords):
                color = selected_colors[i] if i < len(selected_colors) else 'gray'
                circle = Circle((x, y), radius=15, fill=False, 
                                color=color, linewidth=1.2, alpha=0.9)
                ax1.add_patch(circle)
            
            # Plot traces up to current frame
            ax2.set_facecolor('#FFFFFF')
            ax2.patch.set_alpha(0.9)
            
            time_values = self.frame_times[:frame_idx+1] if self.frame_times is not None else np.arange(frame_idx+1) / float(self.original_fps_var.get())
            
            # Plot traces with enhanced styling
            pairs = self.get_ordered_signal_pairs(selected_names)
            base_spacing = 70 * 0.7
            signals_plotted = 0
            
            for i, (left_signal, right_signal) in enumerate(reversed(pairs)):
                base_offset = signals_plotted * base_spacing
                signals_plotted += 1
                
                # Plot signals up to current frame
                if left_signal is not None:
                    left_idx = selected_names.index(left_signal)
                    y_left = normalized_traces[left_idx, :frame_idx+1] + base_offset
                    ax2.plot(time_values, y_left, 'red', alpha=0.8, linewidth=1.5)
                
                if right_signal is not None:
                    right_idx = selected_names.index(right_signal)
                    y_right = normalized_traces[right_idx, :frame_idx+1] + base_offset
                    ax2.plot(time_values, y_right, 'blue', alpha=0.8, linewidth=1.5)
            
            # Format trace plot
            ax2.set_xlabel('Time (s)', fontsize=14)
            ax2.set_ylabel('ΔF/F₀ (%)', fontsize=14)
            ax2.set_title('Fluorescence Traces', fontsize=14)
            ax2.grid(True, axis='both', linestyle='--', alpha=0.3)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            
            plt.tight_layout()
            
            # Convert to OpenCV format
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            img = np.frombuffer(buf, dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            img = cv2.resize(img, (frame_width, frame_height))
            
            out.write(img)
            plt.close(fig)
        
        out.release()

    def on_closing(self):
            """Handle window close event, wait for video generation to complete"""
            if self.is_generating_video and self.video_thread and self.video_thread.is_alive():
                messagebox.showwarning("Warning", "Please wait for video generation to complete.")
            else:
                if self.video_thread:
                    self.video_thread.join(timeout=1.0)  # Wait briefly for thread to finish
                self.root.destroy()

    def _generate_with_matplotlib(self, imaging_data, selected_traces, selected_names, 
                                selected_coords, selected_colors, file_path, output_fps, spacing):
        """Generate video using matplotlib animation with time axes"""
        n_frames = imaging_data.shape[0]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        im = ax1.imshow(imaging_data[0], cmap='hot')
        ax1.set_title('Calcium Imaging with ROIs')
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        
        for i, (x, y) in enumerate(selected_coords):
            circle = Circle((x, y), radius=8, fill=False, 
                          color=selected_colors[i], linewidth=3, alpha=0.9)
            ax1.add_patch(circle)
        
        ax2.set_title('Fluorescence Traces')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Fluorescence Intensity')
        
        trace_lines = []
        for i in range(len(selected_names)):
            line, = ax2.plot([], [], color=selected_colors[i], linewidth=2, 
                           label=selected_names[i])
            trace_lines.append(line)
        
        max_time = self.frame_times[-1] if len(self.frame_times) > 0 else n_frames / float(self.original_fps_var.get())
        ax2.set_xlim(0, max_time)
        ax2.set_ylim(np.min(selected_traces) - spacing,
                    np.max(selected_traces) + len(selected_names) * spacing)
        ax2.legend()
        
        plt.tight_layout()
        
        def animate(frame):
            progress = int((frame / n_frames) * 100)
            self.progress_var.set(f"Generating video... {progress}%")
            
            im.set_array(imaging_data[frame])
            
            current_times = self.frame_times[:frame+1]
            current_time = self.frame_times[frame] if frame < len(self.frame_times) else 0
            
            for i in range(len(selected_names)):
                y_data = selected_traces[i, :frame+1] + (i * spacing)
                trace_lines[i].set_data(current_times, y_data)
            
            ax1.set_title(f'Frame {frame+1}/{n_frames} (t={current_time:.2f}s)')
            return [im] + trace_lines
        
        anim = animation.FuncAnimation(fig, animate, frames=n_frames,
                                     interval=1000/output_fps, blit=True)
        
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=output_fps, metadata=dict(artist='CalciumImaging'), bitrate=1800)
        anim.save(file_path, writer=writer)
        
        self.progress_var.set(f"Video saved successfully!")
        self.root.after(0, lambda: messagebox.showinfo(
            "Success", f"Video generated successfully!\n{file_path}"))
        
        plt.close(fig)


# Here are all the supporting methods needed - add these to your CalciumImagingGUI class:

    def get_ordered_signal_pairs(self, selected_names):
        """Get paired signals organized properly - FIXED VERSION"""
        import re
        
        def sort_key(signal_name):
            # Handle names like "Mean(t1l)" or "Mean(a2r)"
            match = re.search(r'mean\(([ta])(\d+)([lr])\)', signal_name.lower())
            if match:
                sig_type = match.group(1)  # 't' or 'a'
                num = int(match.group(2))
                return (0 if sig_type == 't' else 1, num)
            return (99, 99)
        
        # Separate left and right signals
        left_signals = []
        right_signals = []
        
        for signal in selected_names:
            if 'l)' in signal.lower():
                left_signals.append(signal)
            elif 'r)' in signal.lower():
                right_signals.append(signal)
        
        # Sort them
        left_signals.sort(key=sort_key)
        right_signals.sort(key=sort_key)
        
        # Create pairs based on matching base names
        pairs = []
        used_right = set()
        
        for left_sig in left_signals:
            # Extract base name (e.g., 't1' from 'Mean(t1l)')
            left_match = re.search(r'mean\(([ta]\d+)l\)', left_sig.lower())
            if left_match:
                left_base = left_match.group(1)
                
                # Find matching right signal
                right_sig = None
                for right in right_signals:
                    if right in used_right:
                        continue
                    right_match = re.search(r'mean\(([ta]\d+)r\)', right.lower())
                    if right_match and right_match.group(1) == left_base:
                        right_sig = right
                        used_right.add(right)
                        break
                
                pairs.append((left_sig, right_sig))
        
        # Add any unpaired right signals
        for right_sig in right_signals:
            if right_sig not in used_right:
                pairs.append((None, right_sig))
        
        # If no pairs found, create individual signals
        if not pairs:
            for signal in selected_names:
                pairs.append((signal, None))
        
        return pairs
    
    def apply_fluorescence_normalization(self, trace_data):
        """Apply ΔF/F₀ normalization - FIXED VERSION"""
        if trace_data is None or trace_data.size == 0:
            return trace_data
        
        normalized_data = np.zeros_like(trace_data)
        
        for i in range(trace_data.shape[0]):
            signal = trace_data[i, :]
            
            # Calculate F₀ as mean of first 10% of data (minimum 5 points)
            baseline_length = max(5, int(len(signal) * 0.1))
            f0 = np.mean(signal[:baseline_length])
            
            # Apply ΔF/F₀ normalization: ((F - F₀) / F₀) * 100
            if f0 != 0:
                normalized_data[i, :] = ((signal - f0) / f0) * 100
            else:
                # If F₀ is zero, just use the original signal scaled
                normalized_data[i, :] = signal - np.mean(signal)
        
        return normalized_data
    
    def generate_synthetic_imaging(self, n_frames, selected_coords):
        """Generate synthetic imaging data - IMPROVED VERSION"""
        # Create realistic background dimensions
        height, width = 600, 800  # Larger synthetic image
        
        # Start with background noise
        imaging_data = np.random.randn(n_frames, height, width) * 5 + 50
        
        # Add some structured background patterns
        y_coords, x_coords = np.ogrid[:height, :width]
        
        # Add gradient background
        gradient = (x_coords / width + y_coords / height) * 30
        for t in range(n_frames):
            imaging_data[t] += gradient
        
        # Add activity at ROI locations
        for i, (x, y) in enumerate(selected_coords):
            # Ensure coordinates are within bounds
            x_int = int(np.clip(x, 0, width-1))
            y_int = int(np.clip(y, 0, height-1))
            
            for t in range(n_frames):
                # Create gaussian activity around each ROI
                activity = np.exp(-((x_coords - x_int)**2 + (y_coords - y_int)**2) / (2 * 20**2))
                
                # Add time-varying calcium activity
                calcium_signal = 100 + 50 * np.sin(0.1 * t + i) + 30 * np.sin(0.05 * t + i * 0.5)
                imaging_data[t] += activity * calcium_signal
        
        # Ensure all values are positive and reasonable
        imaging_data = np.clip(imaging_data, 0, 255)
        
        return imaging_data.astype(np.uint8)
    
    def load_roi_data(self):
        """COMPLETE FIXED VERSION of load_roi_data"""
        file_path = filedialog.askopenfilename(
            title="Select ROI Data File",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            # Load the file
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            # Use enhanced sorting
            original_roi_names = df.columns.tolist()
            sorted_indices, sorted_names = self.sort_roi_names_enhanced(original_roi_names)
            
            # Reorder the data according to sorted indices
            sorted_data = df.iloc[:, sorted_indices].values.T  # Transpose to get (n_rois, time)
            
            # Store sorted data and names
            self.trace_data = sorted_data
            self.roi_names = sorted_names
            
            # Generate default ROI coordinates
            n_rois = len(self.roi_names)
            self.roi_coordinates = self.generate_default_coordinates(n_rois)
            
            # Generate enhanced colors for left/right
            self.colors = self.generate_enhanced_colors(self.roi_names)
            
            # Create time array based on sampling rate
            n_timepoints = self.trace_data.shape[1]
            self.frame_times = self.convert_frames_to_time(n_timepoints)
            
            # Check for pending ImageJ ROI coordinates
            if hasattr(self, 'pending_roi_coordinates'):
                self._match_rois_with_traces(self.pending_roi_coordinates)
                delattr(self, 'pending_roi_coordinates')
            
            # Update displays
            self.update_info_display()
            self.update_roi_selection()
            
            # Show success message
            sampling_rate = float(self.sampling_rate_var.get()) if self.sampling_rate_var.get() else 0
            duration = self.frame_times[-1] if len(self.frame_times) > 0 else 0
            
            messagebox.showinfo("Success", 
                              f"Loaded {n_rois} ROIs with {df.shape[0]} time points\n"
                              f"Sampling rate: {sampling_rate} Hz\n"
                              f"Duration: {duration:.2f} seconds\n"
                              f"Order: {', '.join(sorted_names[:6])}{'...' if len(sorted_names) > 6 else ''}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def sort_roi_names_enhanced(self, roi_names):
        """Enhanced ROI sorting - FIXED VERSION"""
        import re
        
        def extract_roi_info(name):
            # Handle different naming patterns like "Mean(t1l)", "t1l", etc.
            patterns = [
                r'mean\(([ta])(\d+)([lr])\)',  # Mean(t1l) format
                r'([ta])(\d+)([lr])',          # t1l format
            ]
            
            for pattern in patterns:
                match = re.search(pattern, name.lower())
                if match:
                    segment_type = match.group(1)  # 't' or 'a'
                    segment_num = int(match.group(2))  # 1, 2, 3, etc.
                    side = match.group(3)  # 'l' or 'r'
                    
                    # Sort priority: t segments first, then a segments
                    type_priority = 0 if segment_type == 't' else 1
                    return (type_priority, segment_num, side, name)
            
            # If name doesn't match pattern, put at end
            return (2, 999, 'z', name)
        
        # Sort ROI names and get the indices
        sorted_items = sorted(enumerate(roi_names), key=lambda x: extract_roi_info(x[1]))
        sorted_indices = [item[0] for item in sorted_items]
        sorted_names = [item[1] for item in sorted_items]
        
        return sorted_indices, sorted_names
    
    def generate_enhanced_colors(self, roi_names):
        """Generate colors for left/right ROIs - FIXED VERSION"""
        colors = []
        for name in roi_names:
            name_lower = name.lower()
            if 'l)' in name_lower or 'l' in name_lower:
                colors.append('red')  # Left side = red
            elif 'r)' in name_lower or 'r' in name_lower:
                colors.append('blue')  # Right side = blue
            else:
                colors.append('gray')  # Unknown side = gray
        return colors
    
    def generate_default_coordinates(self, n_rois):
        """Generate realistic default ROI coordinates - IMPROVED VERSION"""
        coordinates = []
        
        # Parameters for positioning
        image_width = 800
        image_height = 600
        margin = 80
        
        if n_rois <= 12:
            # For reasonable numbers, create organized pattern
            if n_rois <= 6:
                # Two rows
                rows = 2
                cols = (n_rois + 1) // 2
            else:
                # Three rows
                rows = 3
                cols = (n_rois + 2) // 3
            
            # Calculate spacing
            x_spacing = (image_width - 2 * margin) / max(1, cols - 1) if cols > 1 else 0
            y_spacing = (image_height - 2 * margin) / max(1, rows - 1) if rows > 1 else 0
            
            for i in range(n_rois):
                row = i // cols
                col = i % cols
                
                if cols == 1:
                    x = image_width / 2
                else:
                    x = margin + col * x_spacing
                    
                if rows == 1:
                    y = image_height / 2
                else:
                    y = margin + row * y_spacing
                
                # Add small random offset
                x += np.random.randint(-20, 20)
                y += np.random.randint(-20, 20)
                
                # Keep within bounds
                x = np.clip(x, 30, image_width - 30)
                y = np.clip(y, 30, image_height - 30)
                
                coordinates.append((float(x), float(y)))
        
        else:
            # For many ROIs, use circular arrangement
            center_x, center_y = image_width / 2, image_height / 2
            radius = min(image_width, image_height) / 3
            
            for i in range(n_rois):
                angle = 2 * np.pi * i / n_rois
                x = center_x + radius * np.cos(angle)
                y = center_y + radius * np.sin(angle)
                
                # Add some randomness
                x += np.random.randint(-15, 15)
                y += np.random.randint(-15, 15)
                
                # Keep within bounds
                x = np.clip(x, 30, image_width - 30)
                y = np.clip(y, 30, image_height - 30)
                
                coordinates.append((float(x), float(y)))
        
        return coordinates








  
    def create_enhanced_plot(self, imaging_data, selected_traces, selected_names, selected_coords, selected_colors):
        """Create plot with enhanced styling like widgets_V5.py"""
        n_frames = selected_traces.shape[1]
        
        # Create figure with proper sizing
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.patch.set_facecolor('white')
        
        # Configure subplot spacing
        fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.1, wspace=0.15)
        
        # Setup imaging plot (left side)
        ax1.set_title('Calcium Imaging with ROIs', fontsize=14, pad=15)
        im = ax1.imshow(imaging_data[0], cmap='hot', animated=True)
        ax1.set_xlabel('X (pixels)', fontsize=12)
        ax1.set_ylabel('Y (pixels)', fontsize=12)
        
        # Add ROI circles with enhanced styling
        roi_circles = []
        for i, (x, y) in enumerate(selected_coords):
            color = selected_colors[i] if i < len(selected_colors) else 'gray'
            circle = Circle((x, y), radius=16, fill=False, 
                            color=color, linewidth=1.5, alpha=0.9)
            ax1.add_patch(circle)
            roi_circles.append(circle)
        
        # Setup trace plot (right side) with enhanced styling
        ax2.set_title('Fluorescence Traces', fontsize=14, pad=15)
        ax2.set_facecolor('#FFFFFF')
        ax2.patch.set_alpha(0.9)
        
        # Get signal pairs for proper organization
        pairs = self.get_ordered_signal_pairs(selected_names)
        
        # Calculate time values
        if self.frame_times is not None:
            time_values = self.frame_times
        else:
            time_values = np.arange(n_frames) / float(self.original_fps_var.get())
        
        # Enhanced spacing system
        base_spacing = 70  # Base vertical spacing between signals
        pair_spacing_multiplier = 0.7  # Reduce spacing within pairs
        
        # Plot signals with enhanced layout
        signals_plotted = 0
        signal_positions = {}  # Track positions for potential connections
        
        for i, (left_signal, right_signal) in enumerate(reversed(pairs)):
            base_offset = signals_plotted * base_spacing * pair_spacing_multiplier
            signals_plotted += 1
            
            # Determine base name for labeling
            if left_signal is not None:
                base_name = re.search(r'([ta]\d+)l', left_signal.lower())
                base_name = base_name.group(1).upper() if base_name else left_signal[:6]
            elif right_signal is not None:
                base_name = re.search(r'([ta]\d+)r', right_signal.lower())
                base_name = base_name.group(1).upper() if base_name else right_signal[:6]
            else:
                continue
            
            # Plot left signal if it exists
            if left_signal is not None:
                left_idx = selected_names.index(left_signal)
                y_left = selected_traces[left_idx, :] + base_offset
                ax2.plot(time_values, y_left, 'red', alpha=0.8, linewidth=1.5, label=f'{base_name}L' if i == 0 else "")
                signal_positions[left_signal] = {'offset': base_offset, 'y': selected_traces[left_idx, :]}
            
            # Plot right signal if it exists
            if right_signal is not None:
                right_idx = selected_names.index(right_signal)
                y_right = selected_traces[right_idx, :] + base_offset
                ax2.plot(time_values, y_right, 'blue', alpha=0.8, linewidth=1.5, label=f'{base_name}R' if i == 0 else "")
                signal_positions[right_signal] = {'offset': base_offset, 'y': selected_traces[right_idx, :]}
            
            # Add signal label with enhanced positioning
            x_label_pos = -time_values[-1] * 0.08  # Position labels further left
            
            # Calculate label y position based on first signal that exists
            if left_signal is not None:
                left_idx = selected_names.index(left_signal)
                first_points_avg = np.mean(selected_traces[left_idx, :5])
            else:
                right_idx = selected_names.index(right_signal)
                first_points_avg = np.mean(selected_traces[right_idx, :5])
            
            y_label_pos = base_offset + first_points_avg
            
            ax2.text(x_label_pos, y_label_pos, base_name,
                    verticalalignment='center', horizontalalignment='right',
                    color='black', fontsize=14, fontweight='bold')
        
        # Enhanced axis configuration
        ax2.set_xlabel('Time (s)', fontsize=14)
        ax2.set_ylabel('ΔF/F₀ (%)', fontsize=14)
        
        # Configure axis limits and ticks
        if time_values is not None:
            max_time = time_values[-1]
            ax2.set_xlim(-max_time * 0.12, max_time * 1.02)  # Extra space for labels
            
            # Set up time axis ticks
            time_ticks = np.arange(0, int(max_time) + 40, 40)
            ax2.set_xticks(time_ticks)
            ax2.set_xticklabels([f"{int(t)}" for t in time_ticks])
        
        # Set up y-axis
        total_range = base_spacing * pair_spacing_multiplier * signals_plotted
        y_ticks = np.arange(0, total_range + base_spacing/2, 50)
        ax2.set_yticks(y_ticks)
        ax2.set_yticklabels([f"{int(y)}" for y in y_ticks])
        
        # Enhanced grid and styling
        ax2.grid(True, axis='both', linestyle='--', alpha=0.3, linewidth=0.8)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_linewidth(1.2)
        ax2.spines['left'].set_linewidth(1.2)
        
        # Enhanced tick styling
        ax2.tick_params(axis='both', which='major', length=6, width=1.2, labelsize=12)
        ax2.tick_params(axis='x', colors='black')
        ax2.tick_params(axis='y', colors='black')
        
        return fig, (ax1, ax2), signal_positions
    
    def load_imagej_rois(self):
        """Load ImageJ ROI files from a folder or zip file"""
        if not READROI_AVAILABLE and not ROIFILE_AVAILABLE:
            messagebox.showerror("Error", 
                               "ROI reading libraries not available.\n"
                               "Install with: pip install read-roi\n"
                               "or: pip install roifile")
            return
        
        # Ask user to choose folder or zip file
        choice = messagebox.askyesno("ROI Source", 
                                    "Load ROIs from ZIP file?\n"
                                    "Yes = ZIP file\n"
                                    "No = Folder with .roi files")
        
        if choice:
            # Load from zip file
            file_path = filedialog.askopenfilename(
                title="Select ImageJ ROI ZIP File",
                filetypes=[("ZIP files", "*.zip"), ("All files", "*.*")]
            )
            if file_path:
                self._load_rois_from_zip(file_path)
        else:
            # Load from folder
            folder_path = filedialog.askdirectory(title="Select Folder with ROI Files")
            if folder_path:
                self._load_rois_from_folder(folder_path)
    
    def _load_rois_from_zip(self, zip_path):
        """Load ROIs from ImageJ zip file"""
        try:
            if READROI_AVAILABLE:
                rois = read_roi_zip(zip_path)
            else:
                messagebox.showerror("Error", "ZIP loading requires read-roi library")
                return
            
            self._process_loaded_rois(rois, zip_path)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load ROI zip file: {str(e)}")
    
    def _load_rois_from_folder(self, folder_path):
        """Load ROIs from folder containing .roi files"""
        try:
            import os
            roi_files = [f for f in os.listdir(folder_path) if f.endswith('.roi')]
            
            if not roi_files:
                messagebox.showwarning("Warning", "No .roi files found in selected folder")
                return
            
            rois = {}
            
            for roi_file in roi_files:
                file_path = os.path.join(folder_path, roi_file)
                roi_name = os.path.splitext(roi_file)[0]
                
                try:
                    if READROI_AVAILABLE:
                        roi_data = read_roi_file(file_path)
                        rois[roi_name] = roi_data
                    elif ROIFILE_AVAILABLE:
                        roi = roifile.ImagejRoi.fromfile(file_path)
                        rois[roi_name] = self._convert_roifile_to_dict(roi)
                    else:
                        messagebox.showerror("Error", "No ROI reading library available")
                        return
                        
                except Exception as e:
                    print(f"Warning: Could not load {roi_file}: {str(e)}")
                    continue
            
            self._process_loaded_rois(rois, folder_path)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load ROI folder: {str(e)}")
    
    def _convert_roifile_to_dict(self, roi):
        """Convert roifile ROI object to dictionary format"""
        # Extract coordinates based on ROI type
        if hasattr(roi, 'coordinates'):
            coords = roi.coordinates()
            if len(coords) > 0:
                x_coords = coords[:, 1]  # Note: roifile uses (y, x) format
                y_coords = coords[:, 0]
                
                return {
                    'type': 'polygon',
                    'x': x_coords.tolist(),
                    'y': y_coords.tolist(),
                    'left': min(x_coords),
                    'top': min(y_coords),
                    'width': max(x_coords) - min(x_coords),
                    'height': max(y_coords) - min(y_coords)
                }
        
        # Fallback for other ROI types
        return {
            'type': 'point',
            'x': [getattr(roi, 'left', 0)],
            'y': [getattr(roi, 'top', 0)],
            'left': getattr(roi, 'left', 0),
            'top': getattr(roi, 'top', 0),
            'width': getattr(roi, 'width', 10),
            'height': getattr(roi, 'height', 10)
        }
    
    def _process_loaded_rois(self, rois, source_path):
        """Process loaded ROIs and extract coordinates"""
        try:
            roi_coordinates = {}
            roi_info = []
            
            for roi_name, roi_data in rois.items():
                # Calculate centroid coordinates
                centroid_x, centroid_y = self._calculate_roi_centroid(roi_data)
                roi_coordinates[roi_name] = (centroid_x, centroid_y)
                
                # Store info for display
                roi_info.append(f"{roi_name}: ({centroid_x:.1f}, {centroid_y:.1f})")
            
            # Match ROIs with trace data
            if hasattr(self, 'roi_names') and self.roi_names:
                self._match_rois_with_traces(roi_coordinates)
            else:
                messagebox.showwarning("Warning", 
                                     "Load trace data first, then load ROIs.\n"
                                     f"Loaded {len(roi_coordinates)} ROIs from {source_path}")
                # Store for later use
                self.pending_roi_coordinates = roi_coordinates
            
            # Show info
            info_text = f"Loaded {len(roi_coordinates)} ROIs:\n" + "\n".join(roi_info[:10])
            if len(roi_info) > 10:
                info_text += f"\n... and {len(roi_info)-10} more"
            
            messagebox.showinfo("ROI Loading Success", info_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process ROIs: {str(e)}")
    
    def _calculate_roi_centroid(self, roi_data):
        """Calculate centroid coordinates from ROI data"""
        try:
            if 'x' in roi_data and 'y' in roi_data:
                # Polygon or freehand ROI
                x_coords = roi_data['x']
                y_coords = roi_data['y']
                
                if isinstance(x_coords, list) and len(x_coords) > 0:
                    centroid_x = np.mean(x_coords)
                    centroid_y = np.mean(y_coords)
                else:
                    centroid_x = float(x_coords) if not isinstance(x_coords, list) else x_coords[0]
                    centroid_y = float(y_coords) if not isinstance(y_coords, list) else y_coords[0]
                    
            elif 'left' in roi_data and 'top' in roi_data:
                # Rectangle ROI
                centroid_x = roi_data['left'] + roi_data.get('width', 0) / 2
                centroid_y = roi_data['top'] + roi_data.get('height', 0) / 2
                
            else:
                # Fallback - use first available coordinates
                centroid_x = 100  # Default value
                centroid_y = 100
                print(f"Warning: Could not determine coordinates for ROI, using defaults")
            
            return float(centroid_x), float(centroid_y)
            
        except Exception as e:
            print(f"Warning: Error calculating centroid: {str(e)}")
            return 100.0, 100.0
    
    def _match_rois_with_traces(self, roi_coordinates):
        """Match loaded ROI coordinates with trace data"""
        try:
            updated_coordinates = []
            matched_count = 0
            unmatched_traces = []
            
            for trace_name in self.roi_names:
                # Try different matching strategies
                matched_coord = None
                
                # Strategy 1: Exact name match
                if trace_name in roi_coordinates:
                    matched_coord = roi_coordinates[trace_name]
                    matched_count += 1
                
                # Strategy 2: Try without "Mean(" and ")" wrapper
                elif trace_name.startswith('Mean(') and trace_name.endswith(')'):
                    inner_name = trace_name[5:-1]  # Remove "Mean(" and ")"
                    if inner_name in roi_coordinates:
                        matched_coord = roi_coordinates[inner_name]
                        matched_count += 1
                
                # Strategy 3: Try partial matching (e.g., "a1l" matches "a1l.roi")
                if matched_coord is None:
                    for roi_name, coord in roi_coordinates.items():
                        if trace_name.lower() in roi_name.lower() or roi_name.lower() in trace_name.lower():
                            matched_coord = coord
                            matched_count += 1
                            break
                
                if matched_coord is not None:
                    updated_coordinates.append(matched_coord)
                else:
                    # Use default coordinates for unmatched traces
                    default_coords = self.generate_default_coordinates(1)
                    updated_coordinates.append(default_coords[0])
                    unmatched_traces.append(trace_name)
            
            # Update the coordinates
            self.roi_coordinates = updated_coordinates
            
            # Show matching results
            result_msg = f"Matched {matched_count}/{len(self.roi_names)} traces with ROIs"
            if unmatched_traces:
                result_msg += f"\n\nUnmatched traces (using default positions):\n"
                result_msg += "\n".join(unmatched_traces[:5])
                if len(unmatched_traces) > 5:
                    result_msg += f"\n... and {len(unmatched_traces)-5} more"
            
            messagebox.showinfo("ROI Matching Results", result_msg)
            
            # Update display
            self.update_info_display()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to match ROIs with traces: {str(e)}")
    
    # Update the setup_gui method to include the ImageJ ROI button
    def setup_gui_with_imagej_button(self):
        """Updated setup_gui method with ImageJ ROI loading"""
        # ... (keep all existing code, just add to the file loading section)
        
        # File loading section - ADD this button
        file_frame = ttk.LabelFrame(control_frame, text="Load Data", padding="5")
        file_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Button(file_frame, text="Load ROI Data (CSV/Excel)", 
                  command=self.load_roi_data).pack(fill="x", pady=2)
        ttk.Button(file_frame, text="Load Original Video (AVI/MP4)", 
                  command=self.load_video_data).pack(fill="x", pady=2)
        ttk.Button(file_frame, text="Load ImageJ ROIs", 
                  command=self.load_imagej_rois).pack(fill="x", pady=2)
        ttk.Button(file_frame, text="Load ROI Coordinates (CSV)", 
                  command=self.load_roi_coordinates).pack(fill="x", pady=2)
    
    # Update the load_roi_data method to check for pending ROI coordinates
    def load_roi_data_with_roi_check(self):
        """Updated load_roi_data that checks for pending ROI coordinates"""
        # ... (existing load_roi_data code) ...
        
        # After successfully loading trace data, add this:
        try:
            # ... existing loading code ...
            
            # Check if we have pending ROI coordinates from ImageJ
            if hasattr(self, 'pending_roi_coordinates'):
                self._match_rois_with_traces(self.pending_roi_coordinates)
                delattr(self, 'pending_roi_coordinates')
            
            # ... rest of existing code ...
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")

def main():
    if not CORE_LIBRARIES_AVAILABLE:
        print("Core libraries not available. Please fix NumPy/pandas installation.")
        return
        
    if not TKINTER_AVAILABLE:
        print("GUI not available. Using command-line version...")
        command_line_version()
        return
    
    root = tk.Tk()
    app = CalciumImagingGUI(root)
    root.mainloop()

def command_line_version():
    """Simple command-line version"""
    print("\n=== Calcium Imaging Video Generator (Command Line) ===")
    print("Please fix NumPy compatibility issue first:")
    print("Run: pip install 'numpy<2'")
    print("Then restart and try again.")

if __name__ == "__main__":
    main()