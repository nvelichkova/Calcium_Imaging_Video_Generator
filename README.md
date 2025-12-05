# Calcium Imaging Trace Video Generator

A Python-based GUI application for generating synchronized videos of calcium imaging data with fluorescence traces. This tool overlays time-series trace data onto video frames, making it easy to visualize and present calcium imaging results with ROI positions and activity plots.

![Application Screenshot](images/Ca_image_vid_gen.PNG)
## Features

### Video Generation
- **Synchronized Visualization**: Creates videos showing both the original imaging data and corresponding fluorescence traces
- **ROI Overlay**: Displays ROI positions with colored markers on the video frames
- **Real-time Trace Display**: Shows calcium traces that update in sync with the video frames
- **Multi-threaded Processing**: Video generation runs in a separate thread to keep the GUI responsive

### Data Loading
- **Flexible Trace Input**: Load calcium trace data from CSV or Excel files
- **Video Import**: Import original imaging videos (AVI, MP4 formats)
- **ImageJ ROI Support**: Load ROI definitions directly from ImageJ ROI files (.roi or .zip)
- **Manual ROI Coordinates**: Import custom ROI coordinates from CSV files

### ROI Management
- **Automatic Matching**: Intelligent matching of trace names with ROI coordinates
- **Selective Display**: Choose which ROIs to include in the output video
- **Default Positioning**: Automatic circular positioning for ROIs without coordinates
- **Centroid Calculation**: Extracts centroid positions from ImageJ polygon ROIs

### Customization Options
- **Adjustable Sampling Rate**: Configure the temporal resolution of your data (Hz)
- **Output FPS Control**: Set playback speed for the output video
- **Video Duration**: Specify custom duration or use full data length
- **Color Coding**: Each ROI gets a unique color for easy identification
- **Preview Mode**: Preview animation before generating the final video

### Data Compatibility
- **Anatomical Ordering**: Supports anatomically ordered column names (e.g., a1l, a1r, a2l)
- **Flexible Column Naming**: Works with "Mean(ROI_name)" format from ImageJ
- **Multiple File Formats**: CSV, Excel (.xlsx, .xls)
- **ImageJ Integration**: Direct ROI file import (.roi, .zip)

## Requirements

### Required Dependencies
```bash
pip install pandas numpy matplotlib pillow tkinter
```

### Optional Dependencies
```bash
# For video output (highly recommended)
pip install opencv-python

# For ImageJ ROI loading
pip install read-roi
# or alternatively
pip install roifile
```

## Installation

1. Ensure you have Python 3.7+ installed

2. Install required packages:
```bash
pip install pandas numpy matplotlib pillow opencv-python
```

3. Install optional ROI support:
```bash
pip install read-roi
```

4. Download `Calcium_Imaging_Trace_Video_Generator.py`

5. Run the application:
```bash
python Calcium_Imaging_Trace_Video_Generator.py
```

## Quick Start Guide

### 1. Load Your Data

**Option A: Load Trace Data First (Recommended)**
1. Click **"Load ROI Data (CSV/Excel)"**
2. Select your calcium trace data file
   - Rows should be time points/frames
   - Columns should be ROI names
3. Enter your **Sampling Rate** (e.g., 5.0 Hz means frame 5 = 1 second)

**Option B: Load Video First**
1. Click **"Load Original Video (AVI/MP4)"**
2. Select your imaging video
3. Video information will be displayed automatically

### 2. Load ROI Positions

**Option A: ImageJ ROIs**
1. Click **"Load ImageJ ROIs"**
2. Select your .roi or RoiSet.zip file
3. ROIs will be automatically matched with trace names

**Option B: Custom Coordinates**
1. Click **"Load ROI Coordinates (CSV)"**
2. CSV should have columns: ROI_name, x, y

**Option C: Use Defaults**
- If no ROI coordinates are provided, ROIs will be arranged in a circle

### 3. Configure Settings

**Data Settings:**
- **Sampling Rate (Hz)**: How many frames per second your data represents
  - Example: 5 Hz means 5 frames = 1 second of real time

**Video Settings:**
- **Output Video FPS**: Playback speed for the final video (typically 20-30)
- **Video Duration**: Leave blank for full duration, or specify seconds

### 4. Select ROIs
- Use checkboxes to select which ROIs to include in the video
- Selected ROIs will appear in the output with unique colors

### 5. Generate Video

**Preview First (Recommended):**
1. Click **"Generate Preview"** to see a few frames
2. Check that everything looks correct
3. Close the preview window when satisfied

**Generate Final Video:**
1. Click **"Generate Video"**
2. Choose output location and filename
3. Wait for video generation to complete
4. Progress updates will appear in the status section

## Input File Formats

### Trace Data (CSV/Excel)

The trace data file should be organized as:

```
Frame,a1l,a1r,a2l,a2r,a3l,a3r
1,100.5,98.3,105.2,102.1,95.8,97.5
2,103.2,101.5,108.1,104.3,98.2,100.1
3,106.8,104.2,110.5,107.2,101.3,102.8
...
```

Or with "Mean()" format from ImageJ:
```
Frame,Mean(ROI1),Mean(ROI2),Mean(ROI3)
1,100.5,98.3,105.2
2,103.2,101.5,108.1
...
```

**Requirements:**
- First column can be frame numbers or time (will be interpreted based on sampling rate)
- Subsequent columns are ROI fluorescence values
- Column headers are ROI names

### ROI Coordinates (CSV)

If manually providing coordinates:
```
ROI_name,x,y
a1l,150,200
a1r,250,200
a2l,150,300
a2r,250,300
```

**Requirements:**
- Three columns: ROI_name, x, y
- ROI_name must match column names in trace data
- Coordinates are in pixels

### ImageJ ROI Files

- Export ROIs from ImageJ as .roi files or RoiSet.zip
- Supports polygon, freehand, and rectangle ROIs
- Centroids are automatically calculated from ROI shapes

## Output Format

The generated video includes:

**Left Side: Original Video**
- Original imaging frames (if loaded)
- Or black background if no video provided
- Colored circles marking ROI positions
- ROI names labeled

**Right Side: Trace Plot**
- Time-series plot of calcium traces
- Each ROI shown in its unique color
- Vertical line indicating current time point
- Legend with ROI names
- X-axis: Time (seconds)
- Y-axis: Fluorescence intensity (ΔF/F or raw)

## Tips & Best Practices

### For Best Results
1. **Match sampling rates**: Ensure your trace sampling rate matches the video frame rate
2. **Check ROI matching**: Verify that ROI names in your trace data match those in coordinate files
3. **Preview first**: Always preview before generating the full video to catch issues early
4. **Use consistent naming**: Keep ROI names consistent between ImageJ, trace files, and coordinate files

### Anatomical Ordering
If your ROI names follow a pattern (e.g., a1l, a1r, a2l, a2r), the application will:
- Detect the pattern automatically
- Order traces anatomically in the plot
- Group left/right pairs together

### Performance Optimization
1. **Reduce video duration** for testing
2. **Lower output FPS** if file size is too large
3. **Select fewer ROIs** if video generation is slow
4. **Use video compression** in post-processing if needed

### File Size Management
- Output videos can be large (hundreds of MB)
- Consider generating shorter clips for presentations
- Use video editing software to compress if needed
- MP4 format is typically smaller than AVI

## Troubleshooting

### "NumPy compatibility issue"
```bash
pip install 'numpy<2.0'
```

### "No module named cv2"
```bash
pip install opencv-python
```
Without OpenCV, videos will be generated using matplotlib (slower but functional)

### "Could not load ImageJ ROIs"
```bash
pip install read-roi
```
Or alternatively:
```bash
pip install roifile
```

### ROI coordinates don't match video
- Check that coordinate system matches (origin at top-left)
- Verify that ROI coordinates are for the same image dimensions as your video
- Try loading ROIs directly from ImageJ to ensure correct extraction

### Traces not updating during video
- Verify sampling rate is correct
- Check that trace data length matches expected frames
- Ensure trace file has no missing values (NaN)

### Video generation is slow
- Reduce output FPS (20-25 is usually sufficient)
- Generate shorter duration for testing
- Close preview window before generating final video
- Ensure video thread is not already running

### Blank/black video frames
- If no original video loaded, frames will be black (this is normal)
- ROI markers and traces should still appear
- To see original imaging: load video file before generating

## Advanced Features

### Multi-threaded Generation
- Video generation runs in background thread
- GUI remains responsive during generation
- Can cancel generation by closing the application
- Progress updates appear in status window

### Smart ROI Matching
The application tries multiple strategies to match ROI names:
1. Exact name match
2. Remove "Mean()" wrapper and match
3. Partial/substring matching
4. Case-insensitive matching

### Flexible Time Handling
- Frame-based or time-based data supported
- Automatic conversion using sampling rate
- Handles both integer frame numbers and decimal time values

## Example Workflows

### Workflow 1: Complete Pipeline (with video)
1. Export trace data from ImageJ/Fiji as CSV
2. Export ROI set as .zip file from ROI Manager
3. Load trace CSV in application
4. Load original video file
5. Load ImageJ ROI .zip file
6. Set sampling rate (e.g., 5 Hz)
7. Preview to verify
8. Generate final video

### Workflow 2: Traces Only (no video)
1. Load trace CSV
2. Either load ROI coordinates or use default positioning
3. Set sampling rate
4. Select ROIs to display
5. Generate video (will have black background with ROI markers and traces)

### Workflow 3: Custom ROI Positions
1. Load trace data
2. Create CSV with custom ROI coordinates
3. Load ROI coordinate CSV
4. Configure settings
5. Generate video

## Known Limitations

- Large videos (>1000 frames) may take several minutes to generate
- Video dimensions are fixed based on input video or default size
- ROI markers are circles only (not full ROI outlines)
- One trace per ROI (no multi-channel support per ROI)

## Citation

If you use this tool in your research, please acknowledge it in your methods section.

## Author

Created by: bsmsa18b

## Support

For bugs, feature requests, or questions, please contact the author.

---

**Version**: 1.0 (August 2025)

**Last Updated**: December 2025

**License**: This software is provided as-is for research purposes.
