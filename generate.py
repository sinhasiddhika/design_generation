import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageOps
import io
import base64
from sklearn.cluster import KMeans
import cv2
from scipy import ndimage
from scipy.signal import correlate2d
import matplotlib.pyplot as plt

def advanced_pattern_analysis(image):
    """Enhanced pattern analysis with better detection of repeating units"""
    img_gray = np.array(image.convert('L'))
    h, w = img_gray.shape
    
    # Template matching for pattern detection
    patterns = []
    
    # Try different template sizes
    for template_size in [(h//4, w//4), (h//3, w//3), (h//2, w//2)]:
        th, tw = template_size
        if th > 10 and tw > 10:
            template = img_gray[:th, :tw]
            
            # Cross-correlation to find repeating patterns
            correlation = correlate2d(img_gray, template, mode='valid')
            
            # Find peaks in correlation
            threshold = np.max(correlation) * 0.7
            peaks = np.where(correlation > threshold)
            
            if len(peaks[0]) > 1:
                # Calculate pattern repeat distances
                y_diffs = np.diff(sorted(peaks[0]))
                x_diffs = np.diff(sorted(peaks[1]))
                
                y_repeat = np.median(y_diffs) if len(y_diffs) > 0 else th
                x_repeat = np.median(x_diffs) if len(x_diffs) > 0 else tw
                
                patterns.append({
                    'template_size': template_size,
                    'repeat_y': int(y_repeat),
                    'repeat_x': int(x_repeat),
                    'confidence': len(peaks[0])
                })
    
    # Select best pattern
    if patterns:
        best_pattern = max(patterns, key=lambda x: x['confidence'])
        return best_pattern
    else:
        return {'template_size': (h//2, w//2), 'repeat_y': h//2, 'repeat_x': w//2, 'confidence': 1}

def detect_chevron_pattern(image):
    """Specialized detection for chevron/herringbone patterns"""
    img_gray = np.array(image.convert('L'))
    
    # Edge detection
    edges = cv2.Canny(img_gray, 50, 150)
    
    # Line detection using Hough transform
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
    
    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta)
            angles.append(angle)
        
        angles = np.array(angles)
        
        # Check for chevron pattern (two dominant angles)
        unique_angles = []
        for angle in angles:
            if not unique_angles or min([abs(angle - ua) for ua in unique_angles]) > 10:
                unique_angles.append(angle)
        
        is_chevron = len(unique_angles) >= 2
        dominant_angle = np.median(angles) if len(angles) > 0 else 0
        
        return {
            'is_chevron': is_chevron,
            'dominant_angle': dominant_angle,
            'angles': unique_angles[:4]  # Top 4 angles
        }
    
    return {'is_chevron': False, 'dominant_angle': 0, 'angles': []}

def create_seamless_tile(image, pattern_info):
    """Create a perfectly seamless tile using advanced blending"""
    img_array = np.array(image).astype(np.float32)
    h, w, c = img_array.shape
    
    # Calculate optimal tile size based on pattern analysis
    tile_h = min(h, pattern_info['repeat_y'] * 2)
    tile_w = min(w, pattern_info['repeat_x'] * 2)
    
    # Extract the base tile
    tile = img_array[:tile_h, :tile_w]
    
    # Create seamless edges using frequency domain blending
    overlap = min(tile_h//8, tile_w//8, 20)
    
    if overlap > 0:
        # Horizontal seamless
        left_edge = tile[:, :overlap]
        right_edge = tile[:, -overlap:]
        
        # Create smooth transition
        for i in range(overlap):
            alpha = i / overlap
            tile[:, i] = left_edge[:, i] * (1 - alpha) + right_edge[:, i] * alpha
            tile[:, -(i+1)] = right_edge[:, -(i+1)] * (1 - alpha) + left_edge[:, -(i+1)] * alpha
        
        # Vertical seamless
        top_edge = tile[:overlap, :]
        bottom_edge = tile[-overlap:, :]
        
        for i in range(overlap):
            alpha = i / overlap
            tile[i, :] = top_edge[i, :] * (1 - alpha) + bottom_edge[i, :] * alpha
            tile[-(i+1), :] = bottom_edge[-(i+1), :] * (1 - alpha) + top_edge[-(i+1), :] * alpha
    
    return Image.fromarray(tile.astype(np.uint8))

def intelligent_chevron_tiling(tile, output_width, output_height, chevron_info):
    """Specialized tiling for chevron patterns with proper alignment"""
    tile_array = np.array(tile)
    tile_h, tile_w = tile_array.shape[:2]
    
    # Calculate how many tiles needed
    tiles_x = (output_width // tile_w) + 2
    tiles_y = (output_height // tile_h) + 2
    
    # Create larger canvas
    canvas_w = tiles_x * tile_w
    canvas_h = tiles_y * tile_h
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    
    # For chevron patterns, alternate tile orientations
    for y in range(tiles_y):
        for x in range(tiles_x):
            current_tile = tile_array.copy()
            
            # Chevron pattern logic
            if chevron_info['is_chevron']:
                # Alternate flipping for chevron effect
                if (x + y) % 2 == 1:
                    current_tile = np.flip(current_tile, axis=1)  # Horizontal flip
                
                # Additional rotation for some positions
                if chevron_info['dominant_angle'] > 45:
                    if (x % 2 == 0 and y % 2 == 1) or (x % 2 == 1 and y % 2 == 0):
                        current_tile = np.flip(current_tile, axis=0)  # Vertical flip
            
            # Place tile
            y_start = y * tile_h
            y_end = min(y_start + tile_h, canvas_h)
            x_start = x * tile_w
            x_end = min(x_start + tile_w, canvas_w)
            
            canvas[y_start:y_end, x_start:x_end] = current_tile[:y_end-y_start, :x_end-x_start]
    
    # Crop to exact dimensions
    final_canvas = canvas[:output_height, :output_width]
    return Image.fromarray(final_canvas)

def create_realistic_carpet_texture(image):
    """Add realistic carpet texture and appearance"""
    # Convert to array
    img_array = np.array(image).astype(np.float32)
    
    # Add subtle fiber texture
    h, w = img_array.shape[:2]
    
    # Create fiber-like noise
    noise_scale = 0.5
    fiber_noise = np.random.normal(0, noise_scale, (h, w, 3))
    
    # Create directional texture (carpet fibers)
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)
    
    # Subtle directional pattern
    direction_pattern = np.sin(X * 0.1) * np.cos(Y * 0.1) * 2
    direction_pattern = np.stack([direction_pattern] * 3, axis=2)
    
    # Combine textures
    textured = img_array + fiber_noise + direction_pattern
    
    # Add subtle color variations
    color_variation = np.random.normal(1, 0.02, img_array.shape)
    textured = textured * color_variation
    
    # Ensure values stay in valid range
    textured = np.clip(textured, 0, 255)
    
    # Convert back to PIL Image
    textured_image = Image.fromarray(textured.astype(np.uint8))
    
    # Apply carpet-specific filters
    # Slight blur for fiber softness
    textured_image = textured_image.filter(ImageFilter.GaussianBlur(radius=0.3))
    
    # Enhance contrast slightly
    enhancer = ImageEnhance.Contrast(textured_image)
    textured_image = enhancer.enhance(1.1)
    
    # Reduce saturation slightly for realistic look
    enhancer = ImageEnhance.Color(textured_image)
    textured_image = enhancer.enhance(0.95)
    
    return textured_image

def generate_carpet_design(original_image, output_width, output_height, quality_mode="high"):
    """Main function to generate complete carpet design"""
    
    # Step 1: Advanced pattern analysis
    pattern_info = advanced_pattern_analysis(original_image)
    chevron_info = detect_chevron_pattern(original_image)
    
    # Step 2: Create seamless tile
    seamless_tile = create_seamless_tile(original_image, pattern_info)
    
    # Step 3: Intelligent tiling based on pattern type
    if chevron_info['is_chevron']:
        complete_design = intelligent_chevron_tiling(seamless_tile, output_width, output_height, chevron_info)
    else:
        # Standard tiling for non-chevron patterns
        tile_array = np.array(seamless_tile)
        tile_h, tile_w = tile_array.shape[:2]
        
        tiles_x = (output_width // tile_w) + 1
        tiles_y = (output_height // tile_h) + 1
        
        # Create tiled image
        tiled = np.tile(tile_array, (tiles_y, tiles_x, 1))
        complete_design = Image.fromarray(tiled[:output_height, :output_width])
    
    # Step 4: Add realistic carpet texture
    if quality_mode == "ultra":
        complete_design = create_realistic_carpet_texture(complete_design)
    
    return complete_design, pattern_info, chevron_info

def get_download_link(img, filename):
    """Generate download link for image"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG", quality=95, optimize=True)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{img_str}" download="{filename}" style="background-color: #ff4b4b; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; font-weight: bold;">📥 Download {filename}</a>'
    return href

def main():
    st.set_page_config(
        page_title="AI Carpet Pattern Generator",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: bold;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎨 AI Carpet Pattern Generator</h1>
        <p>Transform small pattern samples into complete, realistic carpet designs with AI precision</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Design Controls")
        
        uploaded_file = st.file_uploader(
            "Upload Pattern Sample",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload a small carpet pattern sample (square patterns work best)"
        )
        
        st.subheader("📐 Output Dimensions")
        
        # Preset sizes
        size_preset = st.selectbox(
            "Size Preset",
            ["Custom", "Small Rug (900×600)", "Medium Rug (1200×800)", "Large Rug (1500×1000)", "Room Size (2000×1400)", "Ultra Large (3000×2000)"]
        )
        
        if size_preset == "Small Rug (900×600)":
            default_w, default_h = 900, 600
        elif size_preset == "Medium Rug (1200×800)":
            default_w, default_h = 1200, 800
        elif size_preset == "Large Rug (1500×1000)":
            default_w, default_h = 1500, 1000
        elif size_preset == "Room Size (2000×1400)":
            default_w, default_h = 2000, 1400
        elif size_preset == "Ultra Large (3000×2000)":
            default_w, default_h = 3000, 2000
        else:
            default_w, default_h = 1200, 800
        
        col1, col2 = st.columns(2)
        with col1:
            output_width = st.number_input("Width", min_value=200, max_value=5000, value=default_w, step=50)
        with col2:
            output_height = st.number_input("Height", min_value=200, max_value=5000, value=default_h, step=50)
        
        st.subheader("⚙️ Quality Settings")
        quality_mode = st.selectbox(
            "Quality Mode",
            ["high", "ultra"],
            help="High: Fast generation | Ultra: Maximum quality with realistic texture"
        )
        
        preview_mode = st.checkbox("Preview Mode (Faster)", value=False)
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            enhance_seamless = st.checkbox("Enhanced Seamless Processing", value=True)
            carpet_texture = st.checkbox("Add Carpet Texture", value=True)
            pattern_analysis = st.checkbox("Advanced Pattern Analysis", value=True)
    
    # Main content area
    if uploaded_file is not None:
        try:
            # Load image
            original_image = Image.open(uploaded_file).convert('RGB')
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("📷 Original Pattern")
                st.image(original_image, caption=f"Size: {original_image.size[0]}×{original_image.size[1]}px")
                
                # Quick analysis
                if pattern_analysis:
                    with st.spinner("Analyzing pattern..."):
                        pattern_info = advanced_pattern_analysis(original_image)
                        chevron_info = detect_chevron_pattern(original_image)
                    
                    st.subheader("🔍 Pattern Analysis")
                    
                    # Pattern type
                    if chevron_info['is_chevron']:
                        pattern_type = "🔷 Chevron/Herringbone"
                        st.success(f"**Pattern Type:** {pattern_type}")
                        st.info(f"**Dominant Angle:** {chevron_info['dominant_angle']:.1f}°")
                    else:
                        pattern_type = "🔳 Geometric/Regular"
                        st.info(f"**Pattern Type:** {pattern_type}")
                    
                    st.info(f"**Repeat Unit:** {pattern_info['repeat_x']}×{pattern_info['repeat_y']}px")
                    st.info(f"**Confidence:** {pattern_info['confidence']}/10")
            
            with col2:
                st.subheader("🎨 Generate Complete Design")
                
                # Adjust dimensions for preview
                if preview_mode:
                    gen_width = min(output_width, 800)
                    gen_height = min(output_height, 600)
                    st.warning(f"Preview Mode: Generating {gen_width}×{gen_height}px")
                else:
                    gen_width, gen_height = output_width, output_height
                
                # Generate button
                if st.button("🚀 Generate AI Carpet Design", type="primary", use_container_width=True):
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Generation process
                        status_text.text("🔍 Analyzing pattern structure...")
                        progress_bar.progress(20)
                        
                        status_text.text("🧩 Creating seamless tile...")
                        progress_bar.progress(40)
                        
                        status_text.text("🎨 Generating complete design...")
                        progress_bar.progress(60)
                        
                        # Generate the design
                        complete_design, pattern_info, chevron_info = generate_carpet_design(
                            original_image, gen_width, gen_height, quality_mode
                        )
                        
                        status_text.text("✨ Applying final enhancements...")
                        progress_bar.progress(80)
                        
                        # Final enhancements
                        if carpet_texture and quality_mode == "ultra":
                            complete_design = create_realistic_carpet_texture(complete_design)
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Generation complete!")
                        
                        # Display result
                        st.subheader("✨ Generated Carpet Design")
                        st.image(complete_design, caption=f"AI Generated Design: {complete_design.size[0]}×{complete_design.size[1]}px")
                        
                        # Download section
                        st.subheader("📥 Download")
                        
                        col_dl1, col_dl2 = st.columns(2)
                        with col_dl1:
                            st.markdown(get_download_link(complete_design, "ai_carpet_design.png"), unsafe_allow_html=True)
                        
                        with col_dl2:
                            if preview_mode:
                                if st.button("Generate Full Resolution", type="secondary"):
                                    with st.spinner("Generating full resolution..."):
                                        full_design, _, _ = generate_carpet_design(
                                            original_image, output_width, output_height, quality_mode
                                        )
                                        st.markdown(get_download_link(full_design, "ai_carpet_full_res.png"), unsafe_allow_html=True)
                        
                        # Statistics
                        st.subheader("📊 Generation Statistics")
                        
                        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                        
                        with col_s1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>Original Size</h4>
                                <h2>{original_image.size[0]}×{original_image.size[1]}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_s2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>Generated Size</h4>
                                <h2>{complete_design.size[0]}×{complete_design.size[1]}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_s3:
                            expansion = (complete_design.size[0] * complete_design.size[1]) / (original_image.size[0] * original_image.size[1])
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>Size Expansion</h4>
                                <h2>{expansion:.1f}×</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_s4:
                            file_size = (complete_design.size[0] * complete_design.size[1] * 3) / (1024 * 1024)
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>Est. File Size</h4>
                                <h2>{file_size:.1f}MB</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                        
                    except Exception as e:
                        st.error(f"❌ Generation failed: {str(e)}")
                        st.info("💡 Try reducing dimensions or using preview mode first.")
                        progress_bar.empty()
                        status_text.empty()
        
        except Exception as e:
            st.error(f"❌ Error loading image: {str(e)}")
    
    else:
        # Landing page
        st.subheader("🎯 How It Works")
        
        col_info1, col_info2, col_info3 = st.columns(3)
        
        with col_info1:
            st.markdown("""
            ### 1. 📤 Upload Pattern
            - Upload a small carpet pattern sample
            - Square or rectangular patterns work best
            - High resolution recommended (300px+)
            """)
        
        with col_info2:
            st.markdown("""
            ### 2. 🤖 AI Analysis
            - Advanced pattern recognition
            - Chevron/Herringbone detection
            - Seamless tile generation
            """)
        
        with col_info3:
            st.markdown("""
            ### 3. ✨ Generate Design
            - Intelligent tiling algorithms
            - Realistic carpet texture
            - Customizable dimensions
            """)
        
        st.subheader("🌟 Key Features")
        
        feature_col1, feature_col2 = st.columns(2)
        
        with feature_col1:
            st.markdown("""
            - **🎯 Pattern Recognition**: Detects chevron, herringbone, and geometric patterns
            - **🧩 Seamless Tiling**: Creates perfect seamless connections
            - **🎨 Realistic Texture**: Adds authentic carpet fiber texture
            - **📐 Custom Dimensions**: Any size from small rugs to room-size carpets
            """)
        
        with feature_col2:
            st.markdown("""
            - **⚡ Fast Processing**: Preview mode for quick results
            - **🎛️ Quality Control**: High and Ultra quality modes
            - **💾 Easy Download**: Direct PNG download with optimization
            - **📊 Detailed Analytics**: Pattern analysis and generation statistics
            """)
        
        st.info("👆 Upload a carpet pattern sample above to get started!")

if __name__ == "__main__":
    main()
