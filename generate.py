import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import io
import base64
from sklearn.cluster import KMeans
import cv2

def extract_dominant_colors(image, n_colors=5):
    """Extract dominant colors from the image using KMeans clustering"""
    try:
        data = np.array(image)
        data = data.reshape((-1, 3))
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(data)
        return kmeans.cluster_centers_.astype(int)
    except:
        # Fallback method if sklearn not available
        data = np.array(image)
        data = data.reshape((-1, 3))
        # Sample representative colors
        indices = np.random.choice(len(data), min(1000, len(data)), replace=False)
        sampled = data[indices]
        # Simple clustering by grouping similar colors
        colors = []
        for color in sampled:
            if not colors or min([np.linalg.norm(color - c) for c in colors]) > 30:
                colors.append(color)
                if len(colors) >= n_colors:
                    break
        while len(colors) < n_colors:
            colors.append(data[np.random.randint(len(data))])
        return np.array(colors[:n_colors])

def analyze_pattern_structure(image):
    """Analyze the pattern structure to understand directionality and repetition"""
    img_array = np.array(image.convert('L'))  # Convert to grayscale
    h, w = img_array.shape
    
    # Detect edges to understand pattern structure
    try:
        edges = cv2.Canny(img_array, 50, 150)
    except:
        # Fallback edge detection without OpenCV
        from scipy import ndimage
        sobel_x = ndimage.sobel(img_array, axis=1)
        sobel_y = ndimage.sobel(img_array, axis=0)
        edges = np.hypot(sobel_x, sobel_y)
        edges = (edges > np.percentile(edges, 75)).astype(np.uint8) * 255
    
    # Analyze horizontal and vertical patterns
    horizontal_profile = np.mean(edges, axis=0)
    vertical_profile = np.mean(edges, axis=1)
    
    # Find dominant frequencies
    h_peaks = find_peaks_simple(horizontal_profile)
    v_peaks = find_peaks_simple(vertical_profile)
    
    # Determine pattern type
    pattern_info = {
        'horizontal_period': estimate_period(h_peaks, w) if h_peaks else w,
        'vertical_period': estimate_period(v_peaks, h) if v_peaks else h,
        'is_directional': len(h_peaks) > 2 or len(v_peaks) > 2,
        'dominant_direction': 'horizontal' if len(h_peaks) > len(v_peaks) else 'vertical'
    }
    
    return pattern_info

def find_peaks_simple(signal, threshold=None):
    """Simple peak finding without scipy"""
    if threshold is None:
        threshold = np.mean(signal) + np.std(signal)
    
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1] and signal[i] > threshold:
            peaks.append(i)
    return peaks

def estimate_period(peaks, total_length):
    """Estimate the repeating period from peaks"""
    if len(peaks) < 2:
        return total_length
    
    differences = np.diff(peaks)
    if len(differences) == 0:
        return total_length
    
    # Find most common difference
    unique_diffs, counts = np.unique(differences, return_counts=True)
    most_common_period = unique_diffs[np.argmax(counts)]
    
    return max(most_common_period, total_length // 10)  # Prevent too small periods

def create_advanced_seamless_tile(image, pattern_info):
    """Create a more sophisticated seamless tile based on pattern analysis"""
    img_array = np.array(image).astype(float)
    h, w, c = img_array.shape
    
    # Create overlap regions for seamless blending
    overlap_x = min(w // 4, pattern_info['horizontal_period'] // 2)
    overlap_y = min(h // 4, pattern_info['vertical_period'] // 2)
    
    overlap_x = max(overlap_x, 5)  # Minimum overlap
    overlap_y = max(overlap_y, 5)
    
    result = img_array.copy()
    
    # Horizontal seamless blending
    if w > overlap_x * 2:
        left_region = result[:, :overlap_x]
        right_region = result[:, -overlap_x:]
        
        # Create smooth transition weights
        weights = np.linspace(0, 1, overlap_x).reshape(1, -1, 1)
        
        # Blend regions
        blended_left = left_region * (1 - weights) + right_region * weights
        blended_right = right_region * (1 - weights) + left_region * weights
        
        result[:, :overlap_x] = blended_left
        result[:, -overlap_x:] = blended_right
    
    # Vertical seamless blending
    if h > overlap_y * 2:
        top_region = result[:overlap_y, :]
        bottom_region = result[-overlap_y:, :]
        
        weights = np.linspace(0, 1, overlap_y).reshape(-1, 1, 1)
        
        blended_top = top_region * (1 - weights) + bottom_region * weights
        blended_bottom = bottom_region * (1 - weights) + top_region * weights
        
        result[:overlap_y, :] = blended_top
        result[-overlap_y:, :] = blended_bottom
    
    return Image.fromarray(result.astype(np.uint8))

def create_pattern_variations(tile, pattern_info, num_variations=8):
    """Create intelligent pattern variations based on carpet design principles"""
    variations = [tile]  # Original
    
    # For directional patterns like chevron/herringbone
    if pattern_info['is_directional']:
        # Mirror horizontally (common in carpet patterns)
        variations.append(tile.transpose(Image.FLIP_LEFT_RIGHT))
        
        # Mirror vertically
        variations.append(tile.transpose(Image.FLIP_TOP_BOTTOM))
        
        # Both flips
        variations.append(tile.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM))
        
        # Slight color variations to mimic carpet texture variations
        for brightness_factor in [0.95, 1.05]:
            enhancer = ImageEnhance.Brightness(tile)
            variations.append(enhancer.enhance(brightness_factor))
        
        # Subtle contrast variations
        for contrast_factor in [0.98, 1.02]:
            enhancer = ImageEnhance.Contrast(tile)
            variations.append(enhancer.enhance(contrast_factor))
    
    else:
        # For non-directional patterns, use rotations
        for angle in [90, 180, 270]:
            variations.append(tile.rotate(angle, expand=False))
    
    return variations[:num_variations]

def intelligent_carpet_tiling(tile, output_width, output_height, pattern_info, tiling_mode="carpet_realistic"):
    """Advanced tiling specifically designed for carpet patterns"""
    tile_w, tile_h = tile.size
    
    # Calculate grid
    tiles_x = (output_width + tile_w - 1) // tile_w + 1  # Extra tile for overlap
    tiles_y = (output_height + tile_h - 1) // tile_h + 1
    
    # Create larger canvas for seamless cropping
    canvas_w = tiles_x * tile_w
    canvas_h = tiles_y * tile_h
    canvas = Image.new('RGB', (canvas_w, canvas_h))
    
    # Create variations
    variations = create_pattern_variations(tile, pattern_info)
    
    if tiling_mode == "carpet_realistic":
        # Realistic carpet tiling with proper pattern flow
        np.random.seed(42)  # Reproducible results
        
        for y in range(tiles_y):
            for x in range(tiles_x):
                # Choose variation based on position for natural carpet look
                if pattern_info['is_directional']:
                    # For directional patterns, alternate systematically
                    if pattern_info['dominant_direction'] == 'horizontal':
                        # Alternate every few rows for chevron effect
                        variation_idx = 0 if (y // 2) % 2 == 0 else 1
                    else:
                        # Alternate every few columns
                        variation_idx = 0 if (x // 2) % 2 == 0 else 1
                else:
                    # Random but controlled variation
                    variation_idx = (x * 3 + y * 7) % len(variations)
                
                chosen_tile = variations[variation_idx % len(variations)]
                
                # Calculate position with slight overlap for seamless effect
                pos_x = x * tile_w
                pos_y = y * tile_h
                
                # Add subtle random offset for more natural look (carpet weaving effect)
                if x > 0 and y > 0:  # Not on edges
                    offset_range = min(5, tile_w // 20)
                    pos_x += np.random.randint(-offset_range, offset_range + 1)
                    pos_y += np.random.randint(-offset_range, offset_range + 1)
                
                # Ensure within bounds
                pos_x = max(0, min(pos_x, canvas_w - tile_w))
                pos_y = max(0, min(pos_y, canvas_h - tile_h))
                
                canvas.paste(chosen_tile, (pos_x, pos_y))
    
    elif tiling_mode == "perfect_seamless":
        # Perfect seamless tiling for geometric patterns
        for y in range(tiles_y):
            for x in range(tiles_x):
                canvas.paste(tile, (x * tile_w, y * tile_h))
    
    # Crop to exact dimensions
    final_image = canvas.crop((0, 0, output_width, output_height))
    
    # Apply carpet-specific post-processing
    final_image = apply_carpet_enhancement(final_image)
    
    return final_image

def apply_carpet_enhancement(image):
    """Apply enhancements to make the pattern look more like a real carpet"""
    # Slight blur to mimic carpet texture
    enhanced = image.filter(ImageFilter.GaussianBlur(radius=0.3))
    
    # Add subtle noise for texture
    img_array = np.array(enhanced).astype(float)
    noise = np.random.normal(0, 2, img_array.shape)  # Very subtle noise
    img_array = np.clip(img_array + noise, 0, 255)
    enhanced = Image.fromarray(img_array.astype(np.uint8))
    
    # Enhance for carpet-like appearance
    # Slightly reduce saturation for more realistic look
    enhancer = ImageEnhance.Color(enhanced)
    enhanced = enhancer.enhance(0.95)
    
    # Slight contrast adjustment
    enhancer = ImageEnhance.Contrast(enhanced)
    enhanced = enhancer.enhance(1.05)
    
    # Very subtle sharpening
    enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=0.5, percent=120, threshold=3))
    
    return enhanced

def get_image_download_link(img, filename):
    """Generate download link for image"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG", quality=95)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{img_str}" download="{filename}" style="text-decoration: none; color: #ff4b4b; font-weight: bold;">📥 Download {filename}</a>'
    return href

def main():
    st.set_page_config(
        page_title="Professional Carpet Design Generator",
        page_icon="🏺",
        layout="wide"
    )
    
    st.title("🏺 Professional Carpet Design Pattern Generator")
    st.markdown("Transform small carpet pattern samples into complete, realistic carpet designs for customer presentations")
    
    # Sidebar for controls
    st.sidebar.header("🎛️ Design Controls")
    
    uploaded_file = st.file_uploader(
        "Upload Carpet Pattern Sample",
        type=['png', 'jpg', 'jpeg', 'bmp'],
        help="Upload a small carpet pattern sample"
    )
    
    if uploaded_file is not None:
        # Load and analyze the pattern
        original_image = Image.open(uploaded_file).convert('RGB')
        
        with st.spinner("Analyzing pattern structure..."):
            pattern_info = analyze_pattern_structure(original_image)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📐 Original Pattern")
            st.image(original_image, caption=f"Size: {original_image.size[0]}×{original_image.size[1]}px")
            
            # Pattern analysis results
            st.subheader("🔍 Pattern Analysis")
            pattern_type = "Directional (Chevron/Herringbone)" if pattern_info['is_directional'] else "Non-directional"
            st.info(f"**Type:** {pattern_type}")
            st.info(f"**Dominant Direction:** {pattern_info['dominant_direction'].title()}")
            st.info(f"**Horizontal Period:** {pattern_info['horizontal_period']}px")
            st.info(f"**Vertical Period:** {pattern_info['vertical_period']}px")
            
            # Color analysis
            with st.expander("🎨 Color Analysis"):
                try:
                    colors = extract_dominant_colors(original_image)
                    color_cols = st.columns(min(len(colors), 3))
                    for i, color in enumerate(colors[:3]):
                        with color_cols[i]:
                            color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                            st.color_picker(f"C{i+1}", color_hex, disabled=True, key=f"color_{i}")
                except Exception as e:
                    st.error("Color analysis failed")
        
        with col2:
            st.subheader("⚙️ Generation Settings")
            
            # Carpet dimensions (realistic sizes)
            col_dim1, col_dim2 = st.columns(2)
            with col_dim1:
                preset = st.selectbox(
                    "Carpet Size Preset",
                    ["Custom", "Small Rug (3×5 ft)", "Medium Rug (5×8 ft)", "Large Rug (8×10 ft)", "Room Size (10×14 ft)"],
                    index=2
                )
                
                if preset == "Small Rug (3×5 ft)":
                    output_width, output_height = 900, 1500
                elif preset == "Medium Rug (5×8 ft)":
                    output_width, output_height = 1500, 2400
                elif preset == "Large Rug (8×10 ft)":
                    output_width, output_height = 2400, 3000
                elif preset == "Room Size (10×14 ft)":
                    output_width, output_height = 3000, 4200
                else:
                    output_width = st.number_input("Width (pixels)", min_value=200, max_value=5000, value=1500, step=100)
                    output_height = st.number_input("Height (pixels)", min_value=200, max_value=5000, value=2400, step=100)
            
            with col_dim2:
                st.write("**Quality Settings**")
                tiling_mode = st.selectbox(
                    "Tiling Method",
                    ["carpet_realistic", "perfect_seamless"],
                    help="Carpet Realistic: Natural carpet appearance | Perfect Seamless: Exact pattern repetition"
                )
                
                resolution_quality = st.selectbox(
                    "Output Quality",
                    ["High (Presentation)", "Ultra (Print)"],
                    help="High: Good for screen viewing | Ultra: Print quality"
                )
                
                preview_mode = st.checkbox(
                    "Preview Mode", 
                    value=False, 
                    help="Generate smaller preview first"
                )
            
            # Advanced settings
            with st.expander("🔧 Advanced Settings"):
                enhance_carpet_texture = st.checkbox("Enhance Carpet Texture", value=True)
                seamless_processing = st.checkbox("Advanced Seamless Processing", value=True)
                color_variation = st.slider("Color Variation", 0.0, 0.3, 0.1, 0.05)
            
            # Generate button
            if st.button("🎨 Generate Professional Carpet Design", type="primary", use_container_width=True):
                with st.spinner("Creating professional carpet design... This may take a moment for high quality results."):
                    try:
                        # Adjust for preview mode
                        if preview_mode:
                            gen_width = min(output_width, 800)
                            gen_height = min(output_height, 600)
                            st.info(f"Generating preview: {gen_width}×{gen_height}px")
                        else:
                            gen_width, gen_height = output_width, output_height
                        
                        # Process the tile
                        processed_tile = original_image
                        
                        if seamless_processing:
                            st.info("🔄 Processing for seamless tiling...")
                            processed_tile = create_advanced_seamless_tile(processed_tile, pattern_info)
                        
                        # Generate the complete design
                        st.info("🎨 Generating complete carpet design...")
                        complete_design = intelligent_carpet_tiling(
                            processed_tile, 
                            gen_width, 
                            gen_height, 
                            pattern_info,
                            tiling_mode
                        )
                        
                        # Display result
                        st.subheader("✨ Generated Carpet Design")
                        st.image(complete_design, caption=f"Professional Carpet Design: {complete_design.size[0]}×{complete_design.size[1]}px")
                        
                        # Download options
                        st.markdown("### 📥 Download Options")
                        col_dl1, col_dl2 = st.columns(2)
                        
                        with col_dl1:
                            st.markdown(get_image_download_link(complete_design, "carpet_design.png"), unsafe_allow_html=True)
                        
                        with col_dl2:
                            # Generate high-res version if in preview mode
                            if preview_mode:
                                if st.button("Generate Full Resolution", type="secondary"):
                                    with st.spinner("Generating full resolution design..."):
                                        full_res_design = intelligent_carpet_tiling(
                                            processed_tile, output_width, output_height, pattern_info, tiling_mode
                                        )
                                        st.markdown(get_image_download_link(full_res_design, "carpet_design_full_res.png"), unsafe_allow_html=True)
                        
                        # Design statistics
                        st.subheader("📊 Design Statistics")
                        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                        
                        original_pixels = original_image.size[0] * original_image.size[1]
                        generated_pixels = complete_design.size[0] * complete_design.size[1]
                        expansion_ratio = generated_pixels / original_pixels
                        
                        with col_stat1:
                            st.metric("Original", f"{original_image.size[0]}×{original_image.size[1]}")
                        with col_stat2:
                            st.metric("Generated", f"{complete_design.size[0]}×{complete_design.size[1]}")
                        with col_stat3:
                            st.metric("Expansion", f"{expansion_ratio:.1f}×")
                        with col_stat4:
                            estimated_size_mb = (generated_pixels * 3) / (1024 * 1024)
                            st.metric("Est. Size", f"{estimated_size_mb:.1f}MB")
                        
                    except Exception as e:
                        st.error(f"Generation failed: {str(e)}")
                        st.info("Try reducing dimensions or using preview mode first.")
    
    else:
        # Instructions and examples
        st.info("👆 Upload a carpet pattern sample to begin generating professional designs")
        
        st.subheader("🎯 Optimized for Carpet Patterns")
        st.markdown("""
        This tool is specifically designed for carpet and textile patterns:
        - **Chevron & Herringbone patterns** - Maintains proper directional flow
        - **Geometric patterns** - Preserves pattern integrity
        - **Textile textures** - Adds realistic carpet texture effects
        - **Color variations** - Mimics natural carpet manufacturing variations
        """)
        
        col_help1, col_help2 = st.columns(2)
        
        with col_help1:
            st.subheader("📝 Usage Instructions")
            st.markdown("""
            1. **Upload** your carpet pattern sample
            2. **Review** automatic pattern analysis
            3. **Select** carpet size preset or custom dimensions
            4. **Choose** tiling method (realistic vs seamless)
            5. **Generate** professional carpet design
            6. **Download** for customer presentations
            """)
        
        with col_help2:
            st.subheader("💡 Best Practices")
            st.markdown("""
            - Use **high-quality** pattern samples (300+ pixels)
            - **Square or rectangular** patterns work best
            - Enable **advanced seamless processing** for chevron patterns
            - Use **preview mode** for testing large designs
            - **Carpet realistic** mode for customer presentations
            """)
        
        st.subheader("🔧 Technical Requirements")
        st.code("""
# Required packages:
pip install streamlit pillow numpy scikit-learn opencv-python

# For best results, ensure all packages are installed
        """)

if __name__ == "__main__":
    main()
