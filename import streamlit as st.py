import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2
from sklearn.cluster import KMeans
from scipy import ndimage
import io
import base64

def extract_dominant_colors(image, n_colors=5):
    """Extract dominant colors from the image"""
    data = np.array(image)
    data = data.reshape((-1, 3))
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(data)
    return kmeans.cluster_centers_.astype(int)

def create_seamless_tile(image):
    """Create a seamless tile from the input image"""
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    
    # Apply edge blending to make it more tileable
    h, w = img_array.shape[:2]
    
    # Create fade masks for edges
    fade_width = min(w, h) // 8
    
    # Create gradients for seamless tiling
    left_fade = np.linspace(0, 1, fade_width).reshape(1, -1, 1)
    right_fade = np.linspace(1, 0, fade_width).reshape(1, -1, 1)
    top_fade = np.linspace(0, 1, fade_width).reshape(-1, 1, 1)
    bottom_fade = np.linspace(1, 0, fade_width).reshape(-1, 1, 1)
    
    # Apply fading to edges
    result = img_array.copy().astype(float)
    
    # Blend left edge with right edge
    if w > fade_width * 2:
        left_part = result[:, :fade_width]
        right_part = result[:, -fade_width:]
        blended = left_part * (1 - left_fade) + right_part * left_fade
        result[:, :fade_width] = blended
        result[:, -fade_width:] = right_part * (1 - right_fade) + left_part * right_fade
    
    # Blend top edge with bottom edge
    if h > fade_width * 2:
        top_part = result[:fade_width, :]
        bottom_part = result[-fade_width:, :]
        blended = top_part * (1 - top_fade) + bottom_part * top_fade
        result[:fade_width, :] = blended
        result[-fade_width:, :] = bottom_part * (1 - bottom_fade) + top_part * bottom_fade
    
    return Image.fromarray(result.astype(np.uint8))

def apply_pattern_variations(tile, variation_strength=0.3):
    """Apply subtle variations to create more natural patterns"""
    variations = []
    
    # Original
    variations.append(tile)
    
    # Slightly rotated versions
    for angle in [90, 180, 270]:
        rotated = tile.rotate(angle, expand=False)
        variations.append(rotated)
    
    # Flipped versions
    variations.append(tile.transpose(Image.FLIP_LEFT_RIGHT))
    variations.append(tile.transpose(Image.FLIP_TOP_BOTTOM))
    
    # Color adjusted versions
    enhancer = ImageEnhance.Brightness(tile)
    variations.append(enhancer.enhance(1 + variation_strength * 0.2))
    variations.append(enhancer.enhance(1 - variation_strength * 0.2))
    
    enhancer = ImageEnhance.Contrast(tile)
    variations.append(enhancer.enhance(1 + variation_strength * 0.3))
    
    return variations

def intelligent_tiling(tile, output_width, output_height, pattern_type="smart"):
    """Create intelligent tiling patterns"""
    tile_w, tile_h = tile.size
    
    # Calculate number of tiles needed
    tiles_x = (output_width + tile_w - 1) // tile_w
    tiles_y = (output_height + tile_h - 1) // tile_h
    
    # Create canvas
    canvas = Image.new('RGB', (tiles_x * tile_w, tiles_y * tile_h))
    
    if pattern_type == "simple":
        # Simple repetitive tiling
        for y in range(tiles_y):
            for x in range(tiles_x):
                canvas.paste(tile, (x * tile_w, y * tile_h))
    
    elif pattern_type == "smart":
        # Smart tiling with variations
        variations = apply_pattern_variations(tile)
        
        for y in range(tiles_y):
            for x in range(tiles_x):
                # Choose variation based on position for more natural look
                variation_idx = (x + y * 2) % len(variations)
                chosen_tile = variations[variation_idx]
                canvas.paste(chosen_tile, (x * tile_w, y * tile_h))
    
    elif pattern_type == "organic":
        # More organic, less predictable pattern
        variations = apply_pattern_variations(tile, variation_strength=0.5)
        np.random.seed(42)  # For reproducible results
        
        for y in range(tiles_y):
            for x in range(tiles_x):
                # Random but seeded selection
                variation_idx = np.random.randint(0, len(variations))
                chosen_tile = variations[variation_idx]
                
                # Add slight random offset for more organic feel
                offset_x = np.random.randint(-tile_w//20, tile_w//20)
                offset_y = np.random.randint(-tile_h//20, tile_h//20)
                
                paste_x = max(0, min(x * tile_w + offset_x, canvas.width - tile_w))
                paste_y = max(0, min(y * tile_h + offset_y, canvas.height - tile_h))
                
                canvas.paste(chosen_tile, (paste_x, paste_y))
    
    # Crop to exact dimensions
    canvas = canvas.crop((0, 0, output_width, output_height))
    return canvas

def enhance_pattern_quality(image):
    """Apply post-processing to enhance pattern quality"""
    # Slight blur to smooth any harsh edges
    enhanced = image.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    # Enhance contrast slightly
    enhancer = ImageEnhance.Contrast(enhanced)
    enhanced = enhancer.enhance(1.1)
    
    # Enhance color saturation
    enhancer = ImageEnhance.Color(enhanced)
    enhanced = enhancer.enhance(1.1)
    
    return enhanced

def get_image_download_link(img, filename):
    """Generate download link for image"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{img_str}" download="{filename}">Download {filename}</a>'
    return href

def main():
    st.set_page_config(
        page_title="AI Design Pattern Completion",
        page_icon="🎨",
        layout="wide"
    )
    
    st.title("🎨 AI Design Pattern Completion Tool")
    st.markdown("Upload a small design pattern and generate complete designs with customizable dimensions!")
    
    # Sidebar for controls
    st.sidebar.header("Configuration")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your design pattern",
        type=['png', 'jpg', 'jpeg', 'bmp'],
        help="Upload a small design or pattern piece"
    )
    
    if uploaded_file is not None:
        # Load and display original image
        original_image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Original Pattern")
            st.image(original_image, caption=f"Size: {original_image.size[0]}x{original_image.size[1]}")
            
            # Display dominant colors
            colors = extract_dominant_colors(original_image)
            st.subheader("Dominant Colors")
            color_cols = st.columns(len(colors))
            for i, color in enumerate(colors):
                with color_cols[i]:
                    color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                    st.color_picker(f"Color {i+1}", color_hex, disabled=True)
        
        with col2:
            # Configuration options
            st.subheader("Output Configuration")
            
            col_config1, col_config2 = st.columns(2)
            with col_config1:
                output_width = st.number_input("Output Width", min_value=100, max_value=4000, value=1200, step=50)
                pattern_type = st.selectbox(
                    "Pattern Type",
                    ["smart", "simple", "organic"],
                    help="Smart: Varied repetition, Simple: Direct tiling, Organic: Natural variations"
                )
            
            with col_config2:
                output_height = st.number_input("Output Height", min_value=100, max_value=4000, value=800, step=50)
                enhance_quality = st.checkbox("Enhance Quality", value=True, help="Apply post-processing for better quality")
            
            # Advanced options
            with st.expander("Advanced Options"):
                make_seamless = st.checkbox("Make Seamless", value=True, help="Process pattern for seamless tiling")
                preview_mode = st.checkbox("Preview Mode", value=False, help="Generate smaller preview for faster processing")
            
            # Generate button
            if st.button("🎨 Generate Complete Design", type="primary"):
                with st.spinner("Processing your design..."):
                    try:
                        # Process the image
                        processed_tile = original_image
                        
                        if make_seamless:
                            st.info("Creating seamless tile...")
                            processed_tile = create_seamless_tile(processed_tile)
                        
                        # Adjust dimensions for preview mode
                        if preview_mode:
                            preview_width = min(output_width, 600)
                            preview_height = min(output_height, 400)
                            st.info(f"Generating preview: {preview_width}x{preview_height}")
                            final_width, final_height = preview_width, preview_height
                        else:
                            final_width, final_height = output_width, output_height
                        
                        # Generate the complete design
                        st.info("Generating complete design...")
                        complete_design = intelligent_tiling(
                            processed_tile, 
                            final_width, 
                            final_height, 
                            pattern_type
                        )
                        
                        # Enhance quality if requested
                        if enhance_quality:
                            st.info("Enhancing quality...")
                            complete_design = enhance_pattern_quality(complete_design)
                        
                        # Display result
                        st.subheader("Generated Complete Design")
                        st.image(complete_design, caption=f"Generated Design: {complete_design.size[0]}x{complete_design.size[1]}")
                        
                        # Download link
                        st.markdown(get_image_download_link(complete_design, "complete_design.png"), unsafe_allow_html=True)
                        
                        # Statistics
                        st.subheader("Generation Statistics")
                        original_pixels = original_image.size[0] * original_image.size[1]
                        generated_pixels = complete_design.size[0] * complete_design.size[1]
                        expansion_ratio = generated_pixels / original_pixels
                        
                        stats_col1, stats_col2, stats_col3 = st.columns(3)
                        with stats_col1:
                            st.metric("Original Size", f"{original_image.size[0]}×{original_image.size[1]}")
                        with stats_col2:
                            st.metric("Generated Size", f"{complete_design.size[0]}×{complete_design.size[1]}")
                        with stats_col3:
                            st.metric("Expansion Ratio", f"{expansion_ratio:.1f}x")
                        
                    except Exception as e:
                        st.error(f"Error generating design: {str(e)}")
                        st.info("Try reducing the output dimensions or using a different pattern type.")
    
    else:
        # Show example and instructions
        st.info("👆 Upload a design pattern to get started!")
        
        st.subheader("How it works:")
        st.markdown("""
        1. **Upload** a small design pattern or tile
        2. **Configure** your desired output dimensions
        3. **Choose** a pattern type:
           - **Smart**: Creates variations for natural-looking patterns
           - **Simple**: Direct repetitive tiling
           - **Organic**: More random, natural variations
        4. **Generate** your complete design
        5. **Download** the result
        """)
        
        st.subheader("Tips for best results:")
        st.markdown("""
        - Use square or rectangular patterns that can tile well
        - Higher contrast patterns work better
        - Enable 'Make Seamless' for smoother transitions
        - Use 'Preview Mode' for testing before full generation
        - Try different pattern types to see what works best
        """)

if __name__ == "__main__":
    main()