import streamlit as st
from PIL import ImageDraw, ImageFont
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

def kmeans_color_quantization(image, n_colors=5):
    """
    Perform k-means clustering for color quantization.

    Args:
    - image: PIL.Image object representing the input image
    - n_colors: Number of colors to quantize the image into

    Returns:
    - quantized_img: Quantized image after clustering
    - colors: Dominant colors obtained from clustering
    - color_counts: Number of pixels corresponding to each dominant color
    """
    pixels = np.asarray(image)
    original_shape = pixels.shape

    # Extract RGB or RGBA channels
    if len(original_shape) == 3 and original_shape[2] in [3, 4]:
        pixels = pixels.reshape(-1, original_shape[-1])
    else:
        raise ValueError("Unsupported image format. Image must be RGB or RGBA.")

    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    quantized_pixels = colors[labels]

    quantized_img = quantized_pixels.reshape(original_shape).astype(np.uint8)
    quantized_img = np.clip(quantized_img, 0, 255)

    color_counts = np.bincount(labels, minlength=n_colors)

    return quantized_img, colors, color_counts

def create_color_palette(dominant_colors, color_counts, palette_size=(300, 50)):
    """
    Create a color palette image.

    Args:
    - dominant_colors: Dominant colors obtained from clustering
    - color_counts: Number of pixels corresponding to each dominant color
    - palette_size: Size of the color palette image

    Returns:
    - palette: PIL.Image object representing the color palette
    """
    sorted_colors = [color for _, color in sorted(zip(color_counts, dominant_colors), reverse=True)]

    palette = Image.new("RGB", (palette_size[0] + 100, palette_size[1] * len(sorted_colors)), color=(255, 255, 255))
    draw = ImageDraw.Draw(palette)

    swatch_width = palette_size[0]
    swatch_height = palette_size[1]

    font = ImageFont.load_default()
    font_color = (0, 0, 0)
    background_color = (255, 255, 255)

    for i, color in enumerate(sorted_colors):
        color_hex = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])

        draw.rectangle([0, i * swatch_height, swatch_width, (i + 1) * swatch_height], fill=color_hex)

        count_text = f"{color_counts[i]} units"
        text_width, text_height = draw.textsize(count_text, font=font)
        draw.rectangle([swatch_width, i * swatch_height, swatch_width + text_width + 20, (i + 1) * swatch_height],
                       fill=background_color)
        draw.text((swatch_width + 10, i * swatch_height + (swatch_height - text_height) // 2),
                  count_text, fill=font_color, font=font)

    return palette

def render_ui():
    """
    Render the Streamlit user interface.
    """
    st.title("🎨 Image Color Clustering App 🖼️")
    st.markdown("<h4 style='color: #336699; text-align: right;'>Created by Harshad Thombre</h4>", unsafe_allow_html=True)
    st.markdown("<h5 style='color: #336699; text-align: right;'>GitHub: <a href='https://github.com/Harshad1025' style='color: #336699;'>Harshad1025</a></h5>", unsafe_allow_html=True)
    st.markdown("---")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        image = image.convert("RGBA")  # Ensure image is RGBA format for PNG with alpha channel
        image.thumbnail((400, 400))  # Resizing image for user-friendly display
        st.image(image, caption="Original Image", use_column_width=True)

        n_colors = st.slider("Number of Colors", min_value=2, max_value=10, value=5, step=1)

        if st.button("Apply Clustering"):
            clustered_img, dominant_colors, color_counts = kmeans_color_quantization(image, n_colors)
            clustered_image = Image.fromarray(clustered_img)
            clustered_image.thumbnail((400, 400))  # Resizing clustered image
            st.image(clustered_image, caption=f"Clustered Image (Colors: {n_colors})", use_column_width=True)

            st.markdown("<h3 style='color: #F98A28;'>🎨 Dominant Colors:</h3>", unsafe_allow_html=True)
            palette_img = create_color_palette(dominant_colors, color_counts)
            st.image(palette_img, caption="Color Palette", use_column_width=True)

            st.markdown("---")
            st.markdown("<h3 style='color: #336699;'>📧 Contact Me:</h3>", unsafe_allow_html=True)
            st.write("🌐 GitHub: [Harshad1025](https://github.com/Harshad1025)")
            st.write("📞 Mobile: +91 9112172447")
            st.markdown("---")
            st.markdown("Thank you for visiting! 👋")

def main():
    """
    Main function to run the application.
    """
    render_ui()

if __name__ == "__main__":
    main()
