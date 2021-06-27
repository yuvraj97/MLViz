import streamlit as st
import numpy as np
from PIL import Image
from Algos.K_Means_Clustering.simulation import kMean
np.seterr(divide='ignore', invalid='ignore')


def get_inputs():
    st.write("# n-color")
    st.info("""
    An image carries millions of colors, here this program fetch $n$ colors from an image, 
    (using **K-Mean-Clustering** algorithm)
    that best represent the image, then recreate image with those $n$ colors.   
    So image with millions of colors reduced to image with only $n$ colors.
    """)
    image = st.file_uploader("Choose an Image", type=["png", "jpg"])
    n_colors = st.slider("Number of colors", 1, 10, 2, 1)
    seed: int = st.sidebar.number_input("Enter seed (-1 mean seed is disabled)", -1, 1000, 1, 1)

    return {
        "image": image,
        "n_colors": n_colors,
        "seed": seed
    }

def resize(image):
    if np.product(image.size) <= 320*320:
        return image
    width, height = image.size
    percent = 320.0/max(width, height)
    _width, _height = int(width*percent), int(height*percent)
    st.warning(f"""
    Your image's resolution is **{width} x {height}** that is greater then **320 x 320**.    
    So the resultant image's resolution is **{_width} x {_height}**
    """)
    return image.resize((_width, _height))

def run(state):
    inputs = get_inputs()
    image, n_colors, seed = inputs["image"], inputs["n_colors"], inputs["seed"]

    if image is None:
        return

    real_image = resize(Image.open(image))
    image_np = np.array(real_image)
    shape = (image_np.shape[0], image_np.shape[1])
    coordinates = np.reshape(image_np, (shape[0] * shape[1], image_np.shape[2]))
    coordinates = coordinates / 255

    begin = st.empty()
    if not begin.button("Begin ▶"):
        return
    begin.empty()

    plot_images = st.empty()
    progress, progress_percent = st.beta_columns([9, 1])
    progress = st.progress(0)
    progress_percent = progress_percent.empty()

    for labels, (centroids, percent) in kMean(n_colors, coordinates, max_epochs=10, delay=0, seed=seed):
        new_img = centroids[labels, :]
        new_img = np.reshape(new_img, (shape[0], shape[1], image_np.shape[2]))
        plot_real_image, plot_new_image = plot_images.beta_columns([1, 1])
        plot_real_image.image(real_image, caption="Real Image")
        plot_new_image.image(new_img, "New Image")
        progress.progress(percent)
        progress_percent.markdown(f"${percent*100:.2f}\\%$")
    progress.progress(1.0)
    progress_percent.empty()
    st.success("Completed 😊")
