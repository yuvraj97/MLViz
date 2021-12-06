import json
import math

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_drawable_canvas import st_canvas


class Params:

    @staticmethod
    def get_random_data_inputs():
        """
        Here we get all inputs from user
        :return: Dict[str, Union[int, float]]
        """

        with st.sidebar.expander("Generate n dimensional synthetic data"):
            st.write("")
            st_seed, st_n = st.columns([1, 1])
            seed: int = int(st_seed.text_input("Enter seed (-1 mean seed is disabled)", "0"))
            n: int = int(st_n.text_input("N (number of training examples)", "100"))
            n_features: int = int(st.text_input("Number of features", "2"))
            st_lower_limit, st_upper_limit = st.columns([0.5, 0.5])
            lower_limit: float = float(st_lower_limit.text_input("Lower Limit", "-10.0"))
            upper_limit: float = float(st_upper_limit.text_input("Upper Limit", "10.0"))

            st.write("### Classes proportions")
            st_prop = st.columns([1, 1])
            classes_proportions = [
                float(st_prop[0].text_input(f"Class: 1", "0.5")),
                float(st_prop[1].text_input(f"Class: 2", "0.5"))
            ]

            if not math.isclose(sum(classes_proportions), 1.0, abs_tol=0.01):
                st.error("Proportions should sum to $1$")
                raise ValueError("Algos.Logistic_Regression.run: Proportions should sum to $1$")

            st.write("### Gaussian Noise $\\mathcal{N}(\\mu,\\sigma^2)$")
            st_mean, st_std = st.columns([1, 1])
            mean: float = float(st_mean.text_input("Mean", "0.0"))
            std: float = float(st_std.text_input("Standard deviation", "1.0"))

        return {
            "seed": seed,
            "n": n,
            "n_classes": 2,
            "n_features": n_features,
            "lower_limit": lower_limit,
            "upper_limit": upper_limit,
            "classes_proportions": classes_proportions,
            "mean": mean,
            "std": std,
        }

    @staticmethod
    def get_nn_inputs(n):
        method: str = st.sidebar.selectbox("Which method you want to use", [
            "Batch Gradient Ascent",
            "Mini Batch Gradient Ascent"
        ])

        with st.sidebar.expander("Logistic Regression Parameters", True):

            st_lr, st_epsilon, st_epochs = st.columns([1, 1, 0.8])
            lr: float = float(st_lr.text_input("Learning Rate", "0.01"))
            epochs: int = int(st_epochs.text_input("epochs", "50"))
            epsilon: float = float(st_epsilon.text_input("Epsilon", "0.05"))

            batch_size = None
            if method == "Mini Batch Gradient Descent":
                batch_size = st.number_input("Batch size", 1, n, 10, 1)

            nn_method: str = st.radio("Choose method", ["Implementation From Scratch", "PyTorch Implementation"])

        return {
            "method": method,
            "lr": lr,
            "epsilon": epsilon,
            "epochs": epochs,
            "batch_size": batch_size,
            "nn_method": nn_method
        }


class Canvas:

    colors = ["red", "blue", "green", "black", "orange", "brown", "indigo", "pink", "violet", "purple", "navy"]

    @staticmethod
    def get_canvas_data(default_canvas="./Algos/Neural_Networks/canvas_result.json"):
        if st.checkbox("clear_canvas", False):
            initial_drawing = None
        else:
            with open(default_canvas, "r") as fp:
                initial_drawing = json.load(fp)

        color = st.sidebar.selectbox("Choose class color", Canvas.colors)

        # Create a canvas component
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=13,
            stroke_color=color,
            background_color="#eee",
            # background_image=Image.open(bg_image) if bg_image else None,
            update_streamlit=False,
            height=400, width=800,
            drawing_mode="freedraw",
            key="canvas",
            initial_drawing=initial_drawing
        )

        if canvas_result.json_data is None or len(canvas_result.json_data["objects"]) == 0: return None, None, None, None

        classes, classes_labels, classes_labels_mapping, norm_params = Canvas.get_refined_data(canvas_result.json_data["objects"])
        return classes, classes_labels, classes_labels_mapping, norm_params

    @staticmethod
    def get_refined_data(canvas_objects_data):
        data = pd.DataFrame([[obj["left"], -obj["top"], obj["stroke"]] for obj in canvas_objects_data])
        norm_params = {
            "mean": data.mean().values,
            "std": data.std().values
        }
        unique_labels = data[2].unique()
        groups = data.groupby(2)
        classes = [groups.get_group(class_) for class_ in unique_labels]
        classes_labels = np.hstack([class_[2].values for class_ in classes])
        classes_labels = pd.Series(classes_labels).astype('category')
        classes_labels_mapping = {i: label for i, label in enumerate(classes_labels.cat.categories)}
        [class_.drop(2, axis=1, inplace=True) for class_ in classes]
        classes = np.vstack([((class_ - norm_params["mean"]) / norm_params["std"]).values for class_ in classes])
        return classes, classes_labels.cat.codes.values, classes_labels_mapping, norm_params


class DrawNN:

    @staticmethod
    def draw_nn():
        if "nn_def" not in st.session_state["Neural Networks"]:
            st.session_state["Neural Networks"]["nn_def"] = {
                "n_hidden_layers": 1,
                "hidden_units_per_layer": [[True]]
            }

        st.write(f"""
        ### Create Neural Network Structure
        """)

        layers = [1 for _ in range(st.session_state["Neural Networks"]["nn_def"]["n_hidden_layers"])]
        st_layers = st.columns(layers + [0.1])
        for li in range(len(layers)):
            layer = st.session_state["Neural Networks"]["nn_def"]["hidden_units_per_layer"][li]
            with st_layers[li]:
                for ui in range(len(layer)):
                    st.write(f"$z_{{{ui + 1}}}^{{[{li + 1}]}}$ {'✅' if layer[ui] else '⛔'}")
                    if st.button(
                            f"{'✅' if not layer[ui] else '⛔'}",
                            key=f"Remove layers for l{li}-{ui}",
                            help=f"""
                            Currently this node `{ui + 1}` of layer `{li + 1}` is `{'Activate' if layer[ui] else 'Deactivated'}`      
                            Click to `{'Deactivate' if layer[ui] else 'Activated'}` node `{ui + 1}` of layer `{li + 1}`    
                            """
                    ):
                        layer[ui] = not layer[ui]
                        st.experimental_rerun()
                    st.write("  ")
                    st.write("  ")
                if st.button("➕", key=f"add more layers for l{li}", help=f"Add more neuron in layer {li + 1}"):
                    layer.append(True)
                    st.experimental_rerun()
        with st_layers[-1]:

            if st.button("➕", key="add more layers", help="Add one more layer"):
                st.session_state["Neural Networks"]["nn_def"]["n_hidden_layers"] += 1
                st.session_state["Neural Networks"]["nn_def"]["hidden_units_per_layer"].append([True])
                st.experimental_rerun()
            if st.button("❌", key="Remove Deactivated layers", help="Remove All Deactivated Neurons"):
                for li in range(len(layers)):
                    st.session_state["Neural Networks"]["nn_def"]["hidden_units_per_layer"][li] = \
                        [_ for _ in st.session_state["Neural Networks"]["nn_def"]["hidden_units_per_layer"][li] if _]
                st.session_state["Neural Networks"]["nn_def"]["hidden_units_per_layer"] = \
                [_ for _ in st.session_state["Neural Networks"]["nn_def"]["hidden_units_per_layer"] if _]
                st.session_state["Neural Networks"]["nn_def"]["n_hidden_layers"] = \
                len(st.session_state["Neural Networks"]["nn_def"]["hidden_units_per_layer"])
                st.experimental_rerun()
