import streamlit as st

def get_nn_def():
    if "nn_def" not in st.session_state["Neural Networks"]:
        st.session_state["Neural Networks"]["nn_def"] = {
            "n_hidden_layers": 1,
            "hidden_units_per_layer": [1]
        }
    layers = [1 for _ in range(st.session_state["Neural Networks"]["nn_def"]["n_hidden_layers"])]
    st_layers = st.columns(layers + [1])
    for li in range(len(layers)):
        with st_layers[li]:
            for ui in range(st.session_state["Neural Networks"]["nn_def"]["hidden_units_per_layer"][li]):
                st.write(f"$z_{{{ui + 1}}}^{{[{li + 1}]}}$")
            if st.button("+", key=f"add more layers for l{li}"):
                st.session_state["Neural Networks"]["nn_def"]["hidden_units_per_layer"][li] += 1
                st.experimental_rerun()
    if st_layers[-1].button("+", key="add more layers"):
        st.session_state["Neural Networks"]["nn_def"]["n_hidden_layers"] += 1
        st.session_state["Neural Networks"]["nn_def"]["hidden_units_per_layer"].append(1)
        st.experimental_rerun()


def run():
    if "Neural Networks" not in st.session_state:
        st.session_state["Neural Networks"] = {}
    get_nn_def()
