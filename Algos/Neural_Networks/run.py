import streamlit as st

def get_nn_def():
    if "nn_def" not in st.session_state["Neural Networks"]:
        st.session_state["Neural Networks"]["nn_def"] = {
            "n_hidden_layers": 1,
            "hidden_units_per_layer": [[True]]
        }

    st.write(st.session_state["Neural Networks"])
    layers = [1 for _ in range(st.session_state["Neural Networks"]["nn_def"]["n_hidden_layers"])]
    st_layers = st.columns(layers + [1])
    for li in range(len(layers)):
        layer = st.session_state["Neural Networks"]["nn_def"]["hidden_units_per_layer"][li]
        # st.write(layer)
        with st_layers[li]:
            for ui in range(len(layer)):
                st.write(f"$z_{{{ui + 1}}}^{{[{li + 1}]}}$ {'✅' if layer[ui] else '❌'}")
                if st.button(
                        f"{'✅' if not layer[ui] else '❌'}",
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
            if st.button("+", key=f"add more layers for l{li}"):
                layer.append(True)
                st.experimental_rerun()
    if st_layers[-1].button("+", key="add more layers"):
        st.session_state["Neural Networks"]["nn_def"]["n_hidden_layers"] += 1
        st.session_state["Neural Networks"]["nn_def"]["hidden_units_per_layer"].append([True])
        st.experimental_rerun()


def run():
    if "Neural Networks" not in st.session_state:
        st.session_state["Neural Networks"] = {}
    get_nn_def()
