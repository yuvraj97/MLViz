import streamlit as st
from Fun_Projects.Research_Paper_Summarizer.document import Document
from Fun_Projects.Research_Paper_Summarizer.preprocess import processPDF

def run(state):
    st.write("""
    # Top k sentences, while maintaining a maximum distance between sentences
    """)
    st.info("""
    Here you can input any **research paper** and 
    this program will fetch the top **k** most important **sentences** from that research paper.  
    So you can quickly see what the research paper is about.   
    
    Most interesting part of this program is the variable **D**.  
    **D**: Maximum distance between sentences.   
    You can specify that how far those **k** most important **sentences** should be.  
    (**Dynamic Programing** approach is used to achieve it)
    
    If input is not provided then [this research paper](https://arxiv.org/pdf/1507.02672v1.pdf) will be used.
    """)

    if "Research_Paper_Summarizer" not in state["main"]:
        state["main"]["Research_Paper_Summarizer"] = {"fp_name": None}

    fp = st.file_uploader("Input a Research paper", ["pdf"])
    if fp is None:
        fp = open('./Fun_Projects/Research_Paper_Summarizer/1507.02672v1.pdf', 'rb')

    if state["main"]["Research_Paper_Summarizer"]["fp_name"] == fp.name:
        data = state["main"]["Research_Paper_Summarizer"]["data"]
    else:
        data = processPDF(fp)
        state["main"]["Research_Paper_Summarizer"]["fp_name"] = fp.name
        state["main"]["Research_Paper_Summarizer"]["data"] = data

    st_k, st_max_sentence_distance = st.beta_columns([1, 1])

    k = st_k.number_input(
        "Enter value of K",
        min_value=1,
        max_value=100,
        value=15,
        step=1
    )

    D = st_max_sentence_distance.number_input(
        "Enter maximum distance between sentences (D)",
        min_value=1,
        max_value=100,
        value=10,
        step=1
    )

    begin = st.empty()
    if not begin.button("Begin ▶"):
        return
    begin.empty()

    p = Document(data)

    with st.beta_expander(f"See Summary in {k} sentences"):
        sentences = p.get_top_k_sentence(k=k, D=1)
        st.info("  \n\n".join([sentence for idx, (weight, sentence) in sentences]))
        st.write(f"To get this result we use **k**: ${k}$ and **D**: $1$")

    sentences = p.get_top_k_sentence(
        k=k,
        D=D
    )

    for idx, (weight, sentence) in sentences:
        st.success(f"""
        (index: ${idx}$, weight: ${weight:.4f}$)    
        **{sentence}**
        """)
