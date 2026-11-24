import streamlit as st

import app_core as core
import app_data
import retrieval

METHODS = {"hybrid": "Hybrid", "bm25": "BM25 keywords", "dense": "Dense (MiniLM)"}

f = app_data.forecast(st.session_state.site, st.session_state.time)
pred, hazard = f["pred"], f["hazard"]
auto = f["query"] or ""

# Refill the search box when the forecast changes, but keep the user's edits otherwise.
if st.session_state.get("query_auto") != auto or "query" not in st.session_state:
    st.session_state.query_auto = auto
    st.session_state.query = auto

st.caption(f"{core.site_label(st.session_state.site)} · {core.time_label(st.session_state.time)} · forecast "
           f"**{core.CLASS_NAMES[pred]}**" + (f", driven mainly by {'waves' if hazard == 'wave' else 'wind'}" if pred > 0 else ""))

if pred == 0:
    st.info("The forecast is Normal, so no guidance is suggested. You can still search the corpus below.",
            title="No hazard forecast", icon=":material/check_circle:")

with st.container(horizontal=True, vertical_alignment="bottom"):
    st.text_input("Search the HSE and MAIB corpus", key="query", icon=":material/search:",
                  placeholder="for example: crew transfer in heavy swell", width=560, persist_state="session")
    st.segmented_control("Method", list(METHODS), key="method", default="hybrid", required=True,
                         format_func=METHODS.get, persist_state="session")
    st.toggle("Offshore documents only", key="core_only", value=True, persist_state="session",
              help="Search only documents flagged as offshore operations, as in the retrieval benchmark.")

query = (st.session_state.query or "").strip()
if not query:
    st.stop()

R = app_data.retrievers()
hits = R.top_chunks(st.session_state.method, query, k=3, core_only=st.session_state.core_only)
if not hits:
    st.warning("No documents found for this search.")
    st.stop()

terms = set(retrieval.tokenize(query))
for i, c in enumerate(hits, start=1):
    text, cut_start, cut_end = core.trim_to_sentences(c["text"])
    with st.container(border=True):
        st.markdown(f"**{i}. [{core.md_escape(core.tidy_title(c['title']))}](<{c['url']}>)** "
                    ":material/open_in_new:")
        st.caption(f"{c['source']} · {core.doc_type_label(c['doc_type'])}")
        st.markdown(f"“{core.preview(text, terms)}”")
        with st.expander("Show full passage"):
            st.markdown(core.format_passage(text, terms, cut_start=cut_start, cut_end=cut_end))

st.caption(f"Top 3 documents by {METHODS[st.session_state.method]} search. Words from your search are in bold. "
           "Read the full source before acting on it.")
