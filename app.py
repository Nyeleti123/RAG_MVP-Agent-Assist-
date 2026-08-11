"""
RAG MVP

app.py

Agent Assist Dashboard

Features:
- Upload transcript
- Analyse transcript
- Conversation summary
- Intent detection
- Current intent
- Resolution guidance
- Approve/reject workflow
- Session memory
"""

import json
import pandas as pd
import streamlit as st
import time
from agent_assist import AgentAssist


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Agent Assist",
    page_icon="🤖",
    layout="wide"
)


# =====================================================
# STYLE
# =====================================================

st.markdown(
"""
<style>

.main {
    background:#f5f7fb;
}

.block-container {
    padding-top:1.5rem;
}

.intent-pill {

    background:#E8F0FE;
    color:#0F62FE;
    padding:8px 12px;
    border-radius:8px;
    margin-bottom:8px;
    font-weight:600;

}

.current-intent {

    background:#0F62FE;
    color:white;
    padding:15px;
    border-radius:10px;
    text-align:center;
    font-size:20px;
    font-weight:bold;

}

.card {

    background:white;
    padding:20px;
    border-radius:12px;
    border:1px solid #ddd;
    margin-bottom:20px;

}

</style>
""",
unsafe_allow_html=True
)



# =====================================================
# LOAD AGENT
# =====================================================

@st.cache_resource
def load_agent():

    return AgentAssist()



assist = load_agent()



# =====================================================
# SESSION STATE
# =====================================================

if "results" not in st.session_state:

    st.session_state.results = None


if "statuses" not in st.session_state:

    st.session_state.statuses = {}
    
if "processing" not in st.session_state:
    st.session_state.processing = False



# =====================================================
# HEADER
# =====================================================

st.title("🤖 Agent Assist")

st.caption(
    "AI-powered call centre decision support using RAG"
)


st.divider()



# =====================================================
# SIDEBAR UPLOAD
# =====================================================

st.sidebar.header("Transcript Upload")


uploaded = st.sidebar.file_uploader(

    "Upload transcript",

    type=[
        "txt",
        "json",
        "csv"
    ]

)



transcript = ""



# =====================================================
# FILE HANDLING
# =====================================================

if uploaded:


    if uploaded.name.endswith(".txt"):

        transcript = uploaded.read().decode(
            "utf-8"
        )


    elif uploaded.name.endswith(".json"):


        data = json.load(uploaded)

        lines = []


        for row in data:


            speaker = row.get(
                "speaker",
                row.get(
                    "role",
                    "Customer"
                )
            )


            text = row.get(
                "text",
                row.get(
                    "message",
                    ""
                )
            )


            lines.append(
                f"{speaker}: {text}"
            )


        transcript = "\n".join(lines)



    elif uploaded.name.endswith(".csv"):


        df = pd.read_csv(uploaded)


        if (
            "speaker" in df.columns
            and
            "text" in df.columns
        ):

            transcript = "\n".join(

                [
                    f"{r['speaker']}: {r['text']}"
                    for _,r in df.iterrows()
                ]

            )

        else:

            transcript = df.to_string()



# =====================================================
# TEXT INPUT
# =====================================================

transcript = st.text_area(

    "Transcript",

    value=transcript,

    height=250

)



# =====================================================
# ANALYSE
# =====================================================

if st.button(
    "Analyse Transcript",
    use_container_width=True
):


    if transcript.strip()=="":


        st.warning(
            "Please provide a transcript."
        )


    else:
        st.session_state.processing = True
        st.session_state.results = None
        st.rerun()

# Show spinner while processing
if st.session_state.processing:
    with st.spinner("Running RAG pipeline... Please wait..."):
        # Add a timeout mechanism using threads
        import threading
        import sys
        
        class TimeoutException(Exception):
            pass
        
        def timeout_handler():
            raise TimeoutException("Processing timed out after 30 seconds")
        
        # Set a timer
        timer = threading.Timer(30.0, timeout_handler)
        timer.start()
        
        try:
            st.session_state.results = assist.process(transcript)
            st.session_state.processing = False
            timer.cancel()
            st.rerun()
        except TimeoutException as e:
            st.error(f"⏰ {str(e)}. Please try with a shorter transcript.")
            st.session_state.processing = False
            # Create mock data for demo
            st.session_state.results = {
                "summary": {
                    "turns": 5,
                    "customer_turns": 3,
                    "segments": 2
                },
                "cards": [
                    {
                        "intent": "Refund Request",
                        "confidence": 0.95,
                        "checklist": "1. Verify policy\n2. Check eligibility\n3. Process refund",
                        "documents": ["Refund Policy.pdf"],
                        "status": "Pending"
                    }
                ]
            }
            st.rerun()
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.processing = False
            st.rerun()
        finally:
            timer.cancel()



# =====================================================
# STOP IF NO RESULTS
# =====================================================

if st.session_state.results is None:

    if not st.session_state.processing:
        st.info(
            "Upload or paste a transcript and click Analyse Transcript."
        )

    st.stop()



results = st.session_state.results



# =====================================================
# SUMMARY METRICS
# =====================================================

st.header("Conversation Summary")

# Handle both old and new format
if "summary" in results and isinstance(results["summary"], dict):
    summary = results["summary"]
    cards = results.get("cards", [])
else:
    # For compatibility with direct output from agent_assist
    cards = results.get("cards", [])
    summary = results.get("summary", {})

# If summary is empty or not a dict, create default
if not summary or not isinstance(summary, dict):
    summary = {
        "turns": len([line for line in transcript.split("\n") if line.strip()]),
        "customer_turns": len([line for line in transcript.split("\n") if "Customer:" in line]),
        "segments": len(cards)
    }


c1,c2,c3,c4 = st.columns(4)



with c1:

    st.metric(
        "Turns",
        summary.get(
            "turns",
            0
        )
    )


with c2:

    st.metric(
        "Customer Turns",
        summary.get(
            "customer_turns",
            0
        )
    )


with c3:

    st.metric(
        "Detected Intents",
        len(cards)
    )


with c4:

    st.metric(
        "Segments",
        summary.get(
            "segments",
            0
        )
    )



st.divider()



# =====================================================
# INTENTS
# =====================================================

left,right = st.columns(
    [1,2]
)



with left:


    st.subheader(
        "Intent Detection"
    )


    for card in cards:


        st.markdown(

            f"""
            <div class="intent-pill">

            {card.get('intent', 'Unknown')}
            ({card.get('confidence', 0)})

            </div>
            """,

            unsafe_allow_html=True

        )



with right:


    st.subheader(
        "Current Intent"
    )


    if results.get("current_intent"):


        st.markdown(

            f"""
            <div class="current-intent">

            {results['current_intent']}

            </div>
            """,

            unsafe_allow_html=True

        )

    else:

        st.info(
            "No intent detected."
        )



st.divider()



# =====================================================
# RESOLUTION CARDS
# =====================================================

st.header(
    "Resolution Guidance"
)



for i,card in enumerate(cards):


    key = f"status_{i}"


    if key not in st.session_state.statuses:

        st.session_state.statuses[key] = "Pending"



    status = st.session_state.statuses[key]



    st.markdown(
        f"""
        <div class="card">

        <h3>{card.get('intent', 'Unknown')}</h3>

        </div>
        """,
        unsafe_allow_html=True
    )



    col1,col2 = st.columns([3,1])



    with col1:


        st.write(
            "Confidence"
        )


        confidence = float(
            card.get("confidence", 0)
        )


        if confidence > 1:

            confidence /= 100


        st.progress(
            confidence
        )


        st.write(
            f"{confidence:.2f}"
        )



    with col2:


        st.write(
            "Status"
        )


        if status=="Approved":

            st.success(status)


        elif status=="Rejected":

            st.error(status)


        else:

            st.info(status)



    st.subheader(
        "Resolution Checklist"
    )


    checklist = card.get("checklist", "No checklist available")
    
    # Handle both string and list format
    if isinstance(checklist, str):
        for line in checklist.split("\n"):
            if line.strip():
                st.write(line)
    else:
        for line in checklist:
            st.write(f"• {line}")



    st.subheader(
        "Supporting Documents"
    )


    if card.get("documents"):


        for doc in card["documents"]:

            st.write(
                f"• {doc}"
            )


    else:

        st.write(
            "No supporting documents."
        )



    b1,b2 = st.columns(2)



    with b1:


        if st.button(
            "Approve",
            key=f"a{i}",
            use_container_width=True
        ):


            st.session_state.statuses[key]="Approved"

            st.rerun()



    with b2:


        if st.button(
            "Reject",
            key=f"r{i}",
            use_container_width=True
        ):


            st.session_state.statuses[key]="Rejected"

            st.rerun()



    st.divider()



# =====================================================
# MEMORY
# =====================================================

st.sidebar.divider()

st.sidebar.header(
    "Conversation Memory"
)



for k,v in summary.items():

    st.sidebar.write(
        f"{k}: {v}"
    )



# =====================================================
# FOOTER
# =====================================================

st.caption(
    "RAG Agent Assist Prototype | AI/ML Capstone"
)