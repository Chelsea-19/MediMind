import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import json
import logging
import re
from src.utils.config_loader import load_config
from src.retrieval.vector_store import VectorStore
from src.generation.model import MultimodalMedicalModel
from src.generation.prompt_builder import PromptBuilder
from src.safety.safety_checker import SafetyChecker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="MediMind Pro Research", layout="wide", page_icon="🩺")

@st.cache_resource
def init_system():
    logger.info("Initializing MediMind System...")
    config = load_config()
    
    # 1. Init Database
    vector_store = VectorStore(
        db_path=config['retrieval']['db_path'],
        embedding_model=config['retrieval']['embedding_model'],
        collection_name=config['retrieval']['collection_name']
    )
    
    # 2. Init Safety Checker
    safety_checker = SafetyChecker()
    
    # 3. Init Model (Requires GPU, wrap in try/except if testing on CPU logic-only)
    try:
        model = MultimodalMedicalModel(
            model_name=config['model']['name'],
            quantization=config['model']['quantization'],
            device_map=config['model']['device_map']
        )
    except Exception as e:
        logger.warning(f"Failed to load heavy model. Skipping model init. Error: {e}")
        model = None
        
    return config, vector_store, safety_checker, model

config, vector_store, safety_checker, model = init_system()

st.title("MediMind Pro (Research Framework) 🩺")
st.caption("Patient Education and Preliminary Triage Support System. *Not a diagnostic tool.*")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar - Research Tools
with st.sidebar:
    st.header("Research Controls")
    
    st.subheader("Data Ingestion")
    new_doc_content = st.text_area("Add Reference Document (Text):")
    doc_source = st.text_input("Source/Document Name:", value="Unknown")
    
    if st.button("Ingest Document Chunk"):
        if new_doc_content.strip():
            from src.retrieval.ingestion import DocumentIngestionPipeline
            pipeline = DocumentIngestionPipeline(vector_store)
            ids = pipeline.ingest_text(new_doc_content, doc_source)
            st.success(f"Ingested {len(ids)} chunks!")
        else:
            st.warning("Document content cannot be empty.")
            
    st.divider()
    st.write("System Settings:")
    st.json(config)

# Chat History rendering
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

col1, col2 = st.columns([1, 5])
with col1:
    img = st.file_uploader("Upload Image (Optional)", type=['jpg','png', 'jpeg'])
with col2:
    q = st.chat_input("Enter your symptoms or medical questions...")

if q:
    # 1. Record input
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.write(q)
        if img: st.image(img, width=200)

    # 2. Safety / Risk Routing Gating
    safety_context = safety_checker.build_safety_context(q)
    
    if safety_context["escalate"]:
        st.error(safety_context["message"])

    if safety_context["is_unsafe"]:
        with st.chat_message("assistant"):
            safe_reply = "⚠️ 本系统仅用于初步健康教育科普，无法提供用药处方或危险诊断。请咨询专业执业医师。"
            st.warning(safe_reply)
            st.session_state.messages.append({"role": "assistant", "content": safe_reply})
        st.stop()
        
    # 3. Retrieval
    with st.spinner("Retrieving evidence..."):
        docs = vector_store.search(q, top_k=config['retrieval']['top_k'])
    
    with st.expander(f"📚 Evidence Retrieved ({len(docs)} chunks)", expanded=False):
        for d in docs: 
            st.markdown(f"**Source:** {d['metadata'].get('source_name')} | **ID:** {d['metadata'].get('chunk_id')}")
            st.text(d['content'])
            st.divider()

    # 4. Generation
    with st.chat_message("assistant"):
        if model is None:
            st.error("Model is not loaded (likely due to missing GPU). This is a mock response.")
            st.stop()
            
        with st.spinner("Analyzing..."):
            system_prompt = PromptBuilder.build_system_prompt(docs) + "\n" + PromptBuilder.build_structured_output_instruction()
            
            # handle temporary image saving for the model framework
            img_path = None
            if img:
                img_path = "temp_query.jpg"
                with open(img_path, "wb") as f:
                    f.write(img.getbuffer())

            raw_response = model.generate_response(system_prompt, q, img_path, json_mode=True)
            
            try:
                # Naive json parsing using regex to find the block
                json_str = raw_response
                match = re.search(r'\{(?:[^{}]|(?R))*\}', json_str, re.DOTALL)
                if match:
                    json_str = match.group(0)
                else: 
                     # Fallback simple search
                     start = json_str.find('{')
                     end = json_str.rfind('}') + 1
                     if start != -1 and end != 0:
                         json_str = json_str[start:end]

                json_obj = json.loads(json_str)
                st.json(json_obj) # Display nicely
                
                clean_record = json.dumps(json_obj, ensure_ascii=False, indent=2)
                st.session_state.messages.append({"role": "assistant", "content": clean_record})
                
            except Exception as e:
                # Fallback to display raw output if JSON generation failed
                logger.error(f"JSON Parsing Error: {e}")
                st.write("未能生成结构化 JSON，原始输出：")
                st.write(raw_response)
                st.session_state.messages.append({"role": "assistant", "content": raw_response})
