# ==========================================
# 0. INSTALL DEPENDENCIES 
# ==========================================
import os
print("⏳ Installing libraries... Please wait (approx 1 min)...")
os.system("pip install -q gradio openai-whisper transformers accelerate bitsandbytes gtts langchain langchain-community langchain-core")
print("✅ Installation complete!")

# ==========================================
# 1. IMPORTS & CONFIGURATION
# ==========================================
import gradio as gr
import whisper
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline
# استفاده از langchain_core برای سازگاری با نسخه‌های جدید
from langchain_core.prompts import PromptTemplate 
from gtts import gTTS
import tempfile

LANG = "it"
WHISPER_MODEL_SIZE = "small"
LLM_MODEL_ID = "sapienzanlp/Minerva-7B-instruct-v1.0"

# Default system instruction
DEFAULT_SYSTEM_PROMPT = """
You are an advanced Italian Grammar Assistant.
Your task is to analyze the user's spoken sentence based on the provided context (examples).
- If the input is a question with options (A/B), choose the correct one.
- If the input requires grammatical analysis (subject, verb, etc.), perform the analysis.
- If the input is a general sentence, correct any errors.
Keep the answer concise and strictly follow the format of the provided examples.
"""

current_knowledge_base = ""

# ==========================================
# 2. MODEL LOADING
# ==========================================
print("--- [1/2] Loading Whisper Model (Hearing) ---")
whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)

print("--- [2/2] Loading Minerva LLM (Brain) ---")
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

try:
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        quantization_config=quant_config,
        device_map="auto"
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.3,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    llm = HuggingFacePipeline(pipeline=pipe)
except Exception as e:
    print(f"Error loading LLM: {e}")
    print("Ensure you are using T4 GPU runtime (Runtime > Change runtime type > T4 GPU)")

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def update_context(files):
    global current_knowledge_base
    context_text = ""
    if not files:
        return "No files uploaded. Using default prompt."

    for file in files:
        try:
            with open(file.name, "r", encoding="utf-8") as f:
                content = f.read()
                context_text += f"\n--- Content from {os.path.basename(file.name)} ---\n{content}\n"
        except Exception as e:
            return f"Error reading file: {e}"
            
    current_knowledge_base = context_text
    return "✅ Knowledge Base Updated! AI is ready."

def text_to_speech(text, lang="it"):
    try:
        if not text: return None
        tts = gTTS(text=text, lang=lang)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            return fp.name
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

# ==========================================
# 4. CORE LOGIC
# ==========================================

def process_pipeline(audio_path):
    global current_knowledge_base
    
    if audio_path is None:
        return "No audio recorded.", None

    # Step 1: Transcribe
    print("🎤 Listening...")
    transcription_result = whisper_model.transcribe(audio_path, language="it")
    user_text = transcription_result["text"]
    print(f"📝 Transcribed: {user_text}")

    # Step 2: Prepare Context
    final_context = DEFAULT_SYSTEM_PROMPT
    if current_knowledge_base:
        final_context += f"\n\n### REFERENCE EXAMPLES:\n{current_knowledge_base}"

    # Step 3: Prompt Engineering
    prompt_template = PromptTemplate.from_template(
        "Instructions:\n{system_context}\n\n"
        "User Input:\n{user_text}\n\n"
        "Assistant Response (Italian):"
    )
    
    formatted_prompt = prompt_template.format(
        system_context=final_context,
        user_text=user_text
    )

    # Step 4: Generate
    print("🤖 Thinking...")
    full_response = llm.invoke(formatted_prompt)
    clean_response = full_response.replace(formatted_prompt, "").strip()
    
    if "Assistant Response (Italian):" in clean_response:
        clean_response = clean_response.split("Assistant Response (Italian):")[-1].strip()

    # Step 5: Speak
    print("🔊 Speaking...")
    audio_output_path = text_to_speech(clean_response)

    display_text = (
        f"🗣️ **Transcription:** {user_text}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🤖 **AI Analysis:**\n{clean_response}"
    )
    
    return display_text, audio_output_path

# ==========================================
# 5. GRADIO APP
# ==========================================
with gr.Blocks(theme=gr.themes.Soft(), title="Thesis: Italian AI") as demo:
    gr.Markdown("# 🇮🇹 Thesis Project: Italian Grammar AI")
    gr.Markdown("1. Upload your `.txt` files (grammar rules).\n2. Record your voice.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Upload Context")
            file_uploader = gr.File(label="Upload .txt files", file_count="multiple", file_types=[".txt"])
            upload_status = gr.Textbox(label="Status", interactive=False)
            upload_btn = gr.Button("Update Knowledge Base", variant="secondary")
            upload_btn.click(fn=update_context, inputs=file_uploader, outputs=upload_status)

        with gr.Column(scale=2):
            gr.Markdown("### 2. Speak")
            audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Input")
            analyze_btn = gr.Button("Analyze", variant="primary")
            text_output = gr.Markdown(label="Result")
            audio_output = gr.Audio(label="Audio Response", autoplay=True)

    analyze_btn.click(fn=process_pipeline, inputs=audio_input, outputs=[text_output, audio_output])

print("🚀 Starting app...")
demo.launch(share=True, debug=True)
