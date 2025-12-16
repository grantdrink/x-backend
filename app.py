import os
import re
import uuid
import threading
import subprocess
import platform
from flask import Flask, request, jsonify, send_file, after_this_request
from flask_cors import CORS
from werkzeug.utils import secure_filename
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import pdfplumber
from io import BytesIO
import spacy
import fitz  # PyMuPDF - superior text extraction
import pypdfium2 as pdfium  # Another excellent extraction library
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
import ftfy  # Fix text encoding issues
from cleantext import clean  # Professional text cleaning library
from typing import Dict, List, Tuple

app = Flask(__name__, static_folder='../frontend/dist', static_url_path='')
CORS(app)

UPLOAD_FOLDER = 'backend/uploads'
AUDIO_FOLDER = 'backend/audio'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['AUDIO_FOLDER'] = AUDIO_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

documents = {}

tts_lock = threading.Lock()

# Load spaCy model for AI-powered text processing
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# ============================================================================
# EDGE TTS SYSTEM - Free, Fast, High-Quality
# ============================================================================
print("🎙️ Loading Edge TTS...")
print("=" * 70)

# Dictionary to store TTS engines
tts_engines = {
    'edge': None,      # Edge TTS (Microsoft, free API, fast)
    'xtts_custom': None,  # Custom XTTS voice (your voice!)
}

# EDGE TTS - Microsoft's free TTS API (fast, cloud-based)
print("\n1️⃣ Loading Edge TTS (Microsoft)...")
try:
    import edge_tts
    tts_engines['edge'] = 'available'
    print("   ✅ Edge TTS loaded - Fast, cloud-based, free")
    print("   📜 License: Free for personal/educational use")
    print("   🎭 Voices: 400+ voices in 100+ languages")
except Exception as e:
    print(f"   ⚠️ Edge TTS failed: {e}")

# Check if custom XTTS voice is available
print("\n2️⃣ Checking Custom XTTS Voice...")
CUSTOM_VOICE_DIR = os.path.expanduser("~/Desktop/voice/dataset/wavs")
if os.path.exists(CUSTOM_VOICE_DIR):
    wav_files = [f for f in os.listdir(CUSTOM_VOICE_DIR) if f.endswith('.wav')]
    if wav_files:
        tts_engines['xtts_custom'] = 'available'
        print(f"   ✅ Custom voice found - {len(wav_files)} reference audio files")
        print(f"   📁 Location: {CUSTOM_VOICE_DIR}")
    else:
        print(f"   ⚠️ No .wav files found in {CUSTOM_VOICE_DIR}")
else:
    print(f"   ⚠️ Custom voice directory not found: {CUSTOM_VOICE_DIR}")

print("\n" + "=" * 70)
active_voices = ["Sonia (UK Female)", "Ryan (UK Male)"]
if tts_engines.get('xtts_custom'):
    active_voices.append("Grant's Voice (Custom XTTS)")
print(f"✅ TTS System Ready")
print(f"🎯 Active Voices: {', '.join(active_voices)}")
print("🆓 100% FREE - No API costs, no limits")
print("=" * 70)

# Load marker-pdf models and create converter for AI-powered PDF extraction
print("Loading marker-pdf models for AI-powered PDF extraction...")
print("   This may take a minute on first run (downloading models)...")
try:
    from marker.models import create_model_dict
    from marker.config.parser import ConfigParser
    
    # Create model dictionary (this loads all the AI models)
    # device=None means it will auto-detect CPU/GPU
    marker_models = create_model_dict(device="cpu")  # Force CPU for compatibility
    
    # Create minimal config options for marker
    marker_config_options = {
        "output_format": "markdown",
        "output_dir": "backend/audio",  # Not used but required
        "debug": False,
        "processors": None,
        "config_json": None,
        "disable_multiprocessing": True,  # Safer for web server
        "disable_image_extraction": False,
        "page_range": None,
        "converter_cls": None,
        "llm_config": None,
    }
    
    # Create config parser
    marker_config_parser = ConfigParser(marker_config_options)
    
    # Create the PDF converter with all required parameters
    pdf_converter = PdfConverter(
        config=marker_config_parser.generate_config_dict(),
        artifact_dict=marker_models,  # This is the models dict!
        processor_list=marker_config_parser.get_processors(),
        renderer=marker_config_parser.get_renderer(),
        llm_service=marker_config_parser.get_llm_service(),
    )
    
    print("✅ marker-pdf loaded successfully! AI-powered PDF extraction ready.")
    print("   Models loaded: Detection, OCR, Layout Analysis, Text Recognition")
except Exception as e:
    print(f"⚠️ Error loading marker-pdf: {e}")
    print("   marker-pdf will not be available, falling back to traditional extractors")
    import traceback
    traceback.print_exc()
    pdf_converter = None
    marker_models = None

# Pre-defined voice samples for different personas
# You can add your own voice samples here (6+ seconds of clean audio)
VOICE_SAMPLES = {
    "female_1": None,  # Will use default XTTS speaker
    "male_1": None,    # Will use default XTTS speaker
    "female_2": None,  # Add path to custom voice sample WAV file
    "male_2": None,    # Add path to custom voice sample WAV file
}

async def generate_tts_edge(text: str, voice: str, output_path: str, rate: str = "+0%"):
    """Generate speech using Edge TTS (Microsoft, free, fast)"""
    import edge_tts
    
    # Map voice_id to Edge TTS voice names
    # ONLY UK voices (US/AU removed - not good quality)
    voice_map = {
        'edge_female_uk': 'en-GB-SoniaNeural',     # British female
        'edge_male_uk': 'en-GB-RyanNeural',        # British male
        # Legacy support (fallback to UK defaults)
        'edge_female': 'en-GB-SoniaNeural',
        'edge_male': 'en-GB-RyanNeural',
        'female': 'en-GB-SoniaNeural',
        'male': 'en-GB-RyanNeural',
        'female_uk': 'en-GB-SoniaNeural',
        'male_uk': 'en-GB-RyanNeural',
    }
    
    edge_voice = voice_map.get(voice, 'en-US-AriaNeural')
    
    communicate = edge_tts.Communicate(text, edge_voice, rate=rate)
    await communicate.save(output_path)

def generate_tts_piper(text: str, voice: str, output_path: str):
    """Generate speech using Piper TTS (local, MIT license, ultra-fast)"""
    # Piper implementation - load on-demand
    # Will download voice models automatically on first use
    import subprocess
    
    # Map voice_id to Piper voice models
    voice_map = {
        'piper_female': 'en_US-lessac-medium',     # Clear female
        'piper_male': 'en_US-danny-low',           # Male voice
        'piper_female_uk': 'en_GB-alan-medium',    # British
        # Legacy support
        'female': 'en_US-lessac-medium',
        'male': 'en_US-danny-low',
        'female_uk': 'en_GB-alan-medium',
    }
    
    piper_voice = voice_map.get(voice, 'en_US-lessac-medium')
    
    # Call piper CLI (installed with pip)
    cmd = f'echo "{text}" | piper --model {piper_voice} --output_file {output_path}'
    subprocess.run(cmd, shell=True, check=True)

def generate_tts_xtts_custom(text: str, output_path: str, speed: float = 1.0, language: str = "en"):
    """Generate speech using YOUR custom cloned voice via XTTS v2."""
    import subprocess
    
    # Path to the custom XTTS generator script
    script_path = os.path.join(os.path.dirname(__file__), 'xtts_custom_generator.py')
    
    if not os.path.exists(script_path):
        raise Exception(f"Custom XTTS generator not found at {script_path}")
    
    # Try to find Python with TTS installed
    # First try the conda environment, then system python
    python_paths = [
        os.path.expanduser('~/miniconda3/envs/oddioo-xtts/bin/python'),
        os.path.expanduser('~/anaconda3/envs/oddioo-xtts/bin/python'),
        os.path.expanduser('~/miniconda3/bin/python'),
        os.path.expanduser('~/anaconda3/bin/python'),
        'python3',
        'python',
    ]
    
    python_exe = None
    for path in python_paths:
        if os.path.exists(path) or '/' not in path:
            python_exe = path
            break
    
    if not python_exe:
        raise Exception("No suitable Python interpreter found for XTTS")
    
    cmd = [
        python_exe,
        script_path,
        text,
        output_path,
        "--speed", str(speed),
        "--language", language
    ]
    
    print(f"🎤 Generating speech with YOUR custom voice...")
    print(f"   Text: '{text[:60]}...'")
    
    # Run with longer timeout for XTTS (it can be slow on first load)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    
    if result.returncode != 0:
        print(f"Custom XTTS stderr: {result.stderr}")
        raise Exception(f"Custom XTTS generation failed: {result.stderr}")
    
    print(f"Custom XTTS output: {result.stdout}")
    
    if not os.path.exists(output_path):
        raise Exception("Custom XTTS did not create output file")


def generate_tts_xtts(text: str, voice: str, output_path: str, language: str = "en"):
    """Generate speech using VITS or Silero in separate environment"""
    import subprocess
    
    # Path to multi-engine generator script
    script_path = os.path.join(os.path.dirname(__file__), 'xtts_generator.py')
    
    # Path to TTS conda environment python
    xtts_python = os.path.expanduser('~/miniconda3/envs/oddioo-xtts/bin/python')
    
    # Check if environment exists
    if not os.path.exists(xtts_python):
        raise Exception("TTS environment not found. Run: conda create -n oddioo-xtts python=3.11")
    
    # Determine engine and speaker from voice ID
    if voice.startswith('silero_'):
        # Silero TTS (v3_en has 118 speakers: en_0 to en_117)
        engine = "silero"
        silero_speakers = {
            'silero_0': 'en_0',     # Female 1
            'silero_1': 'en_1',     # Female 2
            'silero_2': 'en_2',     # Male 1
            'silero_3': 'en_3',     # Female 3
            'silero_4': 'en_4',     # Male 2
            'silero_5': 'en_5',     # Female 4
            'silero_10': 'en_10',   # Male 3
            'silero_15': 'en_15',   # Female 5
            'silero_20': 'en_20',   # Male 4
            'silero_25': 'en_25',   # Female 6
            'silero_random': 'random',  # Random voice each time
        }
        speaker = silero_speakers.get(voice, 'en_0')
        
        cmd = [
            xtts_python,
            script_path,
            text,
            output_path,
            "--engine", "silero",
            "--speaker", speaker,
            "--language", language
        ]
        print(f"🎨 Calling Silero TTS in separate environment (speaker: {speaker})...")
        
    else:
        # VITS with specific speaker IDs
        engine = "vits"
        vits_speakers = {
            'vits_p225': 'p225',  # Female British 1
            'vits_p226': 'p226',  # Male British 1
            'vits_p227': 'p227',  # Male British 2
            'vits_p228': 'p228',  # Female British 2
            'vits_p229': 'p229',  # Female British 3
            'vits_p230': 'p230',  # Male British 3
            'vits_p231': 'p231',  # Female British 4
            'vits_p232': 'p232',  # Male British 4
            'vits_p233': 'p233',  # Female British 5
            'vits_p234': 'p234',  # Male British 5
            # Legacy support
            'xtts_female': 'p225',
            'xtts_male': 'p226',
        }
        speaker = vits_speakers.get(voice, 'p225')
        
        cmd = [
            xtts_python,
            script_path,
            text,
            output_path,
            "--engine", "vits",
            "--speaker", speaker,
            "--language", language
        ]
        print(f"🎨 Calling VITS in separate environment (speaker: {speaker})...")
    
    # Set longer timeout for generation
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    
    if result.returncode != 0:
        print(f"TTS stderr: {result.stderr}")
        raise Exception(f"TTS generation failed: {result.stderr}")
    
    print(f"TTS output: {result.stdout}")
    
    if not os.path.exists(output_path):
        raise Exception("TTS did not create output file")


def generate_coqui_tts(text: str, voice_id: str, output_path: str, speed: float = 1.0):
    """
    UNIFIED TTS GENERATION - Routes to correct engine based on voice_id.
    
    Voice ID format: {engine}_{voice_type}
    Examples: edge_male, piper_female, coqui_female
    
    Args:
        text: Text to synthesize
        voice_id: Voice identifier (edge_male, piper_female, etc.)
        output_path: Where to save the audio file
        speed: Speech speed multiplier
    """
    import asyncio
    
    # Determine which engine to use based on voice_id prefix
    
    # Custom XTTS voice (Grant's voice) - highest priority
    if voice_id.startswith('xtts_grant') or voice_id.startswith('xtts_custom') or voice_id == 'grant':
        if tts_engines.get('xtts_custom'):
            try:
                generate_tts_xtts_custom(text, output_path, speed)
                return
            except Exception as e:
                print(f"⚠️ Custom XTTS failed: {e}")
                # Fall through to Edge TTS as backup
                print("   Falling back to Edge TTS...")
        else:
            print("⚠️ Custom XTTS voice not available, falling back to Edge TTS")
    
    if voice_id.startswith('edge_'):
        # Edge TTS voices
        if tts_engines.get('edge'):
            try:
                # Convert speed multiplier to rate percentage for Edge TTS
                rate_value = int((speed - 1.0) * 100)
                rate = f"{rate_value:+d}%" if rate_value != 0 else "+0%"
                asyncio.run(generate_tts_edge(text, voice_id, output_path, rate))
                return
            except Exception as e:
                print(f"⚠️ Edge TTS failed: {e}")
                raise Exception(f"Edge TTS generation failed: {str(e)}")
        else:
            raise Exception("Edge TTS not available")
    
    # Piper TTS - REMOVED (not working)
    # elif voice_id.startswith('piper_'):
    #     if tts_engines.get('piper'):
    #         try:
    #             generate_tts_piper(text, voice_id, output_path)
    #             return
    #         except Exception as e:
    #             print(f"⚠️ Piper TTS failed: {e}")
    #             raise Exception(f"Piper TTS generation failed: {str(e)}")
    #     else:
    #         raise Exception("Piper TTS not available")
    
    # VITS and Silero - REMOVED (not good quality)
    # elif voice_id.startswith('xtts_') or voice_id.startswith('vits_'):
    #     if tts_engines.get('coqui_xtts'):
    #         try:
    #             generate_tts_xtts(text, voice_id, output_path)
    #             return
    #         except Exception as e:
    #             print(f"⚠️ VITS TTS failed: {e}")
    #             raise Exception(f"VITS generation failed: {str(e)}")
    #     else:
    #         raise Exception("VITS not available")
    # 
    # elif voice_id.startswith('silero_'):
    #     if tts_engines.get('silero'):
    #         try:
    #             generate_tts_xtts(text, voice_id, output_path)
    #             return
    #         except Exception as e:
    #             print(f"⚠️ Silero TTS failed: {e}")
    #             raise Exception(f"Silero TTS generation failed: {str(e)}")
    #     else:
    #         raise Exception("Silero TTS not available")
    
    else:
        # Legacy support - no prefix, default to Edge TTS
        print(f"⚠️ Legacy voice_id '{voice_id}' detected, defaulting to Edge TTS")
        if tts_engines.get('edge'):
            try:
                # Convert speed multiplier to rate percentage for Edge TTS
                rate_value = int((speed - 1.0) * 100)
                rate = f"{rate_value:+d}%" if rate_value != 0 else "+0%"
                asyncio.run(generate_tts_edge(text, voice_id, output_path, rate))
                return
            except Exception as e:
                print(f"⚠️ Edge TTS failed: {e}")
        
        raise Exception("No TTS engine available for this voice")

def detect_page_numbers_for_removal(text: str) -> List[Tuple[int, int]]:
    """
    Detect standalone page numbers in text for removal in TTS.
    Returns list of (start_pos, end_pos) tuples to remove.
    Only detects isolated numbers, not numbers within sentences.
    """
    page_number_positions = []
    
    # Pattern 1: Numbers at end of text preceded/followed by lots of whitespace
    # e.g., "...text.\n\n   42   \n\n"
    for match in re.finditer(r'\n\s{2,}(\d{1,3})\s{2,}\n', text):
        page_number_positions.append((match.start(), match.end()))
    
    # Pattern 2: Standalone numbers on their own line at end of paragraphs
    # e.g., "sentence.\n42\n"  
    for match in re.finditer(r'([.!?])\s*\n+\s*(\d{1,3})\s*\n', text):
        # Only capture the number part, keep the sentence ending
        num_start = match.start() + len(match.group(1))
        page_number_positions.append((num_start, match.end()))
    
    return page_number_positions

def insert_spaces_in_camelcase(text: str) -> str:
    """
    Aggressively add spaces to text where words are joined together.
    Example: "providedby" -> "provided by", "thepersonal" -> "the personal"
    """
    # Insert space before capital letter: "ReportContains" -> "Report Contains"
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    # AGGRESSIVE: Insert spaces before common small words that got joined
    common_words = [
        'the', 'and', 'for', 'are', 'was', 'not', 'has', 'can', 'all', 'may', 'any', 
        'our', 'one', 'two', 'new', 'own', 'use', 'see', 'get', 'its', 'now', 'way',
        'from', 'this', 'that', 'have', 'will', 'been', 'were', 'with', 'they', 'your',
        'more', 'when', 'time', 'than', 'each', 'also', 'both', 'only', 'such', 'into',
        'some', 'make', 'does', 'part', 'made', 'like', 'over', 'year', 'work', 'back',
        'after', 'being', 'other', 'their', 'these', 'there', 'which', 'would', 'could',
        'should', 'about', 'author', 'report', 'public', 'herein', 'shall', 'advice',
        'offers', 'securities', 'company', 'payment', 'material', 'trading', 'covered',
        'period', 'hours', 'before', 'ending', 'release', 'factual', 'believed', 'accurate',
        'writing', 'completeness', 'accuracy', 'recipients', 'conduct', 'diligence', 
        'performance', 'indicative', 'future', 'results'
    ]
    
    # Build pattern to insert space before these words if preceded by letters
    for word in common_words:
        # Insert space before word if it's preceded by lowercase letters
        text = re.sub(rf'([a-z])({word})\b', r'\1 \2', text, flags=re.IGNORECASE)
    
    # Also handle common word endings + starts of new words
    endings = ['ed', 'ly', 'ing', 'ion', 'tion', 'sion', 'ness', 'ment', 'able', 'ible', 'ful', 'less', 'ous', 'ive', 'ist', 'ism']
    starts = ['for', 'the', 'and', 'to', 'of', 'in', 'on', 'at', 'by', 'is', 'it', 'or', 'as', 'be', 'an', 'if', 'no', 'do']
    
    for end in endings:
        for start in starts:
            text = re.sub(rf'\b(\w+{end})({start})\b', r'\1 \2', text, flags=re.IGNORECASE)
    
    return text

def clean_text_basic(text: str) -> str:
    """
    Minimal PDF-specific cleaning (used as fallback for OCR post-processing).
    Most cleaning is now done by professional libraries (ftfy + cleantext).
    """
    if not text:
        return ""
    
    # Fix hyphenated words at line breaks (PDF-specific)
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # Fix broken hyphenated phrases
    text = re.sub(r'\s+-\s*(\w+)', r' \1', text)
    text = re.sub(r'(\w+)\s*-\s+', r'\1 ', text)
    
    # Normalize whitespace
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    
    return text.strip()

def ai_clean_text_for_display(text: str) -> str:
    """
    Clean text for display using professional libraries.
    Keeps all meaningful content, fixes encoding issues.
    """
    if not text:
        return ""
    
    # Step 1: Fix unicode and encoding issues with ftfy
    text = ftfy.fix_text(text)
    
    # Step 2: Use cleantext library for comprehensive cleaning
    text = clean(
        text,
        fix_unicode=True,  # Fix unicode errors
        to_ascii=False,  # Keep unicode characters (don't force ASCII)
        lower=False,  # Preserve case
        no_line_breaks=False,  # Keep line breaks
        no_urls=False,  # Keep URLs if any
        no_emails=False,  # Keep emails
        no_phone_numbers=False,  # Keep phone numbers
        no_numbers=False,  # Keep numbers (important for documents!)
        no_digits=False,  # Keep digits
        no_currency_symbols=False,  # Keep $ symbols
        no_punct=False,  # Keep punctuation
        replace_with_punct="",
        replace_with_url="",
        replace_with_email="",
        replace_with_phone_number="",
        replace_with_number="",
        replace_with_digit="",
        replace_with_currency_symbol="",
        lang="en"
    )
    
    # Step 3: Minimal post-processing for PDF artifacts
    # Remove excessive whitespace
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def ai_clean_text_for_tts(text: str) -> str:
    """
    Clean text specifically for TTS using professional libraries.
    More aggressive cleaning for better speech synthesis.
    """
    if not text:
        return ""
    
    # Step 1: Fix unicode and encoding with ftfy
    text = ftfy.fix_text(text)
    
    # Step 2: Use cleantext library with TTS-optimized settings
    text = clean(
        text,
        fix_unicode=True,  # Fix unicode errors
        to_ascii=False,  # Keep unicode for proper pronunciation
        lower=False,  # Preserve case (helps TTS with proper nouns)
        no_line_breaks=False,  # Keep some structure
        no_urls=True,  # Remove URLs (sound bad in TTS) ✅
        no_emails=True,  # Remove emails (sound bad) ✅
        no_phone_numbers=True,  # Remove phone numbers (sound bad) ✅
        no_numbers=False,  # Keep numbers (TTS reads them well)
        no_digits=False,  # Keep digits
        no_currency_symbols=False,  # Keep $ (TTS says "dollars")
        no_punct=False,  # Keep punctuation (helps TTS pausing)
        no_emoji=True,  # Remove emojis (TTS can't read them) ✅
        replace_with_url="",
        replace_with_email="",
        replace_with_phone_number="",
        lang="en"
    )
    
    # Step 3: TTS-specific post-processing (minimal)
    # Remove page numbers at start of lines
    text = re.sub(r'^\s*\d{1,4}\s+(?=[A-Z])', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'Page\s+\d+', '', text, flags=re.IGNORECASE)
    
    # Remove footnote markers (sound bad)
    text = re.sub(r'\[\d+\]', '', text)  # [1], [2]
    text = re.sub(r'([a-z])(\d{1,2})\s*$', r'\1', text, flags=re.MULTILINE)  # word12 at end
    
    # Normalize whitespace
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def clean_text(text):
    """Clean and normalize text from PDF extraction."""
    if not text:
        return ""
    
    # Remove form feed and other control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    
    # Fix hyphenated words at line breaks (e.g., "exam-\nple" -> "example")
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # Fix words split across lines without hyphen (common in PDFs)
    # Be careful to preserve intentional line breaks (like after periods)
    text = re.sub(r'(\w+)\s*\n\s*(\w)', lambda m: m.group(1) + (' ' if m.group(1)[-1].islower() else '\n') + m.group(2), text)
    
    # Replace multiple newlines with double newline (paragraph breaks)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Replace single newlines with space (joining lines within paragraph)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r' {2,}', ' ', text)
    
    # Remove spaces before punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    
    # Remove common OCR artifacts and special characters that cause pauses
    text = re.sub(r'[•●○■□▪▫◦⦿⦾]', '', text)  # Bullet points
    text = re.sub(r'[\u200b-\u200f\u202a-\u202e]', '', text)  # Zero-width chars
    
    # Fix common OCR errors
    text = text.replace('ﬁ', 'fi')  # ligature fi
    text = text.replace('ﬂ', 'fl')  # ligature fl
    text = text.replace('ﬀ', 'ff')  # ligature ff
    text = text.replace('ﬃ', 'ffi') # ligature ffi
    text = text.replace('ﬄ', 'ffl') # ligature ffl
    text = text.replace(''', "'")    # smart quote
    text = text.replace(''', "'")    # smart quote
    text = text.replace('"', '"')    # smart quote
    text = text.replace('"', '"')    # smart quote
    text = text.replace('–', '-')    # en dash
    text = text.replace('—', '-')    # em dash
    text = text.replace('…', '...')  # ellipsis
    
    # Remove page numbers at start or end of text (common pattern)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-–—]\s*\d+\s*[-–—]\s*$', '', text, flags=re.MULTILINE)
    
    # Remove headers/footers that are all caps and short
    text = re.sub(r'^[A-Z\s]{3,30}$', '', text, flags=re.MULTILINE)
    
    # Final cleanup
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Max 2 newlines
    text = text.strip()
    
    return text

def split_into_sentences(text):
    """Split text into sentences using improved regex."""
    # Clean the text first
    text = clean_text(text)
    
    if not text:
        return []
    
    # Split on sentence boundaries but keep abbreviations intact
    # This regex looks for periods, exclamation marks, or question marks
    # followed by whitespace and a capital letter (or end of string)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$', text)
    
    # Clean and filter sentences
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        # Skip very short sentences (likely artifacts) or page numbers
        if len(sentence) > 3 and not sentence.isdigit():
            # Remove any remaining excessive whitespace
            sentence = re.sub(r'\s+', ' ', sentence)
            cleaned_sentences.append(sentence)
    
    return cleaned_sentences

def split_into_sentences_simple(text: str) -> List[str]:
    """
    Split text into sentences using simple, reliable regex.
    Preserves all content without cutting off.
    """
    if not text:
        return []
    
    # Split on sentence boundaries (., !, ?) followed by space and capital letter
    # or end of string
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    # Clean and filter
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        # Skip very short fragments (likely artifacts) but keep everything else
        if len(sentence) > 2:
            cleaned_sentences.append(sentence)
    
    return cleaned_sentences

def extract_with_forced_ocr(pdf_path):
    """
    Force PURE OCR by converting PDF to images and ignoring text layer completely.
    This "sees" the text visually which should preserve proper spacing.
    Uses Tesseract OCR on high-resolution images.
    """
    pages = []
    
    try:
        print("🔍 Extracting with FORCED OCR (ignoring broken text layer)...")
        print("   Converting PDF to images for visual text recognition...")
        
        # Convert PDF pages to high-resolution images
        images = convert_from_path(pdf_path, dpi=300, fmt='jpeg', thread_count=4)
        print(f"   Converted to {len(images)} images, running OCR...")
        
        # Run OCR on each image
        for page_idx, image in enumerate(images):
            print(f"   OCR page {page_idx + 1}/{len(images)}...", end=' ')
            
            # Run Tesseract OCR with optimized config
            custom_config = r'--oem 3 --psm 1'  # LSTM OCR, Automatic page segmentation
            raw_text = pytesseract.image_to_string(image, config=custom_config, lang='eng')
            
            if raw_text and len(raw_text.strip()) > 50:
                print("✓")
                # Clean for display and TTS
                display_text = ai_clean_text_for_display(raw_text)
                tts_text = ai_clean_text_for_tts(raw_text)
                
                if display_text or tts_text:
                    sentences = split_into_sentences_simple(tts_text if tts_text else display_text)
                    pages.append({
                        "page_number": page_idx + 1,
                        "text": display_text,
                        "tts_text": tts_text,
                        "sentences": sentences,
                    })
            else:
                print("(empty)")
        
        if pages:
            print(f"✅ Forced OCR extracted {len(pages)} pages visually!")
            print(f"   Text reconstructed from images - spacing should be preserved.")
            return pages
            
    except Exception as e:
        print(f"⚠️ Forced OCR failed: {e}")
        import traceback
        traceback.print_exc()

        return []

def extract_with_pypdfium2(pdf_path):
    """
    Extract text using pypdfium2 - known for excellent text extraction quality.
    This library often produces the best results for PDFs with encoding issues.
    """
    pages = []
    
    try:
        print("🔍 Extracting with pypdfium2 (highest quality method)...")
        pdf = pdfium.PdfDocument(pdf_path)
        
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            textpage = page.get_textpage()
            
            # Get text with proper spacing
            raw_text = textpage.get_text_range()
            
            if raw_text and len(raw_text.strip()) > 50:
                # Clean for display and TTS
                display_text = ai_clean_text_for_display(raw_text)
                tts_text = ai_clean_text_for_tts(raw_text)
                
                if display_text or tts_text:
                    sentences = split_into_sentences_simple(tts_text if tts_text else display_text)
                    pages.append({
                        "page_number": page_num + 1,
                        "text": display_text,
                        "tts_text": tts_text,
                        "sentences": sentences,
                    })
        
        pdf.close()
        
        if pages:
            print(f"✅ pypdfium2 extracted {len(pages)} pages successfully")
            return pages
            
    except Exception as e:
        print(f"⚠️ pypdfium2 extraction failed: {e}")
    
    return []

def extract_with_pymupdf_blocks(pdf_path):
    """
    Extract text using PyMuPDF's 'blocks' mode which preserves layout better.
    """
    pages = []
    
    try:
        print("🔍 Extracting with PyMuPDF blocks mode...")
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Get blocks - this preserves layout and spacing much better
            blocks = page.get_text("blocks")
            
            # Reconstruct text from blocks
            # blocks format: (x0, y0, x1, y1, "text", block_no, block_type)
            # Sort by vertical position (y0) then horizontal (x0)
            blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
            
            # Join blocks with proper spacing
            text_parts = []
            for block in blocks:
                block_text = block[4]  # Text content
                if block_text.strip():
                    text_parts.append(block_text.strip())
            
            raw_text = '\n'.join(text_parts)
            
            if raw_text and len(raw_text.strip()) > 50:
                # Clean for display and TTS
                display_text = ai_clean_text_for_display(raw_text)
                tts_text = ai_clean_text_for_tts(raw_text)
                
                if display_text or tts_text:
                    sentences = split_into_sentences_simple(tts_text if tts_text else display_text)
                    pages.append({
                        "page_number": page_num + 1,
                        "text": display_text,
                        "tts_text": tts_text,
                        "sentences": sentences,
                    })
        
        doc.close()
        
        if pages:
            print(f"✅ PyMuPDF blocks extracted {len(pages)} pages successfully")
        return pages
            
    except Exception as e:
        print(f"⚠️ PyMuPDF blocks extraction failed: {e}")
    
    return []

def extract_text_from_pdf(pdf_path):
    """
    PDF extraction with FORCED OCR to fix badly-encoded PDFs.
    
    Strategy:
    1. FORCED OCR (Tesseract on images) - Ignores broken text layer, "sees" text visually
    2. pypdfium2 (Chrome's engine) - Fast fallback for well-encoded PDFs  
    3. PyMuPDF blocks - Last resort
    """
    pages: List[Dict] = []
    
    # PRIMARY: Force OCR - converts to images, runs visual OCR, ignores text layer
    print("🎯 Using FORCED OCR as primary method to fix spacing issues...")
    pages = extract_with_forced_ocr(pdf_path)
    if pages:
        return pages
    
    # BACKUP 1: Try pypdfium2 - fast and good for well-encoded PDFs
    print("⚠️ Forced OCR failed, trying pypdfium2...")
    pages = extract_with_pypdfium2(pdf_path)
    if pages:
        return pages
    
    # BACKUP 2: Try PyMuPDF with blocks mode
    print("⚠️ pypdfium2 failed, trying PyMuPDF blocks...")
    pages = extract_with_pymupdf_blocks(pdf_path)
    if pages:
        return pages
    
    print("❌ All extraction methods failed")
    return []


@app.route('/api/upload', methods=['POST'])
def upload_pdf():
    """Upload and parse a PDF file."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if not file.filename or file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are allowed'}), 400
    
    filename = secure_filename(file.filename)
    doc_id = str(uuid.uuid4())
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{doc_id}_{filename}")
    
    file.save(filepath)
    
    try:
        pages = extract_text_from_pdf(filepath)
        
        documents[doc_id] = {
            'id': doc_id,
            'filename': filename,
            'filepath': filepath,
            'pages': pages,
            'total_pages': len(pages)
        }
        
        return jsonify({
            'document_id': doc_id,
            'filename': filename,
            'total_pages': len(pages),
            'pages': pages
        }), 200
        
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Error processing PDF: {str(e)}'}), 500

@app.route('/api/documents/<doc_id>', methods=['GET'])
def get_document(doc_id):
    """Retrieve a parsed document."""
    if doc_id not in documents:
        return jsonify({'error': 'Document not found'}), 404
    
    doc = documents[doc_id]
    return jsonify({
        'document_id': doc['id'],
        'filename': doc['filename'],
        'total_pages': doc['total_pages'],
        'pages': doc['pages']
    }), 200

@app.route('/api/tts', methods=['POST'])
def text_to_speech():
    """Convert text to speech using Coqui XTTS v2 and return audio file."""
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    voice_name = data.get('voice_name', 'female_1')
    speed_setting = data.get('speed', 200)  # 150-250 range from UI
    
    if not text or not text.strip():
        return jsonify({'error': 'Text cannot be empty'}), 400
    
    # Check if any TTS engine is available
    if not any(tts_engines.values()):
        return jsonify({'error': 'No TTS engines available. Please restart the server.'}), 500
    
    # Convert speed from UI range (150-250) to Coqui speed (0.5-1.5)
    # 150 -> 0.75, 200 -> 1.0, 250 -> 1.25
    speed = 0.5 + ((speed_setting - 150) / 100) * 0.75
    speed = max(0.5, min(1.5, speed))  # Clamp between 0.5 and 1.5
    
    audio_id = str(uuid.uuid4())
    wav_path = os.path.join(app.config['AUDIO_FOLDER'], f"{audio_id}.wav")
    
    try:
        with tts_lock:
            # Generate speech with Coqui TTS
            generate_coqui_tts(
                text=text,
                voice_id=voice_name,
                output_path=wav_path,
                speed=speed
            )
            
            if not os.path.exists(wav_path):
                    return jsonify({'error': 'Audio file was not created'}), 500
        
        @after_this_request
        def cleanup(response):
            try:
                if os.path.exists(wav_path):
                    os.remove(wav_path)
            except Exception:
                pass
            return response
        
        return send_file(wav_path, mimetype='audio/wav')
        
    except Exception as e:
        if os.path.exists(wav_path):
            os.remove(wav_path)
        return jsonify({'error': f'Error generating speech: {str(e)}'}), 500

@app.route('/api/voices', methods=['GET'])
def get_voices():
    """Get available TTS voices from all engines."""
    try:
        voice_list = []
        
        # Custom XTTS Voice (Grant's Voice) - LOCAL, highest quality
        if tts_engines.get('xtts_custom'):
            voice_list.extend([
                {
                    'id': 'xtts_grant',
                    'name': '⭐ Grant\'s Voice (Custom) - XTTS v2',
                    'engine': 'xtts_custom',
                    'languages': ['en'],
                    'premium': False,
                    'description': 'YOUR cloned voice - High quality, runs locally, free!'
                },
            ])
        
        # Edge TTS voices (Microsoft - free, fast, best quality)
        # ONLY UK voices - US/AU voices removed (not good quality)
        if tts_engines.get('edge'):
            voice_list.extend([
                {
                    'id': 'edge_female_uk',
                    'name': '🎯 Sonia (UK Female) - Edge TTS',
                    'engine': 'edge',
                    'languages': ['en-GB'],
                    'premium': False,
                    'description': 'British English - High quality, fast, free (RECOMMENDED)'
                },
                {
                    'id': 'edge_male_uk',
                    'name': '🎯 Ryan (UK Male) - Edge TTS',
                    'engine': 'edge',
                    'languages': ['en-GB'],
                    'premium': False,
                    'description': 'British English male - High quality, fast, free'
                },
            ])
        
        # Piper TTS - REMOVED (not working properly)
        
        # VITS and Silero - REMOVED (not good quality, only keeping Sonia/Ryan)
        # Coqui VITS voices (high quality, 109 speakers from VCTK dataset!)
        if False and tts_engines.get('coqui_xtts'):
            voice_list.extend([
                # Female VITS speakers
                {
                    'id': 'vits_p225',
                    'name': '🎨 VITS P225 (Female UK 1) - High Quality',
                    'engine': 'vits',
                'languages': ['en'],
                    'premium': False,
                    'description': 'British female, clear and articulate'
                },
                {
                    'id': 'vits_p228',
                    'name': '🎨 VITS P228 (Female UK 2) - High Quality',
                    'engine': 'vits',
                    'languages': ['en'],
                    'premium': False,
                    'description': 'British female, warm tone'
                },
                {
                    'id': 'vits_p229',
                    'name': '🎨 VITS P229 (Female UK 3) - High Quality',
                    'engine': 'vits',
                    'languages': ['en'],
                    'premium': False,
                    'description': 'British female, professional'
                },
                {
                    'id': 'vits_p231',
                    'name': '🎨 VITS P231 (Female UK 4) - High Quality',
                    'engine': 'vits',
                    'languages': ['en'],
                    'premium': False,
                    'description': 'British female, expressive'
                },
                {
                    'id': 'vits_p233',
                    'name': '🎨 VITS P233 (Female UK 5) - High Quality',
                    'engine': 'vits',
                    'languages': ['en'],
                    'premium': False,
                    'description': 'British female, soft spoken'
                },
                # Male VITS speakers
                {
                    'id': 'vits_p226',
                    'name': '🎨 VITS P226 (Male UK 1) - High Quality',
                    'engine': 'vits',
                    'languages': ['en'],
                    'premium': False,
                    'description': 'British male, deep voice'
                },
                {
                    'id': 'vits_p227',
                    'name': '🎨 VITS P227 (Male UK 2) - High Quality',
                    'engine': 'vits',
                    'languages': ['en'],
                    'premium': False,
                    'description': 'British male, narrator style'
                },
                {
                    'id': 'vits_p230',
                    'name': '🎨 VITS P230 (Male UK 3) - High Quality',
                    'engine': 'vits',
                    'languages': ['en'],
                    'premium': False,
                    'description': 'British male, authoritative'
                },
                {
                    'id': 'vits_p232',
                    'name': '🎨 VITS P232 (Male UK 4) - High Quality',
                    'engine': 'vits',
                    'languages': ['en'],
                    'premium': False,
                    'description': 'British male, friendly'
                },
                {
                    'id': 'vits_p234',
                    'name': '🎨 VITS P234 (Male UK 5) - High Quality',
                    'engine': 'vits',
                    'languages': ['en'],
                    'premium': False,
                    'description': 'British male, professional'
                },
                # Legacy support
                {
                    'id': 'xtts_female',
                    'name': '🎨 VITS Female (Default) - High Quality',
                    'engine': 'vits',
                    'languages': ['en'],
                    'premium': False,
                    'description': 'High-quality (4/5), 100% free, commercial OK, fast'
                },
                {
                    'id': 'xtts_male',
                    'name': '🎨 VITS Male (Default) - High Quality',
                    'engine': 'vits',
                    'languages': ['en'],
                    'premium': False,
                    'description': 'High-quality male voice, natural prosody, commercial OK'
                },
            ])
        
        # Silero TTS voices (MIT license, very fast, 118 voices!) - REMOVED
        if False and tts_engines.get('silero'):
            voice_list.extend([
                {
                    'id': 'silero_0',
                    'name': '⚡ Silero #0 (Female) - Ultra Fast',
                    'engine': 'silero',
                    'languages': ['en'],
                    'premium': False,
                    'description': 'MIT license, very fast (~0.5s), 118 voices total!'
                },
                {
                    'id': 'silero_1',
                    'name': '⚡ Silero #1 (Female 2) - Ultra Fast',
                    'engine': 'silero',
                    'languages': ['en'],
                    'premium': False,
                    'description': 'MIT license, different female voice'
                },
                {
                    'id': 'silero_2',
                    'name': '⚡ Silero #2 (Male) - Ultra Fast',
                    'engine': 'silero',
                    'languages': ['en'],
                    'premium': False,
                    'description': 'MIT license, male voice'
                },
                {
                    'id': 'silero_3',
                    'name': '⚡ Silero #3 (Female 3) - Ultra Fast',
                    'engine': 'silero',
                    'languages': ['en'],
                    'premium': False,
                    'description': 'MIT license, third female option'
                },
                {
                    'id': 'silero_4',
                    'name': '⚡ Silero #4 (Male 2) - Ultra Fast',
                    'engine': 'silero',
                    'languages': ['en'],
                    'premium': False,
                    'description': 'MIT license, second male voice'
                },
                {
                    'id': 'silero_5',
                    'name': '⚡ Silero #5 (Female 4) - Ultra Fast',
                    'engine': 'silero',
                    'languages': ['en'],
                    'premium': False,
                    'description': 'MIT license, fourth female voice'
                },
                {
                    'id': 'silero_10',
                    'name': '⚡ Silero #10 (Male 3) - Ultra Fast',
                    'engine': 'silero',
                    'languages': ['en'],
                    'premium': False,
                    'description': 'MIT license, third male voice'
                },
                {
                    'id': 'silero_15',
                    'name': '⚡ Silero #15 (Female 5) - Ultra Fast',
                    'engine': 'silero',
                    'languages': ['en'],
                    'premium': False,
                    'description': 'MIT license, fifth female voice'
                },
                {
                    'id': 'silero_20',
                    'name': '⚡ Silero #20 (Male 4) - Ultra Fast',
                    'engine': 'silero',
                    'languages': ['en'],
                    'premium': False,
                    'description': 'MIT license, fourth male voice'
                },
                {
                    'id': 'silero_25',
                    'name': '⚡ Silero #25 (Female 6) - Ultra Fast',
                    'engine': 'silero',
                    'languages': ['en'],
                    'premium': False,
                    'description': 'MIT license, sixth female voice'
                },
                {
                    'id': 'silero_random',
                    'name': '🎲 Silero Random (Surprise!) - Ultra Fast',
                    'engine': 'silero',
                    'languages': ['en'],
                    'premium': False,
                    'description': 'MIT license, picks a random voice each time!'
                },
            ])
        
        
        # Add engine status info
        engine_status = {
            'edge': bool(tts_engines.get('edge')),
            'xtts_custom': bool(tts_engines.get('xtts_custom')),
        }
        
        # Set default engine - prefer custom voice if available
        default_engine = 'xtts_custom' if tts_engines.get('xtts_custom') else 'edge'
        
        return jsonify({
            'voices': voice_list,
            'engines': engine_status,
            'default_engine': default_engine,
            'free_commercial_engines': ['xtts_custom', 'edge'],  # 100% free
            'best_quality_free': 'xtts_custom' if tts_engines.get('xtts_custom') else 'edge'
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Error fetching voices: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'}), 200

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    """Serve frontend static files (for Railway deployment)."""
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_file(os.path.join(app.static_folder, path))
    else:
        return send_file(os.path.join(app.static_folder, 'index.html'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
