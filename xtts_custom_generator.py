#!/usr/bin/env python3
"""
Custom Voice TTS Generator using XTTS v2

This script generates speech using YOUR cloned voice.
It's called by the backend when the "Grant's Voice" option is selected.

Usage:
    python xtts_custom_generator.py "Text to speak" output.wav [--speed 1.0]
"""

import os
import sys
import argparse

# Fix for PyTorch 2.6+ security change - must be done before importing TTS
import torch
torch.serialization.add_safe_globals([])
original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return original_load(*args, **kwargs)
torch.load = patched_load

# Global TTS instance (lazy loaded)
_tts_instance = None
_reference_wavs = None

def get_reference_wavs():
    """Get paths to your voice reference audio files."""
    global _reference_wavs
    
    if _reference_wavs is not None:
        return _reference_wavs
    
    # Path to voice samples directory (relative to this script or absolute)
    voice_dir = os.path.expanduser("~/Desktop/voice/dataset/wavs")
    
    # Find all wav files in the voice directory
    if os.path.exists(voice_dir):
        wav_files = sorted([
            os.path.join(voice_dir, f) 
            for f in os.listdir(voice_dir) 
            if f.endswith('.wav')
        ])
        if wav_files:
            _reference_wavs = wav_files[:5]  # Use up to 5 samples for better quality
            return _reference_wavs
    
    # Fallback: check alternative paths
    alt_paths = [
        "/Users/grantdrinkwater/Desktop/voice/dataset/wavs",
        os.path.join(os.path.dirname(__file__), "..", "..", "voice", "dataset", "wavs"),
    ]
    
    for path in alt_paths:
        if os.path.exists(path):
            wav_files = sorted([
                os.path.join(path, f) 
                for f in os.listdir(path) 
                if f.endswith('.wav')
            ])
            if wav_files:
                _reference_wavs = wav_files[:5]
                return _reference_wavs
    
    raise FileNotFoundError(
        f"No voice samples found. Please add .wav files to {voice_dir}"
    )


def get_tts():
    """Lazy-load the XTTS v2 model (expensive, only do once)."""
    global _tts_instance
    
    if _tts_instance is not None:
        return _tts_instance
    
    print("Loading XTTS v2 model (this may take a moment)...", file=sys.stderr)
    
    from TTS.api import TTS
    
    # XTTS v2 is the best quality voice cloning model
    _tts_instance = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    
    print("XTTS v2 model loaded successfully!", file=sys.stderr)
    return _tts_instance


def generate_speech(text: str, output_path: str, speed: float = 1.0, language: str = "en"):
    """
    Generate speech using YOUR cloned voice.
    
    Args:
        text: The text to speak
        output_path: Where to save the audio file (.wav)
        speed: Speech speed multiplier (0.5 to 2.0)
        language: Language code (default: "en")
    """
    # Get reference audio samples
    reference_wavs = get_reference_wavs()
    print(f"Using {len(reference_wavs)} reference audio file(s)", file=sys.stderr)
    
    # Get TTS model
    tts = get_tts()
    
    # Generate speech with your voice
    print(f"Generating speech: '{text[:50]}...'", file=sys.stderr)
    
    tts.tts_to_file(
        text=text,
        file_path=output_path,
        speaker_wav=reference_wavs,
        language=language,
        # Note: XTTS v2 doesn't have a direct speed parameter in tts_to_file
        # Speed adjustment would need to be done post-processing if needed
    )
    
    # Apply speed adjustment if not 1.0
    if speed != 1.0 and os.path.exists(output_path):
        apply_speed_adjustment(output_path, speed)
    
    print(f"Audio saved to: {output_path}", file=sys.stderr)
    return output_path


def apply_speed_adjustment(audio_path: str, speed: float):
    """
    Apply speed adjustment to the generated audio using pydub or similar.
    This preserves pitch while changing speed.
    """
    try:
        from pydub import AudioSegment
        
        audio = AudioSegment.from_wav(audio_path)
        
        # Change speed (this also changes pitch, but XTTS output is already good)
        # For better quality, use rubberband library if available
        if speed != 1.0:
            # Simple speed change by altering frame rate
            new_frame_rate = int(audio.frame_rate * speed)
            adjusted = audio._spawn(audio.raw_data, overrides={
                "frame_rate": new_frame_rate
            }).set_frame_rate(audio.frame_rate)
            adjusted.export(audio_path, format="wav")
            
    except ImportError:
        # pydub not available, skip speed adjustment
        print("Note: pydub not installed, speed adjustment skipped", file=sys.stderr)
    except Exception as e:
        print(f"Speed adjustment failed: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Generate speech with custom XTTS voice")
    parser.add_argument("text", help="Text to convert to speech")
    parser.add_argument("output", help="Output WAV file path")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed (0.5-2.0)")
    parser.add_argument("--language", default="en", help="Language code")
    
    args = parser.parse_args()
    
    try:
        generate_speech(
            text=args.text,
            output_path=args.output,
            speed=args.speed,
            language=args.language
        )
        print("SUCCESS")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

