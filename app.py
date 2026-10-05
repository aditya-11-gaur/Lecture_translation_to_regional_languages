"""
NPTEL Lecture Translation Pipeline – Streamlit GUI
───────────────────────────────────────────────────
Launch with:  streamlit run app.py
"""

import os
import sys
import time
import shutil
import tempfile
import logging
import streamlit as st

# ── Make project root importable ───────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import setup_logging

# Initialise logging once (before any pipeline imports)
setup_logging(level=logging.INFO)

from config.settings import (
    OUTPUT_DIR,
    TARGET_LANGUAGES,
    ALL_LANGUAGES,
    ENGINE_REGISTRY,
    get_available_engines,
    get_default_engine,
    DEFAULT_TARGET_LANGS,
)
from src.tts_generator import (
    is_tts_available,
    get_available_tts_engines,
    get_default_tts_engine,
    TTS_ENGINE_REGISTRY,
)
from src.pipeline import run_pipeline
from src.video_muxer import create_preview_mp4
from src.voice_converter import is_openvoice_available
from src.audio_separator import is_demucs_available
from src.lip_sync import is_wav2lip_available, is_wav2lip_repo_present

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Page config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="NPTEL Lecture Translator",
    page_icon="🎓",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-title  { font-size: 2.2rem; font-weight: 700; margin-bottom: 0; }
    .sub-title   { color: #888; font-size: 1rem; margin-top: 0; }
    .step-header { font-size: 1.1rem; font-weight: 600; }
    .success-box { padding: 1rem; background: #d4edda; border-radius: .5rem;
                   border: 1px solid #c3e6cb; margin: .5rem 0; }
    .info-box    { padding: 1rem; background: #d1ecf1; border-radius: .5rem;
                   border: 1px solid #bee5eb; margin: .5rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Header
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown('<p class="main-title">🎓 NPTEL Lecture Translator</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Translate English lectures into Hindi, Telugu &amp; Odia — with subtitles &amp; dubbed audio</p>',
    unsafe_allow_html=True,
)
st.divider()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sidebar – settings
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.header("⚙️ Settings")

    # ── Detect which API keys are configured ────────────────
    available_stt_engines = get_available_engines(capability="stt")
    available_translate_engines = get_available_engines(capability="translate")

    if not available_stt_engines:
        st.error("No STT engines available! Add an API key or install faster-whisper.")
        st.stop()
    if not available_translate_engines:
        st.error("No translation engines available! Add at least one API key to your `.env` file.")
        st.stop()

    def _engine_label(eid: str) -> str:
        return ENGINE_REGISTRY[eid]["name"]

    default_stt = get_default_engine(capability="stt")
    default_translate = get_default_engine(capability="translate")
    stt_idx = available_stt_engines.index(default_stt) if default_stt in available_stt_engines else 0
    translate_idx = available_translate_engines.index(default_translate) if default_translate in available_translate_engines else 0

    stt_method = st.selectbox(
        "Speech-to-Text engine",
        options=available_stt_engines,
        index=stt_idx,
        format_func=_engine_label,
        help="Whisper (Local) gives the most accurate timestamps",
    )

    translate_method = st.selectbox(
        "Translation engine",
        options=available_translate_engines,
        index=translate_idx,
        format_func=_engine_label,
        help="Only engines with a configured API key are shown",
    )

    target_langs = st.multiselect(
        "Target languages",
        options=list(TARGET_LANGUAGES.keys()),
        default=list(DEFAULT_TARGET_LANGS),
        format_func=lambda k: f"{TARGET_LANGUAGES[k]['name']} ({k})",
        help="Select which languages to translate into",
    )

    # ── Advanced Settings ─────────────────────────────
    with st.expander("🛠️ Advanced Audio Settings", expanded=False):
        enable_glossary = st.checkbox(
            "Enable Domain-Specific Glossary",
            value=True,
            help="Bypass translation for technical terms to keep them in English (Code-Mixing).",
        )
        
        enable_prosody = st.checkbox(
            "Enable Praat Prosody Transfer",
            value=False,
            help="Experimental: Match the professor's exact pitch and intonation using Praat.",
        )
        
        enable_enhancer = st.checkbox(
            "Enable Neural Audio Enhancer",
            value=False,
            help="Experimental: Use AI to hallucinate high-frequency resolution (Super-Resolution).",
        )

    tts_possible = is_tts_available()
    do_tts = st.checkbox(
        "Generate dubbed audio (TTS)",
        value=tts_possible,
        disabled=not tts_possible,
        help="Generate speech in target languages" if tts_possible else "No TTS engine available",
    )

    tts_engine = None
    if tts_possible:
        available_tts = get_available_tts_engines()
        default_tts = get_default_tts_engine()
        default_tts_idx = available_tts.index(default_tts) if default_tts in available_tts else 0

        def _tts_label(eid: str) -> str:
            return TTS_ENGINE_REGISTRY[eid]["name"]

        tts_engine = st.selectbox(
            "TTS engine",
            options=available_tts,
            index=default_tts_idx,
            format_func=_tts_label,
            help="Choose the Text-to-Speech engine",
        )
    else:
        st.caption("⚠️ No TTS engine available.")

    st.divider()
    _ov_available = is_openvoice_available()
    with st.expander("🎙️ Voice Cloning (Stage 2)", expanded=False):
        st.markdown(
            "**Stage 1** — Language-correct dubbed speech (Sarvam / edge-tts)  \n"
            "**Stage 2** — Tone-color transfer: makes the dubbed voice **sound like** the original lecturer.  \n"
        )
        
        st.divider()
        if _ov_available:
            enable_voice_cloning = st.checkbox(
                "Enable Stage 2: Voice Cloning",
                value=True,
                help=(
                    "Uses OpenVoice v2 to transfer the original lecturer's voice "
                    "characteristics onto the dubbed audio. Language-agnostic — "
                    "works for Hindi, Telugu, and Odia.\\n\\n"
                    "First run downloads ~300 MB of model weights automatically."
                ),
            )
        else:
            enable_voice_cloning = False
            st.info(
                "OpenVoice v2 is **not installed**.\n\n"
                "**Python 3.12 compatible install:**\n"
                "```\n"
                "pip install git+https://github.com/myshell-ai/OpenVoice.git --no-deps\n"
                "pip install librosa wavmark huggingface_hub\n"
                "```\n\n"
                "Note: use `--no-deps` to skip numpy==1.22.0 which does not build"
                " on Python 3.12 (not needed for tone color transfer)."
            )

    # ── Music / Background Audio ─────────────────────────────
    _demucs_available = is_demucs_available()
    with st.expander("🎵 Background Music / Music Videos", expanded=False):
        st.markdown(
            "Use this when the video has **background music or is a music video**.\n\n"
            "Demucs separates the vocal track before transcription, "
            "then mixes the dubbed voice back with the **original background music**."
        )
        if _demucs_available:
            separate_music = st.checkbox(
                "Preserve Background Music",
                value=True,
                help=(
                    "Separates vocals from background music using Demucs before "
                    "transcription, then mixes the dubbed voice back with the "
                    "original music. Works for NPTEL lectures with intros as well "
                    "as full music videos."
                ),
            )
        else:
            separate_music = False
            st.info(
                "Demucs is **not installed**. "
                "To enable music separation, run:\n"
                "```\npip install demucs\n```",
            )
    # ── Lip Synchronisation (Wav2Lip) ─────────────────────
    _wav2lip_available    = is_wav2lip_available()
    _wav2lip_repo_present = is_wav2lip_repo_present()
    with st.expander("👄 Lip Synchronisation (Wav2Lip)", expanded=False):
        st.markdown(
            "Makes the speaker's **mouth movements match the dubbed audio** using "
            "[Wav2Lip](https://github.com/Rudrabha/Wav2Lip).  "
            "Produces one MP4 per target language with synced facial animation.  \n\n"
            "When Wav2Lip is not installed the option still works but only swaps "
            "the audio (no visual changes)."
        )
        if _wav2lip_available:
            enable_lip_sync = st.checkbox(
                "Enable Lip Synchronisation",
                value=True,
                help=(
                    "Produces one MP4 per target language where the speaker's lips "
                    "are animated to match the dubbed audio. "
                    "A GPU is strongly recommended for speed."
                ),
            )
            st.caption("✅ Wav2Lip checkpoint found — full lip sync ready.")
        elif _wav2lip_repo_present:
            enable_lip_sync = st.checkbox(
                "Enable Lip Synchronisation",
                value=True,
                help=(
                    "Wav2Lip repo found. Checkpoints will be downloaded automatically "
                    "(~500 MB, one-time) when lip sync first runs."
                ),
            )
            st.info(
                "✅ Wav2Lip repo detected — checkpoints will **auto-download (~500 MB)** "
                "the first time lip sync runs."
            )
        else:
            enable_lip_sync = st.checkbox(
                "Enable Lip Synchronisation (audio-swap only)",
                value=True,
                help=(
                    "Wav2Lip not found — will only swap the audio track. "
                    "Install Wav2Lip with its checkpoint for actual facial animation."
                ),
            )
            st.info(
                "Wav2Lip is **not installed** — lip sync will run in "
                "**audio-swap mode** (no visual face changes).\n\n"
                "To enable full lip sync:\n"
                "```\n"
                "git clone https://github.com/Rudrabha/Wav2Lip\n"
                "pip install -r Wav2Lip/requirements.txt\n"
                "```"
            )
    # ── API key status ─────────────────────────────────────
    st.divider()
    st.markdown("##### 🔑 Engine Status")
    for eid, info in ENGINE_REGISTRY.items():
        if info["key"] == "local":
            st.markdown(f"✅  **{info['name']}** — local (no key needed)")
        elif info["key"]:
            st.markdown(f"✅  **{info['name']}** — configured")
        else:
            st.markdown(f"⬜  **{info['name']}** — not set")
    st.caption("Add keys to `.env` and restart to enable more engines.")
    st.caption("Make sure `ffmpeg` is installed and on PATH.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main area – file upload
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
col_upload, col_info = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload a lecture video",
        type=["mp4", "mkv", "avi", "webm", "mov"],
        help="Supported formats: MP4, MKV, AVI, WebM, MOV",
    )

with col_info:
    st.markdown("##### How it works")
    st.markdown(
        """
        1. **Upload** your NPTEL lecture video  
        2. **Configure** options in the sidebar  
        3. **Click Run** and wait for results  
        4. **Download** subtitles & audio  
        """
    )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pipeline execution
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _save_uploaded_file(uploaded) -> str:
    """Save the Streamlit UploadedFile to a temp directory and return path."""
    tmp_dir = os.path.join(tempfile.gettempdir(), "nptel_pipeline")
    os.makedirs(tmp_dir, exist_ok=True)
    dest = os.path.join(tmp_dir, uploaded.name)
    with open(dest, "wb") as f:
        f.write(uploaded.getbuffer())
    return dest


if uploaded_file is not None:
    st.divider()

    # Show video preview
    st.video(uploaded_file)

    if st.button("🚀  Run Pipeline", type="primary", use_container_width=True):

        if not target_langs:
            st.error("Please select at least one target language in the sidebar.")
            st.stop()

        video_path = _save_uploaded_file(uploaded_file)
        video_name = os.path.splitext(uploaded_file.name)[0]

        # Create a dedicated output folder for this run
        run_output = os.path.join(OUTPUT_DIR, video_name)
        os.makedirs(run_output, exist_ok=True)

        # Progress tracking
        progress = st.progress(0, text="Starting pipeline…")
        status = st.status("Pipeline running…", expanded=True)

        # Map pipeline steps to Streamlit progress percentages
        _STEP_PCT = {0: 2, 1: 15, 2: 35, 3: 55, 4: 62, 5: 78, 6: 90, 7: 98, 8: 99}

        def _st_progress(step: int, total: int, message: str) -> None:
            pct = _STEP_PCT.get(step, int(step / max(total, 1) * 100))
            progress.progress(pct, text=message)
            with status:
                st.write(f"**Step {step}/{total}** – {message}")

        try:
            result = run_pipeline(
                video_path=video_path,
                output_dir=run_output,
                stt_method=stt_method,
                translate_method=translate_method,
                target_langs=target_langs,
                do_tts=do_tts,
                tts_engine=tts_engine,
                separate_music=separate_music,
                voice_profile=None,  # Legacy 
                enable_prosody=enable_prosody,
                enable_glossary=enable_glossary,
                enable_enhancer=enable_enhancer,
                enable_lip_sync=enable_lip_sync,
                progress=_st_progress,
            )

            subtitle_paths = result.get("subtitles", {})
            aligned_audio = result.get("aligned_audio", {})
            mkv_path = result.get("dubbed_video")
            lip_synced_videos = result.get("lip_synced_videos", {})
            lip_sync_used_wav2lip = result.get("lip_sync_used_wav2lip", False)
            lip_sync_errors = result.get("lip_sync_errors", {})

            progress.progress(100, text="Pipeline complete! ✅")

            with status:
                st.write("✅ **All done!**")
            status.update(label="Pipeline complete!", state="complete")

            # Store results in session state for persistent access
            lip_synced_mkvs = result.get("lip_synced_mkvs", {})
            st.session_state["pipeline_result"] = {
                "subtitle_paths": subtitle_paths,
                "aligned_audio": aligned_audio,
                "mkv_path": mkv_path,
                "lip_synced_videos": lip_synced_videos,
                "lip_sync_used_wav2lip": lip_sync_used_wav2lip,
                "lip_sync_errors": lip_sync_errors,
                "lip_synced_mkvs": lip_synced_mkvs,
                "video_path": video_path,
                "video_name": video_name,
                "run_output": run_output,
                "target_langs": target_langs,
            }

        except NotImplementedError as e:
            progress.progress(0, text="Error")
            status.update(label="Pipeline failed", state="error")
            st.error(
                f"**Not yet implemented:** {e}\n\n"
                "This module still has a placeholder. "
                "Wire up the API calls first, then try again."
            )
        except ValueError as e:
            progress.progress(0, text="Error")
            status.update(label="Pipeline failed", state="error")
            _emsg = str(e).lower()
            if "no speech" in _emsg or "empty transcript" in _emsg:
                st.error("🎵 **No speech detected in the audio**")
                from src.audio_separator import is_demucs_available as _ida
                if _ida():
                    st.info(
                        "Demucs is installed. The pipeline will **automatically** "
                        "try vocal separation the next time you run. "
                        "If it already tried and still failed, the audio may "
                        "contain no spoken words at all."
                    )
                else:
                    st.warning(
                        "Install **Demucs** to enable automatic music/vocal separation."
                        " After installing, re-run and the pipeline will auto-detect"  
                        " and separate vocals from background music automatically.\n\n"
                        "```\npip install demucs\n```"
                    )
            else:
                st.error(f"**Error:** {e}")
                st.exception(e)

        except Exception as e:
            progress.progress(0, text="Error")
            status.update(label="Pipeline failed", state="error")
            st.error(f"**Error:** {e}")
            st.exception(e)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Results display – persistent via session state
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if "pipeline_result" in st.session_state:
    res = st.session_state["pipeline_result"]
    subtitle_paths = res["subtitle_paths"]
    aligned_audio = res["aligned_audio"]
    mkv_path = res["mkv_path"]
    lip_synced_videos = res.get("lip_synced_videos", {})
    lip_sync_used_wav2lip = res.get("lip_sync_used_wav2lip", False)
    lip_sync_errors = res.get("lip_sync_errors", {})
    lip_synced_mkvs = res.get("lip_synced_mkvs", {})
    video_path = res["video_path"]
    video_name = res["video_name"]
    run_output = res["run_output"]
    target_langs_done = res["target_langs"]

    st.divider()
    st.subheader("📦 Results")

    # ── Video preview with dubbed audio + subtitle overlay ──────
    st.markdown("##### 🎬 Video Preview")
    st.markdown(
        "Preview the dubbed video with burned-in subtitles. "
        "Choose the audio language and subtitle language below."
    )

    # Build language options for dropdowns
    audio_options = {}
    if aligned_audio:
        for lc in sorted(aligned_audio.keys()):
            lang_name = TARGET_LANGUAGES.get(lc, {}).get("name", lc)
            audio_options[f"{lang_name} ({lc})"] = lc

    subtitle_options = {"None (no subtitles)": None}
    for lc in sorted(subtitle_paths.keys()):
        lang_name = ALL_LANGUAGES.get(lc, {}).get("name", lc)
        subtitle_options[f"{lang_name} ({lc})"] = lc

    col_audio_sel, col_sub_sel = st.columns(2)

    with col_audio_sel:
        if audio_options:
            selected_audio_label = st.selectbox(
                "🔊 Audio language",
                options=list(audio_options.keys()),
                index=0,
                help="Select which dubbed audio track to preview",
            )
            selected_audio_lang = audio_options[selected_audio_label]
        else:
            st.info("No dubbed audio tracks available")
            selected_audio_lang = None

    with col_sub_sel:
        selected_sub_label = st.selectbox(
            "📝 Subtitle language",
            options=list(subtitle_options.keys()),
            index=0,
            help="Select which subtitle language to burn into the preview",
        )
        selected_sub_lang = subtitle_options[selected_sub_label]

    # Generate / display preview
    if selected_audio_lang and selected_audio_lang in aligned_audio:
        # Build a unique preview filename based on audio + subtitle selection
        sub_tag = selected_sub_lang or "nosub"
        preview_name = f"{video_name}_preview_{selected_audio_lang}_{sub_tag}.mp4"
        preview_path = os.path.join(run_output, preview_name)

        if not os.path.isfile(preview_path):
            with st.spinner("Creating preview video…"):
                try:
                    sub_file = None
                    if selected_sub_lang and selected_sub_lang in subtitle_paths:
                        sub_file = subtitle_paths[selected_sub_lang]
                    create_preview_mp4(
                        original_video=video_path,
                        dubbed_audio=aligned_audio[selected_audio_lang],
                        output_path=preview_path,
                        subtitle_path=sub_file,
                        subtitle_lang_code=selected_sub_lang,
                    )
                except Exception as exc:
                    st.warning(f"Preview generation failed: {exc}")
                    preview_path = None

        if preview_path and os.path.isfile(preview_path):
            st.video(preview_path)
        else:
            st.info("Preview not available. Download the MKV and open in VLC.")
    elif not aligned_audio:
        # No dubbed audio — show original video
        st.video(video_path)

    # ── Subtitle display ────────────────────────────────────────
    if selected_sub_lang and selected_sub_lang in subtitle_paths:
        sub_path = subtitle_paths[selected_sub_lang]
        if os.path.isfile(sub_path):
            with st.expander("📄 View subtitle text", expanded=False):
                with open(sub_path, "r", encoding="utf-8-sig") as f:
                    st.code(f.read(), language=None)

    # ── Final dubbed video download ──
    st.markdown("##### 📥 Downloads")

    # Per-language lip-synced MKVs (preferred when lip sync ran)
    if lip_synced_mkvs:
        st.caption(
            "👄 **Lip-synced MKV** — video frames animated with Wav2Lip + "
            "enhanced dubbed audio (open in VLC / mpv and select audio/subtitle tracks)"
        )
        ls_dl_cols = st.columns(len(lip_synced_mkvs))
        for i, (lang_code, ls_mkv) in enumerate(sorted(lip_synced_mkvs.items())):
            lang_name = ALL_LANGUAGES.get(lang_code, {}).get("name", lang_code)
            with ls_dl_cols[i]:
                if os.path.isfile(ls_mkv):
                    with open(ls_mkv, "rb") as f:
                        ls_bytes = f.read()
                    size_mb = len(ls_bytes) / (1024 * 1024)
                    st.download_button(
                        f"⬇ {lang_name} lip-sync MKV ({size_mb:.1f} MB)",
                        data=ls_bytes,
                        file_name=os.path.basename(ls_mkv),
                        mime="video/x-matroska",
                        type="primary",
                        use_container_width=True,
                    )
                else:
                    st.info(f"{lang_name} lip-sync MKV not found")

    # Original multi-language MKV (all audio tracks, no lip-sync video)
    if mkv_path and os.path.isfile(mkv_path):
        label = "⬇ Download multi-language MKV (original video)" if lip_synced_mkvs else "⬇ Download dubbed video"
        with open(mkv_path, "rb") as f:
            mkv_bytes = f.read()
        size_mb = len(mkv_bytes) / (1024 * 1024)
        st.download_button(
            f"{label} ({size_mb:.1f} MB)",
            data=mkv_bytes,
            file_name=os.path.basename(mkv_path),
            mime="video/x-matroska",
            type="secondary" if lip_synced_mkvs else "primary",
            use_container_width=True,
        )

    # ── Subtitle downloads ──
    st.markdown("##### 📝 Subtitles")
    sub_cols = st.columns(len(subtitle_paths))
    for i, (lang, path) in enumerate(subtitle_paths.items()):
        lang_name = ALL_LANGUAGES.get(lang, {}).get("name", lang)
        with sub_cols[i]:
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8-sig") as f:
                    srt_data = f.read()
                st.download_button(
                    f"⬇ {lang_name} (.srt)",
                    data=srt_data,
                    file_name=os.path.basename(path),
                    mime="text/plain",
                )
            else:
                st.info(f"{lang_name} subtitle not found")

    # ── Aligned audio downloads ──
    if aligned_audio:
        st.markdown("##### 🔊 Aligned Dubbed Audio")
        tts_cols = st.columns(len(aligned_audio))
        for i, (lang, path) in enumerate(aligned_audio.items()):
            lang_name = TARGET_LANGUAGES.get(lang, {}).get("name", lang)
            with tts_cols[i]:
                if os.path.isfile(path):
                    with open(path, "rb") as f:
                        audio_bytes = f.read()
                    st.audio(audio_bytes, format="audio/mpeg")
                    st.download_button(
                        f"⬇ {lang_name} audio",
                        data=audio_bytes,
                        file_name=os.path.basename(path),
                        mime="audio/mpeg",
                    )
                else:
                    st.info(f"{lang_name} audio not found")

    # ── Lip-synced video downloads ──────────────────────────────
    if lip_synced_videos:
        st.markdown("##### 👄 Lip-Synced Videos")
        if lip_sync_used_wav2lip:
            st.caption("✅ Sync mode: Wav2Lip (facial animation)")
        else:
            st.warning(
                "⚠️ Lip sync fell back to **audio-swap only** (no facial animation). "
                "Check the Streamlit console / terminal for the full error. "
                "Common causes:\n"
                "- No face detected in the video (Wav2Lip needs a visible face)\n"
                "- Wav2Lip process failed during inference\n"
                "See error details below."
            )
            if lip_sync_errors:
                with st.expander("🔍 Wav2Lip error details", expanded=True):
                    for lang, err in lip_sync_errors.items():
                        lang_name = TARGET_LANGUAGES.get(lang, {}).get("name", lang)
                        st.error(f"**{lang_name}**: {err[:1000]}")
        ls_cols = st.columns(len(lip_synced_videos))
        for i, (lang, path) in enumerate(lip_synced_videos.items()):
            lang_name = TARGET_LANGUAGES.get(lang, {}).get("name", lang)
            with ls_cols[i]:
                if os.path.isfile(path):
                    st.video(path)
                    with open(path, "rb") as f:
                        vid_bytes = f.read()
                    size_mb = len(vid_bytes) / (1024 * 1024)
                    st.download_button(
                        f"⬇ {lang_name} lip-sync ({size_mb:.1f} MB)",
                        data=vid_bytes,
                        file_name=os.path.basename(path),
                        mime="video/mp4",
                    )
                else:
                    st.info(f"{lang_name} lip-sync video not found")

elif uploaded_file is None:
    # Empty state
    st.info("👆 Upload a lecture video to get started.", icon="📹")
