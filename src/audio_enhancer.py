"""
audio_enhancer.py
─────────────────
Neural audio de-noising using DeepFilterNet v3.

Removes background noise from dubbed audio segments while preserving
speech quality.  The model is lazy-loaded and cached so the 200 MB
checkpoint is only read once per process.

Requirements:
    pip install deepfilternet
"""

import logging
import os

logger = logging.getLogger("nptel_pipeline")

# Singleton: (model, df_state, _path_str) — populated on first call
_model_cache: tuple | None = None


def _get_model() -> tuple:
    """Lazy-load and cache the DeepFilterNet3 model."""
    global _model_cache
    if _model_cache is None:
        from df.enhance import init_df  # noqa: PLC0415
        logger.info("Loading DeepFilterNet3 model (first run only)…")
        _model_cache = init_df()        # (model, df_state, checkpoint_path)
    return _model_cache


def release_model() -> None:
    """
    Move the DeepFilterNet model to CPU and free GPU memory.

    Call this after all enhancement passes for a pipeline run are done so
    subsequent stages (Demucs, OpenVoice, etc.) can use the full VRAM budget.
    """
    global _model_cache
    if _model_cache is None:
        return
    try:
        import torch  # noqa: PLC0415
        model, _df_state, _path = _model_cache
        model.to("cpu")
        del model
        _model_cache = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.info("[DeepFilterNet] GPU memory released")
    except Exception as exc:
        logger.warning("[DeepFilterNet] GPU cleanup failed: %s", exc)


def apply_neural_enhancement(input_path: str) -> str:
    """
    Denoise *input_path* with DeepFilterNet3.

    The enhanced file is written alongside the input as
    ``<stem>_enhanced.wav``.  Returns *input_path* unchanged on any
    error so the pipeline can continue gracefully.

    GPU memory is freed via ``release_model()`` after each call so the
    VRAM is available for the next pipeline stage.

    Args:
        input_path: Path to any audio file supported by torchaudio
                    (WAV, MP3, FLAC, …).

    Returns:
        Path to the enhanced WAV file, or *input_path* on failure.
    """
    base, _ = os.path.splitext(input_path)
    output_path = base + "_enhanced.wav"

    # No reuse of an existing output: the input track is regenerated on every
    # run, so a file left by an earlier run would hold the old audio.
    logger.info("Enhancing audio with DeepFilterNet3: %s",
                os.path.basename(input_path))
    try:
        from df.enhance import enhance, load_audio, save_audio  # noqa: PLC0415

        model, df_state, _ = _get_model()
        sr = df_state.sr()  # df_state.sr is a method, not a property

        # load_audio returns (Tensor, AudioMetaData); we only need the tensor
        audio, _ = load_audio(input_path, sr=sr, verbose=False)

        enhanced = enhance(model, df_state, audio)

        # save_audio writes a PCM WAV; dtype=torch.int16 gives 16-bit output
        save_audio(output_path, enhanced, sr)

        logger.info("Enhancement complete → %s", os.path.basename(output_path))
        return output_path

    except Exception as exc:
        logger.error("Neural enhancement failed (%s): %s",
                     os.path.basename(input_path), exc)
        # Non-fatal: caller continues with the original file
        return input_path

    finally:
        # Release GPU memory immediately after each enhancement pass
        release_model()
