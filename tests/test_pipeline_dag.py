"""
Tests for src/pipeline_dag.py and run_pipeline(): which stages a job is
split into, which resource each stage holds, and how failures in one
language are reported.  All heavy work (STT, translation, TTS, ffmpeg) is
mocked, so the real DAG executor runs the real stage graph in milliseconds.
"""

import os
from unittest.mock import patch

import pytest

from src.dag_executor import API, CPU, GPU
from src.transcriber import Segment
from src.voice_converter import NO_GPU_MESSAGE

ENGLISH = [
    Segment(0.0, 2.0, "Hello there, ah, everyone."),
    Segment(2.5, 3.0, "Ah."),
    Segment(3.5, 6.0, "Today we study trees."),
]


def build(tmp_output_dir, *, gpu=True, cached=None, **kwargs):
    from src.pipeline_dag import build_pipeline_dag
    opts = dict(stt_method="whisper", translate_method="sarvam", target_langs=["hi", "te"],
                tts_engine="sarvam_vc", separate_music=True)
    opts.update(kwargs)
    with patch("src.pipeline_dag.file_sha256", return_value="0" * 64), \
         patch("src.pipeline_dag._load_cache", return_value=cached), \
         patch("src.pipeline_dag.is_demucs_available", return_value=True), \
         patch("src.pipeline_dag.is_gpu_available", return_value=gpu), \
         patch("src.pipeline_dag.is_openvoice_available", return_value=True):
        return build_pipeline_dag("test", "lecture.mp4", tmp_output_dir, **opts)


class TestStageGraph:

    def test_stages_and_resources(self, tmp_output_dir):
        dag = build(tmp_output_dir)
        res = {nid: n.resource for nid, n in dag.nodes.items()}
        assert res["demucs"] == GPU and res["stt"] == GPU          # Whisper
        assert res["translate_hi"] == API and res["tts_hi"] == API
        assert res["clone_hi"] == GPU and res["clone_te"] == GPU
        assert res["align_hi"] == CPU and res["mix_te"] == CPU and res["mux"] == CPU
        # Each language is its own chain: Telugu never waits for Hindi
        for lc in ("hi", "te"):
            assert dag.nodes[f"tts_{lc}"].deps[0] == f"translate_{lc}"
            assert dag.nodes[f"clone_{lc}"].deps == [f"tts_{lc}", "voice_ref"]
        assert set(dag.nodes["mux"].after) >= {"mix_hi", "mix_te"}
        dag.validate()

    def test_no_gpu_means_no_cloning_stages(self, tmp_output_dir):
        dag = build(tmp_output_dir, gpu=False)
        assert not [nid for nid in dag.nodes if nid.startswith(("clone_", "voice_ref"))]
        assert dag.ctx["voice_cloning_note"] == NO_GPU_MESSAGE

    def test_cached_transcript_does_not_wait_for_the_gpu(self, tmp_output_dir):
        dag = build(tmp_output_dir, cached=[s.to_dict() for s in ENGLISH])
        assert dag.nodes["stt"].resource == CPU

    def test_xtts_synthesis_is_a_gpu_stage(self, tmp_output_dir):
        dag = build(tmp_output_dir, tts_engine="xtts", enable_voice_cloning=False)
        assert dag.nodes["tts_hi"].resource == GPU
        assert "pitch_hi" not in dag.nodes

    def test_subtitles_only_job(self, tmp_output_dir):
        dag = build(tmp_output_dir, do_tts=False)
        assert "mux" not in dag.nodes
        assert not [nid for nid in dag.nodes if nid.startswith(("tts_", "align_"))]


@pytest.fixture
def mocked_stages(tmp_output_dir):
    """Patch every heavy stage function; returns recorded calls."""
    calls = {"srt": {}, "mux": []}
    failing_tts: set[str] = set()
    missing_segment: dict[str, int] = {}

    def translate(english, langs, method):
        lc = langs[0]
        return {lc: [Segment(s.start, s.end, f"[{lc}] {s.text}") for s in english]}

    def synthesize(segments, lc, output_dir, **kwargs):
        if lc in failing_tts:
            return {}
        return {i: f"{lc}_{i}.mp3" for i, s in enumerate(segments)
                if s.text.strip() and i != missing_segment.get(lc)}

    def srt(segments, lc, *args):
        calls["srt"][lc] = [s.text for s in segments]
        return os.path.join(tmp_output_dir, f"{lc}.srt")

    patches = [
        patch("src.pipeline.run_preflight_checks"),
        patch("src.pipeline.ENABLE_CACHE", False),
        patch("src.pipeline_dag.file_sha256", return_value="0" * 64),
        patch("src.pipeline_dag._get_media_duration", return_value=10.0),
        patch("src.pipeline_dag.extract_audio", return_value="lecture.wav"),
        patch("src.pipeline_dag.is_demucs_available", return_value=False),
        patch("src.pipeline_dag.is_gpu_available", return_value=False),
        patch("src.pipeline_dag.ENABLE_VOICE_PRESERVATION", False),
        patch("src.pipeline_dag.transcribe_audio", return_value=list(ENGLISH)),
        patch("src.pipeline_dag.translate_segments", side_effect=translate),
        patch("src.pipeline_dag.generate_srt", side_effect=srt),
        patch("src.pipeline_dag.generate_vtt"),
        patch("src.pipeline_dag.synthesize_segments", side_effect=synthesize),
        patch("src.pipeline_dag.align_dubbed_audio"),
        patch("src.pipeline_dag.encode_final_track", side_effect=lambda src, out: out),
        patch("src.pipeline_dag.mux_to_mkv",
              side_effect=lambda **kw: calls["mux"].append(kw)),
    ]
    for p in patches:
        p.start()
    yield {"calls": calls, "failing_tts": failing_tts, "missing_segment": missing_segment,
           "output_dir": tmp_output_dir}
    for p in patches:
        p.stop()


def run(mocked, **kwargs):
    from src.pipeline import run_pipeline
    opts = dict(output_dir=mocked["output_dir"], stt_method="whisper",
                translate_method="sarvam", tts_engine="sarvam", target_langs=["hi", "te"])
    opts.update(kwargs)
    return run_pipeline("lecture.mp4", **opts)


class TestRunPipeline:

    def test_all_languages_succeed(self, mocked_stages):
        result = run(mocked_stages)
        assert set(result["aligned_audio"]) == {"hi", "te"}
        assert result["language_errors"] == {} and result["warnings"] == []
        assert result["voice_cloning_note"] == NO_GPU_MESSAGE
        assert result["dubbed_video"].endswith("lecture_dubbed.mkv")
        assert {s["status"] for s in result["stages"]} == {"done"}

    def test_one_language_failing_still_delivers_the_others(self, mocked_stages):
        mocked_stages["failing_tts"].add("te")
        result = run(mocked_stages)
        assert list(result["aligned_audio"]) == ["hi"]
        assert result["language_errors"]["te"].startswith("TTS → Telugu failed")
        mux = mocked_stages["calls"]["mux"][0]
        assert list(mux["audio_tracks"]) == ["hi"]
        assert set(mux["subtitle_tracks"]) == {"en", "hi", "te"}   # subtitles still made

    def test_no_audio_for_any_language_fails_the_job(self, mocked_stages):
        mocked_stages["failing_tts"].update({"hi", "te"})
        with pytest.raises(RuntimeError, match="No dubbed audio was produced for any language"):
            run(mocked_stages)

    def test_partly_missing_tts_is_reported(self, mocked_stages):
        mocked_stages["missing_segment"]["hi"] = 2
        result = run(mocked_stages)
        assert any("Hindi: 1 of 3 sentences have no dubbed audio" in w for w in result["warnings"])

    def test_english_subtitles_leave_out_fillers(self, mocked_stages):
        run(mocked_stages, do_tts=False)
        assert mocked_stages["calls"]["srt"]["en"] == [
            "Hello there, everyone.", "", "Today we study trees.",
        ]
