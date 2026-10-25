# NPTEL Lecture Translation Pipeline

Dub English NPTEL lecture videos into **Hindi**, **Telugu** and **Odia** — subtitles,
dubbed audio in the lecturer's own voice, optional lip sync, all in one MKV.

## Pipeline

Each job is split into stages that form a DAG (`src/pipeline_dag.py`). Every stage
needs one resource — **GPU**, **CPU** or **API** — and the executor
(`src/dag_executor.py`) starts it as soon as its inputs are ready and a slot of that
resource is free. Languages run side by side, and when two jobs are active they share
the slots: one job's TTS calls run while the other job uses the GPU, but the GPU
itself only ever runs one stage at a time.

```
extract (CPU) ─ demucs (GPU) ─ stt: Whisper (GPU) ─ subtitles_en (CPU)
                                 │
                                 └─ per language, in parallel:
                                    translate (API) ─ subtitles (CPU)
                                    tts (API) ─ clone voice (GPU) ─ align (CPU)
                                    ─ match pitch (CPU) ─ mix music (CPU) ─ [lip sync (GPU)]
                                                                           │
                                                          mux MKV (CPU) ◄──┘
```

| Stage | Tool |
|-------|------|
| Music separation | Demucs (`htdemucs`) |
| Speech-to-text | faster-whisper (local) or Sarvam STT |
| Translation | Sarvam `mayura:v1` in **code-mixed** mode (keeps technical terms in English) or Google Cloud Translation |
| Text-to-speech | Sarvam `bulbul:v3`, Google Cloud TTS Neural2, or Coqui XTTS v2 — voice picked to match the lecturer's gender |
| Voice cloning | OpenVoice v2 (needs a CUDA GPU) |
| Alignment | One natural speaking speed (±8–15 % WSOLA nudges via ffmpeg `atempo`); pauses absorb timing differences; sentences start at the real speech onset; loudness evened out |
| Lip sync | Wav2Lip |

Hesitation fillers ("ah", "um") are removed before translation and from the subtitles.
A failure in one language only drops that language; the job fails if a shared stage
fails or no language produced audio.

## Setup (Linux)

```bash
python3.10 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/myshell-ai/OpenVoice.git   # voice cloning (optional)

# ffmpeg must be on PATH
sudo apt install ffmpeg

# Wav2Lip (optional): clone the submodule, then download its weights
git submodule update --init --recursive
python setup_wav2lip.py

cp .env.example .env    # add SARVAM_API_KEY, GCP_API_KEY, APP_PASSWORD
```

## Usage

```bash
streamlit run app.py                       # web app (password = APP_PASSWORD)
./start_ngrok.sh                           # … exposed through ngrok
python main.py input/lecture.mp4 --langs hi te
python main.py input/lecture.mp4 --stt whisper --translate gcp --no-tts
```

In the web app, "Past Results" lists the jobs started from your browser (the key is in
the page URL — bookmark it) plus shared examples from before per-user keys existed.

### Scheduling settings (`config/settings.py`)

| Setting | Default | Meaning |
|---------|---------|---------|
| `MAX_CONCURRENT_JOBS` | 2 | jobs whose stages run at the same time |
| `MAX_WAITING_JOBS` | 1 | further jobs allowed to queue; beyond that the app reports "server busy" |
| `DAG_GPU_SLOTS` | 1 | GPU stages at once (4 GB card) |
| `DAG_CPU_SLOTS` | half the cores | ffmpeg / alignment stages at once |
| `DAG_API_SLOTS` | 4 | Sarvam / Google stages at once |

Whisper and OpenVoice run in child processes: CTranslate2 never returns the GPU memory
it used, which would starve the next GPU stage.

## Project structure

```
config/settings.py          API keys, engines, languages, scheduling limits
src/pipeline.py             run_pipeline(): builds and runs a job's DAG; cache helpers
src/pipeline_dag.py         the stages of a job and their resources
src/dag_executor.py         resource-slot scheduler shared by all jobs
src/job_queue.py            job queue for the web app
src/transcriber.py          speech-to-text (Whisper / Sarvam)
src/translator.py           translation (Sarvam / Google)
src/tts_generator.py        text-to-speech (Sarvam / Google Neural2 / XTTS)
src/voice_converter.py      OpenVoice voice cloning
src/audio_aligner.py        timing alignment
src/audio_separator.py      Demucs separation and music remix
src/voice_analyzer.py       lecturer pitch / gender, pitch matching
src/subtitle_generator.py   .srt / .vtt
src/video_muxer.py          MKV muxing and previews
src/lip_sync.py             Wav2Lip
app.py                      Streamlit web app
main.py                     CLI
tests/                      pytest suite (python -m pytest)
```

## Languages

| Code | Language |
|------|----------|
| `en` | English (source) |
| `hi` | Hindi |
| `te` | Telugu |
| `od` | Odia |
