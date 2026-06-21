# Speaker Diarization Lab — IMA4511

A failure-driven lab on speaker diarization (*who spoke when?*) for the course
**IMA4511 — Pattern Recognition and Biometrics**.

The whole lab lives in a single notebook: [`speaker_diarization_failure_lab.ipynb`](speaker_diarization_failure_lab.ipynb).

It walks you through recording multi-speaker audio, visualizing it, building a simple
energy-based voice activity detector, running a pretrained `pyannote.audio` diarization
pipeline, and analyzing where and why diarization fails.

---

## Run in Google Colab (recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YannisTevissen/LabSession/blob/master/speaker_diarization_failure_lab.ipynb)

1. Click the **Open in Colab** badge above (or open this link):
   `https://colab.research.google.com/github/YannisTevissen/LabSession/blob/master/speaker_diarization_failure_lab.ipynb`
2. (Optional but faster) Set a GPU runtime: **Runtime → Change runtime type → Hardware accelerator → GPU**.
3. Run the **installation cell** in Section 0 once. It installs `pyannote.audio` and
   `pyannote.metrics` (a few minutes). If Colab asks you to restart the runtime after
   installing, do it, then continue.
4. In Section 1, run the **upload cell** to upload your three `.wav` files
   (`clean.wav`, `overlap.wav`, `degraded.wav`).
5. Run the cells in order, top to bottom.

> Colab tip: you can also open it manually with **File → Open notebook → GitHub**, then
> paste the repository URL `https://github.com/YannisTevissen/LabSession`.

### Troubleshooting the install on Colab

- The install cell prints several red **"dependency resolver"** warnings (about
  `google-colab`, `gradio`, `opentelemetry`, `pandas`). These are harmless and come from
  Colab's own preinstalled packages.
- The install cell pins `numpy<2.1` so that `librosa`/`numba` keep working. If Colab shows
  a **"Restart session"** prompt after installing, click it (**Runtime → Restart session**),
  then continue from **Section 1** — you do not need to re-run the install cell.

---

## Hugging Face token (for the diarization pipeline)

Section 6 uses `pyannote/speaker-diarization-3.1`, which requires:

1. a free [Hugging Face](https://huggingface.co/join) account;
2. accepting the model conditions on the
   [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1)
   model page (and its dependency `pyannote/segmentation-3.0`);
3. an access token with **read** permission (create one
   [here](https://huggingface.co/settings/tokens)).

Provide the token in one of these ways:

- **On Colab:** add it as a Secret named `HF_TOKEN` (the key icon in the left sidebar),
  or simply paste it when the notebook prompts you.
- **Locally:** export it before launching, e.g. `export HF_TOKEN=hf_...`, or paste it
  when prompted.

A token is required to run the diarization pipeline (Section 6): the notebook stops with
an error if none is provided.

---

## Run locally (alternative)

```bash
git clone https://github.com/YannisTevissen/LabSession.git
cd LabSession
python -m venv .venv && source .venv/bin/activate   # optional
pip install numpy pandas matplotlib librosa soundfile scikit-learn pyannote.audio pyannote.metrics jupyter
jupyter notebook speaker_diarization_failure_lab.ipynb
```

Place your `clean.wav`, `overlap.wav`, and `degraded.wav` inside the
`diarization_lab_artifacts/` folder (the notebook creates it on first run).

---

## What to record

Work in groups of 2–3 students and create three short recordings:

| File | Duration | Content |
|---|---:|---|
| `clean.wav` | 25–40 s | 2 speakers alternate, no intentional overlap |
| `overlap.wav` | 20–35 s | same speakers, with a clear overlap region |
| `degraded.wav` | 20–35 s | same speakers, harder condition: noise, distance, soft voice, background music, or fast turns |

Convert `.m4a`/`.mp3` files to `.wav` before running the rest of the notebook.

---

## What to submit

The notebook contains **only code and questions**. Write your answers in a **separate
report**.

Submit a single **zip file** containing:

- your **report** (a separate document with all your answers);
- your **code** (this notebook);
- your **audio files** (`clean.wav`, `overlap.wav`, `degraded.wav`).

The last code cell in Section 10 can bundle your audio and generated outputs into
`diarization_lab_artifacts.zip` for you (and download it automatically on Colab).
