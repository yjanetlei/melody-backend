from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import tempfile
import os
import numpy as np
import logging
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Melody to Sheet Music API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
    expose_headers=["*"],
    max_age=3600,
)

@app.options("/{rest_of_path:path}")
async def preflight_handler(request: Request, rest_of_path: str):
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )
SUPPORTED_FORMATS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".webm", ".mp4"}
HAND_SPLIT_MIDI = 60
MIN_NOTE_DURATION = 0.05
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_note_name(m):
    return f"{NOTE_NAMES[m % 12]}{(m // 12) - 1}"


def quantize_duration(duration_sec, tempo_bpm):
    ratio = duration_sec / (60.0 / tempo_bpm)
    durations = [
        (4.0, "1"),
        (3.0, "2d"),
        (2.0, "2"),
        (1.5, "qd"),
        (1.0, "q"),
        (0.75, "8d"),
        (0.5, "8"),
        (0.25, "16"),
        (0.125, "32"),
    ]
    return min(durations, key=lambda x: abs(x[0] - ratio))[1]


def estimate_tempo(starts):
    if len(starts) < 4:
        return 120.0
    intervals = np.diff(sorted(starts))
    intervals = intervals[(intervals > 0.1) & (intervals < 2.0)]
    if len(intervals) == 0:
        return 120.0
    return round(max(40.0, min(200.0, 60.0 / float(np.median(intervals)))), 1)


def assign_hand(midi_pitch):
    return "right" if midi_pitch >= HAND_SPLIT_MIDI else "left"


def group_into_measures(notes, tempo_bpm, beats=4):
    if not notes:
        return []
    measure_dur = (60.0 / tempo_bpm) * beats
    max_end = max(n["end_time"] for n in notes)
    num_m = int(np.ceil(max_end / measure_dur))
    measures = [[] for _ in range(max(num_m, 1))]
    for note in notes:
        idx = min(int(note["start_time"] / measure_dur), num_m - 1)
        measures[idx].append(note)
    return measures


@app.get("/")
def root():
    return {"status": "ok", "message": "Melody to Sheet Music API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    _, ext = os.path.splitext(file.filename or "")
    ext = ext.lower()
    if ext not in SUPPORTED_FORMATS:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(await file.read())

    converted_path = tmp_path

    try:
        # Convert webm/mp4 to wav for Basic Pitch compatibility
        if ext in {".webm", ".mp4"}:
            converted_path = tmp_path.replace(ext, ".wav")
            os.system(f"ffmpeg -i {tmp_path} -ar 22050 -ac 1 {converted_path} -y -loglevel quiet")

        _, _, note_events = predict(
            converted_path,
            ICASSP_2022_MODEL_PATH,
            onset_threshold=0.5,
            frame_threshold=0.3,
            minimum_note_length=MIN_NOTE_DURATION * 1000,
            minimum_frequency=27.5,
            maximum_frequency=4186.0,
            melodia_trick=True,
            multiple_pitch_bends=False,
        )

        if not note_events:
            raise HTTPException(status_code=422, detail="No notes detected.")

        tempo_bpm = estimate_tempo([float(n[0]) for n in note_events])

        notes = []
        for idx, (st, et, mp, vel, _) in enumerate(note_events):
            dur = float(et) - float(st)
            if dur < MIN_NOTE_DURATION:
                continue
            notes.append({
                "id": idx,
                "midi_pitch": int(mp),
                "note_name": midi_to_note_name(int(mp)),
                "start_time": round(float(st), 4),
                "end_time": round(float(et), 4),
                "duration_sec": round(dur, 4),
                "duration_label": quantize_duration(dur, tempo_bpm),
                "velocity": int(vel) if vel else 80,
                "hand": assign_hand(int(mp)),
            })

        notes.sort(key=lambda n: n["start_time"])
        rh = [n for n in notes if n["hand"] == "right"]
        lh = [n for n in notes if n["hand"] == "left"]
        rm = group_into_measures(rh, tempo_bpm)
        lm = group_into_measures(lh, tempo_bpm)
        mx = max(len(rm), len(lm))
        while len(rm) < mx:
            rm.append([])
        while len(lm) < mx:
            lm.append([])

        return JSONResponse(content={
            "status": "success",
            "filename": file.filename,
            "tempo_bpm": tempo_bpm,
            "time_signature": {"numerator": 4, "denominator": 4},
            "total_notes": len(notes),
            "right_hand_notes": len(rh),
            "left_hand_notes": len(lh),
            "num_measures": mx,
            "notes": notes,
            "measures": {"right": rm, "left": lm},
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if converted_path != tmp_path and os.path.exists(converted_path):
            os.unlink(converted_path)
