
import os
import time
import threading
import json
from flask import (Flask, render_template, request, jsonify, Response,
                   send_from_directory, abort)

app = Flask(__name__)

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
GENERATED_DIR = os.path.join(BASE_DIR, "generated")
MODELS_DIR    = os.path.join(BASE_DIR, "models")

os.makedirs(GENERATED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


training_state = {
    "running": False,
    "progress": 0,
    "message": "idle",
    "error": None,
}


def _parse_int(value, default, minimum=None, maximum=None):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _parse_float(value, default, minimum=None, maximum=None):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    return Response(b"", mimetype="image/x-icon")



@app.route("/generate", methods=["POST"])
def generate():
    data        = request.get_json(silent=True) or {}
    genre       = data.get("genre",       "classical")
    n_notes     = _parse_int(data.get("n_notes"), 200, minimum=1, maximum=1000)
    bpm         = _parse_int(data.get("bpm"), 120, minimum=30, maximum=300)
    temperature = _parse_float(data.get("temperature"), 0.8, minimum=0.1, maximum=2.0)
    use_ai      = data.get("use_ai", False)   

    filename = f"generated_{genre}_{int(time.time())}.mid"

    try:
        if use_ai:
           
            from generate import full_pipeline
            path, notes = full_pipeline(
                genre=genre,
                n_generate=n_notes,
                bpm=bpm,
                temperature=temperature,
                output_filename=filename,
            )
        else:
            
            from generate import quick_generate_midi
            path, notes = quick_generate_midi(
                n_notes=n_notes,
                bpm=bpm,
                temperature=temperature,
                genre=genre,
                output_filename=filename,
            )

        return jsonify({
            "success":  True,
            "filename": os.path.basename(path),
            "download": f"/download/{os.path.basename(path)}",
            "mode":     "LSTM" if use_ai else "Rule-based",
            "notes":    notes,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/download/<filename>")
def download(filename):
    
    if not filename.endswith(".mid"):
        abort(400)
    safe = os.path.basename(filename)
    if not os.path.exists(os.path.join(GENERATED_DIR, safe)):
        abort(404)
    return send_from_directory(
        GENERATED_DIR, safe,
        as_attachment=True,
        download_name=safe,
        mimetype="audio/midi",
    )



@app.route("/train", methods=["POST"])
def train():
    if training_state["running"]:
        return jsonify({"success": False, "error": "Training already running"}), 409

    data   = request.get_json(silent=True) or {}
    genre  = data.get("genre",  "classical")
    epochs = _parse_int(data.get("epochs"), 50, minimum=1, maximum=500)

    def _train():
        global training_state
        training_state.update({"running": True, "progress": 0,
                                "message": "Collecting MIDI data...", "error": None})
        try:
            from generate import collect_notes, prepare_sequences, build_model, train_model
            import pickle, os

            training_state["message"] = "Parsing MIDI files..."
            notes = collect_notes(genre=genre)

            training_state["message"] = "Building sequences..."
            training_state["progress"] = 20
            X, y, n_vocab, _ = prepare_sequences(notes)

            training_state["message"] = "Building LSTM model..."
            training_state["progress"] = 35
            model = build_model(n_vocab)

            training_state["message"] = f"Training {epochs} epochs..."
            training_state["progress"] = 40

            import tensorflow as tf
            class ProgressCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    pct = 40 + int((epoch + 1) / epochs * 55)
                    training_state["progress"] = pct
                    loss = f"{logs.get('loss',0):.4f}"
                    training_state["message"] = (
                        f"Epoch {epoch+1}/{epochs} - loss={loss}")

            from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
            from generate import MODEL_WEIGHTS, MAPPING_CACHE
            cbs = [
                ModelCheckpoint(MODEL_WEIGHTS, save_best_only=True,
                                monitor="val_loss", verbose=0),
                EarlyStopping(patience=10, restore_best_weights=True, verbose=0),
                ProgressCallback(),
            ]
            model.fit(X, y, epochs=epochs, batch_size=64,
                      validation_split=0.1, callbacks=cbs, verbose=0)

            training_state.update({"running": False, "progress": 100,
                                    "message": "Training complete"})
        except Exception as e:
            training_state.update({"running": False, "error": str(e),
                                    "message": "Training failed"})

    threading.Thread(target=_train, daemon=True).start()
    return jsonify({"success": True, "message": "Training started"})



@app.route("/train/status")
def train_status():
    from generate import MODEL_WEIGHTS
    return jsonify({
        **training_state,
        "model_ready": os.path.exists(MODEL_WEIGHTS),
    })



@app.route("/status")
def status():
    from generate import MODEL_WEIGHTS, NOTES_CACHE, GENERATED_DIR
    files = [f for f in os.listdir(GENERATED_DIR) if f.endswith(".mid")]
    return jsonify({
        "model_cached":  os.path.exists(MODEL_WEIGHTS),
        "notes_cached":  os.path.exists(NOTES_CACHE),
        "generated_files": len(files),
        "files": sorted(files, reverse=True)[:10],
    })



@app.route("/files")
def list_files():
    files = sorted(
        [f for f in os.listdir(GENERATED_DIR) if f.endswith(".mid")],
        reverse=True,
    )
    return jsonify({"files": files})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
