import os
import pickle
import random
import numpy as np
from music21 import corpus, note, chord, stream, instrument, tempo as m21tempo
from music21 import converter, environment


try:
    us = environment.UserSettings()
    us['warnings'] = 0
except Exception:
    pass


BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR     = os.path.join(BASE_DIR, "models")
GENERATED_DIR  = os.path.join(BASE_DIR, "generated")
NOTES_CACHE    = os.path.join(MODELS_DIR, "notes.pkl")
MAPPING_CACHE  = os.path.join(MODELS_DIR, "mapping.pkl")
MODEL_WEIGHTS  = os.path.join(MODELS_DIR, "lstm.weights.h5")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(GENERATED_DIR, exist_ok=True)



GENRE_COMPOSERS = {
    "classical": ["bach", "beethoven", "mozart"],
    "jazz":      ["joplin"],
    "folk":      ["essenFolksong"],
    "all":       ["bach", "beethoven", "mozart", "joplin"],
}

def collect_notes(genre="classical", max_files=40):
   
    if os.path.exists(NOTES_CACHE):
        with open(NOTES_CACHE, "rb") as f:
            return pickle.load(f)

    notes = []
    composers = GENRE_COMPOSERS.get(genre, GENRE_COMPOSERS["classical"])

    for composer in composers:
        try:
            paths = corpus.getComposer(composer)
        except Exception:
            continue

        for i, path in enumerate(paths):
            if i >= max_files // len(composers):
                break
            try:
                midi = corpus.parse(path)
                parts = midi.flatten()
                for element in parts.notes:
                    if isinstance(element, note.Note):
                        notes.append(str(element.pitch))
                    elif isinstance(element, chord.Chord):
                        notes.append(".".join(
                            str(n) for n in element.normalOrder))
            except Exception:
                continue

    if not notes:
        
        scale = ["C4","D4","E4","G4","A4","C5","D5","E5","G5","A5"]
        notes = (scale * 200)

    with open(NOTES_CACHE, "wb") as f:
        pickle.dump(notes, f)

    return notes



def prepare_sequences(notes, seq_len=100):
    """
    Encode notes as integers and build (X, y) training pairs.
    Returns X, y (numpy), n_vocab, int_to_note mapping.
    """
    pitches      = sorted(set(notes))
    n_vocab      = len(pitches)
    note_to_int  = {n: i for i, n in enumerate(pitches)}
    int_to_note  = {i: n for i, n in enumerate(pitches)}

    X_raw, y_raw = [], []
    for i in range(len(notes) - seq_len):
        seq_in  = notes[i : i + seq_len]
        seq_out = notes[i + seq_len]
        X_raw.append([note_to_int[n] for n in seq_in])
        y_raw.append(note_to_int[seq_out])

    n_patterns = len(X_raw)
    X = np.reshape(X_raw, (n_patterns, seq_len, 1))
    X = X / float(n_vocab)                         

    try:
        import tensorflow as tf
        y = tf.keras.utils.to_categorical(y_raw, num_classes=n_vocab)
    except Exception:
       
        y = np.zeros((len(y_raw), n_vocab))
        for idx, val in enumerate(y_raw):
            y[idx, val] = 1.0

    mapping = {"note_to_int": note_to_int, "int_to_note": int_to_note,
               "pitches": pitches, "n_vocab": n_vocab}
    with open(MAPPING_CACHE, "wb") as f:
        pickle.dump(mapping, f)

    return X, y, n_vocab, int_to_note



def build_model(n_vocab, seq_len=100):
    
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        LSTM, Dense, Dropout, BatchNormalization, Activation)
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        LSTM(512, input_shape=(seq_len, 1), return_sequences=True),
        Dropout(0.3),
        LSTM(256, return_sequences=False),
        Dropout(0.3),
        Dense(256),
        BatchNormalization(),
        Activation("relu"),
        Dense(n_vocab, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model



def train_model(model, X, y, epochs=50, batch_size=64):
    """
    Train with ModelCheckpoint + EarlyStopping.
    Saves best weights to models/lstm.weights.h5
    """
    import tensorflow as tf
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

    callbacks = [
        ModelCheckpoint(MODEL_WEIGHTS, save_best_only=True,
                        save_weights_only=True,
                        monitor="val_loss", verbose=0),
        EarlyStopping(patience=10, restore_best_weights=True, verbose=0),
    ]

    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )
    return history



def generate_notes(model, notes, mapping, n_generate=200,
                   temperature=0.8, seq_len=100):
    """
    Seed the model with a random excerpt, then auto-regressively
    sample n_generate notes applying temperature scaling.
    Returns list of pitch strings.
    """
    note_to_int = mapping["note_to_int"]
    int_to_note = mapping["int_to_note"]
    n_vocab     = mapping["n_vocab"]

    
    start = np.random.randint(0, len(notes) - seq_len)
    seed  = [note_to_int[n] for n in notes[start : start + seq_len]]

    generated = []
    for _ in range(n_generate):
        x   = np.reshape(seed[-seq_len:], (1, seq_len, 1)) / float(n_vocab)
        raw = model.predict(x, verbose=0)[0]

       
        raw = np.log(raw.astype("float64") + 1e-8) / temperature
        raw = np.exp(raw - raw.max())         
        probs = raw / raw.sum()

        idx = np.random.choice(len(probs), p=probs)
        generated.append(int_to_note[idx])
        seed.append(idx)

    return generated



def notes_to_midi(generated_notes, output_filename,
                  bpm=120, offset_step=0.5):
    """
    Build a music21 Stream from generated pitch strings and write MIDI.
    Chords encoded as "0.4.7" (normal-order integers); singles as "C4".
    Returns full path of saved file.
    """
    output_path = os.path.join(GENERATED_DIR, output_filename)

    midi_stream = stream.Stream()
    piano       = instrument.Piano()
    midi_stream.append(piano)
    midi_stream.append(m21tempo.MetronomeMark(number=bpm))

    offset = 0.0
    for idx, pattern in enumerate(generated_notes):
        timing_jitter = random.uniform(-0.035, 0.035)
        placed_offset = max(0.0, offset + timing_jitter)
        duration = offset_step * random.uniform(0.78, 1.08)
        velocity = random.randint(58, 88) + (8 if idx % 8 == 0 else 0)
        velocity = min(100, velocity)

        if "." in pattern:                        
            try:
                notes_in_chord = [
                    note.Note(int(n) + 60) for n in pattern.split(".")
                ]
                for n in notes_in_chord:
                    n.quarterLength = duration
                    n.volume.velocity = velocity
                c = chord.Chord(notes_in_chord)
                c.quarterLength = duration
                c.offset = placed_offset
                midi_stream.append(c)
            except Exception:
                pass
        else:                                     
            try:
                n = note.Note(pattern)
                n.quarterLength = duration
                n.volume.velocity = velocity
                n.offset = placed_offset
                midi_stream.append(n)
            except Exception:
                pass
        offset += offset_step

    midi_stream.write("midi", fp=output_path)
    return output_path


# ─────────────────────────────────────────────
# QUICK-GENERATE (no training — rule-based fallback)
# ─────────────────────────────────────────────
def quick_generate_midi(n_notes=200, bpm=120, temperature=0.8,
                        genre="classical", output_filename=None):
    """
    Lightweight rule-based generator that works without a trained model.
    Uses Markov-style transitions over diatonic scales.
    """
    if output_filename is None:
        import time
        output_filename = f"generated_{int(time.time())}.mid"

    SCALES = {
        "classical": ["C4","D4","E4","F4","G4","A4","B4",
                      "C5","D5","E5","F5","G5","A5","B5"],
        "jazz":      ["C4","Eb4","F4","F#4","G4","Bb4",
                      "C5","Eb5","F5","G5"],
        "folk":      ["G3","A3","B3","C4","D4","E4","F#4",
                      "G4","A4","B4","C5","D5"],
        "blues":     ["C4","Eb4","F4","F#4","G4","Bb4","C5"],
    }
    scale = SCALES.get(genre, SCALES["classical"])

    # Transition weights: prefer stepwise motion
    def next_note(current_idx, scale, temperature):
        n = len(scale)
        weights = np.zeros(n)
        for i in range(n):
            dist = abs(i - current_idx)
            weights[i] = np.exp(-dist * temperature)
        weights /= weights.sum()
        return np.random.choice(n, p=weights)

    idx      = len(scale) // 2
    pitches  = []
    for _ in range(n_notes):
        pitches.append(scale[idx])
        idx = next_note(idx, scale, temperature)

    path = notes_to_midi(pitches, output_filename, bpm=bpm)
    return path, pitches



def full_pipeline(genre="classical", epochs=50, n_generate=200,
                  bpm=120, temperature=0.8, output_filename=None):
    import time
    if output_filename is None:
        output_filename = f"generated_{genre}_{int(time.time())}.mid"

    print("[1/5] Collecting MIDI data...")
    notes = collect_notes(genre=genre)

    print("[2/5] Preparing sequences...")
    X, y, n_vocab, int_to_note = prepare_sequences(notes)

    print("[3/5] Building model...")
    model = build_model(n_vocab)

    if os.path.exists(MODEL_WEIGHTS):
        print("      Loading cached weights...")
        model.load_weights(MODEL_WEIGHTS)
    else:
        print("[4/5] Training model...")
        train_model(model, X, y, epochs=epochs)

    with open(MAPPING_CACHE, "rb") as f:
        mapping = pickle.load(f)

    print("[5/5] Generating music...")
    generated = generate_notes(model, notes, mapping,
                                n_generate=n_generate,
                                temperature=temperature)

    path = notes_to_midi(generated, output_filename, bpm=bpm)
    print(f"      Saved → {path}")
    return path, generated


# ─────────────────────────────────────────────
if __name__ == "__main__":
   
    path, _ = full_pipeline(genre="classical", epochs=10,
                             n_generate=100, bpm=120, temperature=0.8)
    print(f"Done: {path}")
