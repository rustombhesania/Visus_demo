"""
ViSuS — Music Signal Analysis v7
Run: streamlit run app.py
"""
import warnings; warnings.filterwarnings("ignore")
import io, os, tempfile

import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf
import streamlit as st
from scipy.signal import butter, filtfilt, find_peaks, sosfilt
from librosa.sequence import dtw as librosa_dtw

try:
    import pywt; HAS_PYWT = True
except ImportError:
    HAS_PYWT = False

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
SR     = 22050
HOP    = 512
N_MELS = 128
N_MFCC = 13
N_CQT  = 84
CQT_FMIN  = librosa.note_to_hz('C1')
MIDI_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

def midi_name(midi): return f"{MIDI_NAMES[midi%12]}{midi//12-1}"

# ─────────────────────────────────────────────────────────────
# Global color palette — each recording keeps its color everywhere
# ─────────────────────────────────────────────────────────────
REC_COLORS = [
    '#2196F3',  # 1 — blue
    '#FF9800',  # 2 — orange
    '#4CAF50',  # 3 — green
    '#E91E63',  # 4 — pink/red
    '#9C27B0',  # 5 — purple
    '#00BCD4',  # 6 — cyan
    '#FF5722',  # 7 — deep orange
    '#607D8B',  # 8 — blue-grey
]

def rec_color(idx): return REC_COLORS[idx % len(REC_COLORS)]

# Mel frequency band labels for y-axis annotation
MEL_BANDS_HZ = [
    (0,   16,  "Sub-bass\n<80Hz"),
    (16,  32,  "Bass\n80–300Hz"),
    (32,  64,  "Low-mids\n300–1kHz"),
    (64,  96,  "Mids\n1–4kHz"),
    (96,  112, "Presence\n4–8kHz"),
    (112, 128, "Air\n>8kHz"),
]
def cqt_bin_to_note(b):
    freqs = librosa.cqt_frequencies(N_CQT, fmin=CQT_FMIN, bins_per_octave=12)
    midi  = int(round(12*np.log2(freqs[b]/440)+69))
    return midi_name(midi)

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="ViSuS", layout="wide", initial_sidebar_state="expanded")
st.title("ViSuS — Music Signal Analysis")

# ─────────────────────────────────────────────────────────────
# Upload mode selection
# ─────────────────────────────────────────────────────────────
upload_mode = st.radio(
    "Comparison mode",
    ["2 Rehearsals", "N Rehearsals (batch)"],
    horizontal=True,
    key="upload_mode",
    help="2 Rehearsals: detailed pairwise analysis. N Rehearsals: compare multiple takes at once."
)

AUDIO_TYPES = ["mp3","wav","flac","ogg","m4a","mp4","aac"]

# ─────────────────────────────────────────────────────────────
# Score upload mode — supports two usage patterns:
#   "Same setup, multiple takes" — one MIDI score applies to every
#     recording (e.g. same player, same piece, several takes to compare).
#   "Same song, different styles" — each recording may need its own
#     score (e.g. different arrangements/instrumentation of the same piece).
# ─────────────────────────────────────────────────────────────
score_mode = st.radio(
    "Score upload mode",
    ["One score for all recordings", "Different score per recording"],
    index=0, horizontal=True, key="score_upload_mode",
    help="Same setup / multiple takes → one score for all. "
         "Same song / different styles or arrangements → per-recording scores.")

if score_mode == "One score for all recordings":
    shared_midi = st.file_uploader(
        "Score for all recordings (optional)",
        type=["mid","midi"], key="fu_midi_shared",
        help="This single MIDI file will be used as the reference score "
             "for every recording uploaded below.")
else:
    shared_midi = None

# ─────────────────────────────────────────────────────────────
# Upload — 2 recording mode
# ─────────────────────────────────────────────────────────────
if upload_mode == "2 Rehearsals":
    c1, c2 = st.columns(2)
    with c1:
        file_a   = st.file_uploader("Rehearsal 1", type=AUDIO_TYPES, key="fu_a")
        name_a   = st.text_input("Name", value="Rehearsal 1", key="name_a",
                                 help="Give this recording a name")
        midi_a   = (shared_midi if score_mode == "One score for all recordings"
                    else st.file_uploader("Score for Rehearsal 1 (optional)",
                                          type=["mid","midi"], key="fu_midi_a"))
    with c2:
        file_b   = st.file_uploader("Rehearsal 2", type=AUDIO_TYPES, key="fu_b")
        name_b   = st.text_input("Name", value="Rehearsal 2", key="name_b")
        midi_b   = (shared_midi if score_mode == "One score for all recordings"
                    else st.file_uploader("Score for Rehearsal 2 (optional)",
                                          type=["mid","midi"], key="fu_midi_b"))

    files       = [file_a, file_b]
    names       = [name_a, name_b]
    midi_files  = [midi_a, midi_b]
    n_recs = 2

# ─────────────────────────────────────────────────────────────
# Upload — N recording mode
# ─────────────────────────────────────────────────────────────
else:
    st.markdown("Upload between 2 and 8 rehearsal recordings. All pairwise comparisons will be computed.")
    n_recs_input = st.slider("Number of rehearsals", min_value=2, max_value=8, value=3, key="n_recs_slider")
    files = []; names = []; midi_files = []
    cols = st.columns(min(n_recs_input, 4))
    for i in range(n_recs_input):
        col = cols[i % len(cols)]
        with col:
            f = st.file_uploader(f"Rehearsal {i+1}", type=AUDIO_TYPES, key=f"fu_{i}")
            n = st.text_input("Name", value=f"Rehearsal {i+1}", key=f"name_{i}")
            if score_mode == "One score for all recordings":
                m = shared_midi
            else:
                m = st.file_uploader("Score (optional)",
                                      type=["mid","midi"], key=f"fu_midi_{i}")
            files.append(f); names.append(n); midi_files.append(m)
    n_recs = n_recs_input

if score_mode == "One score for all recordings":
    st.caption(
        "📄 **One score for all** — the uploaded MIDI (if any) becomes the "
        "reference for every recording. Best for: same performer/setup, "
        "comparing multiple takes of the same piece.")
else:
    st.caption(
        "📄 **Per-recording scores** — upload a different MIDI for each "
        "recording, or leave any blank to fall back to the auto-generated "
        "consensus. Best for: the same piece played in different styles or "
        "arrangements, where a single shared score wouldn't fit all of them.")

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ViSuS")
    for i,(f,n) in enumerate(zip(files,names)):
        if f: st.caption(f"{n}: {f.name[:30]}")
    st.markdown("---")

    st.subheader("Features")
    with st.expander("Core", expanded=True):
        do_mel      = st.checkbox("Mel Spectrogram",     value=True)
        do_cqt      = st.checkbox("CQT + Key Detection", value=True)
        do_mfcc     = st.checkbox("MFCCs",               value=True)
        do_chroma   = st.checkbox("Chroma",              value=True)
        do_spectral = st.checkbox("Spectral Features",   value=True)
        do_onset    = st.checkbox("Onset & Beats",       value=True)
        do_sms      = st.checkbox("SMS / HPSS",          value=True)
    with st.expander("Optional", expanded=False):
        do_stft      = st.checkbox("STFT",               value=False)
        do_cwt       = st.checkbox("CWT / Scalogram",    value=False)
        do_gammatone = st.checkbox("Gammatone",          value=False)
        do_tonnetz   = st.checkbox("Tonnetz",            value=False)
        do_zcr       = st.checkbox("ZCR",                value=False)
        do_reverb    = st.checkbox("RT60 / Reverb",      value=False)

    st.markdown("---")
    st.markdown("**⏱ DTW Alignment**")
    dtw_strategy_choice = st.radio(
        "Strategy", ["Chroma + Onset (combo)", "Chroma only", "Onset only", "Off"],
        index=0, key="dtw_strategy_radio",
        help="Chroma aligns pitch. Onset aligns rhythm. Combo blends both 50/50.")
    do_dtw = dtw_strategy_choice != "Off"
    dtw_strategy = {"Chroma + Onset (combo)":"combo",
                    "Chroma only":"chroma",
                    "Onset only":"onset",
                    "Off":"none"}[dtw_strategy_choice]

    st.markdown("---")
    st.subheader("🔍 Zoom Window")
    zoom_mode = st.radio("Mode", ["Full recording","Time (seconds)","Musical bars"],
                         key="zoom_mode", label_visibility="collapsed")
    zoom_start_sb = 0.0; zoom_end_sb = None
    bar_start_sb = bar_end_sb = time_sig_sb = None
    if zoom_mode == "Time (seconds)":
        zoom_start_sb = st.number_input("Start (s)", min_value=0.0, value=0.0, step=0.5)
        zoom_end_sb   = st.number_input("End (s)",   min_value=0.0, value=30.0, step=0.5)
    elif zoom_mode == "Musical bars":
        bar_start_sb = st.number_input("From bar", min_value=1, value=1, step=1)
        bar_end_sb   = st.number_input("To bar",   min_value=1, value=4, step=1)
        time_sig_sb  = st.selectbox("Time sig", [4,3,6,5,7], index=0)

    st.markdown("---")
    diff_threshold = st.slider("Waveform diff threshold", 0.0, 0.3, 0.03, 0.01)
    st.markdown("---")
    run_btn = st.button("▶ Run Analysis", type="primary", use_container_width=True)

    st.markdown("---")
    st.caption("ViSuS · HIWI Research Demo")

# ─────────────────────────────────────────────────────────────
# Audio load helper (cached so same file isn't re-decoded)
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_audio(data, fname):
    suf = os.path.splitext(fname)[-1] or ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suf) as f:
        f.write(data); tmp = f.name
    y, _ = librosa.load(tmp, sr=SR, mono=True)
    os.unlink(tmp); return y

def wav_bytes(y):
    y = np.clip(y/(np.max(np.abs(y))+1e-8),-1,1).astype(np.float32)
    b = io.BytesIO(); sf.write(b,y,SR,format='WAV'); b.seek(0); return b.read()

def bp_bytes(y, lo, hi):
    nyq=SR/2; l,h=max(lo,20)/nyq,min(hi,nyq-1)/nyq
    if l>=h: return None
    sos=butter(4,[l,h],btype='band',output='sos')
    yf=sosfilt(sos,y).astype(np.float32)
    yf=np.clip(yf/(np.max(np.abs(yf))+1e-8),-1,1)
    b=io.BytesIO(); sf.write(b,yf,SR,format='WAV'); b.seek(0); return b.read()

def hpss_bytes(y, component='harmonic'):
    D=librosa.stft(y,n_fft=2048,hop_length=HOP)
    H,P=librosa.decompose.hpss(D,margin=3.0)
    yo=librosa.istft(H if component=='harmonic' else P,hop_length=HOP,length=len(y))
    return wav_bytes(yo)

# ─────────────────────────────────────────────────────────────
# Key & Mode detection
# ─────────────────────────────────────────────────────────────
MODE_INTERVALS = {
    "Ionian (Major)":[0,2,4,5,7,9,11],  "Dorian":[0,2,3,5,7,9,10],
    "Phrygian":[0,1,3,5,7,8,10],        "Lydian":[0,2,4,6,7,9,11],
    "Mixolydian":[0,2,4,5,7,9,10],      "Aeolian (Minor)":[0,2,3,5,7,8,10],
    "Locrian":[0,1,3,5,6,8,10],
}
MODE_CHARACTER = {
    "Ionian (Major)":   "Bright, happy, resolved.",
    "Dorian":           "Minor with a raised 6th — jazzy, hopeful.",
    "Phrygian":         "Dark, tense, Spanish flavour.",
    "Lydian":           "Dreamy, floating, raised 4th.",
    "Mixolydian":       "Major but flattened 7th — bluesy, dominant.",
    "Aeolian (Minor)":  "Natural minor — sad, introspective.",
    "Locrian":          "Unstable, dissonant, rarely used.",
}

def detect_key(chroma):
    major=[6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88]
    minor=[6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17]
    cm=chroma.mean(1); cm/=(cm.sum()+1e-10)
    best,score,is_min=0,-np.inf,False
    for k in range(12):
        sm=np.corrcoef(cm,np.roll(major,k))[0,1]
        sn=np.corrcoef(cm,np.roll(minor,k))[0,1]
        if sm>score: score,best,is_min=sm,k,False
        if sn>score: score,best,is_min=sn,k,True
    return MIDI_NAMES[best],is_min

def detect_mode(chroma):
    cm=chroma.mean(1); cm/=(cm.sum()+1e-10)
    best_root,best_mode,best_score=0,"Ionian (Major)",-np.inf
    for mode_name,intervals in MODE_INTERVALS.items():
        for root in range(12):
            pcs=set((root+i)%12 for i in intervals)
            score=sum(cm[p] for p in pcs)-0.5*sum(cm[p] for p in range(12) if p not in pcs)
            if score>best_score: best_score,best_root,best_mode=score,root,mode_name
    return MIDI_NAMES[best_root],best_mode,round(float(best_score),4)

# ─────────────────────────────────────────────────────────────
# RT60
# ─────────────────────────────────────────────────────────────
def estimate_rt60(y):
    D=np.abs(librosa.stft(y,n_fft=2048,hop_length=HOP))**2
    edb=10*np.log10(D.sum(0)+1e-10); edb-=edb.max()
    fd=HOP/SR; peaks,_=find_peaks(edb,prominence=6,distance=int(0.2/fd))
    rates=[]
    for p in peaks:
        win=int(3/fd); seg=edb[p:min(p+win,len(edb))]
        cut=len(seg)
        for k in range(1,len(seg)):
            if seg[k]>seg[k-1]+2: cut=k; break
        seg=seg[:cut]
        if len(seg)<int(0.1/fd) or seg[-1]-seg[0]>-3: continue
        sl,_=np.polyfit(np.arange(len(seg))*fd,seg,1)
        if sl<-2: rates.append(sl)
    if len(rates)>=2:
        rt60=float(np.clip(abs(-60/np.median(rates)),0.05,10))
    else:
        rt60=float(np.clip(abs(np.percentile(edb,5))/40,0.1,3))
    return round(rt60,3),edb

# ─────────────────────────────────────────────────────────────
# Gammatone & CWT
# ─────────────────────────────────────────────────────────────
def gammatone_filterbank(y, n_filters=32, f_min=50, f_max=8000):
    def hz2erb(f): return 21.4*np.log10(1+f/229)
    def erb2hz(e): return 229*(10**(e/21.4)-1)
    cfs=erb2hz(np.linspace(hz2erb(f_min),hz2erb(f_max),n_filters))
    nf=1+(len(y)-1)//HOP; E=np.zeros((n_filters,nf))
    for i,fc in enumerate(cfs):
        bw=1.019*(24.7*(4.37*fc/1000+1)); lo,hi=max(fc-bw/2,1),min(fc+bw/2,SR/2-1)
        if lo>=hi or hi>=SR/2: continue
        try:
            b,a=butter(2,[lo/(SR/2),hi/(SR/2)],btype='band')
            flt=filtfilt(b,a,y)
            for j in range(nf):
                s,e=j*HOP,min((j+1)*HOP,len(flt))
                E[i,j]=np.sqrt(np.mean(flt[s:e]**2)+1e-10)
        except: pass
    E=np.log1p(E); mn,mx=E.min(),E.max()
    if mx>mn: E=(E-mn)/(mx-mn)
    return E

def compute_cwt(y, n_scales=32):
    if not HAS_PYWT: return None
    step=max(1,len(y)//3000); yd=y[::step]
    scales=np.geomspace(2,64,n_scales)
    coef,_=pywt.cwt(yd,scales,'cmor1.5-1.0')
    P=np.abs(coef)**2; mn,mx=P.min(),P.max()
    if mx>mn: P=(P-mn)/(mx-mn)
    return P

# ─────────────────────────────────────────────────────────────
# Run analysis (single recording)
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_analysis(y_bytes,
                 do_mel,do_cqt,do_mfcc,do_chroma,do_spectral,do_onset,do_sms,
                 do_stft,do_cwt,do_gammatone,do_tonnetz,do_zcr,do_reverb):
    y=np.frombuffer(y_bytes,dtype=np.float32)
    R={"dur":len(y)/SR}

    needs_stft = do_stft or do_mel or do_sms or do_reverb or do_onset
    needs_cqt  = do_cqt or do_chroma or do_tonnetz

    if needs_stft:
        D=librosa.stft(y,n_fft=2048,hop_length=HOP)
    if do_stft:
        R["stft"]=librosa.amplitude_to_db(np.abs(D),ref=np.max)
    if do_mel:
        R["mel"]=librosa.power_to_db(librosa.feature.melspectrogram(y=y,sr=SR,n_mels=N_MELS,hop_length=HOP),ref=np.max)
    if needs_cqt:
        C=np.abs(librosa.cqt(y,sr=SR,hop_length=HOP,n_bins=N_CQT,bins_per_octave=12))
    if do_cqt:
        R["cqt"]=librosa.amplitude_to_db(C,ref=np.max)
    if do_mfcc:
        m=librosa.feature.mfcc(y=y,sr=SR,n_mfcc=N_MFCC,hop_length=HOP)
        R["mfcc_means"]=m.mean(1); R["mfcc_stds"]=m.std(1)
        # Normalize per coefficient for display — each coeff scaled to [-1, +1]
        # using its own frame-level range so all 12 are visually comparable.
        # Without this, coefficients 1-4 dominate and 5-12 appear flat near zero.
        coeff_min  = m.min(1)   # min per coefficient across all frames
        coeff_max  = m.max(1)   # max per coefficient across all frames
        coeff_range = coeff_max - coeff_min
        coeff_range[coeff_range < 1e-8] = 1.0   # avoid divide-by-zero
        # Normalize mean to [-1, +1] relative to that coefficient's own range
        R["mfcc_means_z"] = 2 * (m.mean(1) - coeff_min) / coeff_range - 1.0
    if do_chroma:
        R["chroma"]=librosa.feature.chroma_cqt(y=y,sr=SR,hop_length=HOP)
        ka,mi=detect_key(R["chroma"]); R["key"]=f"{ka} {'minor' if mi else 'major'}"
        root,mode,_=detect_mode(R["chroma"])
        R["mode"]=f"{root} {mode}"; R["root"]=root; R["mode_name"]=mode
    if do_spectral:
        R["centroid"] =librosa.feature.spectral_centroid(y=y,sr=SR,hop_length=HOP)[0]
        R["bandwidth"]=librosa.feature.spectral_bandwidth(y=y,sr=SR,hop_length=HOP)[0]
        R["rolloff"]  =librosa.feature.spectral_rolloff(y=y,sr=SR,hop_length=HOP)[0]
        R["flatness"] =librosa.feature.spectral_flatness(y=y,hop_length=HOP)[0]
        R["rms"]      =librosa.feature.rms(y=y,hop_length=HOP)[0]
        if do_zcr:
            R["zcr"]=librosa.feature.zero_crossing_rate(y,hop_length=HOP)[0]
    if do_onset:
        oa=librosa.onset.onset_strength(y=y,sr=SR,hop_length=HOP)
        R["onset"]=oa
        _,beats=librosa.beat.beat_track(onset_envelope=oa,sr=SR,hop_length=HOP)
        R["beats"]=beats
        R["beat_reg"]=round(float(np.std(np.diff(beats))) if len(beats)>2 else 0,3)

        # Tempo octave-error correction: librosa.beat.tempo() can lock
        # onto 2x or 0.5x the true tempo, especially on short/repetitive
        # audio. beat_track() uses a different (dynamic-programming +
        # internal prior) approach and is generally more robust to this
        # failure mode, so cross-check the two and prefer the
        # beat-derived value when they disagree by roughly an octave.
        naive_tempo = float(np.atleast_1d(librosa.beat.tempo(onset_envelope=oa,sr=SR,hop_length=HOP))[0])
        beat_times = librosa.frames_to_time(beats, sr=SR, hop_length=HOP)
        if len(beat_times) > 2:
            from_beats_tempo = 60.0/float(np.median(np.diff(beat_times)))
        else:
            from_beats_tempo = naive_tempo
        _ratio = naive_tempo/from_beats_tempo if from_beats_tempo > 0 else 1.0
        if 1.8 <= _ratio <= 2.2 or 0.45 <= _ratio <= 0.55:
            R["tempo"] = round(from_beats_tempo, 1)
            R["tempo_raw"] = round(naive_tempo, 1)  # kept for debugging/logging only
        else:
            R["tempo"] = round(naive_tempo, 1)
    else:
        R["tempo"]=120.0; R["beat_reg"]=0.0
    if do_sms and needs_stft:
        H,P=librosa.decompose.hpss(D,margin=3.0)
        R["sms_h"]=librosa.amplitude_to_db(np.abs(H),ref=np.max)
        R["sms_p"]=librosa.amplitude_to_db(np.abs(P),ref=np.max)
        he=np.sum(np.abs(H)**2,0)
        R["sms_ratio"]=he/(he+np.sum(np.abs(P)**2,0)+1e-10)
    if do_tonnetz:
        R["tonnetz"]=librosa.feature.tonnetz(y=librosa.effects.harmonic(y),sr=SR)
    if do_reverb:
        R["rt60"],R["decay"]=estimate_rt60(y)
    if do_cwt:
        R["cwt"]=compute_cwt(y)
    if do_gammatone:
        R["gam"]=gammatone_filterbank(y)
    # Global feature ranges (for shared y-axis in plots)
    for k in ["centroid","bandwidth","rolloff","flatness","rms"]:
        if k in R: R[f"{k}_range"]=(float(R[k].min()),float(R[k].max()))
    return R

# ─────────────────────────────────────────────────────────────
# DTW alignment between two result dicts
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def safe_normalize_chroma(c):
    """
    Normalize chroma so each frame has unit L2 norm.
    Frames that are all-zero (silence) get replaced with a uniform
    distribution (1/12 per pitch class) to avoid NaN in cosine distance.
    """
    norms = np.linalg.norm(c, axis=0, keepdims=True)          # shape (1, T)
    silent = (norms < 1e-6).flatten()                          # frames with no energy
    c = c / (norms + 1e-10)                                    # normalize all frames
    c[:, silent] = 1.0 / c.shape[0]                           # replace silent frames
    return c.astype(np.float32)

def dtw_on_feature(feat_a_chroma, feat_b_chroma, strategy='combo',
                   onset_a=None, onset_b=None):
    """
    DTW alignment using one of three strategies:
      'chroma'  — chroma features only
      'onset'   — onset strength envelope only
      'combo'   — chroma + onset blended 50/50 (default)
    feat_a_chroma / feat_b_chroma: (12, T) chroma arrays — always required.
    onset_a / onset_b: (T,) onset arrays — required for onset and combo.
    Returns warping path wp of shape (N, 2).
    """
    def safe_norm1d(x):
        x = np.array(x, dtype=np.float32)
        mx = x.max()
        if mx < 1e-8: return np.full_like(x, 1.0/max(len(x),1))
        normed = x/mx; normed[normed<1e-8] = 1e-8
        return normed

    ca = safe_normalize_chroma(feat_a_chroma.astype(np.float32))
    cb = safe_normalize_chroma(feat_b_chroma.astype(np.float32))

    if strategy == 'chroma':
        fa, fb = ca, cb

    elif strategy == 'onset':
        if onset_a is None or onset_b is None:
            fa, fb = ca, cb  # fallback to chroma
        else:
            fa = safe_norm1d(onset_a).reshape(1, -1)
            fb = safe_norm1d(onset_b).reshape(1, -1)

    elif strategy == 'combo':
        if onset_a is None or onset_b is None:
            fa, fb = ca, cb  # fallback to chroma
        else:
            oa_n = safe_norm1d(onset_a)
            ob_n = safe_norm1d(onset_b)
            nc_a = min(ca.shape[1], len(oa_n))
            nc_b = min(cb.shape[1], len(ob_n))
            # Expand onset to 12×T and blend 50/50 with chroma
            oa_exp = np.tile(oa_n[:nc_a], (ca.shape[0], 1))
            ob_exp = np.tile(ob_n[:nc_b], (cb.shape[0], 1))
            fa = 0.5 * ca[:, :nc_a] + 0.5 * oa_exp
            fb = 0.5 * cb[:, :nc_b] + 0.5 * ob_exp
    else:
        fa, fb = ca, cb  # safe fallback

    fa = np.nan_to_num(fa, nan=1e-8, posinf=1.0, neginf=0.0)
    fb = np.nan_to_num(fb, nan=1e-8, posinf=1.0, neginf=0.0)
    _, wp = librosa_dtw(X=fa, Y=fb, metric='cosine')
    return wp


def apply_warp(arr, wp, n_a):
    pa,pb=wp[:,0],wp[:,1]
    if arr.ndim==2:
        out=np.zeros((arr.shape[0],n_a),dtype=np.float32)
        for i in range(n_a):
            idx=pb[pa==i]
            if len(idx): out[:,i]=arr[:,np.clip(idx,0,arr.shape[1]-1)].mean(1)
    else:
        out=np.zeros(n_a,dtype=np.float32)
        for i in range(n_a):
            idx=pb[pa==i]
            if len(idx): out[i]=arr[np.clip(idx,0,len(arr)-1)].mean()
    return out

def coarse_rms_offset(rms_ref, rms_other, sr=SR, hop=HOP):
    """Coarse alignment via RMS cross-correlation. Returns frame offset."""
    from scipy.signal import correlate
    n = min(len(rms_ref), len(rms_other), 2000)
    a = (rms_ref[:n] - rms_ref[:n].mean()) / (rms_ref[:n].std() + 1e-8)
    b = (rms_other[:n] - rms_other[:n].mean()) / (rms_other[:n].std() + 1e-8)
    corr = correlate(a.astype(np.float64), b.astype(np.float64), mode="full")
    lag = int(np.argmax(corr)) - (n - 1)
    return int(np.clip(lag, -int(30*sr/hop), int(30*sr/hop)))

def align_to(R_ref, R_other, do_dtw_flag, dtw_strategy='combo'):
    """
    Two-stage alignment:
      Stage 1: Coarse RMS cross-correlation offset (shifts all arrays).
      Stage 2: Fine DTW using selected strategy (chroma / onset / combo).
    Stores warping paths: _wp_chroma, _wp_onset, _wp_combo.
    Default warp applied to feature arrays uses the selected strategy.
    """
    if not do_dtw_flag:
        return R_other

    SKIP_KEYS = {
        "mfcc_means","mfcc_stds","mfcc_means_z",
        "tempo","beat_reg","beats","key","mode","root",
        "mode_name","mode_char_a","mode_char_b",
        "rt60","decay","dur",
        "centroid_range","bandwidth_range","rolloff_range",
        "flatness_range","rms_range",
    }
    MIN_FRAMES = 50
    aligned = dict(R_other)

    # Stage 1 — Coarse RMS offset
    rms_offset = 0
    if "rms" in R_ref and "rms" in R_other:
        rms_offset = coarse_rms_offset(R_ref["rms"], R_other["rms"])
        aligned["_rms_offset_frames"]  = rms_offset
        aligned["_rms_offset_seconds"] = round(rms_offset * HOP / SR, 3)
        if rms_offset != 0:
            for k in list(aligned.keys()):
                if k.startswith("_") or k in SKIP_KEYS: continue
                v = aligned[k]
                if not isinstance(v, np.ndarray): continue
                if v.ndim == 1 and len(v) >= MIN_FRAMES:
                    if rms_offset > 0: aligned[k] = v[min(rms_offset,len(v)-1):]
                    else:
                        aligned[k] = np.concatenate([np.zeros(-rms_offset,dtype=v.dtype), v])
                elif v.ndim == 2 and v.shape[1] >= MIN_FRAMES:
                    if rms_offset > 0: aligned[k] = v[:,min(rms_offset,v.shape[1]-1):]
                    else:
                        aligned[k] = np.concatenate(
                            [np.zeros((v.shape[0],-rms_offset),dtype=v.dtype), v], axis=1)

    # Stage 2 — DTW strategies (chroma, onset, combo — no RMS)
    n_a = R_ref["chroma"].shape[1] if "chroma" in R_ref else None
    onset_ref = R_ref.get("onset")
    onset_b   = aligned.get("onset")

    if "chroma" in R_ref and "chroma" in aligned and n_a:
        try:
            aligned["_wp_chroma"] = dtw_on_feature(
                R_ref["chroma"], aligned["chroma"], strategy='chroma')
        except Exception: pass
        try:
            aligned["_wp_onset"] = dtw_on_feature(
                R_ref["chroma"], aligned["chroma"], strategy='onset',
                onset_a=onset_ref, onset_b=onset_b)
        except Exception: pass
        try:
            aligned["_wp_combo"] = dtw_on_feature(
                R_ref["chroma"], aligned["chroma"], strategy='combo',
                onset_a=onset_ref, onset_b=onset_b)
        except Exception: pass

    # Pick default warp based on selected strategy
    wp_key_map = {'chroma':'_wp_chroma','onset':'_wp_onset','combo':'_wp_combo'}
    _wp_pref = aligned.get(wp_key_map.get(dtw_strategy, '_wp_combo'))
    _wp_fall = aligned.get("_wp_chroma")
    wp_default = _wp_pref if _wp_pref is not None else _wp_fall

    if wp_default is not None and n_a:
        keys_2d = [k for k in ["stft","mel","cqt","cwt","gam","chroma",
                                "tonnetz","sms_h","sms_p"] if k in aligned]
        keys_1d = [k for k in ["centroid","bandwidth","rolloff","flatness",
                                "rms","zcr","onset","sms_ratio"] if k in aligned]
        for k in keys_2d:
            if aligned[k].shape[1] != n_a:
                aligned[k] = apply_warp(aligned[k].astype(np.float32), wp_default, n_a)
        for k in keys_1d:
            if len(aligned[k]) != n_a:
                aligned[k] = apply_warp(aligned[k].astype(np.float32), wp_default, n_a)

    aligned["_wp"] = wp_default  # backward compat
    return aligned




# ─────────────────────────────────────────────────────────────
# Song structure detection (on reference recording)
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def detect_structure(y_bytes, tempo):
    y=np.frombuffer(y_bytes,dtype=np.float32)
    mel=librosa.power_to_db(
        librosa.feature.melspectrogram(y=y,sr=SR,n_mels=64,hop_length=HOP),ref=np.max)
    try:
        bounds=librosa.segment.agglomerative(mel,8)
        bt=librosa.frames_to_time(bounds,sr=SR,hop_length=HOP)
    except:
        bt=np.linspace(0,len(y)/SR,9)
    spb=60/max(tempo,1); secs=[]
    labels=["Intro","A","B","A'","C","B'","Outro","Coda"]
    for i in range(len(bt)-1):
        t0,t1=float(bt[i]),float(bt[i+1])
        b0=int(t0/(spb*4))+1; b1=max(b0+1,int(t1/(spb*4))+1)
        secs.append((b0,b1,t0,t1,labels[i%len(labels)]))
    return secs

# ─────────────────────────────────────────────────────────────
# Divergence between two R dicts (1D, frame-level)
# ─────────────────────────────────────────────────────────────
def divergence_1d(Ra, Rb):
    from scipy.signal import savgol_filter
    arrays=[]
    for k in ["centroid","rms","onset","sms_ratio","flatness"]:
        if k in Ra and k in Rb:
            n=min(len(Ra[k]),len(Rb[k]))
            a=(Ra[k][:n]-Ra[k][:n].mean())/(Ra[k][:n].std()+1e-8)
            b=(Rb[k][:n]-Rb[k][:n].mean())/(Rb[k][:n].std()+1e-8)
            arrays.append(np.abs(a-b))
    if not arrays: return None,None
    nm=min(len(d) for d in arrays)
    combined=np.mean([d[:nm] for d in arrays],axis=0)
    if len(combined)>21:
        combined=savgol_filter(combined,min(51,len(combined)//4*2+1),3)
    combined=np.clip(combined,0,None)
    mx=combined.max()
    if mx>0: combined/=mx
    return np.arange(len(combined))*HOP/SR, combined

# ─────────────────────────────────────────────────────────────
# What Changed summary (rule-based, LLM-ready)
# ─────────────────────────────────────────────────────────────
def generate_summary(Ra, Rb, name_a, name_b):
    from scipy.signal import savgol_filter
    lines=[]
    ta,tb=Ra.get("tempo",120),Rb.get("tempo",120); td=tb-ta
    if abs(td)<1:
        lines.append(("🎯","Tempo: identical",f"Both at {ta:.1f} BPM."))
    elif abs(td)<3:
        lines.append(("🎯",f"Tempo: nearly same (Δ{td:+.1f} BPM)",f"{name_a}: {ta:.1f} · {name_b}: {tb:.1f} BPM."))
    else:
        faster=name_b if td>0 else name_a
        lines.append(("⚡",f"Tempo: {faster} is {abs(td):.1f} BPM faster",
            f"{name_a}: {ta:.1f} · {name_b}: {tb:.1f} BPM. Clearly audible difference."))
    ra,rb=Ra.get("beat_reg",0),Rb.get("beat_reg",0)
    if ra>0 and rb>0:
        pct=(ra-rb)/(ra+1e-10)*100
        tighter=name_b if rb<ra else name_a
        feel="locked in" if min(ra,rb)<1.5 else "fairly consistent" if min(ra,rb)<3 else "loose"
        lines.append(("🥁",f"Timing: {tighter} is {abs(pct):.0f}% tighter",
            f"{name_a}={ra:.2f} · {name_b}={rb:.2f} beat regularity. Tighter feel: {feel}."))
    if "mode" in Ra and "mode" in Rb:
        if Ra["mode"]==Rb["mode"]:
            lines.append(("🎹",f"Key & mode: both in {Ra['mode']}",
                MODE_CHARACTER.get(Ra.get("mode_name",""),"")))
        else:
            lines.append(("🎹",f"Key/mode differs: {name_a}={Ra['mode']} · {name_b}={Rb['mode']}",
                f"{name_a}: {MODE_CHARACTER.get(Ra.get('mode_name',''),'')} "
                f"{name_b}: {MODE_CHARACTER.get(Rb.get('mode_name',''),'')}" ))
    if "rms" in Ra and "rms" in Rb:
        rva,rvb=float(Ra["rms"].mean()),float(Rb["rms"].mean())
        pct=(rvb-rva)/(rva+1e-10)*100
        if abs(pct)<5:
            lines.append(("🔊","Loudness: same","Similar average energy."))
        else:
            louder=name_b if pct>0 else name_a
            lines.append(("🔊",f"Loudness: {louder} is {abs(pct):.0f}% louder",
                "Could be gain difference or performance intensity."))
    if "centroid" in Ra and "centroid" in Rb:
        cva,cvb=float(Ra["centroid"].mean()),float(Rb["centroid"].mean())
        dh=cvb-cva
        if abs(dh)>150:
            brighter=name_b if dh>0 else name_a
            lines.append(("🎨",f"Tone: {brighter} sounds brighter",
                f"Brighter recording has more high-frequency energy."))
        else:
            lines.append(("🎨","Tone: same brightness","Similar tonal colour."))
    if "sms_ratio" in Ra and "sms_ratio" in Rb:
        ha,hb=float(Ra["sms_ratio"].mean()),float(Rb["sms_ratio"].mean())
        dh=hb-ha
        if abs(dh)>0.03:
            more=name_b if dh>0 else name_a
            lines.append(("🎼",f"Texture: {more} is more melodic",
                f"{name_a}={ha:.2f} · {name_b}={hb:.2f} harmonic ratio."))
        else:
            lines.append(("🎼","Texture: same balance","Similar melodic/rhythmic content."))
    # Tone & texture summary (folded in from removed tab)
    tone_parts=[]
    if "centroid" in Ra and "centroid" in Rb:
        cva2=float(Ra["centroid"].mean()); cvb2=float(Rb["centroid"].mean())
        dh2=cvb2-cva2
        if abs(dh2)>150:
            tone_parts.append(f"{name_b if dh2>0 else name_a} sounds brighter")
        else:
            tone_parts.append("similar brightness")
    if "sms_ratio" in Ra and "sms_ratio" in Rb:
        def tex_w(h): return "very melodic" if h>0.75 else "melodic" if h>0.55 else "balanced" if h>0.4 else "rhythmic"
        wa2=tex_w(float(Ra["sms_ratio"].mean())); wb2=tex_w(float(Rb["sms_ratio"].mean()))
        if wa2!=wb2: tone_parts.append(f"{name_a} is {wa2}, {name_b} is {wb2}")
        else: tone_parts.append(f"both are {wa2} in texture")
    if "flatness" in Ra and "flatness" in Rb:
        def flat_w(f): return "clean" if f<0.05 else "slightly gritty" if f<0.15 else "noisy"
        wa3=flat_w(float(Ra["flatness"].mean())); wb3=flat_w(float(Rb["flatness"].mean()))
        if wa3!=wb3: tone_parts.append(f"{name_a} is {wa3}, {name_b} is {wb3}")
        else: tone_parts.append(f"both sound {wa3}")
    if tone_parts:
        lines.append(("🎵","Sound character","; ".join(tone_parts)+"."))
    # Most different moment
    times,div=divergence_1d(Ra,Rb)
    if times is not None:
        peak=int(np.argmax(div)); pt=float(times[peak])
        pm,ps=int(pt//60),int(pt%60)
        avg_t2=(Ra.get("tempo",120)+Rb.get("tempo",120))/2
        bar=int(pt/(60/max(avg_t2,1)*4))+1
        sev="large" if div[peak]>0.7 else "moderate" if div[peak]>0.4 else "small"
        lines.append(("📍",f"Most different: {pm}:{ps:02d} (bar {bar})",
            f"Divergence is {sev} at this point. Zoom in to investigate."))
    return lines

# ─────────────────────────────────────────────────────────────
# Shared plot helpers
# ─────────────────────────────────────────────────────────────
def spec_pair(title, ma, mb, la, lb, cmap="magma", ylabel="",
              yticks=None, col_a=None, col_b=None,
              add_mel_bands=False, ref_label="reference", cmp_label="aligned"):
    """
    Three-panel spectrogram: A (reference) | B (aligned) | Difference.
    Difference panel uses recording colors: col_b tint where B>A, col_a tint where A>B.
    col_a / col_b should be hex strings from REC_COLORS.
    """
    import matplotlib.colors as mcolors
    nc=min(ma.shape[1],mb.shape[1]); ma,mb=ma[:,:nc],mb[:,:nc]
    diff=mb-ma; vmax=float(np.percentile(np.abs(diff),95)) or 1.0
    n_rows=ma.shape[0]; ext=[zoom_start,zoom_end,0,n_rows]

    fig,axes=plt.subplots(1,3,figsize=(16,3.5))

    # A panel — border in col_a
    im_a=axes[0].imshow(ma,aspect='auto',origin='lower',cmap=cmap,interpolation='nearest',extent=ext)
    axes[0].set_title(f"{la}  [{ref_label}]",fontsize=8,color=col_a or 'white',fontweight='bold')
    axes[0].set_xlabel("Time (s)",fontsize=7); axes[0].set_ylabel(ylabel,fontsize=7)
    for spine in axes[0].spines.values():
        spine.set_edgecolor(col_a or 'steelblue'); spine.set_linewidth(2)
    plt.colorbar(im_a,ax=axes[0],format="%+.0f",label="dB")

    # B panel — border in col_b
    im_b=axes[1].imshow(mb,aspect='auto',origin='lower',cmap=cmap,interpolation='nearest',extent=ext)
    axes[1].set_title(f"{lb}  [{cmp_label}]",fontsize=8,color=col_b or 'white',fontweight='bold')
    axes[1].set_xlabel("Time (s)",fontsize=7)
    for spine in axes[1].spines.values():
        spine.set_edgecolor(col_b or 'darkorange'); spine.set_linewidth(2)
    plt.colorbar(im_b,ax=axes[1],format="%+.0f",label="dB")

    # Difference panel — custom colormap using recording colors
    ca = mcolors.to_rgb(col_a or '#2196F3')
    cb_ = mcolors.to_rgb(col_b or '#FF9800')
    # Blue (col_a) at -vmax, white at 0, orange (col_b) at +vmax
    diff_cmap = mcolors.LinearSegmentedColormap.from_list(
        'rec_diff', [ca, (1,1,1), cb_], N=256)
    im3=axes[2].imshow(diff,aspect='auto',origin='lower',cmap=diff_cmap,
                        interpolation='nearest',extent=ext,vmin=-vmax,vmax=vmax)
    cb3=plt.colorbar(im3,ax=axes[2],format="%+.0f")
    cb3.set_label(f"(+) = {lb} louder  |  (−) = {la} louder",fontsize=6)
    axes[2].set_title(
        f"Δ = {lb}  MINUS  {la}\n"
        f"orange tint = {lb} has more energy · blue tint = {la} has more energy",
        fontsize=7)
    axes[2].set_xlabel("Time (s)",fontsize=7)

    # Apply yticks to all three panels
    if yticks:
        for ax in axes:
            ax.set_yticks(list(yticks.keys()))
            ax.set_yticklabels(list(yticks.values()),fontsize=5)

    # Mel band labels on y-axis of all panels
    if add_mel_bands:
        for ax in axes:
            for lo,hi,lbl in MEL_BANDS_HZ:
                mid=(lo+hi)/2
                ax.axhline(lo,color='white',lw=0.4,alpha=0.4)
                ax.text(zoom_start+(zoom_end-zoom_start)*0.01, mid, lbl,
                        fontsize=4, color='white', va='center', alpha=0.8)

    fig.suptitle(title,fontsize=10,fontweight='bold'); plt.tight_layout()
    return fig

def line_pair(title, da, db, la, lb, ylabel="", yrange=None, yscale='linear',
              col_a=None, col_b=None):
    ca = col_a or rec_color(0); cb = col_b or rec_color(1)
    n=min(len(da),len(db)); da,db=np.array(da[:n]),np.array(db[:n])
    t=np.linspace(zoom_start,zoom_end,n); diff=db-da

    fig,axes=plt.subplots(3,1,figsize=(14,6),sharex=True)

    # Top two panels — individual signals
    for ax,d,lbl,col in zip(axes[:2],[da,db],[la,lb],[ca,cb]):
        ax.plot(t,d,lw=0.8,color=col)
        ax.set_ylabel(ylabel,fontsize=7)
        ax.set_title(lbl,fontsize=8,color=col,fontweight='bold')
        if yrange: ax.set_ylim(yrange)
        if yscale=='log': ax.set_yscale('log')

    # Difference panel — always autoscale, always zero line, symmetric y-axis
    diff_max = max(abs(diff.max()), abs(diff.min()), 1e-8)
    axes[2].fill_between(t, diff, where=diff>=0, color=cb, alpha=0.55, label=f'{lb} higher')
    axes[2].fill_between(t, diff, where=diff<0,  color=ca, alpha=0.55, label=f'{la} higher')
    axes[2].axhline(0, color='black', lw=1.2, zorder=5)

    # Symmetric y-axis so positive and negative are equally readable
    axes[2].set_ylim(-diff_max * 1.15, diff_max * 1.15)

    # Tick labels: show actual values including negatives
    axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:+.3g}"))
    axes[2].set_ylabel(f"Δ {ylabel}", fontsize=7)
    axes[2].set_xlabel("Time (s)", fontsize=7)
    axes[2].legend(fontsize=7)
    axes[2].set_title(
        f"Δ = {lb}  MINUS  {la}  "
        f"(above zero = {lb} higher · below zero = {la} higher)",
        fontsize=8)

    # Mark the peak positive and negative differences
    if diff.max() > 0:
        peak_pos = t[int(np.argmax(diff))]
        axes[2].annotate(f"+{diff.max():.3g}",
            xy=(peak_pos, diff.max()), xytext=(peak_pos, diff.max()*1.05),
            fontsize=6, ha='center', color=cb)
    if diff.min() < 0:
        peak_neg = t[int(np.argmin(diff))]
        axes[2].annotate(f"{diff.min():.3g}",
            xy=(peak_neg, diff.min()), xytext=(peak_neg, diff.min()*1.05),
            fontsize=6, ha='center', color=ca)

    fig.suptitle(title, fontsize=10, fontweight='bold')
    plt.tight_layout(); return fig

def diff_note(label, va, vb, unit, math_txt, musical_txt):
    d=vb-va; pct=d/(abs(va)+1e-10)*100
    st.markdown(f"**{label}:** 1=`{va:.4g} {unit}` · 2=`{vb:.4g} {unit}` · Δ=`{d:+.4g}` ({pct:+.1f}%)  \n"
                f"*Math:* {math_txt}  \n*Musical:* {musical_txt}")

def diff_formula(name_a, name_b, col_a, col_b):
    """Render a clear one-line formula showing what the difference means."""
    st.markdown(
        f"<div style='background:#111;padding:6px 12px;border-radius:6px;font-size:0.8rem;margin-bottom:4px'>"
        f"<b>Difference =</b> "
        f"<span style='color:{col_b};font-weight:bold'>{name_b}</span>"
        f" <b>minus</b> "
        f"<span style='color:{col_a};font-weight:bold'>{name_a}</span>"
        f" &nbsp;·&nbsp; "
        f"<span style='color:{col_b}'>■</span> tint = {name_b} has more energy &nbsp;·&nbsp; "
        f"<span style='color:{col_a}'>■</span> tint = {name_a} has more energy"
        f"</div>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────
# N-RECORDING MATRIX VIEW (shown when n_recs > 2)
# Stream graph helper — returns a base64 PNG for embedding
# ─────────────────────────────────────────────────────────────
def make_stream_graph_png(Ra, Rb, name_a, name_b,
                          figsize=(5,2.5), fontsize=7, for_matrix=False):
    """
    Build a stream graph for the pair (Ra, Rb) and return it as a
    base64-encoded PNG string suitable for embedding in HTML.
    Uses the same -0.5/+0.5 centred envelope as the Song Map view.
    """
    from scipy.signal import savgol_filter as _sgf
    import io, base64

    STREAM_FEATS = [
        ("rms",       "Volume",     "#4CAF50"),
        ("centroid",  "Brightness", "#2196F3"),
        ("onset",     "Rhythm",     "#FF9800"),
        ("sms_ratio", "Harmony",    "#9C27B0"),
    ]
    streams = {}
    for key, label, col in STREAM_FEATS:
        arr_a = Ra.get(key); arr_b = Rb.get(key)
        if arr_a is None or arr_b is None: continue
        n_s = min(len(arr_a), len(arr_b))
        a_n = (arr_a[:n_s]-arr_a[:n_s].mean())/(arr_a[:n_s].std()+1e-8)
        b_n = (arr_b[:n_s]-arr_b[:n_s].mean())/(arr_b[:n_s].std()+1e-8)
        div_s = np.abs(a_n - b_n)
        if len(div_s) > 21:
            div_s = _sgf(div_s, min(51,len(div_s)//4*2+1), 3)
        streams[label] = (np.clip(div_s, 0, None), col)

    if not streams:
        return None

    min_len   = min(len(v) for v,_ in streams.values())
    times_s   = np.arange(min_len) * HOP / SR
    labels_s  = list(streams.keys())
    vals_s    = np.array([v[:min_len] for v,_ in streams.values()])
    cols_s    = [c for _,c in streams.values()]
    total_s    = vals_s.sum(axis=0)
    peak_total = total_s.max() + 1e-8
    scale      = 0.5 / peak_total
    vals_scaled = vals_s * scale

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#0f0f1a')
    ax.axhline(0, color='white', lw=0.8, alpha=0.5, zorder=2)

    legend_handles = []
    # Stack above zero
    cum_pos = np.zeros(min_len)
    for label, val_sc, col in zip(labels_s, vals_scaled, cols_s):
        upper = cum_pos + val_sc
        patch = ax.fill_between(times_s, cum_pos, upper,
                                color=col, alpha=0.85, label=label)
        legend_handles.append(patch)
        cum_pos = upper
    # Mirror below zero
    cum_neg = np.zeros(min_len)
    for val_sc, col in zip(vals_scaled, cols_s):
        lower = cum_neg - val_sc
        ax.fill_between(times_s, lower, cum_neg, color=col, alpha=0.85)
        cum_neg = lower

    ax.set_xlim(0, times_s[-1] if len(times_s) else 1)
    ax.set_ylim(-0.55, 0.55)
    ax.tick_params(colors='white', labelsize=fontsize-1)
    for sp in ax.spines.values(): sp.set_edgecolor('#333')

    if not for_matrix:
        ax.set_xlabel("Time (s)", fontsize=fontsize, color='white')
        ax.set_ylabel("Divergence", fontsize=fontsize, color='white')
        ax.set_title(f"{name_a} vs {name_b}", fontsize=fontsize,
                     color='white', fontweight='bold')
        ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
        ax.set_yticklabels(["-0.5","-0.25","0","+0.25","+0.5"],
                           fontsize=fontsize-1, color='white')
        ax.legend(handles=legend_handles[::-1], labels=labels_s[::-1],
                  fontsize=fontsize-1, loc='upper right',
                  facecolor='#1a1a2e', labelcolor='white', edgecolor='#333')
    else:
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout(pad=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                facecolor='#0f0f1a')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────────────────────
# Score / MIDI alignment
# ─────────────────────────────────────────────────────────────
def audio_to_midi_chroma(y, sr=SR, hop=HOP):
    """
    Convert a monophonic audio recording to a chroma representation
    via pYIN pitch estimation.

    Pipeline:
      1. pYIN → F0 estimate per frame (fundamental frequency in Hz)
      2. F0 → MIDI note number (round to nearest semitone)
      3. MIDI note → one-hot chroma (12 pitch classes)

    Returns:
      chroma_midi : (12, T) array — pitch class energy per frame
      voiced_flag : (T,) bool array — True where pYIN was confident
      voiced_pct  : float — % of frames where a clear pitch was detected

    Works best for single-instrument monophonic recordings.
    Confidence is lower with noise, vibrato, or fast passages.
    """
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr, hop_length=hop,
        fill_na=None)

    n_frames = len(f0)
    chroma_midi = np.zeros((12, n_frames), dtype=np.float32)

    for t, (freq, voiced) in enumerate(zip(f0, voiced_flag)):
        if voiced and freq is not None and freq > 0:
            midi_note = int(round(12 * np.log2(freq / 440.0) + 69))
            pitch_class = midi_note % 12
            chroma_midi[pitch_class, t] = 1.0

    voiced_pct = float(voiced_flag.sum() / max(len(voiced_flag), 1) * 100)
    return chroma_midi, voiced_flag, voiced_pct


def midi_chroma_to_pretty_midi(chroma_midi, tempo=120.0,
                                hop=HOP, sr=SR):
    """
    Convert a chroma-from-pYIN array to a pretty_midi.PrettyMIDI object.
    Each voiced frame becomes a short MIDI note event.
    Consecutive frames of the same pitch class are merged into one note.
    Requires: pip install pretty_midi
    """
    try:
        import pretty_midi
    except ImportError:
        raise ImportError(
            "pretty_midi is not installed. "
            "Run: pip install pretty_midi")
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst = pretty_midi.Instrument(program=0, name="pYIN transcription")

    frame_dur = hop / sr
    active = {}   # pitch_class → start_time

    n_frames = chroma_midi.shape[1]
    for t in range(n_frames):
        time = t * frame_dur
        active_pcs = set(np.where(chroma_midi[:, t] > 0)[0])

        # End notes that dropped out
        ended = [pc for pc in list(active) if pc not in active_pcs]
        for pc in ended:
            start = active.pop(pc)
            # Use octave 4 as default (MIDI note = pc + 60)
            note = pretty_midi.Note(
                velocity=80,
                pitch=pc + 60,
                start=start,
                end=max(time, start + frame_dur))
            inst.notes.append(note)

        # Start new notes
        for pc in active_pcs:
            if pc not in active:
                active[pc] = time

    # Close any remaining open notes
    end_time = n_frames * frame_dur
    for pc, start in active.items():
        note = pretty_midi.Note(
            velocity=80,
            pitch=pc + 60,
            start=start,
            end=max(end_time, start + frame_dur))
        inst.notes.append(note)

    inst.notes.sort(key=lambda n: n.start)
    pm.instruments.append(inst)
    return pm


def midi_file_to_chroma(midi_bytes, n_frames_target, hop=HOP, sr=SR):
    """
    Parse an uploaded MIDI file (.mid) and build a (12, T) chroma array
    from its note events, matching the frame rate used elsewhere (HOP/SR).
    Used when the user uploads a real score instead of relying on the
    auto-generated consensus.

    Some exported MIDI files contain out-of-range data bytes (e.g. from
    certain DAWs/notation tools) which makes strict parsers raise
    "data byte must be in range 0..127". We sanitize via mido's
    clip=True first (which clamps invalid bytes instead of raising),
    then hand the cleaned file to pretty_midi for note extraction.
    """
    try:
        import pretty_midi
    except ImportError:
        raise ImportError("pretty_midi not installed. Run: pip install pretty_midi")
    import io as _io

    if not midi_bytes or len(midi_bytes) < 4:
        raise ValueError("Uploaded file is empty or too small to be a valid MIDI file.")
    if midi_bytes[:4] != b'MThd':
        raise ValueError(
            "This doesn't look like a standard MIDI file "
            "(missing 'MThd' header). Make sure you uploaded a .mid/.midi file.")

    try:
        # First attempt: parse directly
        pm = pretty_midi.PrettyMIDI(_io.BytesIO(midi_bytes))
    except Exception as e1:
        # Fallback: sanitize with mido (clip=True clamps invalid data bytes
        # instead of raising), re-save, then retry with pretty_midi
        try:
            import mido
            mid = mido.MidiFile(file=_io.BytesIO(midi_bytes), clip=True)
            buf = _io.BytesIO()
            mid.save(file=buf)
            buf.seek(0)
            pm = pretty_midi.PrettyMIDI(buf)
        except Exception as e2:
            raise ValueError(
                f"Could not parse this MIDI file even after attempting repair "
                f"(original error: {e1}; repair error: {e2}). "
                f"Try re-exporting the MIDI from its source application.")

    frame_dur = hop / sr
    total_dur = pm.get_end_time()
    n_frames  = max(int(np.ceil(total_dur / frame_dur)), n_frames_target)
    chroma = np.zeros((12, n_frames), dtype=np.float32)
    for inst in pm.instruments:
        if inst.is_drum: continue
        for note in inst.notes:
            f_start = int(note.start / frame_dur)
            f_end   = max(f_start+1, int(note.end / frame_dur))
            pc = note.pitch % 12
            chroma[pc, f_start:min(f_end, n_frames)] = 1.0
    return chroma[:, :n_frames_target] if n_frames >= n_frames_target else chroma


def synthesize_midi_audio(pm, sr=SR):
    """
    Render a pretty_midi.PrettyMIDI object to audio using simple sine-wave
    synthesis (no soundfont required). Returns float32 waveform.
    Lets the user actually listen to a generated/uploaded MIDI transcription
    to sanity-check it against the source recording.
    """
    y = pm.synthesize(fs=sr)
    return y.astype(np.float32)


def compute_midi_alignment(Rs_aligned, ys, names,
                            dtw_strategy='combo', hop=HOP, sr=SR,
                            uploaded_score_chromas=None):
    """
    For each recording:
      1. Generate a MIDI chroma via pYIN (the "auto-transcription").
      2. Determine that recording's reference:
         - its OWN uploaded score, if provided (uploaded_score_chromas[i])
         - otherwise a consensus MIDI chroma (mean of all recordings'
           auto-transcriptions that don't have their own uploaded score)
      3. DTW-align each recording's AUDIO chroma to its reference.
      4. If a recording has its own uploaded score, also compute
         "transcription accuracy" — chroma cosine similarity between the
         auto-transcription (pYIN) and the uploaded score — as a sanity
         check on how good the automatic transcription is.

    uploaded_score_chromas: list same length as `names`, entries are
    (12,T) chroma arrays or None. Pass None (the whole argument) if no
    recording has an uploaded score.

    Returns:
      midi_chromas: list of (12, T) — pYIN auto-transcriptions, one per recording
      consensus:    (12, T) fallback consensus chroma (used by recordings
                    without their own uploaded score)
      alignments:   list of dicts per recording:
                    wp, mean_offset_s, max_offset_s, score,
                    used_own_score (bool),
                    transcription_accuracy (float or None)
      voiced_pcts:  list of float — pYIN confidence per recording
      any_uploaded_score: bool — whether at least one recording has an uploaded score
    """
    from librosa.sequence import dtw as librosa_dtw
    n = len(names)
    if uploaded_score_chromas is None:
        uploaded_score_chromas = [None] * n

    # Step 1: generate MIDI chroma for each recording (auto-transcription)
    midi_chromas  = []
    voiced_pcts   = []
    for y in ys:
        mc, vf, vp = audio_to_midi_chroma(y.astype(np.float32), sr=sr, hop=hop)
        midi_chromas.append(mc)
        voiced_pcts.append(vp)

    # Step 2: fallback consensus — mean of transcriptions from recordings
    # that do NOT have their own uploaded score (so an uploaded score
    # doesn't bias the group average of everyone else)
    no_own_score_idx = [i for i in range(n) if uploaded_score_chromas[i] is None]
    if no_own_score_idx:
        min_t = min(midi_chromas[i].shape[1] for i in no_own_score_idx)
        stacked = np.array([midi_chromas[i][:, :min_t] for i in no_own_score_idx])
        consensus = stacked.mean(axis=0)
    else:
        # every recording has its own score — build a plain consensus anyway
        # (used only as a display fallback, not for alignment in this case)
        min_t = min(mc.shape[1] for mc in midi_chromas)
        stacked = np.array([mc[:, :min_t] for mc in midi_chromas])
        consensus = stacked.mean(axis=0)
    norms = np.linalg.norm(consensus, axis=0, keepdims=True)
    silent = (norms < 1e-6).flatten()
    consensus = consensus / (norms + 1e-10)
    consensus[:, silent] = 1.0 / 12

    any_uploaded_score = any(c is not None for c in uploaded_score_chromas)

    # Step 3: per-recording alignment against its own reference
    alignments = []
    for i, R in enumerate(Rs_aligned):
        audio_chroma = R.get("chroma")
        if audio_chroma is None:
            alignments.append(None)
            continue

        own_score = uploaded_score_chromas[i]
        used_own  = own_score is not None
        reference = (safe_normalize_chroma(own_score.astype(np.float32))
                     if used_own else consensus)

        # Transcription accuracy: pYIN auto-transcription vs this
        # recording's own uploaded score (only computable if uploaded)
        transcription_accuracy = None
        if used_own:
            mc_i = midi_chromas[i]
            t_common = min(mc_i.shape[1], reference.shape[1])
            if t_common > 0:
                a = mc_i[:, :t_common].mean(1)
                b = reference[:, :t_common].mean(1)
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if na > 1e-8 and nb > 1e-8:
                    cos = float(np.dot(a, b) / (na * nb))
                    transcription_accuracy = float(np.clip((cos + 1) / 2, 0, 1))

        ac = safe_normalize_chroma(audio_chroma.astype(np.float32))
        t_ref = reference.shape[1]
        t_audio = ac.shape[1]
        ac_trim  = ac[:, :min(t_audio, t_ref)]
        ref_trim = reference[:, :min(t_audio, t_ref)]
        try:
            _, wp = librosa_dtw(X=ac_trim, Y=ref_trim, metric='cosine')
            offsets = np.abs(wp[:, 0].astype(float) -
                             wp[:, 1].astype(float)) * hop / sr
            mean_off  = float(offsets.mean())
            max_off   = float(offsets.max())
            max_frames = max(float(wp[:, 0].max()),
                             float(wp[:, 1].max()), 1.0)
            score = float(np.clip(
                1.0 - offsets.mean() / (max_frames * hop / sr), 0, 1))
            alignments.append({
                "wp": wp,
                "mean_offset_s": round(mean_off, 3),
                "max_offset_s":  round(max_off,  3),
                "score":         round(score, 3),
                "offsets":       offsets,
                "used_own_score": used_own,
                "transcription_accuracy": transcription_accuracy,
            })
        except Exception:
            alignments.append(None)

    return midi_chromas, consensus, alignments, voiced_pcts, any_uploaded_score


def render_score_alignment(midi_chromas, consensus, alignments,
                           names, zoom_start, zoom_end,
                           voiced_pcts=None, used_uploaded_score=False,
                           ys=None, Rs_aligned=None, hop=HOP, sr=SR):
    """
    Render the Score/MIDI Alignment tab.
    Shows:
      - pYIN confidence per recording
      - Per-recording reference: own uploaded score, or fallback consensus
      - Transcription accuracy (auto-transcription vs uploaded score) where available
      - Audio playback of each transcription + reference (sanity check)
      - Per-recording: alignment score, offset curve, deviation heatmap
    """
    st.markdown("### Score / MIDI Alignment")
    st.caption(
        "Each recording is transcribed to MIDI via **pYIN** pitch estimation "
        "(monophonic instrument assumed). "
        "If a recording has its own uploaded score, it's aligned against "
        "that score directly. Recordings without their own score are "
        "aligned against a consensus (the average of the other "
        "auto-transcriptions).")
    if used_uploaded_score:
        n_own = sum(1 for a in alignments if a and a.get("used_own_score"))
        st.info(f"📄 {n_own} of {len(names)} recording(s) using their own "
                f"uploaded score. The rest use the auto-generated consensus.")

    n = len(names)
    fs_sc = int(zoom_start * sr / hop)
    fe_sc = int(zoom_end   * sr / hop)

    # ── pYIN confidence warning ───────────────────────────────
    if voiced_pcts is not None:
        st.markdown("**pYIN transcription confidence:**")
        st.caption(
            "pYIN detects pitch frame-by-frame. "
            "**Voiced %** = proportion of frames where a clear pitch was detected. "
            "Low values mean the recording has many silent, noisy, or unpitched frames — "
            "the MIDI transcription and alignment score will be less reliable in those regions. "
            "Best results: >70% voiced.")
        conf_cols = st.columns(n)
        for col, nm, vp in zip(conf_cols, names, voiced_pcts):
            color = "normal" if vp >= 70 else "inverse"
            icon  = "✅" if vp >= 70 else "⚠️" if vp >= 40 else "❌"
            col.metric(f"{icon} {nm}", f"{vp:.0f}% voiced",
                       delta="reliable" if vp>=70 else "low confidence",
                       delta_color=color)
        low_conf = [nm for nm,vp in zip(names,voiced_pcts) if vp < 40]
        if low_conf:
            st.warning(
                f"⚠️ Low pitch detection confidence for: **{', '.join(low_conf)}**. "
                "This may indicate noisy recordings, rests, or very fast passages. "
                "The alignment score for these recordings should be interpreted with caution.")

    # ── Per-recording reference + transcription accuracy ──────
    if any(a and a.get("used_own_score") for a in alignments if a):
        st.markdown("**Transcription accuracy — auto-generated MIDI vs uploaded score:**")
        st.caption(
            "For recordings with their own uploaded score, this compares the "
            "**automatic pYIN transcription** against the **real uploaded MIDI** — "
            "a sanity check on how accurate the auto-transcription actually is. "
            "100% = the pYIN transcription matches the uploaded score's pitch "
            "content exactly. Low values mean the auto-transcription and the "
            "real score disagree — trust the uploaded score, not the auto-transcription, "
            "in that case.")
        acc_cols = st.columns(n)
        for col, nm, aln in zip(acc_cols, names, alignments):
            if aln is None:
                col.caption(f"{nm}: —"); continue
            if aln.get("used_own_score"):
                acc = aln.get("transcription_accuracy")
                if acc is not None:
                    icon = "✅" if acc >= 0.75 else "⚠️" if acc >= 0.5 else "❌"
                    col.metric(f"{icon} {nm}", f"{acc*100:.0f}% match",
                               help="pYIN auto-transcription vs uploaded score")
                else:
                    col.caption(f"{nm}: own score, accuracy N/A")
            else:
                col.caption(f"{nm}: using consensus\n(no own score uploaded)")

    # ── Consensus chroma ─────────────────────────────────────
    with st.expander("📄 Consensus MIDI Score (chroma)", expanded=False):
        cons_w = consensus[:, fs_sc:fe_sc]
        if cons_w.shape[1] > 0:
            fig_c, ax_c = plt.subplots(figsize=(14, 2.5))
            fig_c.patch.set_facecolor('#0f0f1a')
            ax_c.set_facecolor('#0f0f1a')
            t_c = np.linspace(zoom_start, zoom_end, cons_w.shape[1])
            im_c = ax_c.imshow(cons_w, aspect='auto', origin='lower',
                               cmap='YlOrRd',
                               extent=[zoom_start, zoom_end, 0, 12])
            ax_c.set_yticks(range(12))
            ax_c.set_yticklabels(
                ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'],
                fontsize=6, color='white')
            ax_c.set_xlabel("Time (s)", fontsize=7, color='white')
            ax_c.set_title(
                "Consensus MIDI chroma — average across all pYIN transcriptions",
                fontsize=8, color='white')
            ax_c.tick_params(colors='white', labelsize=6)
            for sp in ax_c.spines.values(): sp.set_edgecolor('#333')
            plt.colorbar(im_c, ax=ax_c, label="Pitch energy")
            plt.tight_layout()
            st.pyplot(fig_c, use_container_width=True); plt.close()

    # ── Listen to the MIDI transcriptions ──────────────────────
    with st.expander("🔊 Listen — sanity check the MIDI transcriptions", expanded=False):
        st.caption(
            "Synthesized playback of each pYIN transcription (simple sine-wave "
            "synth, no soundfont) next to the original recording. "
            "If the MIDI sounds musically unrelated to the recording, the "
            "transcription — and therefore its alignment score — should not "
            "be trusted for that recording.")
        try:
            import pretty_midi as _pm_listen
            for idx, (nm, mc) in enumerate(zip(names, midi_chromas)):
                tempo_i = 120.0
                if Rs_aligned is not None and idx < len(Rs_aligned):
                    tempo_i = float(Rs_aligned[idx].get("tempo", 120))
                pm_i = midi_chroma_to_pretty_midi(mc, tempo=tempo_i, hop=hop, sr=sr)
                synth_y = synthesize_midi_audio(pm_i, sr=sr)
                col_orig, col_midi = st.columns(2)
                with col_orig:
                    st.caption(f"{nm} — original")
                    if ys is not None and idx < len(ys):
                        st.audio(wav_bytes(ys[idx].astype(np.float32)), format='audio/wav')
                with col_midi:
                    st.caption(f"{nm} — pYIN MIDI transcription")
                    st.audio(wav_bytes(synth_y), format='audio/wav')
            # Reference score playback
            st.markdown("---")
            st.caption("Reference score used for alignment "
                       f"({'your uploaded MIDI' if used_uploaded_score else 'auto-generated consensus'}):")
            pm_ref = midi_chroma_to_pretty_midi(consensus, tempo=120.0, hop=hop, sr=sr)
            synth_ref = synthesize_midi_audio(pm_ref, sr=sr)
            st.audio(wav_bytes(synth_ref), format='audio/wav')
        except ImportError:
            st.warning("pretty_midi not installed — run: pip install pretty_midi")
        except Exception as _e_listen:
            st.caption(f"Playback unavailable: {_e_listen}")

    # ── Per-recording alignment scores ────────────────────────
    st.markdown("**Alignment scores** (each vs its own score, or the consensus if none was uploaded):")
    score_cols = st.columns(n)
    for col, nm, aln in zip(score_cols, names, alignments):
        if aln is None:
            col.metric(nm, "N/A", help="Alignment failed")
            continue
        score_pct = f"{aln['score']*100:.0f}%"
        delta = f"mean offset {aln['mean_offset_s']:.2f}s"
        col.metric(nm, score_pct, delta,
                   delta_color="inverse")

    # ── Offset curves per recording ───────────────────────────
    st.markdown("**Temporal offset vs consensus score over time:**")
    st.caption("How many seconds each recording drifts from the consensus "
               "score at each moment. Closer to 0 = more on-score.")
    valid_alns = [(nm, aln) for nm, aln in zip(names, alignments)
                  if aln is not None]
    if valid_alns:
        fig_off, ax_off = plt.subplots(figsize=(14, 3))
        fig_off.patch.set_facecolor('#0f0f1a')
        ax_off.set_facecolor('#0f0f1a')
        ax_off.axhline(0, color='white', lw=0.8, alpha=0.4)
        for idx, (nm, aln) in enumerate(valid_alns):
            wp = aln["wp"]
            offsets = aln["offsets"]
            t_wp = wp[::max(1,len(wp)//2000), 0] * hop / sr
            off_ds = offsets[::max(1,len(offsets)//2000)]
            n_plot = min(len(t_wp), len(off_ds))
            ax_off.plot(t_wp[:n_plot], off_ds[:n_plot],
                        lw=1.0, color=rec_color(idx),
                        label=f"{nm} ({aln['score']*100:.0f}%)",
                        alpha=0.85)
        ax_off.set_xlabel("Time (s)", fontsize=8, color='white')
        ax_off.set_ylabel("Offset from score (s)", fontsize=8, color='white')
        ax_off.tick_params(colors='white', labelsize=7)
        for sp in ax_off.spines.values(): sp.set_edgecolor('#333')
        ax_off.legend(fontsize=8, facecolor='#1a1a2e',
                      labelcolor='white', edgecolor='#333')
        plt.tight_layout()
        st.pyplot(fig_off, use_container_width=True); plt.close()

    # ── MIDI chroma vs audio chroma per recording ─────────────
    st.markdown("**MIDI transcription vs audio chroma — per recording:**")
    st.caption("Top row = pYIN MIDI chroma. "
               "Bottom row = audio chroma. "
               "Similar patterns = the recording follows the score closely.")
    for idx, (nm, mc, aln) in enumerate(
            zip(names, midi_chromas, alignments)):
        with st.expander(f"{nm}  —  score {aln['score']*100:.0f}%"
                         if aln else nm, expanded=False):
            mc_w  = mc[:, fs_sc:min(fe_sc, mc.shape[1])]
            aud_c = Rs_aligned[idx].get("chroma")
            if aud_c is not None:
                aud_w = aud_c[:, fs_sc:min(fe_sc, aud_c.shape[1])]
                nc = min(mc_w.shape[1], aud_w.shape[1])
                if nc > 0:
                    fig_cmp, axes_cmp = plt.subplots(
                        2, 1, figsize=(14, 4), sharex=True)
                    fig_cmp.patch.set_facecolor('#0f0f1a')
                    t_cmp = np.linspace(zoom_start, zoom_end, nc)
                    ext = [zoom_start, zoom_end, 0, 12]
                    for ax, mat, title, col in [
                        (axes_cmp[0], mc_w[:, :nc],
                         f"pYIN MIDI chroma — {nm}", rec_color(idx)),
                        (axes_cmp[1], aud_w[:, :nc],
                         f"Audio chroma — {nm}", rec_color(idx)),
                    ]:
                        ax.set_facecolor('#0f0f1a')
                        im = ax.imshow(mat, aspect='auto', origin='lower',
                                       cmap='YlOrRd', extent=ext)
                        ax.set_yticks(range(12))
                        ax.set_yticklabels(
                            ['C','C#','D','D#','E','F','F#',
                             'G','G#','A','A#','B'],
                            fontsize=6, color='white')
                        ax.set_title(title, fontsize=8,
                                     color=col, fontweight='bold')
                        for sp in ax.spines.values():
                            sp.set_edgecolor(col); sp.set_linewidth(1.5)
                        ax.tick_params(colors='white', labelsize=6)
                    axes_cmp[-1].set_xlabel("Time (s)", fontsize=7,
                                             color='white')
                    plt.tight_layout()
                    st.pyplot(fig_cmp, use_container_width=True)
                    plt.close()


def pair_similarity(Ra, Rb):
    """
    Fast similarity score for pair selection.
    Uses MFCC cosine + chroma cosine + tempo proximity + beat regularity.
    Returns score in [0, 1] where 1 = identical.
    """
    scores = []
    if "mfcc_means" in Ra and "mfcc_means" in Rb:
        ma, mb = Ra["mfcc_means"], Rb["mfcc_means"]
        cos = float(np.dot(ma,mb)/(np.linalg.norm(ma)*np.linalg.norm(mb)+1e-10))
        scores.append((cos+1)/2)
    if "chroma" in Ra and "chroma" in Rb:
        ca, cb = Ra["chroma"].mean(1), Rb["chroma"].mean(1)
        cos = float(np.dot(ca,cb)/(np.linalg.norm(ca)*np.linalg.norm(cb)+1e-10))
        scores.append((cos+1)/2)
    ta, tb = Ra.get("tempo",120), Rb.get("tempo",120)
    scores.append(max(0, 1 - abs(ta-tb)/50))
    ra, rb = Ra.get("beat_reg",0), Rb.get("beat_reg",0)
    if ra>0 and rb>0:
        scores.append(max(0, 1 - abs(ra-rb)/max(ra,rb,1e-8)))
    if "centroid" in Ra and "centroid" in Rb:
        ca_v = float(Ra["centroid"].mean())
        cb_v = float(Rb["centroid"].mean())
        scores.append(max(0, 1 - abs(ca_v-cb_v)/max(ca_v,cb_v,1e-8)))
    return float(np.mean(scores)) if scores else 0.5


# Progress Report Generator
# ─────────────────────────────────────────────────────────────
def generate_report(Rs_aligned, names, pairs, sim_matrix,
                    zoom_start, zoom_end, ys, durs,
                    midi_chromas=None, consensus_chroma=None,
                    midi_alignments=None):
    """
    Generate a comprehensive progress report with all figures embedded.
    Returns (markdown_str, html_str) — caller chooses which to download.
    """
    import base64, datetime
    from io import BytesIO

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    n   = len(Rs_aligned)

    # Best pair
    pair_scores = [sim_matrix[i][j] for i,j in pairs]
    best_idx    = int(np.argmax(pair_scores))
    pi, pj      = pairs[best_idx]
    Ra, Rb      = Rs_aligned[pi], Rs_aligned[pj]
    na, nb      = names[pi], names[pj]
    summary_insights = generate_summary(Ra, Rb, na, nb)

    def fig_to_b64(fig, dpi=120):
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    def dark_fig(*args, **kwargs):
        fig, axes = plt.subplots(*args, **kwargs)
        fig.patch.set_facecolor('#0f0f1a')
        if hasattr(axes, '__iter__'):
            for ax in np.array(axes).flatten():
                ax.set_facecolor('#1a1a2e')
                ax.tick_params(colors='white', labelsize=7)
                for sp in ax.spines.values(): sp.set_edgecolor('#333')
        else:
            axes.set_facecolor('#1a1a2e')
            axes.tick_params(colors='white', labelsize=7)
            for sp in axes.spines.values(): sp.set_edgecolor('#333')
        return fig, axes

    # ── Collect all figures ───────────────────────────────────
    sections = []   # list of (title, description, b64_png)

    # 1. Waveform comparison
    try:
        n_wv = min(len(ys[pi]), len(ys[pj])); step = max(1, n_wv//2000)
        wa = ys[pi][:n_wv:step]; wb = ys[pj][:n_wv:step]
        t_wv = np.linspace(0, min(durs[pi], durs[pj]), len(wa))
        fig, axs = dark_fig(3, 1, figsize=(14,5), sharex=True)
        axs[0].plot(t_wv, wa, lw=0.5, color=rec_color(pi)); axs[0].set_title(na, color=rec_color(pi), fontsize=8)
        axs[1].plot(t_wv, wb, lw=0.5, color=rec_color(pj)); axs[1].set_title(nb, color=rec_color(pj), fontsize=8)
        diff_w = wb - wa; dm = max(abs(diff_w.max()), abs(diff_w.min()), 1e-8)
        axs[2].fill_between(t_wv, diff_w, where=diff_w>=0, color=rec_color(pj), alpha=0.6)
        axs[2].fill_between(t_wv, diff_w, where=diff_w<0,  color=rec_color(pi), alpha=0.6)
        axs[2].axhline(0, color='white', lw=0.8); axs[2].set_ylim(-dm*1.15, dm*1.15)
        axs[2].set_title(f"Δ = {nb} minus {na}", color='white', fontsize=8)
        axs[2].set_xlabel("Time (s)", color='white', fontsize=8)
        plt.tight_layout(); sections.append(("Waveform Comparison", f"{na} vs {nb}", fig_to_b64(fig))); plt.close()
    except: pass

    # 2. Pairwise similarity matrix
    try:
        import matplotlib.colors as mcolors
        cmap_sim = mcolors.LinearSegmentedColormap.from_list('sim', ['#E91E63','#FF9800','#4CAF50'], N=256)
        fig, ax = dark_fig(figsize=(max(4,n*1.8), max(3,n*1.6)))
        sim_arr = np.array([[sim_matrix[i][j] for j in range(n)] for i in range(n)])
        im = ax.imshow(sim_arr, cmap=cmap_sim, vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(n)); ax.set_xticklabels(names, fontsize=8, rotation=30, ha='right', color='white')
        ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=8, color='white')
        for i in range(n):
            for j in range(n):
                val = sim_arr[i,j]; txt = "—" if i==j else f"{val*100:.0f}%"
                ax.text(j, i, txt, ha='center', va='center', fontsize=10,
                        color='white' if val<0.55 else 'black', fontweight='bold')
        cb = plt.colorbar(im, ax=ax); cb.set_label("Similarity", color='white'); cb.ax.yaxis.set_tick_params(color='white')
        plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')
        ax.set_title("Pairwise Similarity Matrix", color='white', fontsize=10, fontweight='bold')
        plt.tight_layout(); sections.append(("Pairwise Similarity Matrix", "Green=similar · Red=different", fig_to_b64(fig))); plt.close()
    except: pass

    # 3. MDS — features-based
    try:
        from sklearn.manifold import MDS as _MDS
        dist_f = np.zeros((n,n))
        def _fv(R):
            v=[]
            if "mfcc_means" in R: v.extend(R["mfcc_means"][1:].tolist())
            if "chroma" in R: v.extend(R["chroma"].mean(1).tolist())
            for k in ["tempo","beat_reg","rms","centroid","bandwidth","rolloff","flatness"]:
                vv=R.get(k)
                if vv is not None: v.append(float(vv.mean()) if isinstance(vv,np.ndarray) else float(vv))
            return np.array(v,dtype=np.float32)
        for i in range(n):
            for j in range(n):
                if i==j: continue
                va=_fv(Rs_aligned[i]); vb=_fv(Rs_aligned[j]); nc=min(len(va),len(vb))
                va,vb=va[:nc],vb[:nc]; diffs=np.abs(va-vb)
                norms=np.maximum(np.abs(va)+np.abs(vb),1e-8)/2
                dist_f[i,j]=float(np.clip(np.mean(diffs/norms),0,1))
        np.fill_diagonal(dist_f,0)
        coords=_MDS(n_components=2,dissimilarity='precomputed',random_state=42,n_init=4,normalized_stress='auto').fit_transform(dist_f)
        fig, ax = dark_fig(figsize=(7,5))
        yr = coords[:,1].max()-coords[:,1].min()+1e-8
        for i in range(n):
            for j in range(i+1,n):
                x0,y0=coords[i]; x1,y1=coords[j]; d=dist_f[i,j]
                lc='#4CAF50' if d<=0.25 else '#FF9800' if d<=0.5 else '#E91E63'
                ax.plot([x0,x1],[y0,y1],color=lc,lw=1.5,alpha=0.55)
                ax.text((x0+x1)/2,(y0+y1)/2,f"{d*100:.0f}%",fontsize=7,color='#ccc',ha='center',
                        bbox=dict(fc='#1a1a2e',ec='none',alpha=0.8))
        for i,(nm,(x,y)) in enumerate(zip(names,coords)):
            ax.scatter(x,y,s=200,color=rec_color(i),zorder=3,edgecolors='white',lw=1.5)
            ax.text(x,y+0.04*yr,nm,fontsize=9,color='white',ha='center',va='bottom',fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("Recording Map (MDS — all features)", color='white', fontsize=10, fontweight='bold')
        for sp in ax.spines.values(): sp.set_visible(False)
        plt.tight_layout(); sections.append(("Recording Map (MDS)", "Closer = more similar across all features", fig_to_b64(fig))); plt.close()
    except: pass

    # 4. Stream graph (Song Map)
    try:
        from scipy.signal import savgol_filter as _sgf3
        SFTS=[("rms","Volume","#4CAF50"),("centroid","Brightness","#2196F3"),
              ("onset","Rhythm","#FF9800"),("sms_ratio","Harmony","#9C27B0")]
        streams={}
        for key,label,col in SFTS:
            aa=Ra.get(key); bb=Rb.get(key)
            if aa is None or bb is None: continue
            ns=min(len(aa),len(bb))
            an=(aa[:ns]-aa[:ns].mean())/(aa[:ns].std()+1e-8)
            bn=(bb[:ns]-bb[:ns].mean())/(bb[:ns].std()+1e-8)
            ds=np.abs(an-bn)
            if len(ds)>21: ds=_sgf3(ds,min(51,len(ds)//4*2+1),3)
            streams[label]=(np.clip(ds,0,None),col)
        if streams:
            ml=min(len(v) for v,_ in streams.values())
            ts=np.arange(ml)*HOP/SR
            vs=np.array([v[:ml] for v,_ in streams.values()])
            tot=vs.sum(axis=0); pk=tot.max()+1e-8; sc=vs*(0.5/pk)
            fig, ax = dark_fig(figsize=(14,4))
            ax.axhline(0,color='white',lw=1,alpha=0.5)
            leg=[]; cp=np.zeros(ml)
            for lab,vsc,col in zip(streams.keys(),sc,[c for _,c in streams.values()]):
                p=ax.fill_between(ts,cp,cp+vsc,color=col,alpha=0.82,label=lab); leg.append(p); cp+=vsc
            cn=np.zeros(ml)
            for vsc,col in zip(sc,[c for _,c in streams.values()]):
                ax.fill_between(ts,cn-vsc,cn,color=col,alpha=0.82); cn-=vsc
            ax.set_ylim(-0.55,0.55); ax.set_xlabel("Time (s)",color='white',fontsize=8)
            ax.set_ylabel("Divergence",color='white',fontsize=8)
            ax.set_title(f"Song Map — {na} vs {nb}",color='white',fontsize=10,fontweight='bold')
            ax.legend(handles=leg[::-1],labels=list(streams.keys())[::-1],fontsize=8,
                      facecolor='#1a1a2e',labelcolor='white',edgecolor='#333',loc='upper right')
            plt.tight_layout(); sections.append(("Song Map — Stream Graph", f"What drives the difference: {na} vs {nb}", fig_to_b64(fig))); plt.close()
    except: pass

    # 5. Chroma comparison
    try:
        ca_m=Ra["chroma"].mean(1); cb_m=Rb["chroma"].mean(1)
        MN=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']; x=np.arange(12)
        fig, axs = dark_fig(2,1,figsize=(12,5))
        axs[0].bar(x-0.2,ca_m,0.35,label=na,color=rec_color(pi),alpha=0.85)
        axs[0].bar(x+0.2,cb_m,0.35,label=nb,color=rec_color(pj),alpha=0.85)
        axs[0].set_xticks(x); axs[0].set_xticklabels(MN,color='white')
        axs[0].axhline(0,color='white',lw=0.8); axs[0].legend(facecolor='#1a1a2e',labelcolor='white')
        axs[0].set_title("Pitch class distribution",color='white',fontsize=9,fontweight='bold')
        dc=cb_m-ca_m
        axs[1].bar(x,dc,color=[rec_color(pj) if d>0 else rec_color(pi) for d in dc],alpha=0.85)
        axs[1].axhline(0,color='white',lw=1); axs[1].set_xticks(x); axs[1].set_xticklabels(MN,color='white')
        axs[1].set_title(f"Δ = {nb} minus {na}",color='white',fontsize=8)
        plt.tight_layout(); sections.append(("Harmony (Chroma)", "Pitch class energy distribution", fig_to_b64(fig))); plt.close()
    except: pass

    # 6. MFCC
    try:
        ma_m=Ra["mfcc_means"][1:]; mb_m=Rb["mfcc_means"][1:]; x=np.arange(1,len(ma_m)+1)
        fig, axs = dark_fig(2,1,figsize=(12,5))
        axs[0].bar(x-0.2,ma_m,0.35,label=na,color=rec_color(pi),alpha=0.85)
        axs[0].bar(x+0.2,mb_m,0.35,label=nb,color=rec_color(pj),alpha=0.85)
        axs[0].axhline(0,color='white',lw=0.8); axs[0].legend(facecolor='#1a1a2e',labelcolor='white')
        axs[0].set_title("MFCC Profile (coefficients 1–12)",color='white',fontsize=9,fontweight='bold')
        dm=mb_m-ma_m
        axs[1].bar(x,dm,color=[rec_color(pj) if d>0 else rec_color(pi) for d in dm],alpha=0.85)
        axs[1].axhline(0,color='white',lw=1); axs[1].set_title(f"Δ = {nb} minus {na}",color='white',fontsize=8)
        plt.tight_layout(); sections.append(("Timbre (MFCCs)", "Timbral fingerprint comparison", fig_to_b64(fig))); plt.close()
    except: pass

    # 7. Dynamics (RMS)
    try:
        rms_a=Ra["rms"]; rms_b=Rb["rms"]; nv=min(len(rms_a),len(rms_b))
        t_ax=np.linspace(zoom_start,zoom_end,nv); pk=max(rms_a.max(),rms_b.max(),1e-8)
        fig, axs = dark_fig(2,1,figsize=(14,4),sharex=True)
        axs[0].fill_between(t_ax,rms_a[:nv]/pk,alpha=0.4,color=rec_color(pi),label=na)
        axs[0].fill_between(t_ax,rms_b[:nv]/pk,alpha=0.4,color=rec_color(pj),label=nb)
        axs[0].set_yticks([0,0.5,1]); axs[0].set_yticklabels(["Quiet","","Loud"],color='white')
        axs[0].legend(facecolor='#1a1a2e',labelcolor='white')
        axs[0].set_title("Volume over time",color='white',fontsize=9,fontweight='bold')
        diff_r=rms_b[:nv]/pk-rms_a[:nv]/pk
        axs[1].fill_between(t_ax,diff_r,where=diff_r>=0,color=rec_color(pj),alpha=0.6)
        axs[1].fill_between(t_ax,diff_r,where=diff_r<0, color=rec_color(pi),alpha=0.6)
        axs[1].axhline(0,color='white',lw=0.8); axs[1].set_xlabel("Time (s)",color='white',fontsize=8)
        axs[1].set_title(f"Δ = {nb} minus {na}",color='white',fontsize=8)
        plt.tight_layout(); sections.append(("Dynamics (Volume)", "Normalised RMS energy over time", fig_to_b64(fig))); plt.close()
    except: pass

    # 8. Onset / Rhythm
    try:
        oa=Ra["onset"]; ob=Rb["onset"]; no=min(len(oa),len(ob))
        t_on=np.linspace(zoom_start,zoom_end,no)
        fig, axs = dark_fig(3,1,figsize=(14,5),sharex=True)
        for ax,onset,nm,col in [(axs[0],oa[:no],na,rec_color(pi)),(axs[1],ob[:no],nb,rec_color(pj))]:
            ax.fill_between(t_on,onset,alpha=0.3,color=col); ax.plot(t_on,onset,lw=0.6,color=col)
            ax.set_title(nm,color=col,fontsize=8,fontweight='bold')
        diff_on=ob[:no]-oa[:no]; dm=max(abs(diff_on.max()),abs(diff_on.min()),1e-8)
        axs[2].fill_between(t_on,diff_on,where=diff_on>=0,color=rec_color(pj),alpha=0.6)
        axs[2].fill_between(t_on,diff_on,where=diff_on<0, color=rec_color(pi),alpha=0.6)
        axs[2].axhline(0,color='white',lw=0.8); axs[2].set_ylim(-dm*1.15,dm*1.15)
        axs[2].set_title(f"Δ = {nb} minus {na}",color='white',fontsize=8)
        axs[2].set_xlabel("Time (s)",color='white',fontsize=8)
        plt.tight_layout(); sections.append(("Rhythm (Onset Strength)", "Onset envelope and difference", fig_to_b64(fig))); plt.close()
    except: pass

    # 9. Mel spectrogram (if available)
    try:
        ma=Ra.get("mel"); mb=Rb.get("mel")
        if ma is not None and mb is not None:
            fs_r=int(zoom_start*SR/HOP); fe_r=int(zoom_end*SR/HOP)
            maw=ma[:,fs_r:fe_r]; mbw=mb[:,fs_r:fe_r]; nc=min(maw.shape[1],mbw.shape[1])
            t_mel=np.linspace(zoom_start,zoom_end,nc)
            import matplotlib.colors as _mc
            ca_c=_mc.to_rgb(rec_color(pi)); cb_c=_mc.to_rgb(rec_color(pj))
            dcmap=_mc.LinearSegmentedColormap.from_list('d',[ca_c,(1,1,1),cb_c],N=256)
            vmax=float(np.percentile(np.abs(mbw[:,:nc]-maw[:,:nc]),95)) or 1.0
            fig, axs = plt.subplots(1,3,figsize=(16,3))
            fig.patch.set_facecolor('#0f0f1a')
            ext=[zoom_start,zoom_end,0,128]
            for ax,mat,ttl,cs,zn,zx in [
                (axs[0],maw[:,:nc],na,'magma',None,None),
                (axs[1],mbw[:,:nc],nb,'magma',None,None),
                (axs[2],mbw[:,:nc]-maw[:,:nc],f"Δ = {nb}−{na}",dcmap,-vmax,vmax)]:
                ax.imshow(mat,aspect='auto',origin='lower',cmap=cs,extent=ext,vmin=zn,vmax=zx)
                ax.set_title(ttl,fontsize=8,color='white'); ax.set_facecolor('#0f0f1a')
                ax.tick_params(colors='white',labelsize=6)
                for sp in ax.spines.values(): sp.set_edgecolor('#333')
            plt.tight_layout(); sections.append(("Mel Spectrogram", "128 Mel bands — A, B, and difference", fig_to_b64(fig))); plt.close()
    except: pass

    # 10. Score alignment (if computed)
    try:
        if midi_alignments is not None and consensus_chroma is not None:
            fig, ax = dark_fig(figsize=(14,3))
            ax.axhline(0,color='white',lw=0.8,alpha=0.4)
            for idx,(nm,aln) in enumerate(zip(names,midi_alignments)):
                if aln is None: continue
                wp=aln["wp"]; offsets=aln["offsets"]
                step=max(1,len(wp)//2000)
                t_wp=wp[::step,0]*HOP/SR; off_ds=offsets[::step]
                np_=min(len(t_wp),len(off_ds))
                ax.plot(t_wp[:np_],off_ds[:np_],lw=1.0,color=rec_color(idx),
                        label=f"{nm} ({aln['score']*100:.0f}%)",alpha=0.85)
            ax.set_xlabel("Time (s)",color='white',fontsize=8)
            ax.set_ylabel("Offset from score (s)",color='white',fontsize=8)
            ax.set_title("Score Alignment — offset from consensus MIDI score",color='white',fontsize=9,fontweight='bold')
            ax.legend(fontsize=8,facecolor='#1a1a2e',labelcolor='white',edgecolor='#333')
            plt.tight_layout(); sections.append(("Score Alignment", "Temporal offset of each recording from consensus MIDI", fig_to_b64(fig))); plt.close()
    except: pass

    # ── Build Markdown ────────────────────────────────────────
    md_lines = []
    md_lines.append("# ViSuS Progress Report")
    md_lines.append(f"*Generated: {now}*\n")
    md_lines.append(f"**Recordings:** {', '.join(names)}")
    md_lines.append(f"**Best pair:** {na} vs {nb} ({sim_matrix[pi][pj]*100:.0f}% similar)")
    md_lines.append(f"**Analysis window:** {zoom_start:.1f}s – {zoom_end:.1f}s\n")
    md_lines.append("---\n")
    md_lines.append("## Summary\n")
    for icon,headline,detail in summary_insights:
        md_lines.append(f"**{icon} {headline}**")
        md_lines.append(f"> {detail}\n")
    md_lines.append("## Key Metrics\n")
    md_lines.append("| Recording | Tempo | Key | Mode | Beat Reg |")
    md_lines.append("|---|---|---|---|---|")
    for nm,R in zip(names,Rs_aligned):
        md_lines.append(f"| {nm} | {R.get('tempo','—')} BPM | {R.get('key','—')} | {R.get('mode','—')} | {R.get('beat_reg','—')} |")
    md_lines.append("")
    md_lines.append("## Pairwise Similarity\n")
    md_lines.append("| Pair | Similarity |")
    md_lines.append("|---|---|")
    for (i,j),score in zip(pairs,pair_scores):
        star="⭐ " if (i,j)==(pi,pj) else ""
        md_lines.append(f"| {star}{names[i]} vs {names[j]} | {score*100:.0f}% |")
    md_lines.append("\n## Figures\n")
    for title, desc, b64 in sections:
        md_lines.append(f"### {title}")
        md_lines.append(f"*{desc}*\n")
        md_lines.append(f"![{title}](data:image/png;base64,{b64})\n")
    md_lines.append("---")
    md_lines.append("*Generated by ViSuS — Music Signal Analysis*")
    md_str = "\n".join(md_lines)

    # ── Build HTML ────────────────────────────────────────────
    html_sections = ""
    for title, desc, b64 in sections:
        html_sections += f"""
        <div class="section">
          <h2>{title}</h2>
          <p class="desc">{desc}</p>
          <img src="data:image/png;base64,{b64}" alt="{title}"/>
        </div>"""

    metrics_rows = ""
    for nm,R in zip(names,Rs_aligned):
        metrics_rows += f"<tr><td>{nm}</td><td>{R.get('tempo','—')} BPM</td><td>{R.get('key','—')}</td><td>{R.get('mode','—')}</td><td>{R.get('beat_reg','—')}</td></tr>"

    sim_rows = ""
    for (i,j),score in zip(pairs,pair_scores):
        star="⭐ " if (i,j)==(pi,pj) else ""
        sim_rows += f"<tr><td>{star}{names[i]} vs {names[j]}</td><td>{score*100:.0f}%</td></tr>"

    summary_html = ""
    for icon,headline,detail in summary_insights:
        summary_html += f'<div class="insight"><strong>{icon} {headline}</strong><p>{detail}</p></div>'

    html_str = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>ViSuS Progress Report — {now}</title>
  <style>
    body {{font-family:'Segoe UI',sans-serif;background:#0f0f1a;color:#ccccdd;
           max-width:1100px;margin:0 auto;padding:32px 24px;}}
    h1 {{color:#4a9eff;border-bottom:2px solid #2a2a3e;padding-bottom:12px;}}
    h2 {{color:#4a9eff;margin-top:40px;border-left:4px solid #4a9eff;padding-left:12px;}}
    p.desc {{color:#888;font-size:0.9rem;margin:-8px 0 12px 0;}}
    .meta {{color:#666;font-size:0.9rem;margin-bottom:24px;}}
    table {{border-collapse:collapse;width:100%;margin:16px 0;}}
    th,td {{border:1px solid #2a2a3e;padding:8px 12px;text-align:left;}}
    th {{background:#1a1a2e;color:#4a9eff;}}
    tr:nth-child(even) {{background:#0d0d18;}}
    .insight {{background:#1a1a2e;border-left:3px solid #4a9eff;
               padding:10px 14px;margin:8px 0;border-radius:4px;}}
    .insight strong {{color:white;}}
    .insight p {{color:#aaa;margin:4px 0 0 0;font-size:0.9rem;}}
    .section {{margin:40px 0;}}
    .section img {{width:100%;border-radius:6px;border:1px solid #2a2a3e;
                   margin-top:10px;display:block;}}
    hr {{border:none;border-top:1px solid #2a2a3e;margin:32px 0;}}
    .footer {{color:#444;font-size:0.8rem;text-align:center;margin-top:48px;}}
    @media print {{
      body {{background:white;color:black;}}
      h1,h2 {{color:#1a5fa8;}}
      table th {{background:#e8f0fe;}}
      .insight {{background:#f5f5f5;border-color:#1a5fa8;}}
      .insight strong,.insight p {{color:black;}}
    }}
  </style>
</head>
<body>
  <h1>ViSuS Progress Report</h1>
  <div class="meta">
    Generated: {now} &nbsp;·&nbsp;
    Recordings: {', '.join(names)} &nbsp;·&nbsp;
    Best pair: <strong>{na} vs {nb}</strong> ({sim_matrix[pi][pj]*100:.0f}% similar) &nbsp;·&nbsp;
    Window: {zoom_start:.1f}s – {zoom_end:.1f}s
  </div>

  <h2>Summary</h2>
  {summary_html}

  <h2>Key Metrics</h2>
  <table>
    <tr><th>Recording</th><th>Tempo</th><th>Key</th><th>Mode</th><th>Beat Reg</th></tr>
    {metrics_rows}
  </table>

  <h2>Pairwise Similarity</h2>
  <table>
    <tr><th>Pair</th><th>Similarity</th></tr>
    {sim_rows}
  </table>

  {html_sections}

  <hr/>
  <div class="footer">Generated by ViSuS — Music Signal Analysis · HIWI Research Demo</div>
</body>
</html>"""

    return md_str, html_str

# ─────────────────────────────────────────────────────────────
# Session state — run analysis only when button is clicked.
# Results survive pair-selector / zoom changes without re-running.
# ─────────────────────────────────────────────────────────────
ready = all(f is not None for f in files)
if not ready:
    st.info(f"Upload all {n_recs} recordings to begin."); st.stop()

# Run analysis and store in session_state when button is pressed
if run_btn:
    with st.spinner("Loading audio..."):
        ys_new = []
        for f, n in zip(files, names):
            y = load_audio(f.read(), f.name)
            ys_new.append(y)

    with st.spinner("Analysing all rehearsals..."):
        Rs_new = []
        for y, n in zip(ys_new, names):
            R = run_analysis(
                y.astype(np.float32).tobytes(),
                do_mel, do_cqt, do_mfcc, do_chroma, do_spectral,
                do_onset, do_sms, do_stft, do_cwt, do_gammatone,
                do_tonnetz, do_zcr, do_reverb
            )
            Rs_new.append(R)

    with st.spinner("Aligning timelines..."):
        Rs_aligned_new = [Rs_new[0]]
        for R_other in Rs_new[1:]:
            Rs_aligned_new.append(
                align_to(Rs_new[0], R_other,
                         do_dtw and do_chroma,
                         dtw_strategy=dtw_strategy))

    # Parse each recording's own uploaded score (MIDI), if provided —
    # cached per-recording so every pair tab's Score Alignment view can
    # reuse them without re-uploading. List is same length as recordings;
    # entries are None where no score was uploaded for that recording.
    n_frames_target = max(int(len(y)/HOP) for y in ys_new)
    uploaded_score_chromas = []
    uploaded_score_names   = []
    for i, mf in enumerate(midi_files):
        if mf is not None:
            try:
                ch = midi_file_to_chroma(mf.getvalue(), n_frames_target)
                uploaded_score_chromas.append(ch)
                uploaded_score_names.append(mf.name)
            except Exception as _e_midi:
                uploaded_score_chromas.append(None)
                uploaded_score_names.append(None)
                st.error(f"Could not read MIDI for {names[i]}: {_e_midi}")
        else:
            uploaded_score_chromas.append(None)
            uploaded_score_names.append(None)
    st.session_state["vs_uploaded_score_chromas"] = uploaded_score_chromas
    st.session_state["vs_uploaded_score_names"]   = uploaded_score_names

    # Store everything in session_state — survives widget reruns
    st.session_state["vs_ys"]         = ys_new
    st.session_state["vs_Rs"]         = Rs_new
    st.session_state["vs_Rs_aligned"] = Rs_aligned_new
    st.session_state["vs_names"]      = names[:]
    st.session_state["vs_durs"]       = [len(y)/SR for y in ys_new]
    st.session_state["vs_n_recs"]     = n_recs
    st.session_state["vs_ready"]      = True
    # Clear the color-threshold slider's stored value so it recomputes
    # a fresh, data-driven default for whatever was just uploaded,
    # instead of carrying over a stale value from a previous dataset.
    st.session_state.pop("mds_color_thresholds", None)

# If no results yet (first load, no button pressed), stop here
if not st.session_state.get("vs_ready"):
    st.info("👈 Configure in sidebar, then click **▶ Run Analysis**."); st.stop()

# Restore from session_state — works whether button was just pressed or not
ys         = st.session_state["vs_ys"]
Rs         = st.session_state["vs_Rs"]
Rs_aligned = st.session_state["vs_Rs_aligned"]
names      = st.session_state["vs_names"]
durs       = st.session_state["vs_durs"]
n_recs     = st.session_state["vs_n_recs"]

# ── Sidebar-triggered generation (flags set by sidebar buttons) ──
if st.session_state.get("vs_ready"):
    # Generate report if flagged
    if st.session_state.get("sb_gen_report_flag"):
        st.session_state["sb_gen_report_flag"] = False
        with st.spinner("Generating report with all figures..."):
            try:
                # zoom_start/end not yet computed — derive from sidebar inputs
                _dur_max = max(durs) if durs else 30.0
                if zoom_mode == "Time (seconds)":
                    _zs = float(zoom_start_sb or 0.0)
                    _ze = float(min(zoom_end_sb or _dur_max, _dur_max))
                elif zoom_mode == "Musical bars":
                    _avg_t = float(np.mean([R.get("tempo",120) for R in Rs_aligned]))
                    _spb   = 60.0 / max(_avg_t, 1.0)
                    _zs    = (bar_start_sb - 1) * _spb * (time_sig_sb or 4)
                    _ze    = min(bar_end_sb * _spb * (time_sig_sb or 4), _dur_max)
                else:
                    _zs, _ze = 0.0, _dur_max
                _zs = max(0.0, _zs); _ze = min(_ze, _dur_max)
                if _ze <= _zs: _ze = _zs + 1.0

                _pairs  = [(i,j) for i in range(n_recs) for j in range(i+1,n_recs)]
                _sm     = [[pair_similarity(Rs_aligned[i],Rs_aligned[j])
                            for j in range(n_recs)] for i in range(n_recs)]
                for i in range(n_recs): _sm[i][i]=1.0
                _md, _html = generate_report(
                    Rs_aligned, names, _pairs, _sm,
                    _zs, _ze, ys, durs,
                    midi_chromas=st.session_state.get("midi_chromas_rep"),
                    consensus_chroma=st.session_state.get("consensus_rep"),
                    midi_alignments=st.session_state.get("midi_alignments_rep"))
                st.session_state["report_md"]   = _md
                st.session_state["report_html"] = _html
                st.toast("Report ready — download from sidebar ⬇️", icon="✅")
            except Exception as _e:
                st.error(f"Report generation failed: {_e}")

    # Generate MIDI if flagged
    if st.session_state.get("sb_gen_midi_flag"):
        st.session_state["sb_gen_midi_flag"] = False
        with st.spinner("Transcribing recordings to MIDI..."):
            try:
                try:
                    import pretty_midi as _pm
                except ImportError:
                    st.error("pretty_midi not installed. Run: pip install pretty_midi")
                    st.session_state["midi_ready"] = False
                    st.stop()
                import io as _io
                _mc, _cons, _alns, _vpcts, _used_up = compute_midi_alignment(
                    Rs_aligned, ys, names,
                    uploaded_score_chromas=st.session_state.get("vs_uploaded_score_chromas"))
                st.session_state["midi_chromas_rep"]   = _mc
                st.session_state["consensus_rep"]       = _cons
                st.session_state["midi_alignments_rep"] = _alns
                st.session_state["midi_voiced_pcts"]    = _vpcts
                # Pre-cache MIDI bytes per recording
                for _k, (_nm, _m) in enumerate(zip(names, _mc)):
                    _R    = Rs_aligned[_k]
                    _tempo= float(_R.get("tempo",120))
                    _p    = midi_chroma_to_pretty_midi(_m, tempo=_tempo)
                    _buf  = _io.BytesIO(); _p.write(_buf); _buf.seek(0)
                    st.session_state[f"midi_bytes_{_k}"] = _buf.getvalue()
                # Consensus MIDI
                _avg_t = float(np.mean([R.get("tempo",120) for R in Rs_aligned]))
                _p_c   = midi_chroma_to_pretty_midi(_cons, tempo=_avg_t)
                _buf_c = _io.BytesIO(); _p_c.write(_buf_c); _buf_c.seek(0)
                st.session_state["midi_bytes_consensus"] = _buf_c.getvalue()
                st.session_state["midi_ready"] = True
                st.toast(f"MIDI ready — {len(names)+1} files in sidebar ⬇️", icon="🎵")
            except Exception as _e:
                st.error(f"MIDI generation failed: {_e}")

# ── Sidebar download section — rendered here so vs_ready is confirmed ──
# Using st.sidebar.* from outside the sidebar block renders correctly
# because Streamlit collects all sidebar writes before rendering.
with st.sidebar:
    st.markdown("---")
    st.markdown("**⬇️ Downloads**")
    # Report
    if st.button("📄 Generate Report", use_container_width=True,
                 key="sb_gen_report"):
        st.session_state["sb_gen_report_flag"] = True
        st.rerun()
    if st.session_state.get("report_md"):
        st.download_button("⬇️ Report (.md)",
            data=st.session_state["report_md"],
            file_name="visus_report.md", mime="text/markdown",
            key="sb_dl_md", use_container_width=True)
        st.download_button("⬇️ Report (.html)",
            data=st.session_state["report_html"],
            file_name="visus_report.html", mime="text/html",
            key="sb_dl_html", use_container_width=True)
    st.markdown("---")
    # MIDI
    if st.button("🎵 Generate MIDI", use_container_width=True,
                 key="sb_gen_midi"):
        st.session_state["sb_gen_midi_flag"] = True
        st.rerun()
    if st.session_state.get("midi_ready"):
        _midi_names = st.session_state.get("vs_names", [])
        for _k, _nm in enumerate(_midi_names):
            _mdata = st.session_state.get(f"midi_bytes_{_k}")
            if _mdata:
                st.download_button(
                    f"⬇️ {_nm[:14]}.mid",
                    data=_mdata,
                    file_name=f"{_nm.replace(' ','_')[:20]}.mid",
                    mime="audio/midi",
                    key=f"sb_dl_midi_{_k}",
                    use_container_width=True)
        _cons = st.session_state.get("midi_bytes_consensus")
        if _cons:
            st.download_button("⬇️ Consensus Score.mid",
                data=_cons, file_name="visus_consensus_score.mid",
                mime="audio/midi", key="sb_dl_cons",
                use_container_width=True)

# Status bar — shows what's loaded without blocking anything
st.success(
    f"✓ {n_recs} recording{'s' if n_recs>1 else ''} analysed: "
    + ", ".join(f"**{n}** ({d:.1f}s)" for n,d in zip(names,durs))
    + "  ·  Change settings and click **▶ Run Analysis** to re-analyse.",
    icon="🎵")

# ─────────────────────────────────────────────────────────────
# Signal helpers
# ─────────────────────────────────────────────────────────────
def t2f(t):     return int(np.clip(t*SR/HOP, 0, 999999))
def cl1(a,s,e): return a[s:e]   if len(a)>s   else a[:0]
def cl2(a,s,e): return a[:,s:e] if a.shape[1]>s else a[:,:0]

# ─────────────────────────────────────────────────────────────
# Zoom window (runs after restore — needs durs, Rs, ys)
# ─────────────────────────────────────────────────────────────
dur_max = max(durs)
if zoom_mode == "Time (seconds)":
    zoom_start = float(zoom_start_sb)
    zoom_end   = float(min(zoom_end_sb or dur_max, dur_max))
elif zoom_mode == "Musical bars":
    avg_t = float(np.mean([R.get("tempo",120) for R in Rs]))
    spb=60/avg_t; zoom_start=(bar_start_sb-1)*spb*time_sig_sb
    zoom_end=min(bar_end_sb*spb*time_sig_sb, dur_max)
else:
    zoom_start, zoom_end = 0.0, dur_max
zoom_start=max(0.0,zoom_start); zoom_end=min(zoom_end,dur_max)
if zoom_end<=zoom_start: zoom_end=zoom_start+1.0
fs=t2f(zoom_start); fe=t2f(zoom_end)
segs = [y[int(zoom_start*SR):int(zoom_end*SR)] for y in ys]

# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
def render_matrix():
    """
    Interactive HTML/CSS similarity matrix.
    Each cell shows: similarity %, a mini stream-graph thumbnail.
    Click any cell to expand the stream graph to full size.
    Diagonal = same recording (shown as —).
    Color: green = similar, red = different.
    """
    import streamlit.components.v1 as components

    st.subheader("Pairwise Similarity Matrix")
    st.caption(
        "Each cell shows similarity % and a mini stream graph. "
        "**Click any cell** to expand the stream graph. "
        "Green = similar · Red = different.")

    n = len(Rs_aligned)

    # Build similarity matrix
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_matrix[i,j] = 1.0 if i==j else pair_similarity(Rs_aligned[i], Rs_aligned[j])

    # Pre-render all stream graph thumbnails and full-size PNGs
    thumbs = {}   # (i,j) → base64 PNG (small)
    fulls  = {}   # (i,j) → base64 PNG (large)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            key = (i, j)
            if (j, i) in thumbs:         # reuse symmetric pair
                thumbs[key] = thumbs[(j,i)]
                fulls[key]  = fulls[(j,i)]
            else:
                thumbs[key] = make_stream_graph_png(
                    Rs_aligned[i], Rs_aligned[j], names[i], names[j],
                    figsize=(3.5, 1.6), fontsize=6, for_matrix=True)
                fulls[key]  = make_stream_graph_png(
                    Rs_aligned[i], Rs_aligned[j], names[i], names[j],
                    figsize=(9, 3.5), fontsize=9, for_matrix=False)

    # Color helper: sim score → CSS background
    def cell_bg(sim):
        r = int(233*(1-sim) + 76*sim)
        g = int(30*(1-sim)  + 175*sim)
        b = int(99*(1-sim)  + 80*sim)
        return f"rgb({r},{g},{b})"

    def text_col(sim):
        return "#000" if sim > 0.55 else "#fff"

    # ── Grid: color-coded cells + mini thumbnails (HTML) ────
    cell_size = max(110, min(180, 640 // n))

    def cell_bg(sim):
        r=int(233*(1-sim)+76*sim); g=int(30*(1-sim)+175*sim); b=int(99*(1-sim)+80*sim)
        return f"rgb({r},{g},{b})"
    def text_col(sim): return "#000" if sim>0.55 else "#fff"

    css = f"""<style>
      *{{box-sizing:border-box;margin:0;padding:0;}}
      body{{background:#0f0f1a;font-family:sans-serif;}}
      .sim-grid{{display:grid;
        grid-template-columns:100px repeat({n},{cell_size}px);
        gap:3px;margin-bottom:6px;}}
      .sim-header{{background:#1a1a2e;color:#aaa;font-size:10px;
        padding:4px;text-align:center;border-radius:4px;
        display:flex;align-items:center;justify-content:center;
        font-weight:bold;word-break:break-word;}}
      .sim-row-label{{background:#1a1a2e;color:#aaa;font-size:10px;
        padding:4px 6px;border-radius:4px;display:flex;align-items:center;
        font-weight:bold;word-break:break-word;}}
      .sim-cell{{border-radius:6px;padding:4px;text-align:center;
        height:{cell_size}px;display:flex;flex-direction:column;
        align-items:center;justify-content:center;}}
      .sim-cell .pct{{font-size:13px;font-weight:bold;line-height:1.2;}}
      .sim-cell img{{width:100%;border-radius:3px;margin-top:3px;display:block;}}
      .sim-diag{{background:#1a1a2e;color:#444;font-size:20px;
        border-radius:6px;display:flex;align-items:center;
        justify-content:center;height:{cell_size}px;}}
    </style>"""

    grid = '<div class="sim-grid">'
    grid += '<div class="sim-header"></div>'
    for j in range(n):
        short = names[j] if len(names[j])<=10 else names[j][:9]+"…"
        grid += f'<div class="sim-header">{short}</div>'
    for i in range(n):
        short_row = names[i] if len(names[i])<=10 else names[i][:9]+"…"
        grid += f'<div class="sim-row-label">{short_row}</div>'
        for j in range(n):
            if i == j:
                grid += f'<div class="sim-diag">—</div>'
            else:
                sim       = sim_matrix[i,j]
                bg        = cell_bg(sim)
                tc        = text_col(sim)
                pct       = f"{sim*100:.0f}%"
                thumb_b64 = thumbs.get((i,j))
                img_tag   = (f'<img src="data:image/png;base64,{thumb_b64}" alt=""/>'
                             if thumb_b64 else "")
                grid += (f'<div class="sim-cell" style="background:{bg};color:{tc}">'
                         f'<span class="pct">{pct}</span>{img_tag}</div>')
    grid += '</div>'

    grid_h = 44 + n*(cell_size+4)
    import streamlit.components.v1 as components
    components.html(css + grid, height=grid_h, scrolling=False)

    # ── Pair buttons → stream graph with native fullscreen button ─
    st.caption("Click a pair below to see its full stream graph "
               "(use the ⛶ icon in the top-right of the chart to fullscreen):")

    pairs_list = [(i,j) for i in range(n) for j in range(i+1,n)]
    pair_scores_list = [sim_matrix[i][j] for i,j in pairs_list]
    best_pair_idx = int(np.argmax(pair_scores_list))

    # One button per pair — laid out in a row
    btn_cols = st.columns(len(pairs_list))
    selected_pair = st.session_state.get("matrix_selected_pair",
                                          pairs_list[best_pair_idx])
    for col, (i,j), score in zip(btn_cols, pairs_list, pair_scores_list):
        label = (f"⭐ {names[i]} vs {names[j]}"
                 if (i,j)==pairs_list[best_pair_idx]
                 else f"{names[i]} vs {names[j]}")
        caption = f"{score*100:.0f}%"
        if col.button(label, help=caption, key=f"matbtn_{i}_{j}"):
            st.session_state["matrix_selected_pair"] = (i,j)
            selected_pair = (i,j)

    # Render selected pair's stream graph with native Streamlit fullscreen
    si, sj = selected_pair
    st.markdown(f"**{names[si]} vs {names[sj]}** — "
                f"{sim_matrix[si][sj]*100:.0f}% similar")
    fig_sel = plt.figure(figsize=(14, 4))
    fig_sel.patch.set_facecolor('#0f0f1a')
    ax_sel = fig_sel.add_subplot(111)
    ax_sel.set_facecolor('#0f0f1a')

    from scipy.signal import savgol_filter as _sgf2
    STREAM_FEATS_M = [("rms","Volume","#4CAF50"),("centroid","Brightness","#2196F3"),
                      ("onset","Rhythm","#FF9800"),("sms_ratio","Harmony","#9C27B0")]
    streams_m = {}
    Ra_m = Rs_aligned[si]; Rb_m = Rs_aligned[sj]
    for key, label, col in STREAM_FEATS_M:
        arr_a = Ra_m.get(key); arr_b = Rb_m.get(key)
        if arr_a is None or arr_b is None: continue
        n_s = min(len(arr_a), len(arr_b))
        a_n = (arr_a[:n_s]-arr_a[:n_s].mean())/(arr_a[:n_s].std()+1e-8)
        b_n = (arr_b[:n_s]-arr_b[:n_s].mean())/(arr_b[:n_s].std()+1e-8)
        div_s = np.abs(a_n - b_n)
        if len(div_s)>21: div_s=_sgf2(div_s, min(51,len(div_s)//4*2+1), 3)
        streams_m[label] = (np.clip(div_s,0,None), col)

    if streams_m:
        min_len_m  = min(len(v) for v,_ in streams_m.values())
        times_m    = np.arange(min_len_m)*HOP/SR
        labels_m   = list(streams_m.keys())
        vals_m     = np.array([v[:min_len_m] for v,_ in streams_m.values()])
        cols_m     = [c for _,c in streams_m.values()]
        total_m    = vals_m.sum(axis=0)
        peak_m     = total_m.max()+1e-8
        scaled_m   = vals_m*(0.5/peak_m)

        ax_sel.axhline(0, color='white', lw=1.0, alpha=0.5, zorder=2)
        legend_handles_m = []
        cum_pos = np.zeros(min_len_m)
        for label, val_sc, col in zip(labels_m, scaled_m, cols_m):
            upper = cum_pos + val_sc
            patch = ax_sel.fill_between(times_m, cum_pos, upper,
                                         color=col, alpha=0.82, label=label)
            legend_handles_m.append(patch)
            cum_pos = upper
        cum_neg = np.zeros(min_len_m)
        for val_sc, col in zip(scaled_m, cols_m):
            lower = cum_neg - val_sc
            ax_sel.fill_between(times_m, lower, cum_neg, color=col, alpha=0.82)
            cum_neg = lower

        ax_sel.set_xlim(0, times_m[-1])
        ax_sel.set_ylim(-0.55, 0.55)
        ax_sel.set_yticks([-0.5,-0.25,0,0.25,0.5])
        ax_sel.set_yticklabels(["-0.5","-0.25","0","+0.25","+0.5"],
                                fontsize=7, color='white')
        ax_sel.set_xlabel("Time (s)", fontsize=8, color='white')
        ax_sel.set_ylabel("Divergence", fontsize=8, color='white')
        ax_sel.tick_params(colors='white')
        for sp in ax_sel.spines.values(): sp.set_edgecolor('#333')
        ax_sel.legend(handles=legend_handles_m[::-1], labels=labels_m[::-1],
                      fontsize=8, loc='upper right',
                      facecolor='#1a1a2e', labelcolor='white', edgecolor='#333')
    else:
        ax_sel.text(0.5, 0.5, "Enable Spectral + Onset features",
                    ha='center', va='center', color='#aaa',
                    transform=ax_sel.transAxes)

    plt.tight_layout()
    st.pyplot(fig_sel, use_container_width=True)
    plt.close()

    # ── Two MDS views — DTW-based and all-features-based ─────
    if n >= 3:
        try:
            from sklearn.manifold import MDS

            # ── Shared helper: plot MDS coords ─────────────────────
            def plot_mds(fig, ax, coords, dist_matrix, title, line_label_fn,
                        green_thresh=0.75, red_thresh=0.50, force_labels=False):
                """
                Draw dots + connecting lines on ax, with a similarity colorbar.

                Identical/near-identical recordings land on (almost) the
                exact same MDS coordinate, which would otherwise render as
                one dot silently hiding another. We detect coincident
                groups and nudge them apart in a small ring so every
                recording stays visible, then draw a dashed outline
                connecting the ring back to its true shared position.
                """
                ax.set_facecolor('#0f0f1a')
                x_range = coords[:,0].max() - coords[:,0].min() + 1e-8
                y_range = coords[:,1].max() - coords[:,1].min() + 1e-8
                plot_scale = max(x_range, y_range)

                # ── Detect and separate coincident/near-coincident points ──
                # Threshold: points within 1.5% of the plot's overall scale
                # are considered "the same spot" for display purposes.
                coincide_eps = 0.015 * plot_scale
                display_coords = coords.copy()
                visited = set()
                overlap_groups = []
                for i in range(n):
                    if i in visited: continue
                    group = [i]
                    for j in range(i+1, n):
                        if j in visited: continue
                        if np.linalg.norm(coords[i] - coords[j]) < coincide_eps:
                            group.append(j)
                    if len(group) > 1:
                        overlap_groups.append(group)
                        visited.update(group)
                        # Arrange the group in a small ring around their
                        # shared true position, radius scaled to plot size
                        ring_r = 0.045 * plot_scale
                        cx, cy = coords[group].mean(axis=0)
                        for k, idx in enumerate(group):
                            ang = 2*np.pi*k/len(group)
                            display_coords[idx,0] = cx + ring_r*np.cos(ang)
                            display_coords[idx,1] = cy + ring_r*np.sin(ang)

                # Beyond 4 recordings, per-line numeric labels overlap into
                # unreadable clutter (n=8 → 28 pairwise lines) by default —
                # but force_labels (checkbox) overrides this if the user
                # wants exact numbers regardless of clutter.
                show_labels = (n <= 4) or force_labels

                for i in range(n):
                    for j in range(i+1, n):
                        x0,y0 = display_coords[i]; x1,y1 = display_coords[j]
                        d = dist_matrix[i,j]
                        sim_val = 1.0 - d
                        line_col = ('#4CAF50' if sim_val>=green_thresh
                                    else '#FF9800' if sim_val>=red_thresh else '#E91E63')
                        ax.plot([x0,x1],[y0,y1], color=line_col, lw=1.6, alpha=0.45, zorder=1)
                        if show_labels:
                            lbl = line_label_fn(i, j, d)
                            ax.text((x0+x1)/2, (y0+y1)/2, lbl,
                                    fontsize=6, color='#ccc', ha='center', va='center',
                                    bbox=dict(boxstyle='round,pad=0.15', fc='#1a1a2e',
                                              ec='none', alpha=0.8), zorder=2)

                # Faint marker at the true shared position for each overlap group
                for group in overlap_groups:
                    cx, cy = coords[group].mean(axis=0)
                    ax.scatter(cx, cy, s=40, color='white', marker='+',
                              alpha=0.5, zorder=2)

                for i, (nm, (x, y)) in enumerate(zip(names, display_coords)):
                    ax.scatter(x, y, s=200, color=rec_color(i), zorder=3,
                               edgecolors='white', linewidths=1.5)
                    short_nm = nm if len(nm) <= 14 else nm[:13] + "…"
                    ax.text(x, y + 0.06*y_range, short_nm, fontsize=7.5, color='white',
                            ha='center', va='bottom', fontweight='bold', zorder=4)

                # Pad the axis limits so dots/labels near the edge aren't clipped
                pad_x = 0.20*x_range + 0.05*plot_scale
                pad_y = 0.15*y_range + 0.05*plot_scale
                ax.set_xlim(coords[:,0].min() - pad_x, coords[:,0].max() + pad_x)
                ax.set_ylim(coords[:,1].min() - pad_y, coords[:,1].max() + pad_y*1.3)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(title, fontsize=9, fontweight='bold', color='white', pad=6)
                for sp in ax.spines.values(): sp.set_visible(False)

                caption_lines = []
                if overlap_groups:
                    for group in overlap_groups:
                        gnames = ", ".join(names[k] for k in group)
                        caption_lines.append(f"({gnames}) are near-identical — spread in a ring, marked +")
                if not show_labels:
                    caption_lines.append(
                        "Line labels hidden (too many pairs) — enable "
                        "\"show all pairwise labels\" above, or see the "
                        "similarity matrix for exact %")
                if caption_lines:
                    ax.text(0.5, -0.07, "\n".join(caption_lines),
                            transform=ax.transAxes, fontsize=6.5, color='#888',
                            ha='center', va='top')

                # ── Similarity colorbar — always shown, gives the "axis" ──
                import matplotlib.colors as _mcolors
                cmap_line = _mcolors.LinearSegmentedColormap.from_list(
                    'sim_line', ['#E91E63','#FF9800','#4CAF50'], N=256)
                sm = plt.cm.ScalarMappable(cmap=cmap_line,
                                           norm=plt.Normalize(vmin=0, vmax=1))
                sm.set_array([])
                cb = fig.colorbar(sm, ax=ax, orientation='horizontal',
                                  fraction=0.05, pad=0.16, shrink=0.9)
                cb.set_label("Similarity between recordings",
                            color='white', fontsize=7)
                cb.set_ticks([0, red_thresh, green_thresh, 1])
                cb.set_ticklabels([f'0%\n(different)', f'{red_thresh*100:.0f}%',
                                   f'{green_thresh*100:.0f}%', f'100%\n(identical)'])
                cb.ax.xaxis.set_tick_params(color='white', labelsize=6)
                plt.setp(cb.ax.get_xticklabels(), color='white')
                cb.outline.set_edgecolor('#333')


            # NOTE: an earlier version also computed a "DTW distance" MDS
            # view (temporal offset + chroma cosine, blended). It was
            # dropped per feedback — the all-features view below was the
            # clearer/more useful one in practice — so that computation is
            # no longer run here to avoid the extra DTW cost.

            # ── Feature registry — group_name → (dims, extractor) ────
            # Each entry pulls one feature group out of an R dict as a
            # fixed-length list of floats. This lets the user pick which
            # groups go into the MDS distance (and weight them), instead
            # of always averaging every available feature equally.
            FEATURE_GROUPS = [
                ("MFCC (timbre)",       lambda R: R["mfcc_means"][1:].tolist() if "mfcc_means" in R else None),
                ("Chroma (harmony)",    lambda R: R["chroma"].mean(1).tolist() if "chroma" in R else None),
                ("Tempo",               lambda R: [float(R["tempo"])] if "tempo" in R else None),
                ("Beat regularity",     lambda R: [float(R["beat_reg"])] if "beat_reg" in R else None),
                ("RMS (loudness)",      lambda R: [float(R["rms"].mean())] if "rms" in R else None),
                ("Centroid (brightness)", lambda R: [float(R["centroid"].mean())] if "centroid" in R else None),
                ("Bandwidth",           lambda R: [float(R["bandwidth"].mean())] if "bandwidth" in R else None),
                ("Rolloff",             lambda R: [float(R["rolloff"].mean())] if "rolloff" in R else None),
                ("Flatness",            lambda R: [float(R["flatness"].mean())] if "flatness" in R else None),
                ("Harmonic ratio (SMS)",lambda R: [float(R["sms_ratio"].mean())] if "sms_ratio" in R else None),
                ("Tonnetz (tonal centroid)", lambda R: R["tonnetz"].mean(1).tolist() if "tonnetz" in R else None),
            ]
            # Which groups are actually available given R0's computed features
            R0 = Rs_aligned[0]
            available_groups = [name for name, fn in FEATURE_GROUPS if fn(R0) is not None]

            def build_feature_vector(R, selected_groups, weights):
                """
                One recording's feature vector, restricted to the selected
                groups and scaled by each group's weight (applied AFTER
                z-scoring in build_feature_distance_matrix — weight here
                is just a placeholder position, real scaling happens there
                using the returned group_slices).
                """
                vec = []
                group_slices = {}   # group_name -> (start, end) index in vec
                for name, fn in FEATURE_GROUPS:
                    if name not in selected_groups: continue
                    vals = fn(R)
                    if vals is None: continue
                    start = len(vec)
                    vec.extend(vals)
                    group_slices[name] = (start, len(vec))
                return np.array(vec, dtype=np.float32), group_slices

            def build_feature_distance_matrix(Rs_list, selected_groups, weights):
                """
                Build the full (n,n) feature-distance matrix at once, using
                z-score normalisation ACROSS ALL RECORDINGS (not per-pair),
                then applying the user's per-group weight multiplier, then
                scaling the whole matrix by its own max so the largest
                distance is 1.0 — this preserves relative ordering between
                pairs instead of clipping each pair independently, which
                previously caused many "very different" pairs to collapse
                to the same distance and made MDS squash them together.
                """
                vecs_and_slices = [build_feature_vector(R, selected_groups, weights) for R in Rs_list]
                vecs = [v for v, _ in vecs_and_slices]
                group_slices = vecs_and_slices[0][1] if vecs_and_slices else {}
                min_len = min(len(v) for v in vecs) if vecs else 0
                if min_len == 0:
                    n_ = len(Rs_list)
                    return np.zeros((n_, n_))
                vecs = np.array([v[:min_len] for v in vecs])   # (n, d)
                mean = vecs.mean(axis=0)
                std  = vecs.std(axis=0)
                std[std < 1e-8] = 1e-8
                normed = (vecs - mean) / std                    # z-scored per feature

                # Apply per-group weight multiplier to its slice of columns
                for gname, (s, e) in group_slices.items():
                    w = weights.get(gname, 1.0)
                    if e <= normed.shape[1]:
                        normed[:, s:e] *= w

                n_ = len(Rs_list)
                raw = np.zeros((n_, n_))
                for i in range(n_):
                    for j in range(n_):
                        if i != j:
                            raw[i,j] = float(np.linalg.norm(normed[i] - normed[j]))
                mx = raw.max()
                if mx > 1e-8:
                    raw = raw / mx
                return raw

            # ── Feature selection + weighting controls ──────────────
            st.markdown("**Which features drive this map?**")
            selected_groups = st.multiselect(
                "Include feature groups", options=available_groups,
                default=available_groups, key="mds_feature_groups",
                help="Deselect any feature group to exclude it from the "
                     "distance calculation entirely. Fewer groups = the "
                     "map reflects only what you kept.")
            if not selected_groups:
                st.warning("Select at least one feature group.")
                selected_groups = available_groups

            weights = {g: 1.0 for g in available_groups}
            with st.expander("⚖️ Feature weights (optional — default all equal)",
                              expanded=False):
                st.caption(
                    "After normalisation, each selected group counts equally "
                    "by default (weight 1.0×). Raise a group's weight to make "
                    "it matter more in the map; lower it to matter less. "
                    "This is separate from group size — e.g. MFCC naturally "
                    "has 12 numbers vs. Tempo's 1, so MFCC already has more "
                    "influence even at equal weight; use these sliders to "
                    "compensate if you want tempo to matter as much as timbre.")
                w_cols = st.columns(min(3, max(1, len(selected_groups))))
                for idx, gname in enumerate(selected_groups):
                    with w_cols[idx % len(w_cols)]:
                        weights[gname] = st.slider(
                            gname, 0.1, 3.0, 1.0, 0.1,
                            key=f"mds_weight_{gname}")

            # ── Build the distance matrix ────────────────────────────
            feat_mat = build_feature_distance_matrix(Rs_aligned, selected_groups, weights)

            # ── MDS projection (feature-based only — see note below) ──
            mds = MDS(n_components=2, dissimilarity='precomputed',
                      random_state=42, n_init=4, normalized_stress='auto')
            coords_feat = mds.fit_transform(feat_mat)

            # Dimension count + display labels reflect what the user
            # actually selected above (not just what's available)
            _dims_per_group = {
                "MFCC (timbre)": 12, "Chroma (harmony)": 12,
                "Tonnetz (tonal centroid)": 6,
            }
            feat_list = [
                f"{g} (weight {weights.get(g,1.0):.1f}×)"
                if weights.get(g,1.0) != 1.0 else g
                for g in selected_groups
            ]
            n_dims = sum(_dims_per_group.get(g, 1) for g in selected_groups)

            # ── Explanation ─────────────────────────────────────────
            st.markdown("#### Recording Map (MDS)")
            with st.expander("ℹ️ How this map is built — read before interpreting it",
                              expanded=False):
                st.markdown(f"""
**What you're looking at:** each dot is one recording. **MDS
(Multidimensional Scaling)** takes a table of pairwise distances between
all recordings and finds 2D coordinates that best preserve those
distances — so the *geometry* of the plot (which dots cluster, which sit
far apart) is the real signal, not the absolute position or orientation
of any single dot (MDS output can be rotated/flipped without changing
its meaning).

**What "distance" is made of ({n_dims} dimensions):** every recording is
turned into one numeric vector combining {', '.join(feat_list)}. Each of
those {n_dims} numbers is **z-score normalised across all uploaded
recordings** first (subtract the group mean, divide by the group's
standard deviation) — this is what makes the map fair: a feature like
spectral centroid (measured in Hz, values in the thousands) does not
automatically outweigh a feature like beat regularity (values under 5)
just because its raw numbers are bigger. After normalising, we take the
plain Euclidean distance between every pair of vectors, then divide the
whole distance matrix by its own largest value — so the single most
different pair in your batch always sits at exactly 100% "different",
and everyone else is scaled relative to that. This means the map is
**relative to whatever you uploaded**, not an absolute scale — five very
similar takes will still show visible spread between them, because MDS
always stretches to use the full plot.

**Where the similarity/difference line is drawn:** there is no fixed
numeric cutoff baked into the underlying distance — the 🟢/🟠/🔴
coloring on the connecting lines is a **display threshold only**,
chosen for readability: 🟢 ≥75% similar · 🟠 50–75% · 🔴 <50%. If you
need a different threshold for a specific decision (e.g. "same take,
re-record if below 90%"), that should be set explicitly for your use
case rather than assumed from the color — the color bands are a visual
aid, not a validated pass/fail boundary.

**Which parameters are most sensitive:** in this z-score setup, MFCC (12
dims) and Chroma (12 dims) dominate the vector by sheer count — together
they're 24 of the {n_dims} dimensions — so timbral and harmonic
differences generally move a recording's position more than a single
scalar like tempo or beat regularity does. If you want tempo/rhythm
differences to matter *more* in the map, that would need deliberate
re-weighting (e.g. duplicating those dimensions, or a weighted distance)
— currently every dimension counts equally after normalisation, so more
dimensions from one feature family means more influence from that family.
""")
            st.caption(
                f"Distance = normalised Euclidean distance in a {n_dims}D "
                f"feature space (features: {', '.join(feat_list)}). "
                "Closer dots = more similar. Line labels/colorbar show "
                "**similarity** (100% = identical, 0% = most different "
                "pair in this batch).")

            # ── User-adjustable controls ─────────────────────────────
            # Compute a data-driven default so the initial color split
            # reflects THIS dataset's actual similarity range, rather
            # than always defaulting to a fixed 75/50 -- which puts
            # everything in one color bucket when a batch's similarities
            # are all very high (e.g. near-duplicate takes) or all
            # clustered in a tighter band than 50-75% (e.g. an outlier
            # test where every pair is well above 75% anyway).
            _off_diag = sim_matrix[~np.eye(n, dtype=bool)]
            if len(_off_diag) >= 2 and _off_diag.std() > 1e-4:
                _mean, _std = float(_off_diag.mean()), float(_off_diag.std())
                _green_def = int(round((_mean + 0.5*_std) * 100 / 5) * 5)
                _red_def   = int(round((_mean - 0.5*_std) * 100 / 5) * 5)
                _green_def = min(max(_green_def, 10), 95)
                _red_def   = min(max(_red_def, 5), _green_def - 10)
            else:
                _green_def, _red_def = 75, 50  # not enough spread to adapt -- fall back

            ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2,2,1.4])
            with ctrl_col1:
                green_thresh_pct, red_thresh_pct = st.slider(
                    "Color thresholds (green ≥ upper, red < lower)",
                    min_value=0, max_value=100,
                    value=(_green_def, _red_def), step=5,
                    key="mds_color_thresholds",
                    help="Auto-set from this dataset's similarity range so the "
                         "colors are meaningful by default. Adjust anytime — "
                         "purely visual, doesn't change the underlying distances.")
            with ctrl_col3:
                force_labels = st.checkbox(
                    "Show all pairwise labels", value=False,
                    key="mds_force_labels",
                    help="Force exact similarity % on every connecting line, "
                         "even with many recordings (can get cluttered).")
            green_thresh = green_thresh_pct / 100.0
            red_thresh   = red_thresh_pct / 100.0

            # Figure size scales with number of recordings so dots/labels
            # don't get squeezed together as n grows
            mds_figsize = (max(6.5, 5.0 + 0.6*n), max(5.2, 4.2 + 0.5*n))
            fig2, ax2 = plt.subplots(figsize=mds_figsize)
            fig2.patch.set_facecolor('#0f0f1a')
            plot_mds(fig2, ax2, coords_feat, feat_mat,
                     "All-features distance",
                     lambda i,j,d: f"sim: {(1-d)*100:.0f}%",
                     green_thresh=green_thresh, red_thresh=red_thresh,
                     force_labels=force_labels)
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True); plt.close()
        except Exception as e:
            st.caption(f"MDS unavailable: {e}")

    # Feature comparison across all recordings
    st.subheader("Feature Comparison — All Rehearsals")
    metrics = []
    if all("tempo" in R for R in Rs_aligned):
        tempos = [R["tempo"] for R in Rs_aligned]
        metrics.append(("Tempo (BPM)", tempos,
                        "Lower variation = more consistent tempo across sessions."))
    if all("beat_reg" in R for R in Rs_aligned):
        regs = [R["beat_reg"] for R in Rs_aligned]
        metrics.append(("Beat Regularity (lower=tighter)", regs,
                        "How metronomic each rehearsal was."))
    if all("rms" in R for R in Rs_aligned):
        rms_vals = [float(R["rms"].mean()) for R in Rs_aligned]
        metrics.append(("Average Loudness (RMS)", rms_vals,
                        "Relative energy level of each rehearsal."))
    if all("centroid" in R for R in Rs_aligned):
        cents = [float(R["centroid"].mean()) for R in Rs_aligned]
        metrics.append(("Brightness (Hz)", cents,
                        "Higher = brighter/more treble in that rehearsal."))
    if all("sms_ratio" in R for R in Rs_aligned):
        ratios = [float(R["sms_ratio"].mean()) for R in Rs_aligned]
        metrics.append(("Harmonic Ratio (melodic vs rhythmic)", ratios,
                        "Higher = more melodic/sustained content."))

    if metrics:
        n_m = len(metrics)
        fig_feat, axes_feat = plt.subplots(1, n_m, figsize=(min(16, n_m*3.5), 4))
        if n_m==1: axes_feat=[axes_feat]
        colors = plt.cm.tab10(np.linspace(0,0.9,len(names)))
        for ax, (feat_name, vals, note) in zip(axes_feat, metrics):
            bars = ax.bar(range(len(names)), vals, color=colors, alpha=0.85)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels([n[:12] for n in names], rotation=30, ha='right', fontsize=8)
            ax.set_title(feat_name, fontsize=8, fontweight='bold')
            ax.set_ylabel("")
            # Highlight best (lowest for regularity, highest for others)
            if "Regularity" in feat_name:
                best = int(np.argmin(vals))
            else:
                best = int(np.argmax(vals))
            bars[best].set_edgecolor('gold'); bars[best].set_linewidth(2.5)
        plt.tight_layout(); st.pyplot(fig_feat, use_container_width=True); plt.close()
        st.caption("Gold outline = best performing rehearsal for each metric.")

    # Key/mode table
    if all("key" in R for R in Rs_aligned):
        st.markdown("**Key & Mode per rehearsal:**")
        cols = st.columns(len(names))
        for col, name, R in zip(cols, names, Rs_aligned):
            col.markdown(f"**{name}**  \n{R.get('key','—')}  \n{R.get('mode','—')}")

    # Progression note
    if len(Rs_aligned) >= 3 and all("beat_reg" in R for R in Rs_aligned):
        regs = [R["beat_reg"] for R in Rs_aligned]
        if regs[-1] < regs[0]:
            pct = (regs[0]-regs[-1])/regs[0]*100
            st.success(f"📈 Rhythmic tightness improved by {pct:.0f}% from {names[0]} to {names[-1]}.")
        elif regs[-1] > regs[0]:
            pct = (regs[-1]-regs[0])/regs[0]*100
            st.warning(f"📉 Rhythmic tightness decreased by {pct:.0f}% from {names[0]} to {names[-1]}.")

    # ── N-recording overlay + mean±spread ────────────────
    st.markdown("---")
    st.subheader("All Rehearsals — Group View")

    OVERLAY_FEATS = []
    for key, label in [("rms","Volume (RMS)"),
                        ("centroid","Brightness (Centroid Hz)"),
                        ("onset","Onset Strength"),
                        ("sms_ratio","Harmonic Ratio"),
                        ("bandwidth","Bandwidth (Hz)"),
                        ("flatness","Spectral Flatness")]:
        if all(key in R for R in Rs_aligned):
            OVERLAY_FEATS.append((key, label))

    if not OVERLAY_FEATS:
        st.info("Enable Spectral Features and Onset to see overlay plots.")
    else:
        ov_tab1, ov_tab2 = st.tabs(["📈 All recordings overlaid",
                                     "📊 Group mean ± spread"])

        # ── Tab 1: All recordings on same axes ────────────
        with ov_tab1:
            st.caption("Every rehearsal on the same axes. "
                       "Each keeps its recording color. "
                       "Separating lines = the rehearsals diverge at that moment.")
            n_feats = len(OVERLAY_FEATS)
            fig_ov, axes_ov = plt.subplots(n_feats, 1,
                figsize=(14, 2.5*n_feats), sharex=True)
            fig_ov.patch.set_facecolor('#0f0f1a')
            if n_feats == 1: axes_ov = [axes_ov]
            for ax, (feat, feat_label) in zip(axes_ov, OVERLAY_FEATS):
                ax.set_facecolor('#0f0f1a')
                for idx, (R, nm) in enumerate(zip(Rs_aligned, names)):
                    arr = R.get(feat)
                    if arr is None: continue
                    arr_w = cl1(arr, fs, fe)
                    if len(arr_w) == 0: continue
                    t_feat = np.linspace(zoom_start, zoom_end, len(arr_w))
                    ax.plot(t_feat, arr_w, lw=0.9, color=rec_color(idx),
                            label=nm, alpha=0.85)
                ax.set_ylabel(feat_label, fontsize=7, color='white')
                ax.tick_params(colors='white', labelsize=6)
                for sp in ax.spines.values(): sp.set_edgecolor('#333')
                ax.legend(fontsize=7, loc="upper right",
                          facecolor='#1a1a2e', labelcolor='white', edgecolor='#333')
            axes_ov[-1].set_xlabel("Time (s)", fontsize=8, color='white')
            plt.suptitle(f"All {len(names)} Rehearsals Overlaid",
                         fontsize=10, fontweight="bold", color='white')
            plt.tight_layout()
            st.pyplot(fig_ov, use_container_width=True); plt.close()

        # ── Tab 2: Group mean ± spread ─────────────────────
        with ov_tab2:
            st.caption(
                "**Solid line** = group mean (average across all rehearsals). "
                "**Shaded band** = ±1 standard deviation (how spread out the rehearsals are). "
                "**Colored lines** = each rehearsal's deviation from the mean. "
                "A wide band = rehearsals disagreed here. "
                "A rehearsal line far from the band centre = that rehearsal was the outlier.")

            n_feats = len(OVERLAY_FEATS)
            fig_ms, axes_ms = plt.subplots(n_feats, 1,
                figsize=(14, 3.0*n_feats), sharex=True)
            fig_ms.patch.set_facecolor('#0f0f1a')
            if n_feats == 1: axes_ms = [axes_ms]

            for ax, (feat, feat_label) in zip(axes_ms, OVERLAY_FEATS):
                ax.set_facecolor('#0f0f1a')

                # Collect all arrays aligned to same length
                arrays = []
                for R in Rs_aligned:
                    arr = R.get(feat)
                    if arr is None: continue
                    arr_w = cl1(arr, fs, fe)
                    if len(arr_w) > 0:
                        arrays.append(arr_w)
                if len(arrays) < 2:
                    ax.text(0.5, 0.5, "Not enough data", ha='center',
                            transform=ax.transAxes, color='#aaa')
                    continue

                # Align to shortest
                min_len_ov = min(len(a) for a in arrays)
                arrays = [a[:min_len_ov] for a in arrays]
                t_feat = np.linspace(zoom_start, zoom_end, min_len_ov)
                stacked = np.array(arrays)          # (n_recs, T)
                mean    = stacked.mean(axis=0)
                std     = stacked.std(axis=0)

                # Shade ±1 std
                ax.fill_between(t_feat, mean-std, mean+std,
                                alpha=0.25, color='white',
                                label='±1 std (group spread)')
                # Mean line
                ax.plot(t_feat, mean, lw=2.0, color='white',
                        label='Group mean', zorder=4)
                # Each recording's deviation
                for idx, (arr_w, nm) in enumerate(zip(arrays, names)):
                    ax.plot(t_feat, arr_w, lw=0.8, color=rec_color(idx),
                            label=nm, alpha=0.75, zorder=3)

                # Mark moments where any recording > 1.5 std from mean
                outlier_thresh = 1.5 * std
                for idx, arr_w in enumerate(arrays):
                    dev = np.abs(arr_w - mean)
                    outlier_frames = np.where(dev > outlier_thresh)[0]
                    if len(outlier_frames) > 0:
                        ax.scatter(t_feat[outlier_frames],
                                   arr_w[outlier_frames],
                                   s=12, color=rec_color(idx),
                                   zorder=5, alpha=0.6)

                ax.set_ylabel(feat_label, fontsize=7, color='white')
                ax.tick_params(colors='white', labelsize=6)
                for sp in ax.spines.values(): sp.set_edgecolor('#333')
                ax.legend(fontsize=7, loc="upper right",
                          facecolor='#1a1a2e', labelcolor='white',
                          edgecolor='#333', ncol=2)

            axes_ms[-1].set_xlabel("Time (s)", fontsize=8, color='white')
            plt.suptitle(
                f"Group Mean ± Spread — {len(names)} Rehearsals  "
                f"(dots = outlier moments >1.5σ from group)",
                fontsize=10, fontweight="bold", color='white')
            plt.tight_layout()
            st.pyplot(fig_ms, use_container_width=True); plt.close()

            # Outlier summary — which recording deviated most overall
            st.markdown("**Outlier summary:**")
            for feat, feat_label in OVERLAY_FEATS:
                arrays2 = []
                for R in Rs_aligned:
                    arr = R.get(feat)
                    if arr is None: continue
                    arr_w = cl1(arr, fs, fe)
                    if len(arr_w) > 0: arrays2.append(arr_w)
                if len(arrays2) < 2: continue
                min_len2 = min(len(a) for a in arrays2)
                arrays2 = [a[:min_len2] for a in arrays2]
                mean2 = np.mean(arrays2, axis=0)
                devs = [float(np.mean(np.abs(a - mean2))) for a in arrays2]
                outlier_name = names[int(np.argmax(devs))]
                max_dev = max(devs)
                others  = np.mean(sorted(devs)[:-1]) if len(devs)>1 else 0
                if max_dev > others*1.3:
                    st.markdown(
                        f"- **{feat_label}:** "
                        f"*{outlier_name}* deviated most "
                        f"(mean Δ = {max_dev:.3f}, others avg {others:.3f})")

# ─────────────────────────────────────────────────────────────
# Song Map — two visualizations
# ─────────────────────────────────────────────────────────────
def section_comparison(Ra, Rb, t0, t1, name_a, name_b):
    """
    Compare Ra and Rb in time window t0–t1 on musical dimensions.
    Returns dict of {dimension: (verdict, detail)}.
    verdict: "same" | "slight" | "noticeable"
    All dimensions are neutral — no "better/worse" judgement.
    """
    def safe_mean(arr, fs, fe):
        """Slice array by frame indices and return mean. Handles empty slices."""
        sl = arr[fs:fe] if arr.ndim==1 else arr[:,fs:fe]
        if sl.ndim==1:
            return float(sl.mean()) if len(sl)>0 else float(arr.mean())
        return float(sl.mean()) if sl.size>0 else float(arr.mean())

    fs_s = max(0, int(t0 * SR / HOP))
    fe_s = max(fs_s+1, int(t1 * SR / HOP))
    result = {}

    # Rhythm — beat regularity comparison (scalar, not per-frame)
    if "beat_reg" in Ra and "beat_reg" in Rb:
        ra = float(Ra["beat_reg"]); rb = float(Rb["beat_reg"])
        mx = max(ra, rb, 1e-8)
        diff_pct = abs(ra-rb)/mx*100
        if diff_pct < 8:
            result["🥁 Rhythm"] = ("same",
                f"Both consistently timed ({name_a}={ra:.2f} · {name_b}={rb:.2f})")
        else:
            tighter = name_a if ra < rb else name_b
            sev = "noticeable" if diff_pct>20 else "slight"
            result["🥁 Rhythm"] = (sev,
                f"{tighter} is more consistent ({name_a}={ra:.2f} · {name_b}={rb:.2f})")

    # Harmony — chroma cosine in this section
    if "chroma" in Ra and "chroma" in Rb:
        ca = Ra["chroma"][:,fs_s:fe_s]; cb = Rb["chroma"][:,fs_s:fe_s]
        nc = min(ca.shape[1], cb.shape[1])
        if nc > 2:
            ca_m = ca[:,:nc].mean(1); cb_m = cb[:,:nc].mean(1)
            nca = np.linalg.norm(ca_m); ncb = np.linalg.norm(cb_m)
            if nca > 1e-8 and ncb > 1e-8:
                cos = float(np.dot(ca_m,cb_m)/(nca*ncb))
                sim = int((cos+1)/2*100)
                if sim >= 90:
                    result["🎵 Harmony"] = ("same",
                        f"Nearly identical harmonic content ({sim}% match)")
                elif sim >= 70:
                    result["🎵 Harmony"] = ("slight",
                        f"Similar harmony with some differences ({sim}% match)")
                else:
                    result["🎵 Harmony"] = ("noticeable",
                        f"Diverging harmonic emphasis ({sim}% match)")

    # Harmonic ratio (SMS) — how melodic vs percussive each recording is
    if "sms_ratio" in Ra and "sms_ratio" in Rb:
        ha = safe_mean(Ra["sms_ratio"], fs_s, fe_s)
        hb = safe_mean(Rb["sms_ratio"], fs_s, fe_s)
        diff_h = abs(ha-hb)
        if diff_h < 0.04:
            result["🎼 Melodic balance"] = ("same",
                f"Both equally melodic/rhythmic ({name_a}={ha:.2f} · {name_b}={hb:.2f})")
        else:
            more_mel = name_a if ha > hb else name_b
            sev = "noticeable" if diff_h>0.10 else "slight"
            result["🎼 Melodic balance"] = (sev,
                f"{more_mel} has more sustained/melodic content "
                f"({name_a}={ha:.2f} · {name_b}={hb:.2f})")

    # Volume — RMS difference
    if "rms" in Ra and "rms" in Rb:
        va = safe_mean(Ra["rms"], fs_s, fe_s)
        vb = safe_mean(Rb["rms"], fs_s, fe_s)
        diff_v = abs(va-vb)/(max(va,vb)+1e-10)*100
        if diff_v < 8:
            result["🔊 Volume"] = ("same", "Similar energy in this section")
        else:
            louder = name_a if va > vb else name_b
            sev = "noticeable" if diff_v>25 else "slight"
            result["🔊 Volume"] = (sev,
                f"{louder} is {diff_v:.0f}% louder here")

    # Brightness — spectral centroid
    if "centroid" in Ra and "centroid" in Rb:
        ca_v = safe_mean(Ra["centroid"], fs_s, fe_s)
        cb_v = safe_mean(Rb["centroid"], fs_s, fe_s)
        diff_c = abs(ca_v-cb_v)/(max(ca_v,cb_v)+1e-10)*100
        if diff_c < 5:
            result["🎨 Brightness"] = ("same", f"Same tonal colour ({ca_v:.0f} Hz avg)")
        else:
            brighter = name_a if ca_v > cb_v else name_b
            sev = "noticeable" if diff_c>15 else "slight"
            result["🎨 Brightness"] = (sev,
                f"{brighter} sounds brighter ({name_a}={ca_v:.0f} · {name_b}={cb_v:.0f} Hz)")

    return result

def render_song_map(Ra, Rb, name_a, name_b):
    # Always compute divergence over FULL recording (not just zoom window)
    # so Option 2 line covers the whole song
    from scipy.signal import savgol_filter
    arrays_full = []
    for k in ["centroid","rms","onset","sms_ratio","flatness"]:
        ka = k; kb_w = k + "_w" if k+"_w" in Rb else k
        arr_a = Ra.get(ka); arr_b = Rb.get(kb_w, Rb.get(ka))
        if arr_a is not None and arr_b is not None:
            n = min(len(arr_a),len(arr_b))
            an = (arr_a[:n]-arr_a[:n].mean())/(arr_a[:n].std()+1e-8)
            bn = (arr_b[:n]-arr_b[:n].mean())/(arr_b[:n].std()+1e-8)
            arrays_full.append(np.abs(an-bn))
    if not arrays_full:
        st.info("Enable Spectral Features or Onset to compute the song map.")
        return
    nm = min(len(d) for d in arrays_full)
    div_full = np.mean([d[:nm] for d in arrays_full], axis=0)
    if len(div_full) > 21:
        div_full = savgol_filter(div_full, min(51,len(div_full)//4*2+1), 3)
    div_full = np.clip(div_full, 0, None)
    mx = div_full.max()
    if mx > 0: div_full = div_full / mx
    times_full = np.arange(len(div_full)) * HOP / SR

    tempo_avg = (Ra.get("tempo",120)+Rb.get("tempo",120))/2
    spb = 60/max(tempo_avg,1); spbar = spb*4
    total_dur = max(Ra.get("dur",60), Rb.get("dur",60))

    # Detect sections
    secs = detect_structure(ys[0].astype(np.float32).tobytes(), Ra.get("tempo",120))

    st.markdown(f"**Song Map — {name_a} vs {name_b}**")
    st.caption(f"Full recording: {total_dur:.0f}s · Tempo: {tempo_avg:.0f} BPM avg · "
               f"{len(secs)} sections detected")

    map_tabs = st.tabs(["📋 Section breakdown", "🌊 Stream graph", "📊 Stacked bars"])

    # ── View 1: Per-section framing + traffic light ─────────
    with map_tabs[0]:
        st.caption("Each section is analysed in isolation. "
                   "Differences are flagged neutrally — no judgement of 'better'.")
        if not secs:
            st.info("No sections detected.")
        else:
            VERDICT_ICON = {"same":"🟢","slight":"🟡","noticeable":"🔴"}
            VERDICT_WORD = {"same":"Same","slight":"Slight diff","noticeable":"Notable diff"}

            # Compute once per section — avoid double call
            sec_cmps = {}
            for b0,b1,t0,t1,lbl in secs:
                sec_cmps[(b0,b1,t0,t1,lbl)] = section_comparison(
                    Ra, Rb, t0, t1, name_a, name_b)

            for (b0,b1,t0,t1,lbl), cmp in sec_cmps.items():
                if not cmp: continue
                verdicts = [v for v,_ in cmp.values()]
                if all(v=="same" for v in verdicts):
                    overall_icon="🟢"; overall_word="Very similar"
                elif any(v=="noticeable" for v in verdicts):
                    overall_icon="🔴"; overall_word="Notable differences"
                else:
                    overall_icon="🟡"; overall_word="Minor differences"

                with st.expander(
                    f"{overall_icon} **{lbl}** — bars {b0}–{b1} "
                    f"· {t0:.0f}–{t1:.0f}s · {overall_word}",
                    expanded=(overall_icon=="🔴")
                ):
                    for dim, (verdict, detail) in cmp.items():
                        icon = VERDICT_ICON[verdict]
                        st.markdown(
                            f"{icon} **{dim}:** {VERDICT_WORD[verdict]}  \n"
                            f"<span style='color:#888;font-size:0.8rem'>{detail}</span>",
                            unsafe_allow_html=True)

            # Summary strip at bottom — computed from cached results
            st.markdown("---")
            st.markdown("**Section summary:**")
            summary_cols = st.columns(len(secs))
            for col,((b0,b1,t0,t1,lbl),cmp) in zip(summary_cols,sec_cmps.items()):
                verdicts = [v for v,_ in cmp.values()]
                icon = "🟢" if all(v=="same" for v in verdicts)                        else "🔴" if any(v=="noticeable" for v in verdicts) else "🟡"
                col.markdown(f"**{lbl}**  \n{icon} bars {b0}–{b1}")

    # ── View 2: Stream graph ────────────────────────────────
    with map_tabs[1]:
        st.caption(
            "Stream graph — each coloured band shows how much one feature "
            "contributes to the total difference at each moment. "
            "Wider band = that feature is driving the divergence more.")

        from scipy.signal import savgol_filter as _sgf
        STREAM_FEATS = [
            ("rms",      "Volume",     "#4CAF50"),
            ("centroid", "Brightness", "#2196F3"),
            ("onset",    "Rhythm",     "#FF9800"),
            ("sms_ratio","Harmony",    "#9C27B0"),
        ]
        streams = {}
        for key, label, col in STREAM_FEATS:
            arr_a = Ra.get(key); arr_b = Rb.get(key)
            if arr_a is None or arr_b is None: continue
            n_s = min(len(arr_a), len(arr_b))
            a_n = (arr_a[:n_s]-arr_a[:n_s].mean())/(arr_a[:n_s].std()+1e-8)
            b_n = (arr_b[:n_s]-arr_b[:n_s].mean())/(arr_b[:n_s].std()+1e-8)
            div_s = np.abs(a_n - b_n)
            if len(div_s) > 21:
                div_s = _sgf(div_s, min(51,len(div_s)//4*2+1), 3)
            streams[label] = (np.clip(div_s, 0, None), col)

        if streams:
            min_len   = min(len(v) for v,_ in streams.values())
            times_s   = np.arange(min_len) * HOP / SR
            labels_s  = list(streams.keys())
            vals_s    = np.array([v[:min_len] for v,_ in streams.values()])
            cols_s    = [c for _,c in streams.values()]
            # ── Clamshell envelope — symmetric around zero, -0.5 to +0.5 ──
            # Scale raw divergences so peak total → 0.5.
            # Mirror: stack the same bands both above AND below zero.
            # Result: symmetrical clamshell that opens/closes around the center line.
            total_s    = vals_s.sum(axis=0)
            peak_total = total_s.max() + 1e-8
            scale      = 0.5 / peak_total
            vals_scaled = vals_s * scale   # shape: (n_feats, n_frames), peak sum → 0.5

            fig_sg, ax_sg = plt.subplots(figsize=(14, 4))
            ax_sg.set_facecolor('#0f0f1a'); fig_sg.patch.set_facecolor('#0f0f1a')
            ax_sg.axhline(0, color='white', lw=1.0, alpha=0.5, zorder=2)

            legend_handles = []
            # Stack above zero (positive half)
            cum_pos = np.zeros(min_len)
            for label, val_sc, col in zip(labels_s, vals_scaled, cols_s):
                upper = cum_pos + val_sc
                patch = ax_sg.fill_between(times_s, cum_pos, upper,
                                            color=col, alpha=0.82, label=label)
                legend_handles.append(patch)
                cum_pos = upper
            # Mirror below zero (negative half) — same bands, same order
            cum_neg = np.zeros(min_len)
            for val_sc, col in zip(vals_scaled, cols_s):
                lower = cum_neg - val_sc
                ax_sg.fill_between(times_s, lower, cum_neg,
                                   color=col, alpha=0.82)
                cum_neg = lower

            # Label at moment of peak divergence
            mid_t = int(np.argmax(total_s))
            cum_lbl = np.zeros(min_len)
            for label, val_sc, col in zip(labels_s, vals_scaled, cols_s):
                mid_y = (cum_lbl[mid_t] + cum_lbl[mid_t] + val_sc[mid_t]) / 2
                if val_sc[mid_t] > 0.02:
                    ax_sg.text(times_s[mid_t], mid_y, label,
                               fontsize=8, color='white', ha='center', va='center',
                               fontweight='bold')
                cum_lbl += val_sc

            for b0,b1,t0_s,t1_s,lbl in secs:
                ax_sg.axvline(t0_s, color='white', lw=0.6, alpha=0.4, ls='--')
                ax_sg.text(t0_s+0.3, 0.98, lbl, fontsize=6, color='white',
                           va='top', transform=ax_sg.get_xaxis_transform())

            if zoom_mode != "Full recording":
                ax_sg.axvspan(zoom_start, zoom_end, alpha=0.15, color='yellow', zorder=0)

            ax_sg.set_xlim(0, total_dur)
            ax_sg.set_ylim(-0.55, 0.55)
            ax_sg.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
            ax_sg.set_yticklabels(["-0.5", "-0.25", "0", "+0.25", "+0.5"],
                                   fontsize=7, color='white')
            ax_sg.set_xlabel("Time (s)", fontsize=8, color='white')
            ax_sg.set_ylabel("Divergence", fontsize=8, color='white')
            ax_sg.set_title(f"What drives the difference — {name_a} vs {name_b}",
                            fontweight='bold', fontsize=10, color='white')
            ax_sg.tick_params(colors='white')
            for sp in ax_sg.spines.values(): sp.set_edgecolor('#333')
            # Legend order: reverse so top band appears first in legend
            ax_sg.legend(handles=legend_handles[::-1],
                         labels=labels_s[::-1],
                         fontsize=8, loc='upper right',
                         facecolor='#1a1a2e', labelcolor='white', edgecolor='#333')
            plt.tight_layout()
            st.pyplot(fig_sg, use_container_width=True); plt.close()

            if len(times_full) > 0:
                st.markdown("**Top 3 most different moments:**")
                top3_i = np.argsort(div_full)[::-1][:3]
                for rank, idx in enumerate(top3_i):
                    pt2 = float(times_full[idx]); bar2 = int(pt2/spbar)+1
                    fi = min(int(pt2*SR/HOP), min_len-1)
                    dom_idx = int(np.argmax(vals_scaled[:,fi]))
                    dom = labels_s[dom_idx]
                    st.markdown(
                        f"{rank+1}. **{int(pt2//60)}:{int(pt2%60):02d}** "
                        f"(bar {bar2}) — divergence {div_full[idx]:.2f} "
                        f"· driven by **{dom}**")
                st.caption("Use the zoom window in the sidebar to focus on any of these moments.")
        else:
            st.info("Enable Spectral Features and Onset for the stream graph.")

    # ── View 3: Stacked bar per section ────────────────────
    with map_tabs[2]:
        st.caption(
            "Each bar = one detected section. "
            "Bar is split into coloured segments showing different types of difference. "
            "Taller total bar = more overall divergence. "
            "Segment colours show which dimension drove the difference."
        )
        if not secs:
            st.info("No sections detected.")
        else:
            sec_labels = [f"{lbl}\n{t0:.0f}–{t1:.0f}s" for b0,b1,t0,t1,lbl in secs]
            # Compute per-section per-dimension divergence
            dim_keys   = ["rms","centroid","onset","sms_ratio"]
            dim_labels = ["Volume","Brightness","Rhythm","Harmonic"]
            dim_colors = ["#4CAF50","#2196F3","#FF9800","#9C27B0"]

            sec_dim_vals = []  # list of dicts per section
            for b0,b1,t0,t1,lbl in secs:
                fs_s=max(0,int(t0*SR/HOP)); fe_s=max(fs_s+1,int(t1*SR/HOP))
                row={}
                for dk in dim_keys:
                    arr_a=Ra.get(dk); arr_b=Rb.get(dk)
                    if arr_a is None or arr_b is None:
                        row[dk]=0.0; continue
                    n=min(len(arr_a),len(arr_b))
                    sa=arr_a[fs_s:min(fe_s,n)]; sb=arr_b[fs_s:min(fe_s,n)]
                    if len(sa)==0 or len(sb)==0:
                        row[dk]=0.0; continue
                    # Normalize difference to 0-1 using full-recording range
                    full_range=max(arr_a.max()-arr_a.min(),arr_b.max()-arr_b.min(),1e-8)
                    row[dk]=float(np.abs(sb.mean()-sa.mean())/full_range)
                sec_dim_vals.append(row)

            x=np.arange(len(secs))
            fig_sb, ax_sb = plt.subplots(figsize=(max(8, len(secs)*1.6), 4))
            bottoms=np.zeros(len(secs))
            for dk,dlbl,dcol in zip(dim_keys,dim_labels,dim_colors):
                vals=np.array([row.get(dk,0.0) for row in sec_dim_vals])
                ax_sb.bar(x, vals, bottom=bottoms, label=dlbl, color=dcol, alpha=0.82, width=0.6)
                bottoms+=vals

            ax_sb.set_xticks(x); ax_sb.set_xticklabels(sec_labels, fontsize=8)
            ax_sb.set_ylabel("Divergence (stacked by dimension)", fontsize=9)
            ax_sb.set_title(f"Section divergence — {name_a} vs {name_b}",
                            fontweight="bold", fontsize=10)
            ax_sb.legend(fontsize=8, loc="upper right")
            ax_sb.axhline(0, color="black", lw=0.5)

            # Mark best and worst section
            totals=bottoms
            best_i=int(np.argmin(totals)); worst_i=int(np.argmax(totals))
            ax_sb.annotate("most similar",xy=(best_i,totals[best_i]),
                xytext=(best_i,totals[best_i]+0.05),ha="center",fontsize=7,
                color="green",arrowprops=dict(arrowstyle="-",color="green",lw=0.8))
            ax_sb.annotate("most different",xy=(worst_i,totals[worst_i]),
                xytext=(worst_i,totals[worst_i]+0.05),ha="center",fontsize=7,
                color="red",arrowprops=dict(arrowstyle="-",color="red",lw=0.8))

            plt.tight_layout(); st.pyplot(fig_sb, use_container_width=True); plt.close()

            # Table summary
            st.markdown("**Section breakdown table:**")
            header_cols = st.columns([2]+[1]*len(dim_labels)+[1])
            header_cols[0].markdown("**Section**")
            for i,dl in enumerate(dim_labels): header_cols[i+1].markdown(f"**{dl}**")
            header_cols[-1].markdown("**Total**")
            for (b0,b1,t0,t1,lbl),row in zip(secs,sec_dim_vals):
                total=sum(row.get(dk,0) for dk in dim_keys)
                icon="🟢" if total<0.15 else "🟡" if total<0.35 else "🔴"
                row_cols=st.columns([2]+[1]*len(dim_labels)+[1])
                row_cols[0].markdown(f"**{lbl}** {icon}")
                for i,dk in enumerate(dim_keys):
                    v=row.get(dk,0)
                    row_cols[i+1].markdown(f"{v:.2f}")
                row_cols[-1].markdown(f"**{total:.2f}**")

# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────────────────────
main_col, side_col = st.columns([3,1])

# ── Floating side panel ───────────────────────────────────────
with side_col:
    st.markdown("#### 🎧 Playback")
    for seg, name in zip(segs, names):
        st.caption(f"**{name}**")
        if len(seg) > 0: st.audio(wav_bytes(seg), format='audio/wav')

    # Mini waveforms (first two only if N>2)
    n_ds=300
    fig_w, axes_w = plt.subplots(min(n_recs,3),1,figsize=(3,min(n_recs,3)*1.0),sharex=True)
    if min(n_recs,3)==1: axes_w=[axes_w]
    colors_w=[rec_color(i) for i in range(len(names))]
    for ax,y,name,col in zip(axes_w,ys[:3],names[:3],colors_w):
        ds=y[int(zoom_start*SR):int(zoom_end*SR):max(1,int((zoom_end-zoom_start)*SR/n_ds))]
        tw=np.linspace(zoom_start,zoom_end,len(ds))
        ax.plot(tw,ds,lw=0.5,color=col); ax.set_title(name[:18],fontsize=6); ax.set_yticks([])
    axes_w[-1].set_xlabel("Time (s)",fontsize=6)
    plt.tight_layout(pad=0.2); st.pyplot(fig_w,use_container_width=True); plt.close()

    st.markdown("---")
    for name,R in zip(names,Rs_aligned):
        if "tempo" in R:
            st.markdown(f"**{name}**  \n{R['tempo']} BPM · {R.get('key','—')}")
            if "mode" in R: st.caption(R["mode"])
    st.markdown("---")
    st.caption(f"Window: {zoom_start:.1f}–{zoom_end:.1f}s")
    dtw_status = "✓" if any("_wp" in R for R in Rs_aligned[1:]) else "✗"
    st.caption(f"DTW: {dtw_status}")

# ── Main panel ────────────────────────────────────────────────
with main_col:

    # ── Compute pairwise similarity scores ──────────────────────
    # N-recording matrix (always shown when n_recs > 2)
    if n_recs > 2:
        render_matrix()
        st.markdown("---")

        # Compute all pairwise scores
        pairs = [(i,j) for i in range(n_recs) for j in range(i+1,n_recs)]
        pair_labels = [f"{names[i]} vs {names[j]}" for i,j in pairs]
        pair_scores = [pair_similarity(Rs_aligned[i], Rs_aligned[j]) for i,j in pairs]

        # Find best (most similar) pair
        best_idx = int(np.argmax(pair_scores))
        worst_idx = int(np.argmin(pair_scores))  # most different = likely outlier

        # Show similarity scores transparently
        st.subheader("Pairwise Similarity")
        st.caption("Computed from timbre, harmony, tempo, rhythm, and brightness. "
                   "Higher = more similar. Auto-selects the closest pair for detailed comparison.")

        # Visual score table
        score_cols = st.columns(len(pairs))
        for col, (i,j), score, label in zip(score_cols, pairs, pair_scores, pair_labels):
            is_best  = (i,j)==pairs[best_idx]
            is_worst = (i,j)==pairs[worst_idx]
            tag = " ✅ auto-selected" if is_best else (" ⚠️ most different" if is_worst else "")
            bg = "#1a3a1a" if is_best else ("#3a1a1a" if is_worst else "#1a1a2e")
            col.markdown(
                f"<div style='background:{bg};border-radius:8px;padding:10px;text-align:center'>"
                f"<div style='font-size:0.8rem;color:#aaa'>{label}</div>"
                f"<div style='font-size:2rem;font-weight:bold;color:{'#4CAF50' if is_best else ('#E91E63' if is_worst else '#fff')}'>"
                f"{score*100:.0f}%</div>"
                f"<div style='font-size:0.7rem;color:#888'>{tag}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

        # Feature-level breakdown of scores
        with st.expander("Why these scores? — feature breakdown"):
            breakdown_rows = []
            feat_names = ["Timbre (MFCC)","Harmony (Chroma)","Tempo","Rhythm","Brightness"]
            for (i,j),label in zip(pairs,pair_labels):
                Ra_t, Rb_t = Rs_aligned[i], Rs_aligned[j]
                row = {"Pair": label}
                if "mfcc_means" in Ra_t and "mfcc_means" in Rb_t:
                    ma,mb=Ra_t["mfcc_means"],Rb_t["mfcc_means"]
                    cos=float(np.dot(ma,mb)/(np.linalg.norm(ma)*np.linalg.norm(mb)+1e-10))
                    row["Timbre"]=f"{(cos+1)/2*100:.0f}%"
                if "chroma" in Ra_t and "chroma" in Rb_t:
                    ca,cb=Ra_t["chroma"].mean(1),Rb_t["chroma"].mean(1)
                    cos=float(np.dot(ca,cb)/(np.linalg.norm(ca)*np.linalg.norm(cb)+1e-10))
                    row["Harmony"]=f"{(cos+1)/2*100:.0f}%"
                ta,tb=Ra_t.get("tempo",120),Rb_t.get("tempo",120)
                row["Tempo"]=f"{max(0,1-abs(ta-tb)/50)*100:.0f}%  ({ta:.0f} vs {tb:.0f} BPM)"
                ra2,rb2=Ra_t.get("beat_reg",0),Rb_t.get("beat_reg",0)
                if ra2>0 and rb2>0:
                    row["Rhythm"]=f"{max(0,1-abs(ra2-rb2)/max(ra2,rb2,1e-8))*100:.0f}%"
                breakdown_rows.append(row)
            # Print as markdown table
            if breakdown_rows:
                cols_b=list(breakdown_rows[0].keys())
                header="| "+" | ".join(cols_b)+" |"
                sep="| "+" | ".join(["---"]*len(cols_b))+" |"
                rows_md=[header,sep]
                for row in breakdown_rows:
                    rows_md.append("| "+" | ".join(str(row.get(c,"—")) for c in cols_b)+" |")
                st.markdown("\n".join(rows_md))

        # Identify outlier recording
        outlier_counts = {i:0 for i in range(n_recs)}
        for (i,j),score in zip(pairs,pair_scores):
            if score==min(pair_scores):
                # Both recordings in the worst pair are candidates
                outlier_counts[i]+=1; outlier_counts[j]+=1
        outlier_idx = max(outlier_counts, key=outlier_counts.get)
        best_pi, best_pj = pairs[best_idx]
        outlier_name = names[outlier_idx]
        best_name_a, best_name_b = names[best_pi], names[best_pj]

        st.info(
            f"**Auto-selected:** {best_name_a} vs {best_name_b} "
            f"({pair_scores[best_idx]*100:.0f}% similar) — the closest pair.  \n"
            f"**Outlier:** {outlier_name} differs most from the others. "
            f"It may represent a session that went differently."
        )

        # ── Pair tabs — one tab per pair, no rerun on switch ────
        st.subheader("Detailed Comparison — All Pairs")
        st.caption(
            "Each tab is one pair. Switch freely — no rerun needed. "
            "⭐ = auto-selected (most similar).")
        # Shorten labels when many pairs to avoid tab overflow
        # ≤3 pairs: full names  |  >3 pairs: R1 vs R2 style abbreviations
        n_pairs = len(pairs)
        def pair_label(i, j, score, is_best):
            star = "⭐" if is_best else ""
            sim  = f"{score*100:.0f}%"
            if n_pairs <= 3:
                return f"{star}{names[i]} vs {names[j]} {sim}"
            else:
                # Use short index labels: R1, R2, etc.
                return f"{star}R{i+1}·R{j+1} {sim}"

        tab_labels_pairs = [
            pair_label(i, j, score, idx==best_idx)
            for idx, ((i,j), score) in enumerate(zip(pairs, pair_scores))
        ]

        if n_pairs > 15:
            st.warning(
                f"⚠️ {n_recs} recordings produce {n_pairs} pairs. "
                "Consider using the N-recording overview above to identify "
                "the most relevant pairs first.")

        pair_tabs = st.tabs(tab_labels_pairs)

        # Legend when using short labels
        if n_pairs > 3:
            legend_cols = st.columns(min(n_recs, 6))
            for col, (k, nm) in zip(legend_cols, enumerate(names)):
                col.caption(f"**R{k+1}** = {nm}")

    else:
        # 2 recordings — no pair selection needed
        pair_tabs  = None
        pairs      = [(0, 1)]
        best_idx   = 0

    # ── Render comparison for each pair ─────────────────────
    def _render_pair(pi, pj):
        """Render the full comparison for one pair inside the current context."""
        Ra_sel, Rb_sel = Rs_aligned[pi], Rs_aligned[pj]
        name_a_sel, name_b_sel = names[pi], names[pj]
        summary_insights = generate_summary(Ra_sel, Rb_sel, name_a_sel, name_b_sel)
        summary_insights = generate_summary(Ra_sel, Rb_sel, name_a_sel, name_b_sel)

        # ── TAB DEFINITIONS — single unified set ────────────────
        # ── Tab order: musical first, technical second, advanced last ──
        ALL_TABS = ["Overview", "Song Map", "Alignment"]
        if do_chroma:    ALL_TABS.append("Harmony")
        if do_onset:     ALL_TABS.append("Rhythm")
        if do_mfcc:      ALL_TABS.append("Timbre")
        if do_spectral:  ALL_TABS.append("Dynamics")
        # Spectrograms group
        has_specs = do_mel or do_cqt
        if has_specs:    ALL_TABS.append("Spectrograms")
        # Score/MIDI alignment
        if do_chroma:    ALL_TABS.append("Score Alignment")
        # Advanced optional
        adv = [k for k,v in [("SMS/HPSS",do_sms),("Reverb",do_reverb),
                              ("STFT",do_stft),("CWT",do_cwt and HAS_PYWT),
                              ("Gammatone",do_gammatone),("Tonnetz",do_tonnetz)] if v]
        if adv:          ALL_TABS.append("Advanced")

        tabs = st.tabs(ALL_TABS)
        tab_map = {n:t for n,t in zip(ALL_TABS, tabs)}

        # Which DTW strategies are available for this pair?
        wp_chroma = Rb_sel.get("_wp_chroma")
        wp_onset  = Rb_sel.get("_wp_onset")
        wp_combo  = Rb_sel.get("_wp_combo")
        has_dtw   = any(w is not None for w in [wp_chroma, wp_onset, wp_combo])

        # ── Helper: insight card shown at top of every tab ────────
        def insight_card(icon, headline, detail):
            """Render a plain-English insight card before the technical plot."""
            st.markdown(
                f"<div style='background:#1a1a2e;border-left:4px solid #4a9eff;"
                f"border-radius:6px;padding:10px 14px;margin-bottom:12px'>"
                f"<span style='font-size:1.2rem'>{icon}</span> "
                f"<strong>{headline}</strong><br>"
                f"<span style='color:#aaa;font-size:0.82rem'>{detail}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

        # ─────────────────────────────────────────────────────────────
        # TAB CONTENT
        # ─────────────────────────────────────────────────────────────

        # Helper: plot warping path and offset curve for one DTW strategy
        def dtw_panel(ax_path, ax_offset, wp, label, color, dur_a, dur_b):
            offset = wp[:,0].astype(float) - wp[:,1].astype(float)
            step   = max(1, len(wp)//1000); wp_ds = wp[::step]
            off_s  = offset * HOP / SR
            step2  = max(1, len(off_s)//2000); t_ds = wp[::step2,0]*HOP/SR; off_ds = off_s[::step2]

            ax_path.plot(wp_ds[:,0]*HOP/SR, wp_ds[:,1]*HOP/SR,
                         lw=1.0, color=color, label=label)
            ax_path.set_xlabel(f"Time — {name_a_sel} (s)", fontsize=7)
            ax_path.set_ylabel(f"Time — {name_b_sel} (s)", fontsize=7)

            ax_offset.fill_between(t_ds, off_ds, where=off_ds>=0,
                                   color=rec_color(pj), alpha=0.5)
            ax_offset.fill_between(t_ds, off_ds, where=off_ds<0,
                                   color=rec_color(pi), alpha=0.5)
            ax_offset.axhline(0, color='black', lw=0.8)
            off_max = max(abs(off_ds.max()), abs(off_ds.min()), 1e-8)
            ax_offset.set_ylim(-off_max*1.15, off_max*1.15)
            ax_offset.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda v,_: f"{v:+.2f}s"))
            avg = float(np.mean(np.abs(off_ds)))
            ax_offset.set_title(f"{label} — mean offset ±{avg:.2f}s",
                                fontsize=8, color=color)

        def before_after_panel(feat_key, feat_label, ylabel, is_2d=False):
            """Show before (raw) and after (aligned) comparison for one feature."""
            arr_a   = Ra_sel.get(feat_key)
            arr_b_raw = Rs[pj].get(feat_key)  # raw unaligned B
            arr_b_aln = Rb_sel.get(feat_key)  # aligned B
            if arr_a is None or arr_b_raw is None: return

            if is_2d:
                ma = cl2(arr_a, fs, fe)
                mb_raw = cl2(arr_b_raw, fs, fe)
                mb_aln = cl2(arr_b_aln, fs, fe) if arr_b_aln is not None else mb_raw
                nc = min(ma.shape[1], mb_raw.shape[1], mb_aln.shape[1])
                import matplotlib.colors as mcol
                ca = mcol.to_rgb(rec_color(pi)); cb = mcol.to_rgb(rec_color(pj))
                dcmap = mcol.LinearSegmentedColormap.from_list('d',[ca,(1,1,1),cb],N=256)
                ext = [zoom_start, zoom_end, 0, ma.shape[0]]
                fig, axes = plt.subplots(2, 3, figsize=(16, 5))
                for row, (mb, title_sfx) in enumerate([
                    (mb_raw[:,:nc], "Before alignment"),
                    (mb_aln[:,:nc], "After alignment")
                ]):
                    diff = mb - ma[:,:nc]
                    vmax = float(np.percentile(np.abs(diff),95)) or 1.0
                    axes[row,0].imshow(ma[:,:nc], aspect='auto', origin='lower',
                                       cmap='magma', extent=ext)
                    axes[row,0].set_title(f"{name_a_sel}", fontsize=8,
                                           color=rec_color(pi), fontweight='bold')
                    for sp in axes[row,0].spines.values():
                        sp.set_edgecolor(rec_color(pi)); sp.set_linewidth(2)
                    axes[row,1].imshow(mb, aspect='auto', origin='lower',
                                       cmap='magma', extent=ext)
                    axes[row,1].set_title(f"{name_b_sel} — {title_sfx}", fontsize=8,
                                           color=rec_color(pj), fontweight='bold')
                    for sp in axes[row,1].spines.values():
                        sp.set_edgecolor(rec_color(pj)); sp.set_linewidth(2)
                    im3 = axes[row,2].imshow(diff, aspect='auto', origin='lower',
                                              cmap=dcmap, extent=ext, vmin=-vmax, vmax=vmax)
                    axes[row,2].set_title(f"Difference — {title_sfx}", fontsize=8)
                    plt.colorbar(im3, ax=axes[row,2], format="%+.0f")
                plt.suptitle(feat_label, fontsize=10, fontweight='bold')
                plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
            else:
                da   = cl1(arr_a, fs, fe)
                db_r = cl1(arr_b_raw, fs, fe)
                db_a = cl1(arr_b_aln, fs, fe) if arr_b_aln is not None else db_r
                n    = min(len(da), len(db_r), len(db_a))
                t_ax = np.linspace(zoom_start, zoom_end, n)
                fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
                for ax, db, title_sfx in [
                    (axes[0], db_r[:n], "Before alignment"),
                    (axes[1], db_a[:n], "After alignment"),
                ]:
                    ax.plot(t_ax, da[:n], lw=0.8, color=rec_color(pi), label=name_a_sel)
                    ax.plot(t_ax, db,     lw=0.8, color=rec_color(pj),
                            label=f"{name_b_sel} ({title_sfx})", alpha=0.8)
                    diff = db - da[:n]
                    ax2  = ax.twinx()
                    ax2.fill_between(t_ax, diff, where=diff>=0,
                                     color=rec_color(pj), alpha=0.25)
                    ax2.fill_between(t_ax, diff, where=diff<0,
                                     color=rec_color(pi), alpha=0.25)
                    ax2.axhline(0, color='black', lw=0.5)
                    dm = max(abs(diff.max()), abs(diff.min()), 1e-8)
                    ax2.set_ylim(-dm*1.2, dm*1.2)
                    ax2.set_ylabel(f"Δ {ylabel}", fontsize=7)
                    ax.set_ylabel(ylabel, fontsize=7)
                    ax.legend(fontsize=7); ax.set_title(title_sfx, fontsize=8)
                axes[1].set_xlabel("Time (s)", fontsize=8)
                plt.suptitle(feat_label, fontsize=10, fontweight='bold')
                plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        # ── Overview ─────────────────────────────────────────────────
        with tab_map["Overview"]:
            st.markdown(f"### {name_a_sel} vs {name_b_sel}")
            st.caption(f"Window: {zoom_start:.1f}s → {zoom_end:.1f}s  |  "
                       f"A: {durs[pi]:.1f}s  B: {durs[pj]:.1f}s  |  "
                       f"DTW: {'chroma+onset+RMS' if has_dtw else 'off'}")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric(f"Tempo {name_a_sel}", f"{Ra_sel.get('tempo',120)} BPM")
            c2.metric(f"Tempo {name_b_sel}", f"{Rb_sel.get('tempo',120)} BPM",
                      f"{Rb_sel.get('tempo',120)-Ra_sel.get('tempo',120):+.1f}")
            if "key" in Ra_sel:
                c3.metric(f"Key {name_a_sel}", Ra_sel["key"])
                c4.metric(f"Key {name_b_sel}", Rb_sel.get("key","—"))
            if "mfcc_means" in Ra_sel and "mfcc_means" in Rb_sel:
                mc=float(np.dot(Ra_sel["mfcc_means"],Rb_sel["mfcc_means"])/
                         (np.linalg.norm(Ra_sel["mfcc_means"])*np.linalg.norm(Rb_sel["mfcc_means"])+1e-10))
                st.metric("Timbre Similarity", f"{(mc+1)/2*100:.1f}%")
            for icon,headline,detail in summary_insights:
                insight_card(icon, headline, detail)
            # Waveform diff
            n_wv=min(len(ys[pi]),len(ys[pj])); step_wv=max(1,n_wv//2000)
            wa=ys[pi][:n_wv:step_wv]; wb=ys[pj][:n_wv:step_wv]
            t_wv=np.linspace(0,min(durs[pi],durs[pj]),len(wa))
            diff_w=wb-wa; mag=np.abs(diff_w)
            norm_mag=np.clip(mag/(mag.max()+1e-8),0,1); mask=mag>diff_threshold
            fig_wv,axes_wv=plt.subplots(3,1,figsize=(14,6),sharex=True)
            axes_wv[0].plot(t_wv,wa,lw=0.4,color=rec_color(pi))
            axes_wv[0].set_title(name_a_sel,fontsize=8,color=rec_color(pi),fontweight='bold')
            axes_wv[1].plot(t_wv,wb,lw=0.4,color=rec_color(pj))
            axes_wv[1].set_title(name_b_sel,fontsize=8,color=rec_color(pj),fontweight='bold')
            for i in range(len(t_wv)-1):
                if not mask[i]: continue
                col=(1,0,0,float(norm_mag[i])) if diff_w[i]>0 else (0,0,1,float(norm_mag[i]))
                axes_wv[2].fill_between(t_wv[i:i+2],[diff_w[i],diff_w[i+1]],color=col)
            axes_wv[2].axhline(0,color='black',lw=0.8)
            dw_max=max(abs(diff_w.max()),abs(diff_w.min()),1e-8)
            axes_wv[2].set_ylim(-dw_max*1.15,dw_max*1.15)
            axes_wv[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_:f"{v:+.3g}"))
            axes_wv[2].set_title(f"Δ = {name_b_sel} minus {name_a_sel}",fontsize=8)
            axes_wv[2].set_xlabel("Time (s)",fontsize=7)
            plt.tight_layout(); st.pyplot(fig_wv,use_container_width=True); plt.close()

        # ── Song Map ─────────────────────────────────────────────────
        with tab_map["Song Map"]:
            render_song_map(Ra_sel, Rb_sel, name_a_sel, name_b_sel)

        # ── Alignment tab — three strategies + before/after ──────────
        with tab_map["Alignment"]:
            if not has_dtw:
                st.info("Enable DTW Alignment in the sidebar to see alignment analysis.")
            else:
                st.markdown("### DTW Alignment — Three Strategies")
                st.caption(
                    "Each strategy aligns the recordings using a different musical signal. "
                    "**Chroma**: matches harmonic/melodic content. "
                    "**Onset**: matches rhythmic attack positions. "
                    "**RMS**: matches overall loudness shape. "
                    "For same instrument/same piece recordings, all three should agree closely."
                )
                rms_off_s = Rb_sel.get("_rms_offset_seconds", 0)
                if rms_off_s:
                    st.info(f"Coarse RMS offset detected: **{rms_off_s:+.2f}s** "
                            f"({'B starts later' if rms_off_s>0 else 'B starts earlier'}). "
                            f"Applied before all DTW strategies.")

                # ── Strategy comparison plot ──────────────────────────
                st.subheader("Warping paths — all three strategies")
                fig_cmp, axes_cmp = plt.subplots(1, 3, figsize=(16, 4))
                strategy_info = [
                    (wp_chroma, "Chroma DTW",        rec_color(0)),
                    (wp_onset,  "Onset DTW",         rec_color(1)),
                    (wp_combo,  "Chroma+Onset Combo", rec_color(2)),
                ]
                for ax, (wp, label, col) in zip(axes_cmp, strategy_info):
                    if wp is None:
                        ax.text(0.5,0.5,"Not available",ha='center',va='center',
                                transform=ax.transAxes,fontsize=9,color='gray')
                        ax.set_title(label,fontsize=9)
                        continue
                    step=max(1,len(wp)//1000); wp_ds=wp[::step]
                    ax.plot(wp_ds[:,0]*HOP/SR, wp_ds[:,1]*HOP/SR,
                            lw=1.0, color=col)
                    # Perfect alignment reference
                    max_t=max(durs[pi],durs[pj])
                    ax.plot([0,max_t],[0,max_t],'--',color='gray',lw=0.7,alpha=0.6,
                            label='perfect')
                    avg_off=float(np.mean(np.abs(wp[:,0]-wp[:,1]))*HOP/SR)
                    ax.set_title(f"{label} — mean offset ±{avg_off:.2f}s",
                                 fontsize=8,color=col,fontweight='bold')
                    ax.set_xlabel(f"{name_a_sel} (s)",fontsize=7)
                    ax.set_ylabel(f"{name_b_sel} (s)",fontsize=7)
                    ax.legend(fontsize=6)
                plt.suptitle("DTW Warping Paths — diagonal = perfect alignment",
                             fontsize=10, fontweight='bold')
                plt.tight_layout(); st.pyplot(fig_cmp,use_container_width=True); plt.close()

                # ── Offset curves ─────────────────────────────────────
                st.subheader("Temporal offset over time")
                fig_off, ax_off = plt.subplots(figsize=(14,3))
                for wp, label, col in strategy_info:
                    if wp is None: continue
                    offset = (wp[:,0]-wp[:,1]).astype(float)*HOP/SR
                    step=max(1,len(offset)//2000)
                    t_p=wp[::step,0]*HOP/SR; off_ds=offset[::step]
                    ax_off.plot(t_p, off_ds, lw=1.0, color=col,
                                label=label, alpha=0.85)
                ax_off.axhline(0,color='black',lw=0.8)
                ax_off.set_xlabel("Time (s)",fontsize=8)
                ax_off.set_ylabel("Offset (s)",fontsize=8)
                ax_off.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_:f"{v:+.2f}s"))
                ax_off.legend(fontsize=8)
                ax_off.set_title("How much each strategy shifts B to match A at each moment",
                                 fontsize=9,fontweight='bold')
                ax_off.fill_between([0,max(durs[pi],durs[pj])],[0],[0],alpha=0)
                plt.tight_layout(); st.pyplot(fig_off,use_container_width=True); plt.close()
                st.caption("All three curves close together = recordings are well-aligned by any measure. "
                           "Large divergence between curves = complex timing relationship.")

                # ── Before / After for each strategy ─────────────────
                st.subheader("Before vs After alignment — feature comparison")
                active_label = {"chroma":"Chroma DTW","onset":"Onset DTW",
                                "combo":"Chroma+Onset Combo"}.get(dtw_strategy,"Combo")
                st.caption(f"Active strategy for feature alignment: **{active_label}**. "
                           "All three paths computed and shown for comparison above.")
                strat_tab_names = [s[1] for s in strategy_info if s[0] is not None]
                if strat_tab_names:
                    strat_tabs = st.tabs(strat_tab_names)
                    for stab, (wp, label, col) in zip(strat_tabs, strategy_info):
                        if wp is None: continue
                        with stab:
                            st.caption(f"**{label}** — comparing raw B vs B warped "
                                       f"using {label.lower()}")
                            # Show RMS before/after for this strategy
                            if "rms" in Ra_sel and "rms" in Rs[pj]:
                                n_a_warp = Ra_sel["rms"].shape[0] if "chroma" not in Ra_sel                                        else Ra_sel["chroma"].shape[1]
                                rms_b_warp = apply_warp(
                                    Rs[pj]["rms"].astype(np.float32), wp, n_a_warp)
                                rms_a  = Ra_sel["rms"]
                                rms_b_raw = Rs[pj]["rms"]
                                n = min(len(rms_a), len(rms_b_raw), len(rms_b_warp))
                                t_ax = np.linspace(zoom_start, zoom_end,
                                                   min(n, fe-fs) if fe>fs else n)
                                n_w = len(t_ax)
                                col_a = rec_color(pi)   # recording A — always its identity color
                                col_b = rec_color(pj)   # recording B — always its identity color
                                fig_ba, axes_ba = plt.subplots(2,1,figsize=(14,5),sharex=True)
                                for ax, db, sfx in [
                                    (axes_ba[0], rms_b_raw[:n_w], "Before alignment"),
                                    (axes_ba[1], rms_b_warp[:n_w], "After alignment"),
                                ]:
                                    ax.plot(t_ax, rms_a[:n_w], lw=0.9,
                                            color=col_a, label=name_a_sel)
                                    ax.plot(t_ax, db, lw=0.9,
                                            color=col_b, label=f"{name_b_sel} ({sfx})",
                                            alpha=0.8)
                                    # Shade difference
                                    n_plt = min(len(t_ax), len(rms_a), len(db))
                                    diff_rms = db[:n_plt] - rms_a[:n_plt]
                                    ax.fill_between(t_ax[:n_plt], rms_a[:n_plt], db[:n_plt],
                                        where=diff_rms>=0, alpha=0.18, color=col_b)
                                    ax.fill_between(t_ax[:n_plt], rms_a[:n_plt], db[:n_plt],
                                        where=diff_rms<0,  alpha=0.18, color=col_a)
                                    ax.set_ylabel("RMS Energy",fontsize=7)
                                    ax.legend(fontsize=7)
                                    ax.set_title(sfx, fontsize=8,
                                                 color='#aaa' if sfx=="Before alignment" else '#4CAF50')
                                axes_ba[1].set_xlabel("Time (s)",fontsize=8)
                                plt.suptitle(f"RMS Energy — {label}  "
                                             f"({name_a_sel} = {col_a[:7]}  "
                                             f"{name_b_sel} = {col_b[:7]})",
                                             fontsize=10, fontweight='bold')
                                plt.tight_layout()
                                st.pyplot(fig_ba,use_container_width=True); plt.close()

                            if "chroma" in Ra_sel and "chroma" in Rs[pj]:
                                n_ac = Ra_sel["chroma"].shape[1]
                                ch_b_warp = apply_warp(
                                    Rs[pj]["chroma"].astype(np.float32), wp, n_ac)
                                ca = cl2(Ra_sel["chroma"],fs,fe)
                                cb_raw = cl2(Rs[pj]["chroma"],fs,fe)
                                cb_aln = cl2(ch_b_warp,fs,fe)
                                nc = min(ca.shape[1],cb_raw.shape[1],cb_aln.shape[1])
                                import matplotlib.colors as mcol
                                col_a_ch = rec_color(pi)   # A = recording A color
                                col_b_ch = rec_color(pj)   # B = recording B color
                                ca_c=mcol.to_rgb(col_a_ch)
                                cb_c=mcol.to_rgb(col_b_ch)
                                dcmap=mcol.LinearSegmentedColormap.from_list(
                                    'd',[ca_c,(1,1,1),cb_c],N=256)
                                fig_ch,axes_ch=plt.subplots(2,3,figsize=(16,5))
                                for row,(cb,sfx) in enumerate(
                                        [(cb_raw[:,:nc],"Before"),(cb_aln[:,:nc],"After")]):
                                    diff=cb-ca[:,:nc]
                                    vmax=float(np.percentile(np.abs(diff),95)) or 1.0
                                    ext=[zoom_start,zoom_end,0,12]
                                    for col_i,(mat,nm,c_) in enumerate([
                                        (ca[:,:nc],name_a_sel,col_a_ch),
                                        (cb,f"{name_b_sel} ({sfx})",col_b_ch),
                                        (diff,"Difference",None)
                                    ]):
                                        if col_i<2:
                                            axes_ch[row,col_i].imshow(mat,aspect='auto',
                                                origin='lower',cmap='YlOrRd',extent=ext)
                                            axes_ch[row,col_i].set_title(nm,fontsize=8,
                                                color=c_,fontweight='bold')
                                            for sp in axes_ch[row,col_i].spines.values():
                                                sp.set_edgecolor(c_); sp.set_linewidth(2)
                                        else:
                                            im=axes_ch[row,col_i].imshow(diff,aspect='auto',
                                                origin='lower',cmap=dcmap,extent=ext,
                                                vmin=-vmax,vmax=vmax)
                                            axes_ch[row,col_i].set_title("Difference",fontsize=8)
                                        axes_ch[row,col_i].set_yticks(range(12))
                                        axes_ch[row,col_i].set_yticklabels(MIDI_NAMES,fontsize=6)
                                plt.suptitle(f"Chroma — {label}",fontsize=10,fontweight='bold')
                                plt.tight_layout()
                                st.pyplot(fig_ch,use_container_width=True); plt.close()

        # ── Harmony ───────────────────────────────────────────────────
        if do_chroma and "chroma" in Ra_sel and "Harmony" in tab_map:
            with tab_map["Harmony"]:
                if "key" in Ra_sel:
                    same_key = Ra_sel.get("key")==Rb_sel.get("key")
                    insight_card("🎹",
                        f"Key & mode: {'both in '+Ra_sel['key'] if same_key else Ra_sel['key']+' vs '+Rb_sel.get('key','—')}",
                        MODE_CHARACTER.get(Ra_sel.get("mode_name",""),""))
                c1,c2=st.columns(2)
                for col,R_src,nm,idx in [(c1,Ra_sel,name_a_sel,pi),(c2,Rb_sel,name_b_sel,pj)]:
                    with col:
                        st.markdown(f"**{nm}**")
                        st.markdown(f"Key: {R_src.get('key','—')} · Mode: {R_src.get('mode','—')}")
                        st.caption(MODE_CHARACTER.get(R_src.get("mode_name",""),""))
                        ch_w=cl2(R_src["chroma"],fs,fe)
                        cm=ch_w.mean(1) if ch_w.shape[1]>0 else R_src["chroma"].mean(1)
                        top3=sorted(range(12),key=lambda i:-cm[i])[:3]
                        st.markdown(f"Dominant notes: **{', '.join(MIDI_NAMES[i] for i in top3)}**")
                ca_w=cl2(Ra_sel["chroma"],fs,fe); cb_w=cl2(Rb_sel["chroma"],fs,fe)
                nc=min(ca_w.shape[1],cb_w.shape[1])
                if nc>0:
                    ca_m=ca_w[:,:nc].mean(1); cb_m=cb_w[:,:nc].mean(1)
                    cc=float(np.dot(ca_m,cb_m)/(np.linalg.norm(ca_m)*np.linalg.norm(cb_m)+1e-10))
                    st.markdown(f"**Harmonic similarity: {int((cc+1)/2*100)}%**")
                    st.progress((cc+1)/2)
                    x=np.arange(12); dc=cb_m-ca_m
                    diff_formula(name_a_sel,name_b_sel,rec_color(pi),rec_color(pj))
                    fig_cb,axes_cb=plt.subplots(2,1,figsize=(12,5))
                    axes_cb[0].bar(x-0.2,ca_m,0.35,label=name_a_sel,color=rec_color(pi),alpha=0.85)
                    axes_cb[0].bar(x+0.2,cb_m,0.35,label=name_b_sel,color=rec_color(pj),alpha=0.85)
                    axes_cb[0].set_xticks(x); axes_cb[0].set_xticklabels(MIDI_NAMES)
                    axes_cb[0].set_title("Pitch class energy",fontweight='bold'); axes_cb[0].legend()
                    axes_cb[1].bar(x,dc,color=[rec_color(pj) if d>0 else rec_color(pi) for d in dc],alpha=0.8)
                    axes_cb[1].axhline(0,color='black',lw=1.2,zorder=5)
                    dc_max=max(abs(dc.max()),abs(dc.min()),1e-8)
                    axes_cb[1].set_ylim(-dc_max*1.2,dc_max*1.2)
                    axes_cb[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_:f"{v:+.2f}"))
                    axes_cb[1].set_xticks(x); axes_cb[1].set_xticklabels(MIDI_NAMES)
                    axes_cb[1].set_title(f"Δ = {name_b_sel} minus {name_a_sel}",fontsize=9)
                    plt.tight_layout(); st.pyplot(fig_cb,use_container_width=True); plt.close()

        # ── Rhythm ────────────────────────────────────────────────────
        if do_onset and "onset" in Ra_sel and "Rhythm" in tab_map:
            with tab_map["Rhythm"]:
                ta_v=Ra_sel.get("tempo",120); tb_v=Rb_sel.get("tempo",120)
                ra=Ra_sel.get("beat_reg",0); rb=Rb_sel.get("beat_reg",0)
                td=tb_v-ta_v
                if abs(td)<1: insight_card("🎯","Tempo: identical",f"Both at {ta_v:.1f} BPM.")
                elif abs(td)<3: insight_card("🎯",f"Tempo: nearly same (Δ{td:+.1f} BPM)",f"A:{ta_v:.1f} B:{tb_v:.1f}")
                else:
                    faster=name_b_sel if td>0 else name_a_sel
                    insight_card("⚡",f"Tempo: {faster} is {abs(td):.1f} BPM faster",
                                 f"A:{ta_v:.1f} · B:{tb_v:.1f} BPM")
                c1,c2=st.columns(2)
                with c1:
                    st.metric(f"Tempo {name_a_sel}",f"{ta_v} BPM")
                    feel_a="Locked in 🟢" if ra<1.5 else "Fairly steady 🟡" if ra<3 else "Loose 🔴"
                    st.markdown(f"**Feel:** {feel_a}  \nBeat regularity: `{ra:.3f}`")
                with c2:
                    st.metric(f"Tempo {name_b_sel}",f"{tb_v} BPM",delta=f"{td:+.1f} BPM")
                    feel_b="Locked in 🟢" if rb<1.5 else "Fairly steady 🟡" if rb<3 else "Loose 🔴"
                    st.markdown(f"**Feel:** {feel_b}  \nBeat regularity: `{rb:.3f}`")
                if ra>0 and rb>0:
                    max_reg=max(ra,rb,0.001)
                    ta_t=int((1-ra/max_reg)*10); tb_t=int((1-rb/max_reg)*10)
                    tight_a='█'*ta_t+'░'*(10-ta_t); tight_b='█'*tb_t+'░'*(10-tb_t)
                    st.markdown(f"**Tightness:** {name_a_sel}: `{tight_a}` {ta_t}/10 · {name_b_sel}: `{tight_b}` {tb_t}/10")
                oa=cl1(Ra_sel["onset"],fs,fe); ob=cl1(Rb_sel["onset"],fs,fe)
                n_on=min(len(oa),len(ob)); oa,ob=oa[:n_on],ob[:n_on]
                t_on=np.linspace(zoom_start,zoom_end,n_on)
                diff_formula(name_a_sel,name_b_sel,rec_color(pi),rec_color(pj))
                fig_on,axes_on=plt.subplots(3,1,figsize=(14,6),sharex=True)
                for ax,onset,R_src,nm,col in [
                    (axes_on[0],oa,Ra_sel,name_a_sel,rec_color(pi)),
                    (axes_on[1],ob,Rb_sel,name_b_sel,rec_color(pj))
                ]:
                    ax.fill_between(t_on,onset,alpha=0.3,color=col)
                    ax.set_title(f"{nm} — {R_src.get('tempo',120)} BPM",fontsize=8,color=col,fontweight='bold')
                    ax.set_ylabel("Onset",fontsize=7); ax.set_yticks([])
                    if "beats" in R_src:
                        bt=R_src["beats"]*HOP/SR; bt=bt[(bt>=zoom_start)&(bt<=zoom_end)]
                        if len(bt): ax.scatter(bt,np.interp(bt,t_on,onset),color=col,s=30,zorder=5)
                diff_on=ob-oa
                axes_on[2].fill_between(t_on,diff_on,where=diff_on>=0,color=rec_color(pj),alpha=0.55)
                axes_on[2].fill_between(t_on,diff_on,where=diff_on<0,color=rec_color(pi),alpha=0.55)
                axes_on[2].axhline(0,color='black',lw=1.2,zorder=5)
                don_max=max(abs(diff_on.max()),abs(diff_on.min()),1e-8)
                axes_on[2].set_ylim(-don_max*1.15,don_max*1.15)
                axes_on[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_:f"{v:+.2g}"))
                axes_on[2].set_xlabel("Time (s)",fontsize=7)
                axes_on[2].set_title(f"Δ = {name_b_sel} minus {name_a_sel}",fontsize=8)
                plt.tight_layout(); st.pyplot(fig_on,use_container_width=True); plt.close()

        # ── Timbre ────────────────────────────────────────────────────
        if do_mfcc and "mfcc_means" in Ra_sel and "Timbre" in tab_map:
            with tab_map["Timbre"]:
                ma_raw=Ra_sel["mfcc_means"]; mb_raw=Rb_sel.get("mfcc_means",ma_raw)
                mc=float(np.dot(ma_raw,mb_raw)/(np.linalg.norm(ma_raw)*np.linalg.norm(mb_raw)+1e-10))
                insight_card("🎨",f"Timbre similarity: {(mc+1)/2*100:.1f}%",
                    "MFCCs capture the timbral fingerprint. Coeff 1–4 = broad spectral shape. 5–12 = fine texture.")
                ca_m_col=rec_color(pi); cb_m_col=rec_color(pj)
                mfcc_view=st.radio("View",["Raw values","Normalized per coefficient"],
                                    horizontal=True,key=f"mfcc_view_{pi}_{pj}")
                if mfcc_view=="Raw values":
                    ma_m=ma_raw[1:]; mb_m=mb_raw[1:]; ylabel="Mean MFCC value"
                    caption="Raw values. Coefficients 1–4 are large (broad spectral envelope). 5–12 are small — expected."
                else:
                    ma_m=Ra_sel.get("mfcc_means_z",ma_raw)[1:]
                    mb_m=Rb_sel.get("mfcc_means_z",mb_raw)[1:]; ylabel="Normalized [-1,+1]"
                    caption="Each coefficient scaled to its own [-1,+1] range. All 12 now visually comparable."
                x=np.arange(1,len(ma_m)+1)
                fig_m,axes_m=plt.subplots(2,1,figsize=(14,5))
                axes_m[0].bar(x-0.2,ma_m,0.35,label=name_a_sel,color=ca_m_col,alpha=0.85)
                axes_m[0].bar(x+0.2,mb_m,0.35,label=name_b_sel,color=cb_m_col,alpha=0.85)
                axes_m[0].axhline(0,color='black',lw=1.2,zorder=5)
                axes_m[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_:f"{v:+.2f}"))
                axes_m[0].set_xticks(x); axes_m[0].set_xlabel("Coefficient index",fontsize=8)
                axes_m[0].set_title(f"MFCC Profile — {mfcc_view}",fontweight='bold',fontsize=9)
                axes_m[0].legend()
                dm=mb_m-ma_m
                axes_m[1].bar(x,dm,color=[cb_m_col if d>0 else ca_m_col for d in dm],alpha=0.85)
                axes_m[1].axhline(0,color='black',lw=1.2,zorder=5)
                dm_max=max(abs(dm.max()),abs(dm.min()),1e-8)
                axes_m[1].set_ylim(-dm_max*1.2,dm_max*1.2)
                axes_m[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_:f"{v:+.2f}"))
                axes_m[1].set_xticks(x); axes_m[1].set_xlabel("Coefficient index",fontsize=8)
                n_b_h=sum(1 for d in dm if d>0); n_a_h=sum(1 for d in dm if d<0)
                dominant=name_b_sel if n_b_h>n_a_h else name_a_sel
                top_c=int(np.argmax(np.abs(dm)))+1
                axes_m[1].set_title(f"Δ = {name_b_sel} minus {name_a_sel} — above zero = {name_b_sel} higher",fontsize=8)
                axes_m[1].annotate(f"largest diff (coeff {top_c})",
                    xy=(top_c,dm[top_c-1]),xytext=(min(top_c+1.5,x[-1]),dm[top_c-1]*1.1 if dm[top_c-1]!=0 else dm_max*0.5),
                    fontsize=6,arrowprops=dict(arrowstyle='->',lw=0.8))
                plt.tight_layout(); st.pyplot(fig_m,use_container_width=True); plt.close()
                st.caption(caption)
                st.caption(f"**{dominant}** dominates {max(n_a_h,n_b_h)}/12 coefficients.")

        # ── Dynamics ──────────────────────────────────────────────────
        if do_spectral and "centroid" in Ra_sel and "Dynamics" in tab_map:
            with tab_map["Dynamics"]:
                if "rms" in Ra_sel and "rms" in Rb_sel:
                    rms_a_w=cl1(Ra_sel["rms"],fs,fe); rms_b_w=cl1(Rb_sel["rms"],fs,fe)
                    n_r=min(len(rms_a_w),len(rms_b_w)); rms_a_w,rms_b_w=rms_a_w[:n_r],rms_b_w[:n_r]
                    t_r=np.linspace(zoom_start,zoom_end,n_r)
                    mean_a=float(rms_a_w.mean()); mean_b=float(rms_b_w.mean())
                    loud_pct=(mean_b-mean_a)/(mean_a+1e-10)*100
                    if abs(loud_pct)<5: loud_s="Both at similar volume."
                    elif loud_pct>0: loud_s=f"{name_b_sel} is {abs(loud_pct):.0f}% louder."
                    else: loud_s=f"{name_a_sel} is {abs(loud_pct):.0f}% louder."
                    insight_card("🔊",loud_s,"Volume difference in the selected window.")
                    peak=max(rms_a_w.max(),rms_b_w.max(),1e-8)
                    rms_a_n=rms_a_w/peak; rms_b_n=rms_b_w/peak
                    fig_e,axes_e=plt.subplots(2,1,figsize=(14,4),sharex=True)
                    axes_e[0].fill_between(t_r,rms_a_n,alpha=0.35,color=rec_color(pi),label=name_a_sel)
                    axes_e[0].fill_between(t_r,rms_b_n,alpha=0.35,color=rec_color(pj),label=name_b_sel)
                    axes_e[0].set_ylim(0,1.05); axes_e[0].set_yticks([0,0.5,1.0])
                    axes_e[0].set_yticklabels(["Quiet","","Loud"],fontsize=8)
                    axes_e[0].set_title("Volume over time",fontsize=9,fontweight='bold'); axes_e[0].legend(fontsize=8)
                    diff_e=rms_b_n-rms_a_n
                    axes_e[1].fill_between(t_r,diff_e,where=diff_e>=0,color=rec_color(pj),alpha=0.6,label=f"{name_b_sel} louder")
                    axes_e[1].fill_between(t_r,diff_e,where=diff_e<0,color=rec_color(pi),alpha=0.6,label=f"{name_a_sel} louder")
                    axes_e[1].axhline(0,color='black',lw=1.2,zorder=5)
                    de_max=max(abs(diff_e.max()),abs(diff_e.min()),1e-8)
                    axes_e[1].set_ylim(-de_max*1.15,de_max*1.15)
                    axes_e[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_:f"{v:+.2f}"))
                    axes_e[1].set_xlabel("Time (s)",fontsize=8)
                    axes_e[1].set_title(f"Δ = {name_b_sel} minus {name_a_sel}",fontsize=9)
                    axes_e[1].legend(fontsize=8)
                    plt.tight_layout(); st.pyplot(fig_e,use_container_width=True); plt.close()
                st.markdown("---")
                st.markdown("**Spectral descriptors:**")
                for feat,ylabel,yscale,note in [
                    ("centroid","Hz","linear","Brightness — centre of spectral mass."),
                    ("bandwidth","Hz","linear","Bandwidth — harmonic richness."),
                    ("rolloff","Hz","linear","Rolloff — high-frequency content."),
                    ("flatness","","log","Flatness — 0=tonal, 1=noise-like. Music values are tiny (0.001–0.05); spikes are real events."),
                    ("rms","","linear","RMS energy — loudness."),
                ]:
                    if feat not in Ra_sel or feat not in Rb_sel: continue
                    da=cl1(Ra_sel[feat],fs,fe); db=cl1(Rb_sel[feat],fs,fe)
                    yr=Ra_sel.get(f"{feat}_range")
                    if yr and Rb_sel.get(f"{feat}_range"):
                        yr=(min(yr[0],Rb_sel[f"{feat}_range"][0]),max(yr[1],Rb_sel[f"{feat}_range"][1]))
                    diff_formula(name_a_sel,name_b_sel,rec_color(pi),rec_color(pj))
                    st.pyplot(line_pair(note,da,db,name_a_sel,name_b_sel,
                                        ylabel=ylabel,yrange=yr,yscale=yscale,
                                        col_a=rec_color(pi),col_b=rec_color(pj))); plt.close()

        # ── Spectrograms ──────────────────────────────────────────────
        if has_specs and "Spectrograms" in tab_map:
            with tab_map["Spectrograms"]:
                spec_sub_names=[]
                if do_mel and "mel" in Ra_sel: spec_sub_names.append("Mel")
                if do_cqt and "cqt" in Ra_sel: spec_sub_names.append("CQT")
                if spec_sub_names:
                    spec_sub_tabs=st.tabs(spec_sub_names)
                    spec_sub_map={n:t for n,t in zip(spec_sub_names,spec_sub_tabs)}
                    if "Mel" in spec_sub_map:
                        with spec_sub_map["Mel"]:
                            insight_card("🌊","Mel Spectrogram",
                                "128 Mel bands, x-axis in seconds. Bottom=bass, top=treble. "
                                "Red=B louder, blue=A louder in difference panel.")
                            diff_formula(name_a_sel,name_b_sel,rec_color(pi),rec_color(pj))
                            ma=cl2(Ra_sel["mel"],fs,fe); mb=cl2(Rb_sel.get("mel",Ra_sel["mel"]),fs,fe)
                            nc=min(ma.shape[1],mb.shape[1])
                            st.pyplot(spec_pair("Mel Spectrogram",ma[:,:nc],mb[:,:nc],
                                name_a_sel,name_b_sel,cmap="inferno",ylabel="Mel band",
                                add_mel_bands=True,col_a=rec_color(pi),col_b=rec_color(pj),
                                cmp_label="DTW-aligned")); plt.close()
                    if "CQT" in spec_sub_map:
                        with spec_sub_map["CQT"]:
                            insight_card("🎵","CQT","Semitone-aligned. Y-axis = MIDI note names. "
                                "Differences at specific note rows show which pitches diverge.")
                            diff_formula(name_a_sel,name_b_sel,rec_color(pi),rec_color(pj))
                            ma=cl2(Ra_sel["cqt"],fs,fe); mb=cl2(Rb_sel.get("cqt",Ra_sel["cqt"]),fs,fe)
                            nc=min(ma.shape[1],mb.shape[1])
                            yticks={b:cqt_bin_to_note(b) for b in range(0,N_CQT,12)}
                            st.pyplot(spec_pair("CQT",ma[:,:nc],mb[:,:nc],
                                name_a_sel,name_b_sel,cmap="viridis",ylabel="Note",
                                yticks=yticks,col_a=rec_color(pi),col_b=rec_color(pj),
                                cmp_label="DTW-aligned")); plt.close()

        # ── Advanced ──────────────────────────────────────────────────
        # ── Score / MIDI Alignment ─────────────────────────
        if do_chroma and "Score Alignment" in tab_map:
            with tab_map["Score Alignment"]:
                # Each recording's own score (if uploaded) was parsed once
                # alongside the audio files at the top of the page and
                # cached in session state — no re-uploading needed per tab.
                uploaded_score_chromas_r = st.session_state.get("vs_uploaded_score_chromas")
                uploaded_score_names_r   = st.session_state.get("vs_uploaded_score_names")
                if uploaded_score_chromas_r and any(c is not None for c in uploaded_score_chromas_r):
                    have = [(nm, sn) for nm, c, sn in
                            zip(names, uploaded_score_chromas_r, uploaded_score_names_r or [])
                            if c is not None]
                    st.success("✓ Own score uploaded for: " +
                               ", ".join(f"{nm} ({sn})" for nm, sn in have))
                else:
                    st.caption(
                        "No reference scores uploaded — all recordings use an "
                        "auto-generated consensus score instead. To align a "
                        "recording against its real score, upload a MIDI file "
                        "for it in the upload section above (before running analysis).")

                with st.spinner("Generating MIDI transcriptions..."):
                    try:
                        midi_chromas_r, consensus_r, alignments_r, _vp_r, used_up_r = \
                            compute_midi_alignment(
                                Rs_aligned, ys, names,
                                dtw_strategy=dtw_strategy,
                                uploaded_score_chromas=uploaded_score_chromas_r)
                        render_score_alignment(
                            midi_chromas_r, consensus_r, alignments_r,
                            names, zoom_start, zoom_end,
                            voiced_pcts=st.session_state.get("midi_voiced_pcts"),
                            used_uploaded_score=used_up_r, ys=ys, Rs_aligned=Rs_aligned)
                    except Exception as e:
                        st.error(f"Score alignment failed: {e}")
                        import traceback; st.code(traceback.format_exc())

        if adv and "Advanced" in tab_map:
            with tab_map["Advanced"]:
                adv_tabs=st.tabs(adv)
                adv_map={n:t for n,t in zip(adv,adv_tabs)}

                if "SMS/HPSS" in adv_map and do_sms and "sms_h" in Ra_sel:
                    with adv_map["SMS/HPSS"]:
                        ha=float(Ra_sel["sms_ratio"].mean()) if "sms_ratio" in Ra_sel else 0
                        hb=float(Rb_sel["sms_ratio"].mean()) if "sms_ratio" in Rb_sel else 0
                        insight_card("🎼","Harmonic/Percussive separation",
                            f"Ratio: {name_a_sel}={ha:.2f} · {name_b_sel}={hb:.2f}  (0=rhythmic, 1=melodic)")
                        for key,title in [("sms_h","Harmonic"),("sms_p","Percussive")]:
                            if key in Ra_sel and key in Rb_sel:
                                diff_formula(name_a_sel,name_b_sel,rec_color(pi),rec_color(pj))
                                ma=cl2(Ra_sel[key],fs,fe); mb=cl2(Rb_sel[key],fs,fe)
                                nc=min(ma.shape[1],mb.shape[1])
                                st.pyplot(spec_pair(title,ma[:,:nc],mb[:,:nc],
                                    name_a_sel,name_b_sel,cmap="magma",ylabel="Freq bin",
                                    col_a=rec_color(pi),col_b=rec_color(pj),
                                    cmp_label="DTW-aligned")); plt.close()
                        with st.expander("🔊 Listen"):
                            c1s,c2s=st.columns(2)
                            with c1s:
                                st.caption(f"{name_a_sel} Harmonic"); st.audio(hpss_bytes(segs[pi],'harmonic'),format='audio/wav')
                                st.caption(f"{name_a_sel} Percussive"); st.audio(hpss_bytes(segs[pi],'percussive'),format='audio/wav')
                            with c2s:
                                st.caption(f"{name_b_sel} Harmonic"); st.audio(hpss_bytes(segs[pj],'harmonic'),format='audio/wav')
                                st.caption(f"{name_b_sel} Percussive"); st.audio(hpss_bytes(segs[pj],'percussive'),format='audio/wav')

                if "Reverb" in adv_map and do_reverb and "rt60" in Ra_sel and "rt60" in Rb_sel:
                    with adv_map["Reverb"]:
                        diff_rt=Rb_sel["rt60"]-Ra_sel["rt60"]
                        insight_card("🏠","Room acoustics",
                            f"RT60: {name_a_sel}={Ra_sel['rt60']:.3f}s · {name_b_sel}={Rb_sel['rt60']:.3f}s · Δ={diff_rt:+.3f}s")
                        fig_r,ax_r=plt.subplots(figsize=(14,3))
                        ax_r.plot(np.linspace(0,durs[pi],len(Ra_sel["decay"])),Ra_sel["decay"],
                                  label=name_a_sel,lw=0.8,color=rec_color(pi))
                        ax_r.plot(np.linspace(0,durs[pj],len(Rb_sel["decay"])),Rb_sel["decay"],
                                  label=name_b_sel,lw=0.8,color=rec_color(pj))
                        ax_r.axhline(-60,color='red',ls='--',lw=0.8,label='−60 dB')
                        ax_r.legend(fontsize=7); ax_r.set_xlabel("Time (s)"); ax_r.set_ylabel("dB")
                        plt.tight_layout(); st.pyplot(fig_r,use_container_width=True); plt.close()

                if "STFT" in adv_map and do_stft and "stft" in Ra_sel:
                    with adv_map["STFT"]:
                        diff_formula(name_a_sel,name_b_sel,rec_color(pi),rec_color(pj))
                        ma=cl2(Ra_sel["stft"],fs,fe); mb=cl2(Rb_sel.get("stft",Ra_sel["stft"]),fs,fe)
                        nc=min(ma.shape[1],mb.shape[1])
                        st.pyplot(spec_pair("STFT",ma[:,:nc],mb[:,:nc],
                            name_a_sel,name_b_sel,cmap="magma",ylabel="Freq bin",
                            col_a=rec_color(pi),col_b=rec_color(pj))); plt.close()

                if "CWT" in adv_map and do_cwt and HAS_PYWT and "cwt" in Ra_sel:
                    with adv_map["CWT"]:
                        from scipy.ndimage import zoom as nd_zoom
                        ca_cw=Ra_sel["cwt"]; cb_cw=Rb_sel.get("cwt",ca_cw)
                        if cb_cw is not None and cb_cw.shape[1]!=ca_cw.shape[1]:
                            cb_cw=nd_zoom(cb_cw.astype(np.float32),(1,ca_cw.shape[1]/cb_cw.shape[1]),order=1)
                        n_c=ca_cw.shape[1]; cz_s=int(zoom_start/durs[pi]*n_c); cz_e=int(zoom_end/durs[pi]*n_c)
                        ma=ca_cw[:,cz_s:cz_e]; mb=(cb_cw[:,cz_s:cz_e] if cb_cw is not None else ma)
                        nc=min(ma.shape[1],mb.shape[1])
                        diff_formula(name_a_sel,name_b_sel,rec_color(pi),rec_color(pj))
                        st.pyplot(spec_pair("CWT",ma[:,:nc],mb[:,:nc],
                            name_a_sel,name_b_sel,cmap="plasma",ylabel="Scale",
                            col_a=rec_color(pi),col_b=rec_color(pj))); plt.close()

                if "Gammatone" in adv_map and do_gammatone and "gam" in Ra_sel:
                    with adv_map["Gammatone"]:
                        ma=cl2(Ra_sel["gam"],fs,fe); mb=cl2(Rb_sel.get("gam",Ra_sel["gam"]),fs,fe)
                        nc=min(ma.shape[1],mb.shape[1])
                        diff_formula(name_a_sel,name_b_sel,rec_color(pi),rec_color(pj))
                        st.pyplot(spec_pair("Gammatone",ma[:,:nc],mb[:,:nc],
                            name_a_sel,name_b_sel,cmap="hot",ylabel="ERB filter",
                            col_a=rec_color(pi),col_b=rec_color(pj))); plt.close()

                if "Tonnetz" in adv_map and do_tonnetz and "tonnetz" in Ra_sel:
                    with adv_map["Tonnetz"]:
                        ta_w=cl2(Ra_sel["tonnetz"],fs,fe); tb_w=cl2(Rb_sel.get("tonnetz",Ra_sel["tonnetz"]),fs,fe)
                        nc=min(ta_w.shape[1],tb_w.shape[1]); ta_m=ta_w[:,:nc].mean(1); tb_m=tb_w[:,:nc].mean(1)
                        dist=float(np.linalg.norm(ta_m-tb_m))
                        insight_card("🔮","Tonnetz",f"L2 distance: {dist:.4f}. {'Same harmonic region.' if dist<0.2 else 'Different tonal centre.'}")
                        dim_labels=["Fifths 1","Fifths 2","Maj 3rd 1","Maj 3rd 2","Min 3rd 1","Min 3rd 2"]
                        fig_t,ax_t=plt.subplots(figsize=(10,3)); x=np.arange(6)
                        ax_t.bar(x-0.2,ta_m,0.35,label=name_a_sel,color=rec_color(pi),alpha=0.85)
                        ax_t.bar(x+0.2,tb_m,0.35,label=name_b_sel,color=rec_color(pj),alpha=0.85)
                        ax_t.axhline(0,color='black',lw=1.2,zorder=5)
                        ax_t.set_xticks(x); ax_t.set_xticklabels(dim_labels,fontsize=8); ax_t.legend()
                        plt.tight_layout(); st.pyplot(fig_t,use_container_width=True); plt.close()

        # ── N-recording all-features overlay (shown when n_recs > 2) ─
        if n_recs > 2:
            st.markdown("---")
            st.subheader("All Rehearsals — Feature Overlay")
            st.caption("All recordings on the same axes for each feature. "
                       "Each recording keeps its color throughout.")
            overlay_feats=[
                ("rms","Volume (RMS)","linear"),
                ("centroid","Brightness (Centroid Hz)","linear"),
                ("onset","Onset Strength","linear"),
                ("sms_ratio","Harmonic Ratio","linear"),
            ]
            avail=[f for f in overlay_feats if all(f[0] in R for R in Rs_aligned)]
            if avail:
                n_f=len(avail)
                fig_ov,axes_ov=plt.subplots(n_f,1,figsize=(14,2.5*n_f),sharex=True)
                if n_f==1: axes_ov=[axes_ov]
                for ax,(feat,label,_) in zip(axes_ov,avail):
                    for idx,(R,nm) in enumerate(zip(Rs_aligned,names)):
                        arr=cl1(R[feat],fs,fe)
                        if len(arr)==0: continue
                        ax.plot(np.linspace(zoom_start,zoom_end,len(arr)),
                                arr,lw=0.9,color=rec_color(idx),label=nm,alpha=0.85)
                    ax.set_ylabel(label,fontsize=7); ax.legend(fontsize=7,loc="upper right")
                axes_ov[-1].set_xlabel("Time (s)",fontsize=8)
                plt.suptitle(f"All {n_recs} Rehearsals — {zoom_start:.1f}–{zoom_end:.1f}s",
                             fontsize=10,fontweight='bold')
                plt.tight_layout(); st.pyplot(fig_ov,use_container_width=True); plt.close()
            else:
                st.info("Enable Spectral Features and Onset for overlay plots.")

    if pair_tabs is not None:
        # N recordings: render inside each pair tab
        for pair_tab, (pi, pj) in zip(pair_tabs, pairs):
            with pair_tab:
                _render_pair(pi, pj)
    else:
        # 2 recordings: render directly
        _render_pair(0, 1)

    # ── What Changed card (bottom) — always uses best/only pair ─
    best_pi, best_pj = pairs[best_idx]
    best_Ra, best_Rb = Rs_aligned[best_pi], Rs_aligned[best_pj]
    best_na, best_nb = names[best_pi], names[best_pj]
    best_insights = generate_summary(best_Ra, best_Rb, best_na, best_nb)
    st.markdown("---")
    with st.expander("💡 What Changed? — Quick Summary", expanded=False):
        st.caption(f"Based on auto-selected pair: **{best_na} vs {best_nb}**")
        for icon, headline, detail in best_insights:
            st.markdown(f"**{icon} {headline}**")
            st.caption(detail)
        st.caption("_Rule-based · swap `generate_summary()` for an LLM call_")

# ── Downloads section ────────────────────────────────────────
with main_col:
    st.markdown("---")
    st.subheader("⬇️ Downloads")

    # ── Report generation ─────────────────────────────────────
    st.markdown("**Progress Report**")
    st.caption("Generates a full report with all figures: "
               "similarity matrix, MDS, stream graph, waveform, "
               "chroma, MFCC, dynamics, onset, Mel spectrogram, "
               "and score alignment (if computed).")

    if st.button("Generate Report", key="gen_report"):
        with st.spinner("Rendering all figures..."):
            # Get MIDI alignment data if it was computed this session
            midi_chromas_rep   = st.session_state.get("midi_chromas_rep")
            consensus_rep      = st.session_state.get("consensus_rep")
            midi_alignments_rep= st.session_state.get("midi_alignments_rep")
            report_md, report_html = generate_report(
                Rs_aligned, names, pairs, sim_matrix,
                zoom_start, zoom_end, ys, durs,
                midi_chromas=midi_chromas_rep,
                consensus_chroma=consensus_rep,
                midi_alignments=midi_alignments_rep)
        st.session_state["report_md"]   = report_md
        st.session_state["report_html"] = report_html
        st.success("Report ready — download below.")

    if st.session_state.get("report_md"):
        fname_base = f"visus_report_{names[pairs[best_idx][0]][:8]}_{names[pairs[best_idx][1]][:8]}"
        col_md, col_html = st.columns(2)
        with col_md:
            st.download_button(
                label="⬇️ Download as Markdown (.md)",
                data=st.session_state["report_md"],
                file_name=f"{fname_base}.md",
                mime="text/markdown",
                key="dl_report_md",
                use_container_width=True)
            st.caption("Open in any markdown viewer. "
                       "Images may not render in all viewers.")
        with col_html:
            st.download_button(
                label="⬇️ Download as HTML (.html)",
                data=st.session_state["report_html"],
                file_name=f"{fname_base}.html",
                mime="text/html",
                key="dl_report_html",
                use_container_width=True)
            st.caption("Open in Chrome/Firefox → "
                       "Ctrl+P → Save as PDF to share.")

        with st.expander("Preview (first 4000 chars)"):
            st.markdown(st.session_state["report_md"][:4000] +
                        ("..." if len(st.session_state["report_md"])>4000 else ""))

    st.markdown("---")

    # ── MIDI downloads ────────────────────────────────────────
    st.markdown("**MIDI Files**")
    st.caption("Generate MIDI transcriptions from each recording using pYIN pitch detection. "
               "Download individual files or the consensus score (average of all transcriptions).")

    if st.button("Generate MIDI Files", key="gen_midi"):
        with st.spinner("Transcribing recordings to MIDI via pYIN..."):
            try:
                midi_chromas_dl, consensus_dl, alignments_dl, voiced_pcts_dl, used_upload_dl = compute_midi_alignment(
                    Rs_aligned, ys, names,
                    uploaded_score_chromas=st.session_state.get("vs_uploaded_score_chromas"))
                # Store in session state for report use
                st.session_state["midi_chromas_rep"]    = midi_chromas_dl
                st.session_state["consensus_rep"]        = consensus_dl
                st.session_state["midi_alignments_rep"]  = alignments_dl
                st.session_state["midi_ready"]           = True
            except Exception as e:
                st.error(f"MIDI generation failed: {e}")
                st.session_state["midi_ready"] = False

    if st.session_state.get("midi_ready"):
        midi_chromas_dl    = st.session_state["midi_chromas_rep"]
        consensus_dl       = st.session_state["consensus_rep"]

        # Individual MIDI downloads
        midi_cols = st.columns(min(n_recs, 4))
        try:
            import pretty_midi as _pm_
            import io as _io_
        except ImportError:
            st.error("pretty_midi not installed. Run: **pip install pretty_midi** then restart the app.")
            st.stop()
        for col, nm, mc in zip(midi_cols, names, midi_chromas_dl):
            try:
                import io as _io
                R_nm = Rs_aligned[names.index(nm)]
                tempo = float(R_nm.get("tempo", 120))
                pm = midi_chroma_to_pretty_midi(mc, tempo=tempo)
                buf = _io.BytesIO(); pm.write(buf); buf.seek(0)
                fname_midi = nm.replace(" ","_")[:20] + ".mid"
                col.download_button(
                    label=f"⬇️ {nm[:15]}",
                    data=buf.getvalue(),
                    file_name=fname_midi,
                    mime="audio/midi",
                    key=f"dl_midi_{names.index(nm)}",
                    use_container_width=True)
                col.caption(f"{tempo:.0f} BPM")
            except Exception as e:
                col.caption(f"Failed: {e}")

        # Consensus MIDI
        try:
            import pretty_midi, io as _io
            avg_tempo = float(np.mean([R.get("tempo",120) for R in Rs_aligned]))
            pm_cons = midi_chroma_to_pretty_midi(consensus_dl, tempo=avg_tempo)
            buf_cons = _io.BytesIO(); pm_cons.write(buf_cons); buf_cons.seek(0)
            st.download_button(
                label="⬇️ Download Consensus Score (.mid)",
                data=buf_cons.getvalue(),
                file_name="visus_consensus_score.mid",
                mime="audio/midi",
                key="dl_midi_consensus",
                use_container_width=False)
            st.caption(f"Consensus = average of all {n_recs} MIDI transcriptions · "
                       f"{avg_tempo:.0f} BPM avg. "
                       "Import into any DAW or notation software.")
        except Exception as e:
            st.caption(f"Consensus MIDI failed: {e}")

st.caption("ViSuS · Music Signal Analysis · HIWI Research Demo")
