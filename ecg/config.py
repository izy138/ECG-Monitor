"""Project-wide constants.

Everything that affects what a "beat" looks like to the model lives here, and both the
training pipeline and the inference API import it. That way preprocessing can't silently
drift between training and serving.
"""

# --- Signal ---------------------------------------------------------------
FS = 360                      # MIT-BIH sampling rate (Hz)
LEAD = "MLII"                 # selected by name, not by channel index
WINDOW_BEFORE = 90            # samples before the R-peak
WINDOW_AFTER = 110            # samples after the R-peak
WINDOW_LEN = WINDOW_BEFORE + WINDOW_AFTER

BANDPASS_LOW_HZ = 0.5
BANDPASS_HIGH_HZ = 45.0
BANDPASS_ORDER = 4

LOCAL_RR_BEATS = 10           # beats of history used for the "local rhythm" RR baseline

# --- Records --------------------------------------------------------------
# The 48 MIT-BIH records. (There is no record 211.)
ALL_RECORDS = (
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
)

# Paced records are excluded from the de Chazal inter-patient protocol.
PACED_RECORDS = ("102", "104", "107", "217")

# de Chazal et al. (2004) inter-patient split.
# Known caveat: records 201 and 202 are the same patient, split across DS1/DS2.
# We keep the published split anyway so results stay comparable to the literature.
DS1 = (
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122",
    "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230",
)
DS2 = (
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210",
    "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234",
)

# --- Labels ---------------------------------------------------------------
# Every annotation symbol that marks an actual heartbeat. Used for RR intervals, so a beat
# we later drop from the dataset still counts as a beat when measuring timing.
# Non-beat annotations ('+', '~', '|', 'x', '!', '"', '[', ']') are excluded.
BEAT_SYMBOLS = frozenset("NLRBAaJSVrFejnE/fQ?")

# AAMI EC57 grouping. Class Q (paced / unclassifiable) is dropped because its beats come
# almost entirely from the paced records, which are not in DS1 or DS2.
AAMI_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
}
CLASSES = ("N", "S", "V", "F")
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
CLASS_NAMES = {
    "N": "Normal",
    "S": "Supraventricular ectopic",
    "V": "Ventricular ectopic",
    "F": "Fusion",
}

RR_FEATURE_NAMES = ("pre_rr_s", "post_rr_s", "pre_rr_ratio", "post_rr_ratio")

# --- RR feature physiological bounds --------------------------------------
# MIT-BIH annotation files contain a handful of multi-second-to-100-second gaps between
# consecutive beat annotations (dropped/unreadable annotations, not real asystole). Left
# unclipped, these dominate any mean/std fit on pre_rr_s/post_rr_s/pre_rr_ratio/post_rr_ratio
# and crush the real distribution into a few hundredths of a sigma. Used by
# ecg.preprocessing.fit_rr_scaler/apply_rr_scaler (clip-then-standardize) and by
# ecg.build_dataset.process_record's "implausible_rr_gap" drop gate -- one set of bounds,
# two complementary defenses (see build_dataset docstring for why both are needed).
#
# Evidence (data/processed/train.npz, 40155 train beats, current inter-patient split): the
# real RR distribution is dense and continuous up to ~2.5s (p99.9 = 1.89s; only 1 beat in
# [2.5s, 3.0s)), then falls off a cliff -- the next beat above 3.0s sits at 3.7s, and beyond
# that values jump straight to 5-100s (annotation gaps). pre_rr_ratio/post_rr_ratio show the
# identical cliff: dense up to ~2.5-3.0 (p99.9 = 2.57), then a hard drop to just 3 beats in
# [3, 4). Genuine class medians for pre_rr_ratio -- the main discriminative RR signal -- are
# all in [0.74, 1.00] (N 1.004, S 0.815, V 0.742, F 0.995), far below these bounds, so real
# signal is untouched.
RR_MIN_S = 0.2       # 300 bpm ceiling on heart rate. No training beat is this fast today
                     # (min observed = 0.25s); kept as a defensive floor for a future
                     # streaming beat, where a double-detected peak could read near 0.
RR_MAX_S = 3.0       # 20 bpm floor -- sits in the cliff between the real tail (<=2.5s) and
                     # the annotation-gap outliers (>=3.7s, up to 100s).
RR_RATIO_MAX = 3.0   # Same cliff, expressed as a ratio to the local rhythm baseline.
