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
