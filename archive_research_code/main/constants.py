PAD_SYM = "$$$PAD$$$"
UNK_SYM = "$$$UNK$$$"
PAD_SYM_ID = 0  # can not change
UNK_SYM_ID = 1  # can not change

INS_START_SYM = "<ins>"
INS_END_SYM = "</ins>"
DEL_START_SYM = "<del>"
DEL_END_SYM = "</del>"

INS_START_SYM_ESCAPED = "$ins@"
INS_END_SYM_ESCAPED = "@ins$"
DEL_START_SYM_ESCAPED = "$del@"
DEL_END_SYM_ESCAPED = "@del$"

PADDING_SIZE = 4  # amount of padding before and (where applicable) after each sentence; for uniCNN this could be 0, but need to update the BERT masking

AESW_CLASS_LABELS = [0, 1]

ID_CORRECT = 0  # "negative" class (correct token)
ID_WRONG = 1  # "positive class" (token with error)


UNK_TITLE_ID = -1

# Decision labels:
SUPPORTS_STRING = "Supports:"
REFUTES_STRING = "Refutes:"
MOREINFO_STRING = "Unverifiable:"
CONSIDER_STRING = "Consider:"

SUPPORTS_ID = 0
REFUTES_ID = 1
MOREINFO_ID = 2

# Additional Claims prefixes
CLAIMS_CONSIDER_PREFIX_STRING = "Consider:"  # same as above for title side
CLAIMS_REF_PREFIX_STRING = "Reference:"
CLAIMS_PREDICT_PREFIX_STRING = "Predict:"

UNVERIFIABLE_TITLE_ID = -2


# for ec
EC_CORRECT_ID = 3
