# checked

import os


class Constants:
    Tag_Start = "<START>"
    Tag_End = "<END>"
    Word_Pad = "<PAD>"
    Word_Unknown = "<UNK>"
    Char_Pad = "<C_PAD>"
    Char_Unknown = "<C_UNK>"
    Invalid_Transition = -10000
    Models_Folder = "./models/"
    Logs_Folder = "./logs/"
    Eval_Folder = "./evaluation/"
    Eval_Temp_Folder = os.path.join(Eval_Folder, "tmp")
    Eval_Script = os.path.join(Eval_Folder, "eval.pl")
