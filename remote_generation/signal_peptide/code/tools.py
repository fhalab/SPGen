import numpy as np

class CharacterTable(object):
    """Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    """
    def __init__(self, chars, one_hot=True):
        """Initialize character table.
        # Arguments
            chars: Characters that can appear in the input.
            one_hot: one_hot or tokenize
        """
        self.chars = sorted(set(chars))
        self.one_hot = one_hot
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows, reverse=False):
        """One hot encode given string C.
        # Arguments
            num_rows: Number of rows in the returned one hot encoding. This is
                used to keep the # of rows for each data the same.
        """
        if not self.one_hot:
            x = np.array([self.char_indices[c] for c in C])
            if reverse:
                x = x[::-1]
            return x

        x = np.zeros((num_rows, len(self.chars)))

        for i, c in enumerate(C):
            if reverse:
                j = -1 - i
            else:
                j = i
            x[j, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True, reverse=False):
        if not self.one_hot:
            s = ''.join([self.indices_char[xx] for xx in x])
        else:
            if calc_argmax:
                x = x.argmax(axis=-1)
            s = ''.join(self.indices_char[xx] for xx in x)
        if reverse:
            s = s[::-1]
        return s
