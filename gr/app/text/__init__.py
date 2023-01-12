""" from https://github.com/keithito/tacotron """
import re

from . import cleaners

_symbol_to_id = None
pattern = None


def build_pattern():
    brackets = ['（', '[', '『', '「', '【', ")", "】", "]", "』", "」", "）"]
    pattern = re.compile('|'.join(map(re.escape, brackets)))
    return pattern


def text_to_seq(text: str, hps):
    global pattern
    if not pattern:
        pattern = build_pattern()
    text = pattern.sub(' ', text).strip()
    text_norm = text_to_sequence(
        text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = intersperse(text_norm, 0)
    return text_norm


def text_to_sequence(text, symbols, cleaner_names):
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
      Args:
        text: string to convert to a sequence
        symbols: list of symbols in the text
        cleaner_names: names of the cleaner functions to run the text through
      Returns:
        List of integers corresponding to the symbols in the text


        ATTENTION: unable to access Config variabel , don't know why
    '''

    global _symbol_to_id

    if not _symbol_to_id:
        _symbol_to_id = {s: i for i, s in enumerate(symbols)}

    clean_text = _clean_text(text, cleaner_names)

    sequence = [
        _symbol_to_id[symbol] for symbol in clean_text if symbol in _symbol_to_id.keys()
    ]

    # for symbol in clean_text:
    #     if symbol not in _symbol_to_id.keys():
    #         continue
    #     symbol_id = _symbol_to_id[symbol]
    #     sequence += [symbol_id]
    return sequence


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception('Unknown cleaner: %s' % name)
        text = cleaner(text)
    return text
