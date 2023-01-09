""" from https://github.com/keithito/tacotron """
from loguru import logger
from app.config import Config
from text import cleaners



def text_to_sequence(text, symbols, cleaner_names):
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
      Args:
        text: string to convert to a sequence
        symbols: no use, just to keep same with original code https://huggingface.co/spaces/skytnt/moe-tts/tree/main
        cleaner_names: names of the cleaner functions to run the text through
      Returns:
        List of integers corresponding to the symbols in the text
    '''

    

    # Config.symbol_to_id = Config.symbol_to_id

    logger.debug(f"symbol_to_id: {Config.symbol_to_id}")

    clean_text = _clean_text(text, cleaner_names)

    sequence = [
        Config.symbol_to_id[symbol] for symbol in clean_text if symbol in Config.symbol_to_id.keys()
    ]


    return sequence


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception('Unknown cleaner: %s' % name)
        text = cleaner(text)
    return text
