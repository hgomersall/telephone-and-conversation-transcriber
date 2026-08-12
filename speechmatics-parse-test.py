#!/usr/bin/env python3
"""Tests for parse_speechmatics_transcript in caption_app.py.

The response shape is the part of a new provider most likely to be wrong, and
getting it wrong does not raise — it produces a recogniser that runs, connects,
reports healthy, and silently displays nothing. For a device someone relies on
to follow conversation, that is the worst possible failure, so the parser is
written to survive a shape it did not expect and this checks that it does.

Run: ./speechmatics-parse-test.py
"""

import sys
import types


def stub_pyqt():
    class Any:
        def __init__(self, *a, **k): pass
        def __getattr__(self, n): return Any()
        def __call__(self, *a, **k): return Any()
    mods = {
        'PyQt6.QtWidgets': ['QApplication', 'QMainWindow', 'QTextEdit', 'QLabel',
                            'QVBoxLayout', 'QHBoxLayout', 'QWidget', 'QScroller',
                            'QStackedWidget'],
        'PyQt6.QtCore': ['Qt', 'pyqtSignal', 'QObject', 'QTimer'],
        'PyQt6.QtGui': ['QFont', 'QFontDatabase', 'QTextCursor', 'QTextCharFormat',
                        'QPainter', 'QColor', 'QLinearGradient', 'QPen'],
    }
    sys.modules['PyQt6'] = types.ModuleType('PyQt6')
    for mod, names in mods.items():
        sys.modules[mod] = types.ModuleType(mod)
        for n in names:
            setattr(sys.modules[mod], n, Any)


def word(content, speaker=None):
    alt = {'content': content, 'confidence': 0.9}
    if speaker:
        alt['speaker'] = speaker
    return {'type': 'word', 'start_time': 0.0, 'end_time': 1.0, 'alternatives': [alt]}


CASES = [
    ('metadata transcript preferred',
     {'message': 'AddTranscript', 'metadata': {'transcript': 'hello there'},
      'results': [word('hello', 'S1'), word('there', 'S1')]},
     ('hello there', 'S1')),

    ('rebuilt from results when metadata absent',
     {'message': 'AddTranscript',
      'results': [word('hello', 'S1'), word('there', 'S1')]},
     ('hello there', 'S1')),

    ('majority speaker across a segment that straddles a change',
     {'message': 'AddTranscript', 'metadata': {'transcript': 'a b c'},
      'results': [word('a', 'S1'), word('b', 'S2'), word('c', 'S2')]},
     ('a b c', 'S2')),

    # UU is Speechmatics for "unknown". Treated as a speaker it would read as a
    # turn change and recolour the captions for no reason.
    ('UU is not a speaker',
     {'message': 'AddTranscript', 'metadata': {'transcript': 'mm'},
      'results': [word('mm', 'UU')]},
     ('mm', None)),

    ('no diarization means no speaker',
     {'message': 'AddTranscript', 'metadata': {'transcript': 'plain'},
      'results': [word('plain')]},
     ('plain', None)),

    # Everything below is a shape we did not expect. None may raise.
    ('empty message',
     {'message': 'AddTranscript', 'metadata': {'transcript': ''}, 'results': []},
     ('', None)),

    ('missing keys entirely',
     {'message': 'AddTranscript'},
     ('', None)),

    ('results present but no alternatives',
     {'message': 'AddTranscript', 'results': [{'type': 'word'}]},
     ('', None)),

    ('metadata is not an object',
     {'message': 'AddTranscript', 'metadata': 'unexpected',
      'results': [word('fallback')]},
     ('fallback', None)),

    ('nulls where objects were expected',
     {'message': 'AddTranscript', 'metadata': None, 'results': None},
     ('', None)),
]


def main():
    stub_pyqt()
    sys.path.insert(0, __file__.rsplit('/', 1)[0])
    from caption_app import parse_speechmatics_transcript as parse

    failures = 0
    for name, message, expected in CASES:
        try:
            got = parse(message)
        except Exception as e:
            print(f'  FAIL  {name}\n        raised {e!r}')
            failures += 1
            continue
        if got == expected:
            print(f'  PASS  {name}')
        else:
            print(f'  FAIL  {name}\n        got {got!r}, expected {expected!r}')
            failures += 1

    print()
    if failures:
        print(f'{failures} FAILED')
        return 1
    print('ALL CHECKS PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
