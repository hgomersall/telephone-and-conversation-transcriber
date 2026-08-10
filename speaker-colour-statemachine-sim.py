#!/usr/bin/env python3
"""PyQt-free harness for the interim/final/speaker-change state machine.

Mirrors CaptionView.add_segment / add_text / reset_speakers against a plain
string document, so the ordering logic can be exercised in milliseconds
instead of an hour in a live room.

Keep this in step with caption_app.py — if the display logic changes, change
it here too, or the tests are testing a fiction.

Run: python3 speaker-colour-statemachine-sim.py
"""

import sys

PAL = ['WHITE', 'AMBER', 'BLUE']
MARK = '> '          # stands in for SPEAKER_MARKER ('▸ ')


class Sim:
    """Mirror of the CaptionView document state."""

    def __init__(self):
        self.doc = ''
        self.fmt = []              # per-char colour, parallel to doc
        self.prov = None           # _prov_start
        self.last_speaker = None
        self.cidx = 0
        self.last_speech_final = False
        self.last_text_time = 0

    def reset_speakers(self):
        self.prov = None
        self.last_speaker = None
        self.cidx = 0
        self.last_speech_final = True

    def add_segment(self, text, is_final, speaker, speech_final=False, now=None):
        text = (text or '').strip()
        if now is None:
            now = self.last_text_time + 0.1

        if not text:
            # Empty final: commit any interim already on screen, don't delete it.
            if is_final:
                self.prov = None
                self.last_speech_final = speech_final
            return

        if self.prov is None:
            self.prov = len(self.doc)
        else:
            self.doc = self.doc[:self.prov]
            self.fmt = self.fmt[:self.prov]

        cidx, turn = self.cidx, False
        if is_final and speaker is not None and speaker != self.last_speaker:
            if self.last_speaker is not None:
                cidx = (self.cidx + 1) % len(PAL)
                turn = True

        at_start = (self.prov == 0)
        if not at_start:
            if turn:
                sep = '\n\n'
            elif self.last_speech_final:
                sep = '\n'
            elif self.last_text_time > 0 and (now - self.last_text_time) > 2:
                sep = '\n'
            else:
                sep = ' '
            self.doc += sep
            self.fmt += ['-'] * len(sep)

        marker = MARK if (turn or at_start) else ''
        body = marker + text
        self.doc += body
        self.fmt += [PAL[cidx]] * len(body)

        if is_final:
            self.prov = None
            self.cidx = cidx
            self.last_speech_final = speech_final
            if speaker is not None:
                self.last_speaker = speaker
        self.last_text_time = now

    def add_text(self, t, now=None):
        """Non-streaming path — offline engines and the non-Deepgram providers."""
        if now is None:
            now = self.last_text_time + 0.1
        self.prov = None           # commit provisional text where it stands
        has_newline = t.endswith('\n')
        t = t.rstrip('\n')
        if self.doc:
            sep = '\n\n' if (self.last_text_time > 0 and (now - self.last_text_time) > 2) else ' '
            self.doc += sep
            self.fmt += ['-'] * len(sep)
        self.doc += t
        self.fmt += ['-'] * len(t)
        if has_newline:
            self.doc += '\n'
            self.fmt += ['-']
        self.last_text_time = now

    def lines(self):
        """(colour, line) for each non-blank line."""
        out, i = [], 0
        for line in self.doc.split('\n'):
            if line.strip():
                col = next((self.fmt[j] for j in range(i, i + len(line))
                            if self.fmt[j] != '-'), '-')
                out.append((col, line))
            i += len(line) + 1
        return out

    def render(self):
        return '\n'.join(f'[{col:5}] {line}' for col, line in self.lines())


# ─── Tests ───────────────────────────────────────────────────────────────────

FAILURES = []


def check(name, cond, detail=''):
    if cond:
        print(f'  PASS  {name}')
    else:
        print(f'  FAIL  {name}' + (f'\n        {detail}' if detail else ''))
        FAILURES.append(name)


def test_single_speaker_utterance_breaks():
    """Two utterances from one speaker must not run together on one line.

    Regression: with speech_final dropped, consecutive finals were joined by a
    space and the whole conversation became one paragraph.
    """
    s = Sim()
    s.add_segment('so I', False, 0)
    s.add_segment('so I said to him', True, 0, speech_final=True)
    s.add_segment('that appointment was', False, 0)
    s.add_segment('that appointment was Tuesday', True, 0, speech_final=True)
    lines = s.lines()
    check('single speaker: one line per utterance', len(lines) == 2,
          f'got {len(lines)} line(s): {[l for _, l in lines]}')
    check('single speaker: utterances not run together',
          'him that appointment' not in s.doc, s.doc)


def test_phone_tap_has_breaks():
    """The phone tap has no diarization, so speaker is always None.

    Breaks must still come from speech_final, or a whole call renders as one
    unbroken wall of text.
    """
    s = Sim()
    for utt in ['hello love', 'yes I got the letter', 'see you Tuesday']:
        s.add_segment(utt[:5], False, None)
        s.add_segment(utt, True, None, speech_final=True)
    lines = s.lines()
    check('phone tap: three utterances, three lines', len(lines) == 3,
          f'got {len(lines)}: {[l for _, l in lines]}')
    check('phone tap: all one colour (no diarization)',
          {c for c, _ in lines} == {'WHITE'}, str(lines))


def test_interims_do_not_duplicate():
    s = Sim()
    s.add_segment('the', False, 0)
    s.add_segment('the cat', False, 0)
    s.add_segment('the cat sat', True, 0, speech_final=True)
    check('interims overwrite rather than append',
          s.doc.count('the cat sat') == 1 and s.doc.count('the cat') == 1, s.doc)


def test_speaker_change_marks_and_colours():
    s = Sim()
    s.add_segment('so I said to him', True, 0, speech_final=True)
    s.add_segment('no they moved it', True, 1, speech_final=True)
    s.add_segment('she is right', True, 2, speech_final=True)
    lines = s.lines()
    check('speaker change: three turns', len(lines) == 3, str(lines))
    check('speaker change: consecutive turns differ in colour',
          all(lines[i][0] != lines[i + 1][0] for i in range(len(lines) - 1)),
          str([c for c, _ in lines]))
    check('speaker change: every turn carries the structural marker',
          all(line.startswith(MARK) for _, line in lines),
          str([l for _, l in lines]))


def test_reconnect_resets_cycle():
    s = Sim()
    s.add_segment('first speaker', True, 0, speech_final=True)
    s.add_segment('second speaker', True, 1, speech_final=True)
    check('before reconnect: cycled off turn A', s.cidx != 0, f'cidx={s.cidx}')
    s.reset_speakers()                     # emitter.speakers_reset
    s.add_segment('after reconnect', True, 0, speech_final=True)
    check('reconnect: cycle resets to turn A', s.lines()[-1][0] == PAL[0],
          str(s.lines()[-1]))
    check('reconnect: no crash on cleared _prov_start', s.prov is None)


def test_reconnect_mid_interim_keeps_text():
    """Connection drops with interim text on screen — those words are real."""
    s = Sim()
    s.add_segment('committed sentence', True, 0, speech_final=True)
    s.add_segment('half spoken thou', False, 0)
    s.reset_speakers()
    s.add_segment('new session', True, 0, speech_final=True)
    check('reconnect mid-interim: earlier text survives',
          'committed sentence' in s.doc and 'half spoken thou' in s.doc, s.doc)


def test_empty_final_preserves_interim():
    """Deepgram sends empty finals; they must not wipe the interim."""
    s = Sim()
    s.add_segment('nearly all of it', False, 0)
    s.add_segment('', True, None, speech_final=True)
    check('empty final: interim text preserved', 'nearly all of it' in s.doc, s.doc)
    check('empty final: region committed', s.prov is None)


def test_offline_interleave_does_not_wipe():
    """Online dies mid-interim, offline takes over, online returns.

    Regression: add_text left _prov_start set, so the next interim deleted from
    that stale position to the end — silently wiping the whole offline session.
    """
    s = Sim()
    s.add_segment('online sentence', True, 0, speech_final=True)
    s.add_segment('interrupted mid ut', False, 0)      # connection dies here
    s.add_text('offline caption one\n')                # fallback engine
    s.add_text('offline caption two\n')
    s.reset_speakers()                                 # online comes back
    s.add_segment('back online', True, 0, speech_final=True)
    for expected in ['online sentence', 'offline caption one',
                     'offline caption two', 'back online']:
        check(f'offline interleave: kept "{expected}"', expected in s.doc, s.doc)


def test_add_text_commits_provisional():
    """add_text must clear _prov_start, independently of any reconnect.

    Isolates the fix: the interleave test above is masked by reset_speakers,
    but a trim during a long offline session would shift every position and
    leave a stale _prov_start pointing into unrelated text.
    """
    s = Sim()
    s.add_segment('interim left hanging', False, 0)
    check('add_text: provisional region is open beforehand', s.prov is not None)
    s.add_text('offline caption\n')
    check('add_text: provisional region committed', s.prov is None,
          f'prov={s.prov}')


def test_long_gap_breaks_without_speech_final():
    """Backstop for providers or sessions where speech_final never arrives."""
    s = Sim()
    s.add_segment('first thought', True, 0, now=100.0)
    s.add_segment('much later', True, 0, now=140.0)
    check('long gap: breaks without speech_final', len(s.lines()) == 2,
          str(s.lines()))


def test_greyscale_structure_survives():
    """Colour is unavailable in greyscale — markers and breaks must carry it."""
    s = Sim()
    s.add_segment('turn one', True, 0, speech_final=True)
    s.add_segment('turn two', True, 1, speech_final=True)
    s.add_segment('turn three', True, 2, speech_final=True)
    marked = [line for _, line in s.lines() if line.startswith(MARK)]
    check('greyscale: turn boundaries readable from markers alone',
          len(marked) == 3, str(s.lines()))
    check('greyscale: turns separated by blank line', '\n\n' in s.doc)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        print(f'\n{t.__name__}')
        summary = (t.__doc__ or '').strip().splitlines()
        if summary:
            print(f'  {summary[0]}')
        t()

    print('\n' + '=' * 60)
    print('SAMPLE RENDER — three people alternating in the room')
    print('=' * 60)
    s = Sim()
    for text, is_final, spk, sf in [
        ('so I',                        False, 0, False),
        ('so I said',                   False, 0, False),
        ('so I said to him',            True,  0, True),
        ('that appointment was',        False, 0, False),
        ('that appointment was Tuesday', True, 0, True),
        ('no they',                     False, 1, False),
        ('no they moved',               False, 1, False),
        ('no they moved it',            True,  1, True),
        ("she's",                       False, 1, False),
        ("she's right I took",          False, 2, False),
        ("she's right, I took the call", True, 2, True),
        ('well nobody',                 False, 0, False),
        ('well nobody told me any of this', True, 0, True),
    ]:
        s.add_segment(text, is_final, spk, speech_final=sf)
    print(s.render())

    print('\n' + '=' * 60)
    if FAILURES:
        print(f'{len(FAILURES)} CHECK(S) FAILED:')
        for f in FAILURES:
            print(f'  - {f}')
        return 1
    print('ALL CHECKS PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
