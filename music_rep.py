import music21
from jazzgen_exceptions import JazzGenException

class MusicInstance:
    """
    A MusicInstance object is a representation of the music happening in some fixed interval of time.
    I.e. one object corresponds to one quarter note unit of time (the exact unit depends on interval specified).
    """
    
    def __init__(self, interval):
        self.objs = frozenset()
        self.interval = interval
    
    def add_m21_obj(self, m21_obj, attack: bool):
        internal_obj = self.m21_to_internal(m21_obj, attack)
        if internal_obj:
            self.objs |= {internal_obj}
    
    def m21_to_internal(self, m21_obj, attack: bool):
        if isinstance(m21_obj, music21.note.Note):
            # return m21_obj.pitch.midi, 1
            return ( (m21_obj.pitch.midi,), attack )
        elif isinstance(m21_obj, music21.chord.Chord):
            # print(m21_obj.quality)
            # return m21_obj.root().midi, m21_obj.multisetCardinality
            return ( tuple(p.midi for p in m21_obj.pitches), attack )
        elif m21_obj is None:
            # return None, 0
            return tuple()
        else:
            raise TypeError("Foreign music21 object, cannot make rep")

    def internal_to_m21(self, internal_obj, obj_duration):
        pitches, _ = internal_obj
        if len(pitches) == 1:
            pitch = pitches[0]
            m21_obj = music21.note.Note(pitch)
        else:
            m21_obj = music21.chord.Chord(pitches)
        m21_obj.duration = music21.duration.Duration(obj_duration)
        return m21_obj

    def to_m21_objs(self, obj_duration):
        m21_objs = []
        for internal_obj in self.objs:
            m21_obj = self.internal_to_m21(internal_obj=internal_obj, obj_duration=obj_duration)
            m21_objs.append(m21_obj)
        return m21_objs

    def get_attack_objs(self):
        return frozenset({ pitches for pitches, attack in self.objs if attack })
    
    def get_sustain_objs(self):
        return frozenset({ pitches for pitches, attack in self.objs if not attack })

    def __hash__(self) -> int:
        return hash(self.objs)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, MusicInstance):
            return False
        if self.interval != other.interval:
            raise JazzGenException("MusicInstance objects with different intervals. Should not ever happen in the same training set.")
        return self.objs == other.objs

    def __repr__(self) -> str:
        return str(self.objs)

def make_m21_stream(music_rep_seq):
    if not music_rep_seq:
        raise JazzGenException("Cannot make stream from empty music_rep_seq")
    interval = music_rep_seq[0].interval

    song_stream = music21.stream.Stream()
    for ix, mi1 in enumerate(music_rep_seq):
        offset = ix * interval

        mi1_attack_objs = mi1.get_attack_objs()
        for attack_obj in mi1_attack_objs:
            full_attack_obj = (attack_obj, True)
            mi1.objs = frozenset({ obj for obj in mi1.objs if obj != full_attack_obj })
            
            ct = 1
            for mi2 in music_rep_seq[ix + 1:]:
                mi2_sustain_objs = mi2.get_sustain_objs()
                if attack_obj not in mi2_sustain_objs:
                    break
                full_sustain_obj = (attack_obj, False)
                mi2.objs = frozenset({ obj for obj in mi2.objs if obj != full_sustain_obj })
                ct += 1
            
            obj_duration = ct * interval
            m21_attack_obj = mi1.internal_to_m21(internal_obj=full_attack_obj, obj_duration=obj_duration)
            song_stream.insert(offset, m21_attack_obj)

        # remaining_m21_objs = mi1.to_m21_objs(obj_duration=interval)
        # for obj in remaining_m21_objs:
        #     song_stream.insert(offset, obj)
    
    return song_stream