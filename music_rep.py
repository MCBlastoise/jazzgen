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
    
    def add_m21_obj(self, m21_obj):
        internal_obj = self.m21_to_internal(m21_obj)
        self.objs |= {internal_obj}
    
    def m21_to_internal(self, m21_obj):
        if isinstance(m21_obj, music21.note.Note):
            return m21_obj.pitch.midi, 1
        elif isinstance(m21_obj, music21.chord.Chord):
            return m21_obj.root().midi, m21_obj.multisetCardinality
        elif m21_obj is None:
            return None, 0
        else:
            raise TypeError("Foreign music21 object, cannot make rep")

    def to_m21_objs(self):
        raise NotImplementedError
    
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
    raise NotImplementedError