class Target:
    """
    This is for the target marker and the id of the person to be tracked.
    
    Attributes
    ----------
    marker: str
        class of the target marker
    track_id: str
        the person who has the target marker
    changed: boolean
        whether the target marker changed in the previous frame
        or whether the track id is lost in the previous frame
    """
    def __init__(self, marker):
        self.marker = marker
        self.track_id = None
        self.changed = True
        print("marker class: ", self.marker)
        
    def set_target(self, marker):
        self.marker = marker
        self.changed = True
        print("new marker class: ", self.marker)
    
    def set_track_id(self, track_id):
        self.track_id = track_id
        self.changed = False