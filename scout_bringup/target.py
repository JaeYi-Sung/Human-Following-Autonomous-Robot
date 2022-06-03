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
    lost_track_id: boolean
        whether the track id is lost in the previous frame
    """
    def __init__(self, marker, track_id = None, changed = True, lost_track_id = True):
        self.marker = marker
        self.track_id = track_id
        self.changed = changed
        self.lost_track_id = lost_track_id
        print("marker class: ", self.marker)
        
    def set_target(self, marker):
        self.marker = marker
        self.changed = True
        print("new marker class: ", self.marker)
    
    def set_track_id(self, track_id):
        self.track_id = track_id
        self.changed = False
        self.lost_track_id = False