from model.sort import *

class Tracker:
    def __init__(self):

        #Creates a SORT instance
        self.mot_tracker = Sort()

    def apply_tracker(self, detections):
        #Track detections
        track_bbs_ids = self.mot_tracker.update(np.asarray(detections))
        
        return track_bbs_ids