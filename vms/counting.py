class LineCounter:
    """Counts unique tracked people crossing a horizontal virtual line."""
    def __init__(self, line_position=0.65, deadband=8):
        self.line_position=float(line_position)
        self.deadband=int(deadband)
        self.previous={}
        self.counted=set()
        self.in_count=0
        self.out_count=0

    def update(self, track_id, center_y, frame_height):
        line_y=int(frame_height*self.line_position)
        side = -1 if center_y < line_y-self.deadband else (1 if center_y > line_y+self.deadband else 0)
        old=self.previous.get(int(track_id),0)
        event=None
        if side and old and side != old and int(track_id) not in self.counted:
            # Moving from below the line to above = IN; opposite = OUT.
            event="IN" if old==1 and side==-1 else "OUT"
            if event=="IN": self.in_count+=1
            else: self.out_count+=1
            self.counted.add(int(track_id))
        if side:
            self.previous[int(track_id)]=side
        return event

    @property
    def total(self):
        return self.in_count-self.out_count
