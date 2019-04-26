class Point(object):
    def __init__(self, loc, global_map):
        self.frames = []
        self.location = loc
        self.idx = []
        self.id = len(global_map.points)
        global_map.points.append(self)

    def add_obervation(self, frame, idx):
        frame.points[idx] = self
        self.frames.append(frame)
        self.idx.append(idx)
