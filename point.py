class Point(object):
    def __init__(self, loc, global_map):
        self.frames = []
        self.location = loc
        self.idx = []
        self.id = len(global_map.points)

    def add_obervation(self, frame, idx):
        self.frames.append(frame)
        self.idx.append(idx)
