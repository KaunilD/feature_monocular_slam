import OpenGL.GL as gl
from multiprocessing import Process, Queue
import pangolin
import numpy as np
class Map(object):
    def __init__(self, W, H):
        self.points = []
        self.frames = []
        self.W = W
        self.H = H
        self.state = None
        self.q = None


    def create_display(self):
        self.q = Queue()
        self.p = Process(target = self.viewer_thread, args = (self.q,))
        self.p.daemon = True
        self.p.start()

    def viewer_thread(self, q):
        self.viewer_init(self.W*2, self.H*2)
        while True:
            self.viewer_refresh(q)

    def viewer_init(self, w, h):
        pangolin.CreateWindowAndBind('3D', w, h)
        gl.glEnable(gl.GL_DEPTH_TEST)

        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(w, h, 420, 420, w//2, h//2, 0.1, 10000),
            pangolin.ModelViewLookAt(0, -10, -20, 0, 0, 0, 0, -1, 0)
        )

        self.handler = pangolin.Handler3D(self.scam)

        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -w/h)
        self.dcam.SetHandler(self.handler)

    def viewer_refresh(self, q):
        if self.state is None or not q.empty():
            self.state = q.get()

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        self.dcam.Activate(self.scam)


        gl.glColor3f(1.0, 0.0, 0.0)
        pangolin.DrawCameras(self.state[0])

        gl.glPointSize(2.0)
        gl.glColor3f(0.0, 1.0, 0.0)
        if len(np.array(self.state[1]).shape) == 2:
            pangolin.DrawPoints(self.state[1])

        pangolin.FinishFrame()

    def display(self):

        if self.q is None:
            return

        poses, pts = [], []

        for f in self.frames:
            poses.append(f.pose)
        for p in self.points:
            pts.append(p.location)
        self.q.put((poses, pts))
