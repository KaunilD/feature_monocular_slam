import OpenGL.GL as gl
from multiprocessing import Process, Queue
import pangolin
import numpy as np
import g2o
from feature_extractor import poseRt

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

        print(self.state[1])


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

    def optimize(self):
        opt = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        opt.set_algorithm(solver)

        robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))

        for fdx, frame in enumerate(self.frames):
            pose = frame.pose

            sbacam = g2o.SBACam(g2o.SE3Quat(pose[0:3, 0:3], pose[0:3, 3]))
            sbacam.set_cam(frame.K[0][0], frame.K[1][1], frame.K[0][2], frame.K[1][2], 1.0)

            v_se3 = g2o.VertexCam()
            v_se3.set_id(frame.id)
            v_se3.set_estimate(sbacam)
            v_se3.set_fixed(frame.id <= 1)
            opt.add_vertex(v_se3)

        PT_ID_OFFSET = 0x10000

        for pdx, point in enumerate(self.points):
            pt = g2o.VertexSBAPointXYZ()
            pt.set_id(point.id + PT_ID_OFFSET)
            pt.set_estimate(point.location[0:3])
            pt.set_marginalized(True)
            pt.set_fixed(False)
            opt.add_vertex(pt)

            for f in point.frames:
                edge = g2o.EdgeProjectP2MC()
                edge.set_vertex(0, pt)
                edge.set_vertex(1, opt.vertex(f.id))
                uv = f.key_points_unsorted[f.points.index(point)]
                edge.set_measurement(uv)
                edge.set_information(np.eye(2))
                edge.set_robust_kernel(robust_kernel)
                opt.add_edge(edge)

        opt.set_verbose(True)
        opt.initialize_optimization()
        opt.optimize(50)

        for frame in self.frames:
            est = opt.vertex(frame.id).estimate()
            R = est.rotation().matrix()
            t = est.translation()
            frame.pose = poseRt(R, t)

        # put points back
        for point in self.points:
            est = opt.vertex(point.id + PT_ID_OFFSET).estimate()
            point.location = np.array(est)
