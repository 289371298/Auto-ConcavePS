# the camera is at (0, 0, -800) looking at (0, 0, 0)
import numpy as np
import open3d as o3d
import argparse
N = 1000 # number of points
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, default="null", help='name of subfolder')
    args = parser.parse_args()
    return args
def f(x, y):
    return -(x * x + y * y) / 300 + np.random.normal() * 10
if __name__ == "__main__":
     args = parse()
     

"""
     cloud = pv.PolyData(coord)
     # cloud.plot()
     volume = cloud.delaunay_3d(alpha=20)
     shell = volume.extract_geometry()
     shell.plot()
     pv.save_meshio("shell.obj", shell, 'obj')
"""