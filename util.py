import Imath
import OpenEXR
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
def exr2jpg(exrfile, jpgfile):
    file = OpenEXR.InputFile(exrfile)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    dw = file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    rgbf = [Image.frombytes("F", size, file.channel(c, pt)) for c in "RGB"]

    extrema = [im.getextrema() for im in rgbf]
    darkest = min([lo for (lo,hi) in extrema])
    lighest = max([hi for (lo,hi) in extrema])
    print("darkest:", darkest, "lighest:", lighest)
    scale = 255 / (lighest - darkest)
    def normalize_0_255(v):
        return (v * scale) + darkest
    rgb8 = [im.point(normalize_0_255).convert("L") for im in rgbf]
    Image.merge("RGB", rgb8).save(jpgfile)

def plot_light_groundtruth(light_groundtruth, dir):
    angle = np.linspace(0, 2 * np.pi, 150)
    radius = 1
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    figure, axes = plt.subplots(1)

    axes.plot(x, y)
    axes.set_aspect(1)

    axes.scatter(light_groundtruth[:, 0], light_groundtruth[:, 1])

    plt.title('Light Ground Truth')
    plt.savefig(dir+"/light_groundtruth.png")