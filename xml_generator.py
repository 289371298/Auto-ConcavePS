import argparse
import time
import os
import numpy as np
import math
import open3d as o3d
import cv2
from compphotofinal.texture_generator import make_texture
import scipy.spatial
from compphotofinal.util import *
from compphotofinal.AABBtree import *
from tqdm import tqdm
import matplotlib.pyplot as plt
UNIT_RADIUS = 30
BRDF_PER_OBJECT = 8
RETAIN_EXR = False
# 512 objects * 8 BRDF per object * 128 pictures per (object, BRDF) pair

SIZE = (128, 128)
N, M = 775, 5 # note: should be divisible by M(M+1)/2. n is #points, m is #layer.

FOV = 40 # field of view
# calculate screen location; for normal vector calculation
actual_x, actual_y = 800 * math.tan((FOV / 2) / 90 * math.pi / 2), 800 * math.tan((FOV / 2) / 90 * math.pi / 2)  # actual maximum/minimum coordinate on the screen.
screendist_x, screendist_y = (SIZE[0] / 2) / actual_x, (SIZE[1] / 2) / actual_y  # the camera is at (0, 0, -800).

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_data', type=int, default=4096, help='number of data points')
    parser.add_argument('-m', '--num_photo', type=int, default=128, help='number of photos')
    parser.add_argument('-d', '--max_depth', type=int, default=10, help='max depth')
    parser.add_argument('-s', '--seed', type=int,default=1588, help='seed')
    args = parser.parse_args()
    return args

def generate_mesh(instance_name, get_normal=True):
    os.system("del cbox/meshes/mesh.obj")
    old = [i * N // (M * (M + 1) // 2)  for i in range(1, M+1)]
    old_sum = [(i - 1) * i * N // (M * (M + 1)) for i in range(1, M+1)]
    centers = (np.random.random(size=(3, 2)) - 0.5) * 150 # np.array([[0, 0]])
    def f(x, y):
        g = 0
        for i in range(centers.shape[0]):
            """
            if (x - centers[i, 0]) ** 2 +  (y - centers[i, 1]) ** 2 < 4900:
                val = math.sqrt((x - centers[i, 0]) ** 2 +  (y - centers[i, 1]) ** 2)) / 70
                g -= (1 - math.sin(math.pi / 2 + math.pi * val)) * 60
            """
            g -= ((x - centers[i, 0]) ** 2 + (y - centers[i, 1] ) ** 2) / 400
        return g + np.random.normal() * 5
    # points = (np.random.random((N, 2)) - 0.5) * 300  # the (x, y) coordinate
    points = np.zeros((N + 1, 2))
    for j in range(len(old)):
        r = (j + 1) * UNIT_RADIUS
        dir = np.random.random(size=(old[j], 2)) - 0.5
        for i in range(old[j]):
            dir[i, :] /= np.linalg.norm(dir[i, :])
            points[old_sum[j] + i, :] = dir[i, :] * r
    points[N, 0], points[N, 1] = 0, 0
    coord = np.zeros((N + 1, 3))
    coord[:, :2] = points
    for i in range(N):
        coord[i, 2] = f(points[i, 0], points[i, 1])
    coord[N, 2] = -11
    print("checkpoint 0")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    alpha = 120

    # use this with ball pivoting / poisson.
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=50, max_nn=30))
    normals = np.asarray(pcd.normals)
    print(normals)
    for i in range(normals.shape[0]):
        if normals[i, 2] < 0: normals[i, :] = -normals[i, :]
    pcd.normals = o3d.utility.Vector3dVector(normals)
    ###################

    radii = [10 * i for i in range(1, 30)] # for ball pivoting.
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=2)
    mesh.compute_vertex_normals()
    """# use this with alpha_shape.
    normals = np.asarray(mesh.triangle_normals)
    print(normals)
    for i in range(normals.shape[0]):
        if normals[i, 2] < 0: normals[i, :] = -normals[i, :]
    mesh.triangle_normals = o3d.utility.Vector3dVector(normals)

    normals = np.asarray(mesh.vertex_normals)
    print(normals)
    for i in range(normals.shape[0]):
        if normals[i, 2] < 0: normals[i, :] = -normals[i, :]
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    ####################
    """
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    # o3d.visualization.draw_geometries([mesh])
    original_vertices = np.asarray(pcd.points)
    vertices = np.asarray(mesh.vertices)
    triangles_id = np.asarray(mesh.triangles)
    normal_pixelwise = np.zeros((SIZE[0], SIZE[1], 3))
    """
    for i in range(len(triangles_id[0])):
        id1, id2, id3 = triangles_id[i, 0], triangles_id[i, 1], triangles_id[i, 2] # the three vertices of triangle i
        tr1, tr2, tr3 = vertices[id1, :], vertices[id2, :], vertices[id3, :] # the three coordinates of vertices
        left = min(tr1[0], min(tr2[0], tr3[0]))
        up = min(tr1[1], min(tr2[1], tr3[1]))
        right = max(tr1[0], max(tr2[0], tr3[0]))
        down = max(tr1[1], max(tr2[1], tr3[1]))
    """
    o3d.io.write_triangle_mesh("cbox/meshes/mesh.obj", mesh)
    o3d.io.write_triangle_mesh(instance_name+"/mesh.obj", mesh)
    vertices_on_screen = np.zeros((vertices.shape[0], 2))
    original_vertices_on_screen = np.zeros((original_vertices.shape[0], 2))
    for i in range(original_vertices.shape[0]):
        original_vertices_on_screen[i, 0], original_vertices_on_screen[i, 1] = original_vertices[i, 0] * screendist_x, original_vertices[i, 1] * screendist_y
    for i in range(vertices.shape[0]):
        vertices_on_screen[i, 0], vertices_on_screen[i, 1] = vertices[i, 0] * screendist_x, vertices[i, 1] * screendist_y
    # deprecated - convex hull mask
    # contours = np.array((np.floor(original_vertices_on_screen[:, :2] + 0.5) + (SIZE[0] / 2) * np.ones_like(original_vertices_on_screen))).astype('int')
    # cv2.fillPoly(image, pts=[cv2.convexHull(contours)], color=(255, 255, 255))
    # cv2.ellipse(image, (0, 0), (np.random.random() * R / 2 + R / 2, np.random.random() * R / 2 + R / 2), 0, 360, (255, 255, 255))
    # print(mask_points.shape)
    # calculate pixel-wise normal vector.
    if get_normal:
        actual_light = np.zeros((SIZE[0], SIZE[1], 2))
        for i in range(SIZE[0]):
            for j in range(SIZE[1]):
                actual_light[i, j] = np.array([(i - SIZE[0] / 2 + 0.5) / screendist_x, (j - SIZE[1] / 2 + 0.5) / screendist_y])
        # light on screen location (i, j) will pass (0, 0, -800) and (actual_light[i, j, 0], actual_light[i, j, 1], 0)
        # building aabb-tree from triangles
        triangles = []
        for i in tqdm(range(triangles_id.shape[0])):
            triangles.append(Triangle0(vertices[triangles_id[i, 0]], vertices[triangles_id[i, 1]], vertices[triangles_id[i, 2]])) # 是normals[i]吗？
        root = Build_AABBtree(triangles)
        # root.debug_dfs()
        fl = open("backlog.txt","w") # for debugging.
        for i in tqdm(range(SIZE[0])):
            for j in range(SIZE[1]):
                v = root.find_collision((np.array([0, 0, -800]), np.array([actual_light[i, j, 0], actual_light[i, j, 1], 0])))[0].reshape(-1)
                # print(i, j, "normal vector =", v)
                fl.write(str(i)+" "+str(j)+" "+str(v)+"\n")
                fl.flush()
                if v[2] < 0: normal_pixelwise[i, j, :] = -v
                else: normal_pixelwise[i, j, :] = v

        fl.close()
        return normal_pixelwise
    else:
        return None

if __name__ == "__main__":
    # global mask_points
    args = parse()
    dataset_name = str(time.time())
    np.random.seed(args.seed)
    os.mkdir(dataset_name)
    obj_fd = open(dataset_name+"/objects.txt", "a")
    for i in tqdm(range(args.num_data)): # for each instance
        instance_name = dataset_name+"/"+str(i)
        obj_fd.write(str(i)+"\n")
        obj_fd.flush()
        os.mkdir(instance_name)
        g = open(instance_name+"/light_groundtruth.txt","w")
        # generate texture
        make_texture(instance_name)
        # generate ellipse for mask.png
        R = UNIT_RADIUS * M
        ELLIPSE_A, ELLIPSE_B = SIZE[0] / 6 * np.random.random() + SIZE[0] / 6, SIZE[1] / 6 * np.random.random() + SIZE[1] / 6
        print("EA:", ELLIPSE_A, "EB:", ELLIPSE_B)
        mask_points = np.zeros((150, 2))
        for j in range(mask_points.shape[0]):
            angle = 2 * math.pi * j / mask_points.shape[0]
            mask_points[j, 0], mask_points[j, 1] = ELLIPSE_A * math.sin(angle) + 2 * (np.random.random() - 0.5), ELLIPSE_B * math.cos(angle) + 2 * (np.random.random() - 0.5)
        mask_points[:, 0] += SIZE[0] / 2
        mask_points[:, 1] += SIZE[1] / 2
        # generate mask
        image = np.zeros((SIZE[0], SIZE[1])).astype('uint8')
        cv2.fillPoly(image, pts=np.array([mask_points]).astype('int'), color=(255, 255, 255))
        cv2.imwrite(instance_name + "/mask.png", image)
        # generate a highly concave object.
        # os.system("python3 mesh_generator.py -n="+instance_name) #这里最后不带反斜杠！
        if i % BRDF_PER_OBJECT == 0: # generate new mesh; 8 BRDF per object
            get_normal = True
            if get_normal:
                normal_map = generate_mesh(instance_name, get_normal)

                mask = (np.max(normal_map, axis=2) > 0).astype('int').reshape(SIZE[0], SIZE[1], 1)
                mask = np.repeat(mask, 3, axis=2)
                np.save(instance_name+'/normal', normal_map)

                color = mask * (normal_map + 1) * 0.5

                debug = open("debug_color.txt", "w")
                for l in range(3):
                    for k in tqdm(range(mask.shape[0])):
                        for j in range(mask.shape[1]):
                            debug.write(str(color[k, j, l]) + " ")
                        debug.write("\n")
                        debug.flush()
                    debug.write("\n")
                debug.close()

                cv2.imwrite(instance_name+"/normal.png", color * 255)
            else:
                generate_mesh(instance_name, get_normal=True)
        else: # inherit from old mesh
            print("skipping mesh generation process...")
            print("copy ", dataset_name + "/" + str(i - 1) + "/normal.npy "+dataset_name+"\\"+str(i))
            os.system("copy " + dataset_name + "\\" + str(i - 1) + "\\normal.npy "+dataset_name+"\\"+str(i))
            os.system("copy " + dataset_name + "\\" + str(i - 1) + "\\normal.png "+dataset_name+"\\"+str(i))
            os.system("copy " + dataset_name + "\\" + str(i - 1) + "\\mesh.obj "+dataset_name+"\\"+str(i))
        names_fd = open(instance_name+"/names.txt","a")
        light_groundtruth = np.zeros((args.num_photo, 3))
        for j in range(args.num_photo): # for each photo
            f = open(instance_name+"/"+str(j)+".xml","w")
            names_fd.write(str(j)+".jpg\n")
            names_fd.flush()
            x = 2 * (np.random.random() - 0.5) # (-0.5, 0.5)
            rest = math.sqrt(1 - x * x)
            y = rest * 2 * (np.random.random() - 0.5) # (-0.5, 0.5)
            z = -math.sqrt(1 - x * x - y * y) # to ensure the light is always behind the camera
            # print(x, y, z, x*x+y*y+z*z)
            g.write(str(-y)+" "+str(-x)+" "+str(-z)+"\n")
            light_groundtruth[j, 0], light_groundtruth[j, 1], light_groundtruth[j, 2] = y, x, z
            g.flush()
            #不开twosided渲染会导致中间有空隙，但是开了twosided好像又会造成一些奇怪的边缘问题
            # generate lighting: (x, y, z) is on a ball whose center is the original point and radius is 1700.
            f.write("\n\
<scene version=\"2.0.0\"> \n\
    <!-- <integrator type=\"direct\"/> --> \n\
    <integrator type=\"path\"> \n\
        <integer name=\"max_depth\" value=\""+str(args.max_depth)+"\"/> \n\
    </integrator> \n\
    <sensor type=\"perspective\"> \n\
        <string name=\"fov_axis\" value=\"smaller\"/> \n\
        <float name=\"near_clip\" value=\"10\"/> \n\
        <float name=\"far_clip\" value=\"1500\"/> \n\
        <float name=\"focus_distance\" value=\"800\"/> \n\
        <float name=\"fov\" value=\""+str(FOV)+"\"/> \n\
        <transform name=\"to_world\"> \n\
            <lookat origin=\"0, 0, -800\" \n\
                    target=\"0, 0, -799\" \n\
                    up    =\"  0,   1,    0\"/>\n\
        </transform>\n\
        <sampler type=\"independent\">  <!-- ldsampler -->\n\
            <integer name=\"sample_count\" value=\"128\"/>\n\
        </sampler>\n\
        <film type=\"hdrfilm\">\n\
            <integer name=\"width\" value=\""+str(SIZE[0])+"\"/> \n\
            <integer name=\"height\" value=\""+str(SIZE[1])+"\"/> \n\
            <rfilter type=\"gaussian\"/>\n\
        </film>\n\
        </sensor>\n\
        <bsdf type=\"diffuse\" id=\"red\">\n\
        <texture type=\"bitmap\" name=\"reflectance\">\n\
        <string name=\"filename\" value=\""+instance_name+"/texture.png\"/>\n\
        </texture>\n\
        </bsdf>\n\
        <bsdf type=\"diffuse\" id=\"light\">\n\
            <spectrum name=\"reflectance\" value= \"400:0.78, 500:0.78, 600:0.78, 700:0.78\"/>\n\
        </bsdf>\n\
        <bsdf type=\"roughconductor\" id=\"gold\">\n\
            <string name=\"material\" value=\"Al\"/>\n\
            <string name=\"distribution\" value=\"ggx\"/>\n\
            <float name=\"alpha\" value=\"0.1\"/>\n\
        </bsdf>\n\
        <shape type=\"obj\">\n\
        <string name=\"filename\" value=\"meshes/lightboard.obj\"/>\n\
        <transform name=\"to_world\">\n\
            <translate x=\""+str(x * 3400)+"\" y=\""+str(y * 3400)+"\" z=\""+str(z * 3400)+"\"/>\n\
            <lookat origin=\""+str(x * 3400)+","+str(y * 3400)+","+str(z * 3400)+"\" target=\"0, 0, 0\"/>\n\
        </transform>\n\
        <ref id=\"light\"/>\n\
        \n\
        <emitter type=\"area\">\n\
            <spectrum name=\"radiance\" value=\"400:200, 500:200, 600:200, 700:200\"/>\n\
        </emitter>\n\
    </shape>\n\
	<shape type=\"obj\">\n\
        <string name=\"filename\" value=\""+instance_name+"/mesh.obj"+"\"/>\n\
            <bsdf type=\"twosided\">\n\
                <ref id=\"red\"/>\n\
            </bsdf>\n\
    </shape>\n\
</scene>"
            )
            f.close()
            os.system("mitsuba "+instance_name+"/"+str(j)+".xml -a cbox -o "+instance_name+"/"+str(j)+".exr")
            # convert_exr_to_jpg(instance_name+"/"+str(j)+".exr", instance_name+"/"+str(j)+".jpg")
            # img = cv2.imread(instance_name+"/"+str(j)+".exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            while not os.path.exists(instance_name+"/"+str(j)+".exr"):
                pass # wait until file system completes the job; there might be a very small chance (<0.5%) of delay.
            exr2jpg(instance_name+"/"+str(j)+".exr", instance_name+"/"+str(j)+".jpg")
            print(dataset_name+"\\"+str(i)+"\\"+str(j)+".exr")
            if not RETAIN_EXR:
                os.system("del "+dataset_name+"\\"+str(i)+"\\"+str(j)+".exr") # to save space.
                os.system("del " + dataset_name + "\\" + str(i) + "\\" + str(j) + ".xml")  # to save space.
        names_fd.close()
        g.close()
        del g
        # plotting light groundtruth
        plot_light_groundtruth(light_groundtruth[:, :2], instance_name)
        """ exr转jpg的时候貌似出现了一些误差，导致外部也有一些被框进来的pixel。还是直接在generate_mesh里求凸包好些。
        g = open(instance_name+"/mask.txt","w") # 我们认为所有的图片加起来，是0的像素就mask。（反正所有图片都是0的像素也没法判断，有平面和没有没区别）
        tot = np.zeros((SIZE[0], SIZE[1], 3))
        for j in range(args.num_photo):
            img = cv2.imread(instance_name+"/"+str(j)+".jpg")
            tot += img
        t = 1 - ((tot.max(axis=2) > 0).astype('int'))
        for j in range(SIZE[0]):
            for k in range(SIZE[1]):
                g.write(str(t[j, k])+" ")
            g.write("\n")
        g.close()
        t *= 255
        cv2.imwrite(instance_name+"/mask.png", t)
        """
    obj_fd.close()