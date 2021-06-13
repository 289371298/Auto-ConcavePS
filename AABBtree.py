import numpy as np
from sympy import Line, Point, Triangle, Polygon, Plane, Point3D
from quickselect import floyd_rivest
class Triangle0:
    def __init__(self, p1, p2, p3):
        self.p1, self.p2, self.p3 = p1, p2, p3  # each is a 3-dimensional point.
        self.center = (self.p1 + self.p2 + self.p3) / 3
        self.normal = Plane(Point3D(p1), Point3D(p2), Point3D(p3)).normal_vector# calculate normal of this
        self.normal = np.array([self.normal[0].p / self.normal[0].q, self.normal[1].p / self.normal[1].q, self.normal[2].p / self.normal[2].q])
        self.normal = self.normal / np.linalg.norm(self.normal)

    def intersect(self, light): # given two points of a ray, test if this triangle intersects with others. if so, return the intersecting point; otherwise return None.
        direction = light[1] - light[0]
        normalT = self.normal.reshape(1, -1)
        if abs(np.matmul(normalT, direction.reshape(-1, 1))) < 1e-8: return None
        b = np.matmul(normalT, self.p2.reshape(-1, 1))
        # normal^T (light[0] + direction * lambda) = b; light[0] + direction * lambda is the intersection point
        lmbda = (b - np.matmul(normalT, light[0].reshape(-1, 1))) / np.matmul(normalT, direction)
        X = light[0] + direction * lmbda # X是与平面交点
        A = np.array([[self.p3[0] - self.p1[0], self.p2[0] - self.p1[0]], [self.p3[1] - self.p1[1], self.p2[1] - self.p1[1]], [self.p3[2] - self.p1[2], self.p2[2] - self.p1[2]]])
        coeff = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), (X.reshape(-1) - self.p1).reshape(-1, 1))
        if coeff[0] > 1e-8 and coeff[1] > 1e-8 and 1 - coeff[0] - coeff[1] > 1e-8: return X.reshape(-1)
        else: return None

def intersection_box(mn, mx, light):
    direction = light[1] - light[0]
    debug_flag = abs(light[1][0] + 9.66795935) < 1e-4 and abs(light[1][1] + 9.66795935) < 1e-4
    if debug_flag:
        print("mn:", mn, "mx:", mx, "light:", light, "direction:", direction)
    if direction[0] != 0:
        # x = mn[0]
        y3, z3 = light[0][1] + (mn[0] - light[0][0]) / direction[0] * direction[1], light[0][2] + (mn[0] - light[0][0]) / direction[0] * direction[2]
        if debug_flag: print("x3:", mn[0], "y3:", y3, "z3:", z3)
        if y3 >= mn[1] and y3 <= mx[1] and z3 >= mn[2] and z3 <= mx[2]: return True
        # x = mx[0]
        y3, z3 = light[0][1] + (mx[0] - light[0][0]) / direction[0] * direction[1], light[0][2] + (mx[0] - light[0][0]) / direction[0] * direction[2]
        if debug_flag: print("x3:", mx[0], "y3:", y3, "z3:", z3)
        if y3 >= mn[1] and y3 <= mx[1] and z3 >= mn[2] and z3 <= mx[2]: return True
    if direction[1] != 0:
        # y = mn[1]
        x3, z3 = light[0][0] + (mn[1] - light[0][1]) / direction[1] * direction[0], light[0][2] + (mn[1] - light[0][1]) / direction[1] * direction[2]
        if debug_flag: print("x3:", x3, "y3:", mn[1], "z3:", z3)
        if x3 >= mn[0] and x3 <= mx[0] and z3 >= mn[2] and z3 <= mx[2]: return True
        # y = mx[1]
        x3, z3 = light[0][0] + (mx[1] - light[0][1]) / direction[1] * direction[0], light[0][2] + (mx[1] - light[0][1]) / direction[1] * direction[2]
        if debug_flag: print("x3:", x3, "y3:", mx[1], "z3:", z3)
        if x3 >= mn[0] and x3 <= mx[0] and z3 >= mn[2] and z3 <= mx[2]: return True
    if direction[2] != 0:
        # z = mn[2]
        x3, y3 = light[0][0] + (mn[2] - light[0][2]) / direction[2] * direction[0], light[0][1] + (mn[2] - light[0][2]) / direction[2] * direction[1]
        if debug_flag: print("x3:", x3, "y3:", y3, "z3:", mn[2])
        if x3 >= mn[0] and x3 <= mx[0] and y3 >= mn[1] and y3 <= mx[1]: return True
        # z = mx[2]
        x3, y3 = light[0][0] + (mx[2] - light[0][2]) / direction[2] * direction[0], light[0][1] + (mx[2] - light[0][1]) / direction[2] * direction[1]
        if debug_flag: print("x3:", x3, "y3:", y3, "z3:", mx[2])
        if x3 >= mn[0] and x3 <= mx[0] and y3 >= mn[1] and y3 <= mx[1]: return True
    return False

class AABBtree_node: # a 3-dimensional AABB tree.
    cnt = 0
    def __init__(self, mn, mx, triangles=None, is_leaf=False):
        self.mn, self.mx = mn, mx
        self.idx = AABBtree_node.cnt
        AABBtree_node.cnt += 1
        assert len(mn) == 3, "mn Dimension Error!"
        self.is_leaf = is_leaf
        self.left_son, self.right_son = None, None
        self.triangles = triangles # the triangles

    def intersect(self, light): # given two points of a ray, test if the box intersects with others.
        return intersection_box(self.mn, self.mx, light)

    def debug_dfs(self):
        print("idx:", self.idx, end=" ")
        if self.left_son is not None:
            print("lson:", self.left_son.idx, end=" ")
        if self.right_son is not None:
            print("rson:", self.right_son.idx, end=" ")
        print("")
        if self.left_son is not None: self.left_son.debug_dfs()
        if self.right_son is not None: self.right_son.debug_dfs()

    def find_collision(self, light):
        # if abs(light[1][0] + 9.66795935) < 1e-4 and abs(light[1][1] + 9.66795935) < 1e-4:
        #     print("finding collision", self.idx, light)
        if not self.intersect(light): return (np.array([0, 0, 0]), None) # if not intersecting with the box
        if self.is_leaf:
            assert len(self.triangles) > 0, "Empty leaf!"
            mn_intersect, rec_normal = None, None
            # print("finding collision on leaf!", self.idx, "light:", light)
            for x in self.triangles:
                p0 = x.intersect(light)
                if p0 is None: continue
                else:
                    # print("intersection:", p0)
                    if mn_intersect is None or mn_intersect[2] > p0[2]:
                        mn_intersect, rec_normal = p0, x.normal
            if mn_intersect is None: return (np.array([0, 0, 0]), None)
            else: return (rec_normal, mn_intersect)
        v1, v2 = None, None
        if self.left_son is not None: v1 = self.left_son.find_collision(light)
        if self.right_son is not None: v2 = self.right_son.find_collision(light)
        # if abs(light[1][0] + 9.66795935) < 1e-4 and abs(light[1][1] + 9.66795935) < 1e-4:
        #    print("v1:", v1, "v2:", v2)
        if v1[1] is None and v2[1] is None: return (np.array([0, 0, 0]), None)
        elif v1[1] is None and v2[1] is not None: return v2
        elif v1[1] is not None and v2[1] is None: return v1
        else: # return the point nearest to (0, 0, -800); the manifold has z coord > -800.
            if v1[1][2] < v2[1][2]: return v1
            else: return v2

def Build_AABBtree(triangles):
    # print("Building AABBtree...", len(triangles))
    coords = np.concatenate([x.center.reshape(-1, 1) for x in triangles], axis=1)
    allvertices = np.concatenate([np.concatenate([x.p1.reshape(-1, 1), x.p2.reshape(-1, 1), x.p3.reshape(-1, 1)], axis=1) for x in triangles], axis=1)
    mx, mn = np.max(allvertices, axis=1), np.min(allvertices, axis=1)
    # print(coords.shape)
    assert len(mn) == 3, "AABBtree Error!"
    delta = mx - mn
    if len(triangles) < 5: # by brute force.
        node = AABBtree_node(mn, mx, triangles, is_leaf=True)
        return node
    # build an AABBtree node; return the root node.
    ax = None
    if delta[0] >= delta[1] and delta[0] >= delta[2]: ax = 0
    elif delta[1] >= delta[0] and delta[1] >= delta[2]: ax = 1
    else: ax = 2
    # partitioning the triangles.
    coord_on_axis = [x.center[ax] for x in triangles]
    mid_rank = (len(triangles) + 1) // 2
    # print("midrank:", mid_rank)
    # print("ax:", ax, "coord_on_axis:", coord_on_axis)
    mid = floyd_rivest.nth_smallest(coord_on_axis, mid_rank)
    left, right = [], []
    for x in triangles:
        if x.center[ax] < mid: left.append(x)
        else: right.append(x)
    # print("lenleft:", len(left), "lenright:",  len(right))
    node = AABBtree_node(mn, mx)
    node.left_son = Build_AABBtree(left)
    node.right_son = Build_AABBtree(right)
    return node