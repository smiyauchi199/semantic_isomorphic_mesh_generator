import numpy as np
import plyfile


def read_from_obj(filename):
    """load model data from .obj file"""
    v_list = []  # vertex coordinate
    vt_list = []  # vertex texture coordinate
    vc_list = []  # vertex color
    vn_list = []  # vertex normal
    f_list = []  # face vertex indices
    fn_list = []  # face normal indices
    ft_list = []  # face texture indices

    # read data
    fp = open(filename, 'r')
    lines = fp.readlines()
    fp.close()

    for line in lines:
        if len(line) < 2:
            continue
        line_data = line.strip().split(' ')
        line_data = list(filter(lambda x: len(x) > 0, line_data))
        # parse vertex cocordinate
        if line_data[0] == 'v':
            v_list.append((float(line_data[1]), float(line_data[2]), float(line_data[3])))
            if len(line_data) == 7:
                vc_list.append((float(line_data[4]), float(line_data[5]), float(line_data[6])))

        # parse vertex texture coordinate
        if line_data[0] == 'vt':
            vt_list.append((float(line_data[1]), float(line_data[2])))

        # parse vertex normal
        if line_data[0] == 'vn':
            vn_list.append((float(line_data[1]), float(line_data[2]), float(line_data[3])))

        # parse face
        if line_data[0] == 'f':
            # used for parsing face element data
            def segElementData(ele_str):
                fv = None
                ft = None
                fn = None
                eles = ele_str.strip().split('/')
                if len(eles) == 1:
                    fv = int(eles[0]) - 1
                elif len(eles) == 2:
                    fv = int(eles[0]) - 1
                    ft = int(eles[1]) - 1
                elif len(eles) == 3:
                    fv = int(eles[0]) - 1
                    fn = int(eles[2]) - 1
                    ft = None if eles[1] == '' else int(eles[1]) - 1
                return fv, ft, fn

            fv0, ft0, fn0 = segElementData(line_data[1])
            fv1, ft1, fn1 = segElementData(line_data[2])
            fv2, ft2, fn2 = segElementData(line_data[3])
            f_list.append((fv0, fv1, fv2))
            if ft0 is not None and ft1 is not None and ft2 is not None:
                ft_list.append((ft0, ft1, ft2))
            if fn0 is not None and fn1 is not None and fn2 is not None:
                fn_list.append((fn0, fn1, fn2))

    v_list = np.asarray(v_list)
    vn_list = np.asarray(vn_list)
    vt_list = np.asarray(vt_list)
    vc_list = np.asarray(vc_list)
    f_list = np.asarray(f_list)
    ft_list = np.asarray(ft_list)
    fn_list = np.asarray(fn_list)

    model = {'v': v_list, 'vt': vt_list, 'vc': vc_list, 'vn': vn_list,
             'f': f_list, 'ft': ft_list, 'fn': fn_list}
    return model


def write_to_obj(filename, model):
    assert 'v' in model and model['v'].size != 0

    with open(filename, 'w') as fp:
        if 'v' in model and model['v'].size != 0:
            if 'vc' in model and model['vc'].size != 0:
                for v, vc in zip(model['v'], model['vc']):
                    fp.write('v %f %f %f %f %f %f\n' % (v[0], v[1], v[2], vc[0], vc[1], vc[2]))
            else:
                for v in model['v']:
                    fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        if 'vn' in model and model['vn'].size != 0:
            for vn in model['vn']:
                fp.write('vn %f %f %f\n' % (vn[0], vn[1], vn[2]))

        if 'vt' in model and model['vt'].size != 0:
            for vt in model['vt']:
                fp.write('vt %f %f\n' % (vt[0], vt[1]))

        if 'f' in model and model['f'].size != 0:
            if 'fn' in model and model['fn'].size != 0 and 'ft' in model and model['ft'].size != 0:
                assert model['f'].size == model['fn'].size
                assert model['f'].size == model['ft'].size
                for f_, ft_, fn_ in zip(model['f'], model['ft'], model['fn']):
                    f = np.copy(f_) + 1
                    ft = np.copy(ft_) + 1
                    fn = np.copy(fn_) + 1
                    fp.write('f %d/%d/%d %d/%d/%d %d/%d/%d\n' %
                             (f[0], ft[0], fn[0], f[1], ft[1], fn[1], f[2], ft[2], fn[2]))
            elif 'fn' in model and model['fn'].size != 0:
                assert model['f'].size == model['fn'].size
                for f_, fn_ in zip(model['f'], model['fn']):
                    f = np.copy(f_) + 1
                    fn = np.copy(fn_) + 1
                    fp.write('f %d//%d %d//%d %d//%d\n' % (f[0], fn[0], f[1], fn[1], f[2], fn[2]))
            elif 'ft' in model and model['ft'].size != 0:
                assert model['f'].size == model['ft'].size
                for f_, ft_ in zip(model['f'], model['ft']):
                    f = np.copy(f_) + 1
                    ft = np.copy(ft_) + 1
                    fp.write('f %d/%d %d/%d %d/%d\n' % (f[0], ft[0], f[1], ft[1], f[2], ft[2]))
            else:
                for f_ in model['f']:
                    f = np.copy(f_) + 1
                    fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))


def read_from_plyfile(filename):
    # use plyfile

    plydata = plyfile.PlyData.read(filename)
    points = []  # plydata.elements[0]
    faces = [] #plydata.elements[1]
    for i in range(plydata.elements[0].count):
        v = plydata.elements[0][i]
        points.append(np.array((v[0], v[1], v[2])))
    for i in range(plydata.elements[1].count):
        f = plydata.elements[1][i]
        faces.append(np.array([f[0], f[1], f[2]]))
    points = np.asarray(points)
    faces = np.asarray(faces)

    return points, faces


def write_to_ply(filename, xyz_points, faces=None, verts_normals=None, rgb_points=None, rgb_faces=None):
    "write ply file"

    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8)*169

    if rgb_faces is None and faces is not None:
        rgb_faces = np.ones(faces.shape).astype(np.uint8)*169

    fout = open(filename, 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex " + str(xyz_points.shape[0]) + "\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    if verts_normals is not None:
        fout.write("property float nx\n")
        fout.write("property float ny\n")
        fout.write("property float nz\n")

    fout.write("property uchar red\n")
    fout.write("property uchar green\n")
    fout.write("property uchar blue\n")
    if faces is not None:
        fout.write("element face " + str(len(faces)) + "\n")
        fout.write("property uchar red\n")
        fout.write("property uchar green\n")
        fout.write("property uchar blue\n")
        fout.write("property list uchar int vertex_index\n")
    fout.write("end_header\n")

    if verts_normals is not None:
        for i in range(xyz_points.shape[0]):
            color = rgb_points[i]
            color = str(color[0]) + ' ' + \
                str(color[1]) + ' ' + str(color[2])
            fout.write(str(xyz_points[i, 0]) + " " + str(xyz_points[i, 1]) + " " + str(
                xyz_points[i, 2]) + " " + str(verts_normals[i, 0]) + " " + str(
                    verts_normals[i, 1]) + " " + str(verts_normals[i, 2]) + " " + color + "\n")
    else:
        for i in range(xyz_points.shape[0]):
            color = rgb_points[i]
            color = str(color[0]) + ' ' + \
                str(color[1]) + ' ' + str(color[2])
            fout.write(str(xyz_points[i, 0]) + " " + str(xyz_points[i, 1]) + " " + str(
                xyz_points[i, 2]) + " " + color + "\n")
    if faces is not None:
        for i in range(len(faces)):
            color = rgb_faces[i]
            color = str(color[0]) + ' ' + \
                str(color[1]) + ' ' + str(color[2])
            fout.write(color + " 3 " + str(faces[i, 0]) + " " +
                       str(faces[i, 1]) + " " + str(faces[i, 2]) + "\n")

    fout.close()


def write_to_plyfile(filename, verts, faces):
    # use plyfile

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(verts[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")
    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(filename)


def read_from_ori(filename):

    def read_line(line):
        line = line.rstrip().split(' ')
        return np.asarray(line[0]), np.asarray(line[1]), np.asarray(line[2])

    with open(filename, 'r') as f:

        num_vtx = f.readline()

        vertices = []
        for i in range(int(num_vtx)):
            line = f.readline()
            line = read_line(line)
            vertices.append(line)
        vertices = np.asarray(vertices)

        num_faces = f.readline()

        faces = []
        normals = []
        for i in range(int(num_faces)):
            line = f.readline()
            line = read_line(line)
            faces.append(line)

            line = f.readline()
            line = read_line(line)
            normals.append(line)

            f.readline()

        faces = np.asarray(faces)
        normals = np.asarray(normals)

    return vertices, faces, normals

