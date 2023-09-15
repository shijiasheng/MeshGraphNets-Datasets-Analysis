import json
import tensorflow as tf
import functools
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def map_color(values, vmin=None, vmax=None):
    colors = np.array([[0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0], [1, 0, 0]])
    positions = np.array([0, 0.25, 0.5, 0.75, 1, 1])

    if vmax is None:
        vmax = values.max()

    if vmin is None:
        vmin = values.min()

    values = np.maximum(values, vmin)
    values = np.minimum(values, vmax)

    vrange = vmax - vmin
    if vrange > 1e-10:
        values = values / vrange

    is_greater = np.ones(len(values), dtype='bool')
    start_index = np.zeros(len(values), dtype='int32')
    for i in range(5):
        is_greater = is_greater & (values >= positions[i])
        start_index[is_greater] = i
    end_index = start_index + 1
    start_color = colors[start_index]
    end_color = colors[end_index]

    t = np.tile(values - positions[start_index], (3, 1)).T * 4
    return (1 - t) * start_color + t * end_color


class Visualizer:
    def __init__(self):
        self.shapes = dict()
        self.cur_id = None
        self.cur_shape = None

        self.fig = None
        self.ax = None
        self.poly = None
        self.poly2 = None
        self.colorbar = None
        self.vmax = 0
        self.vmin = 0

        self.cmap = mpl.colors.LinearSegmentedColormap.from_list('mises',
                                                                 [[0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0],
                                                                  [1, 0, 0]])

    def load(self, elements, positions, id=None, displacement=None):
        if id is not None and self.shapes.get(id) is not None:
            self.cur_id = id
            self.cur_shape = self.shapes.get(id)
            self.init()
            return

        triangles = []
        for e in elements:
            if len(e) == 4:
                triangles.append(np.array(e)[[0, 1, 2]])
                triangles.append(np.array(e)[[0, 1, 3]])
                triangles.append(np.array(e)[[1, 2, 3]])
                triangles.append(np.array(e)[[0, 2, 3]])

        if displacement is not None:
            positions = positions + displacement

        verts = positions[triangles, :]

        vert_max = positions.max(axis=0)
        vert_min = positions.min(axis=0)
        size = len(verts)
        elements = triangles

        self.cur_shape = {'elements': elements, 'verts': verts, 'positions': positions,
                          'vert_max': vert_max,
                          'vert_min': vert_min,
                          'size': size}
        self.init()

        if id is not None:
            self.cur_id = id
            self.shapes[id] = self.cur_shape

    def init(self):
        verts = self.cur_shape['verts']
        vert_max = self.cur_shape['vert_max']
        vert_min = self.cur_shape['vert_min']
        axis_ranges = vert_max - vert_min

        self.poly = Poly3DCollection(verts, facecolors='b', edgecolors='black', linewidths=2, alpha=1,
                                     linestyles='solid')

        self.fig = plt.figure(figsize=(27, 27))
        self.ax = self.fig.add_subplot(projection='3d')
        self.ax.set_xlim3d(vert_min[0], vert_max[0])
        self.ax.set_ylim3d(vert_min[1], vert_max[1])
        self.ax.set_zlim3d(vert_min[2], vert_max[2])
        self.ax.set_box_aspect(axis_ranges)
        self.ax.axis('off')
        self.ax.add_collection3d(self.poly)

        self.colorbar = self.fig.add_axes([0.7, 0.3, 0.01, 0.4])
        self.vmin = 0
        self.vmax = 0

    def cur_size(self):
        if self.cur_shape is None:
            return 0
        return self.cur_shape['size']

    def set_mises(self, node_mises, vmin=None, vmax=None):
        self.vmin = vmin
        self.vmax = vmax

        elements = self.cur_shape['elements']
        facecolors = np.mean(np.array(node_mises)[elements], axis=1)
        facecolors = map_color(facecolors, vmin, vmax)

        self.poly.set_facecolor(facecolors)
        return

    def add_marks(self, index):
        positions = self.cur_shape['positions']
        positions = positions[index, :]
        self.ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], marker='o', s=5000, c='r', alpha=1)

    def get_distance(self, index1, index2):
        positions = self.cur_shape['positions']
        point1 = positions[index1, :]
        point2 = positions[index2, :]
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def show(self, colorbar_visible=True):
        self.colorbar.set_visible(colorbar_visible)
        if colorbar_visible:
            self.set_colorbar(self.vmin, self.vmax)
        self.fig.show()

    def save(self, path, colorbar_visible=True):
        self.colorbar.set_visible(colorbar_visible)
        if colorbar_visible:
            self.set_colorbar(self.vmin, self.vmax)
        self.fig.savefig(path)

    def set_colorbar(self, vmin=None, vmax=None):
        plt.rcParams['font.size'] = 80
        vmin = np.round(vmin).astype(int)
        vmax = np.round(vmax).astype(int)
        bounds = [round(elem, 2) for elem in np.linspace(vmin, vmax, 5)]
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        mpl.colorbar.ColorbarBase(self.colorbar, cmap=self.cmap, ticks=bounds, norm=norm)


def _parse(proto, meta):
    """Parses a trajectory from tf.Example."""
    feature_lists = {k: tf.io.VarLenFeature(tf.string)
                     for k in meta['field_names']}
    features = tf.io.parse_single_example(proto, feature_lists)
    out = {}
    for key, field in meta['features'].items():
        data = tf.io.decode_raw(features[key].values, getattr(tf, field['dtype']))
        data = tf.reshape(data, field['shape'])
        if field['type'] == 'static':
            pass
        elif field['type'] == 'dynamic_varlen':
            length = tf.io.decode_raw(features['length_' + key].values, tf.int32)
            length = tf.reshape(length, [-1])
            data = tf.RaggedTensor.from_row_lengths(data, row_lengths=length)
        elif field['type'] == 'dynamic':
            data = tf.transpose(data, perm=[1, 0, 2])  # (num_nodes, length_trajectory, feature_dim)
        elif field['type'] != 'dynamic':
            raise ValueError('invalid data format')
        out[key] = data
    return out


def add_targets(ds, meta, target, add_history):
    """Adds target and optionally history fields to dataframe."""

    def fn(trajectory):
        out = {}
        for key, val in trajectory.items():
            if meta['features'][key]['type'] == 'dynamic':
                out[key] = val[:, 1:-1]
            else:
                out[key] = val
            if key == target:
                if add_history:
                    out['prev_' + key] = val[:, 0:-2]
                out['target_' + key] = val[:, 2:]
        return out

    return ds.map(fn, num_parallel_calls=8)


if __name__ == '__main__':
    for dataset in ['flag_simple', 'sphere_simple']:
        # dataset = 'cylinder_flow'
        for type in ['valid', 'test', 'train']:
            # type = 'valid'
            path = os.path.join('')
            out_path = os.path.join('')
            if not os.path.exists(out_path):
                os.mkdir(out_path)
            with open(os.path.join(path, 'meta.json'), 'r') as fp:
                meta = json.loads(fp.read())
            ds = tf.data.TFRecordDataset(os.path.join(path, f'%s.tfrecord' % type))
            ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=8)
            # ini = 1300
            for idx, data in enumerate(ds):
                d = {}
                for key, value in data.items():
                    # print(key)
                    d[key] = value.numpy()
                # break
                cells = d['cells']
                node_type = d['node_type']
                mesh_pos = d['mesh_pos']
                world_pos = d['world_pos']

                np.savez(os.path.join(out_path, str(idx) + '.npz'), cells=cells, node_type=node_type, mesh_pos=mesh_pos,
                         world_pos=world_pos)
