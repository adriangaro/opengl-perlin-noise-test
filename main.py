import math
import struct
from pathlib import Path

import moderngl as mg
import moderngl_window as mglw
import numpy as np
from matplotlib import pylab
from moderngl_window import resources
from moderngl_window.scene import KeyboardCamera

import matplotlib.pyplot as plt

from pyrr import Matrix44, matrix44, Vector3

resources.register_dir((Path(__file__).parent / 'resources').resolve())


class MainWindow(mglw.WindowConfig):
    window_size = (1920, 1080)
    aspect_ratio = 16 / 9
    gl_version = (4, 3)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wnd.mouse_exclusivity = True

        self.camera = KeyboardCamera(self.wnd.keys, fov=75.0, aspect_ratio=self.wnd.aspect_ratio, near=0.1, far=1000.0)
        self.camera.velocity = 7.0
        self.camera.mouse_sensitivity = 0.3

        self.heightMap = self.ctx.buffer(np.zeros(1024 * 1024, dtype=np.float32).tobytes())

        with open('./resources/simplex_noise_2d.glsl') as f:
            self.compute_shader = self.ctx.compute_shader(f.read()
                                                          .replace("%OCTAVES%", '8')
                                                          .replace("%WIDTH%", '1024')
                                                          .replace("%HEIGHT%", '1024')
                                                          .replace("%PERMUTE_1%", '34.0')
                                                          .replace("%PERMUTE_2%", '1.0'))

        with open('./resources/vertex.glsl') as f1:
            with open('./resources/fragment.glsl') as f2:
                self.program = self.ctx.program(
                    vertex_shader=f1.read(),
                    fragment_shader=f2.read())

        self.heightMap.bind_to_storage_buffer(0, size=1024 * 1024 * 4)
        self.compute_shader.run(32, 32)

        im = np.frombuffer(self.heightMap.read(), dtype=np.float32).reshape(1024, 1024)
        my_cm = pylab.cm.get_cmap('jet_r')
        color_im = my_cm(im).astype(np.float32)

        plt.imshow(im)
        plt.show()

        plt.imshow(color_im)
        plt.show()

        self.tex = self.ctx.texture((1024, 1024), 1, im.tobytes(),
                                    dtype='f4')
        self.color_tex = self.ctx.texture((1024, 1024), 4, color_im.tobytes(), dtype='f4')

        self.tex.build_mipmaps()
        self.color_tex.build_mipmaps()
        self.tex.use(0)
        self.color_tex.use(1)
        self.program['Heightmap'].value = 0
        self.program['Colormap'].value = 1

        index = 0
        self.vertices = bytearray()
        self.indices = bytearray()

        for i in range(1024 - 1):
            for j in range(1024):
                self.vertices += struct.pack('2f', (i + 1) / 1024, j / 1024)
                self.indices += struct.pack('i', index)
                index += 1
                self.vertices += struct.pack('2f', i / 1024, j / 1024)
                self.indices += struct.pack('i', index)
                index += 1

            self.indices += struct.pack('i', -1)

        self.vbo = self.ctx.buffer(self.vertices)
        self.ibo = self.ctx.buffer(self.indices)

        self.vao = self.ctx.vertex_array(self.program, [(self.vbo, '2f', 'vert')], self.ibo)

        # Use this for gltf scenes for better camera controls
        # if self.scene.diagonal_size > 0:
        # self.camera.velocity = self.scene.diagonal_size / 5.0

    def render(self, time: float, frametime: float):
        """Render the scene"""
        self.ctx.enable_only(mg.DEPTH_TEST | mg.CULL_FACE)

        translation = matrix44.create_from_translation((0, 0, -1.5))
        # rotation = matrix44.create_from_eulers((time, time, time))
        rotation = matrix44.create_from_eulers((0, 0, 0))
        model_matrix = matrix44.multiply(rotation, translation)

        camera_matrix = matrix44.multiply(model_matrix, self.camera.matrix)

        self.program['m_proj'].write(self.camera.projection.matrix.astype('f4').tobytes())
        self.program['m_cam'].write(camera_matrix.astype('f4').tobytes())

        self.vao.render(mg.TRIANGLE_STRIP)

        # # Currently only works with GLTF2
        # self.scene.draw_bbox(
        #     projection_matrix=self.camera.projection.matrix,
        #     camera_matrix=camera_matrix,
        #     children=True,
        # )

    def key_event(self, key, action, modifiers):
        self.camera.key_input(key, action, modifiers)

    def mouse_position_event(self, x: int, y: int):
        self.camera.rot_state(x, y)

    def resize(self, width: int, height: int):
        self.camera.projection.update(aspect_ratio=self.wnd.aspect_ratio)


def main():
    mglw.run_window_config(MainWindow)


if __name__ == '__main__':
    main()
