import moderngl as mg
import numpy as np
import matplotlib.pyplot as plt

context = mg.create_standalone_context(require=430)

heightMap = context.buffer(np.zeros(1024 * 1024, dtype=np.float32).tobytes())
print(heightMap.size)

with open('./shaders/simplex_noise_2d.glsl') as f:
    compute_shader = context.compute_shader(f.read()
                                            .replace("%OCTAVES%", '8')
                                            .replace("%WIDTH%", '1024')
                                            .replace("%HEIGHT%", '1024')
                                            .replace("%PERMUTE_1%", '34.0')
                                            .replace("%PERMUTE_2%", '1.0'))

heightMap.bind_to_storage_buffer(0, size=1024*1024)

if __name__ == '__main__':
    compute_shader.run(32, 32)
    plt.imshow(np.frombuffer(heightMap.read(), dtype=np.float32).reshape(1024, 1024))
    plt.show()

