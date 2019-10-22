#version 430
uniform mat4 m_proj;
uniform mat4 m_cam;
uniform sampler2D Heightmap;
uniform sampler2D Colormap;

in vec2 vert;
out vec2 v_text;
void main() {
    vec4 vertex = vec4(vert.x * 32, texture(Heightmap, vert).r * 8, vert.y * 32, 1.0);
    mat4 mv = m_cam;
    vec4 p = mv * vertex;
	gl_Position = m_proj * p;
    v_text = vert;
}