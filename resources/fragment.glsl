#version 430

uniform sampler2D Heightmap;
uniform sampler2D Colormap;

in vec2 v_text;
out vec4 f_color;

void main() {
    vec3 color = texture(Colormap, v_text).rgb;
    f_color = vec4(color,  1.0);
}