from matplotlib import pyplot as plt

from scene import Scene
from scenecomponents import Sphere, Point, Light, Material, Color

scene = Scene()

r = Material(
    diffuse=Color(1, 0, 0),
    ambient=Color(0.2, 0, 0),
    specular=Color(1, 1, 1),
    specularity=100,
    reflection=0.5
)

g = Material(
    diffuse=Color(0, 1, 0),
    ambient=Color(0, 0.2, 0),
    specular=Color(1, 1, 1),
    specularity=100,
    reflection=0.5
)

b = Material(
    diffuse=Color(0, 0, 1),
    ambient=Color(0, 0, 0.2),
    specular=Color(1, 1, 1),
    specularity=100,
    reflection=0.5
)

scene.addComponent(Light(
    Point(5, 5, 5)
))

scene.addComponent(Sphere(
    Point(-0.2, 0, -1),
    0.7,
    r
))
scene.addComponent(Sphere(
    Point(0.1, -0.3, 0),
    0.1,
    g
))
scene.addComponent(Sphere(
    Point(-0.3, 0, 0),
    0.15,
    b
))

plt.imshow(scene.render(1920, 1080))
plt.show()
