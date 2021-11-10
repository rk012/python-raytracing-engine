from abc import ABC, abstractmethod
from typing import Union, Tuple

import numpy as np


class Point:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z
        self.coords = np.array([x, y, z])


class Ray:
    def __init__(self, x: float, y: float, z: float, dx: float, dy: float, dz: float):
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.origin = Point(x, y, z)
        self.vector = np.array([dx, dy, dz])

    def normalize(self):
        norm_vector = self.vector / np.linalg.norm(self.vector)
        return Ray(*self.origin.coords, *norm_vector)

    def pointAt(self, t: float) -> Point:
        return Point(*(t*self.vector + self.origin.coords))


class Color:
    def __init__(self, r: float, g: float, b: float):
        self.r = r
        self.g = g
        self.b = b

        self.arr = np.array([r, g, b])


class Material:
    def __init__(self, ambient: Color, diffuse: Color, specular: Color, specularity: float, reflection: float):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.specularity = specularity
        self.reflection = reflection


class SceneComponent:
    pass


class Light(SceneComponent):
    def __init__(self, pos: Point, ambient: Color=Color(1, 1, 1), diffuse: Color=Color(1, 1, 1), specular: Color=Color(1, 1, 1)):
        self.pos = pos
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular


class SceneObject(SceneComponent, ABC):
    def __init__(self, m: Material):
        self._m = m

    def getMaterial(self):
        return self._m

    @abstractmethod
    def getIntersection(self, ray: Ray) -> Tuple[Union[float, None], Union[Ray, None]]:
        pass


class Sphere(SceneObject):
    def __init__(self, center: Point, radius: float, m: Material):
        super().__init__(m)

        self.c = center
        self.r = radius

    def getIntersection(self, ray: Ray) -> Tuple[Union[float, None], Union[Ray, None]]:
        b = 2 * np.dot(ray.vector, ray.origin.coords - self.c.coords)
        c = np.linalg.norm(ray.origin.coords - self.c.coords) ** 2 - self.r ** 2
        delta = b ** 2 - 4 * c
        if delta > 0:
            t1 = (-b + np.sqrt(delta)) / 2
            t2 = (-b - np.sqrt(delta)) / 2
            if t1 > 0 and t2 > 0:
                t = min(t1, t2)
                p = ray.pointAt(t)
                return min(t1, t2), Ray(*p.coords, *(p.coords - self.c.coords)).normalize()
        return None, None
