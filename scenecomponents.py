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
        return Point(*(t * self.vector + self.origin.coords))


class Color:
    def __init__(self, r: float, g: float, b: float):
        self.r = r
        self.g = g
        self.b = b

        self.arr = np.array([r, g, b])


class Transform:
    def __init__(self, x: float, y: float, z: float, yaw: float, pitch: float, roll: float, sx: float, sy: float, sz: float):
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        self.sx = sx
        self.sy = sy
        self.sz = sz

        self.pos = np.array([x, y, z])
        self.rotation = np.array([yaw, pitch, roll])
        self.scale = np.array([sx, sy, sz])

        self.matrix = np.dot(np.diag(self.scale), np.dot(np.dot(np.array(
            [[np.cos(yaw), 0, np.sin(yaw)],
             [0, 1, 0],
             [-np.sin(yaw), 0, np.cos(yaw)]]
        )
                             , np.array(
                [[1, 0, 0],
                 [0, np.cos(pitch), np.sin(pitch)],
                 [0, -np.sin(pitch), np.cos(pitch)]]
            ))
                             , np.array(
                [[np.cos(roll), -np.sin(roll), 0],
                 [np.sin(roll), np.cos(roll), 0],
                 [0, 0, 1]]
            )
                             ))
        self.inv_matrix = np.linalg.inv(self.matrix)

    def applyTransform(self, p: Union[Point, Ray]) -> Union[Point, Ray]:
        if isinstance(p, Point):
            t = np.dot(self.matrix, p.coords) + self.pos
            return Point(*t)

        else:
            t = np.dot(self.matrix, p.origin.coords) + self.pos
            v = np.dot(self.matrix, p.origin.coords)
            return Ray(*t, *v)

    def undoTransform(self, p: Union[Point, Ray]) -> Union[Point, Ray]:
        if isinstance(p, Point):
            t = np.dot(self.inv_matrix, p.coords - self.pos)
            return Point(*t)

        else:
            t = np.dot(self.inv_matrix, p.origin.coords - self.pos)
            v = np.dot(self.inv_matrix, p.origin.coords)
            return Ray(*t, *v)


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
    def __init__(self, pos: Point, intensity: float, ambient: Color = Color(1, 1, 1), diffuse: Color = Color(1, 1, 1),
                 specular: Color = Color(1, 1, 1)):
        self.pos = pos
        self.intensity = intensity
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
