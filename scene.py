from typing import Dict, Tuple, Union

import numpy as np

from scenecomponents import SceneObject, SceneComponent, Light, Ray, Point


def _reflect(ray: Ray, normal: Ray) -> Ray:
    return Ray(*normal.origin.coords, *(ray.vector - 2 * np.dot(ray.vector, normal.vector) * normal.vector))


class Scene:
    sceneObjects: Dict[int, SceneObject] = {}
    lights: Dict[int, Light] = {}
    objectCount = 0

    def addComponent(self, obj: Union[SceneObject, Light]) -> int:
        if isinstance(obj, SceneObject):
            self.sceneObjects[self.objectCount] = obj
        elif isinstance(obj, Light):
            self.lights[self.objectCount] = obj

        self.objectCount += 1
        return self.objectCount - 1

    def removeComponent(self, obj_id: int):
        if obj_id in self.sceneObjects:
            del self.sceneObjects[obj_id]
        elif obj_id in self.lights:
            del self.lights[obj_id]

    def getComponent(self, obj_id: int) -> SceneComponent:
        if obj_id in self.sceneObjects:
            return self.sceneObjects[obj_id]
        elif obj_id in self.lights:
            return self.lights[obj_id]

    def _nearest_intersected_object(self, ray: Ray) -> Tuple[SceneObject, float, Ray]:
        sceneObjects = list(self.sceneObjects.values())

        intersections = np.array([obj.getIntersection(ray) for obj in sceneObjects])
        distances = intersections[:, 0]
        normals = intersections[:, 1]

        nearest_object = None
        normal = None
        min_distance = np.inf
        for index, distance in enumerate(distances):
            if distance and distance < min_distance:
                min_distance = distance
                nearest_object = sceneObjects[index]
                normal = normals[index]

        return nearest_object, min_distance, normal

    def render(self, width: int, height: int, max_depth=5) -> np.ndarray:
        camera = Point(0, 0, 1)
        ratio = float(width) / height
        screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom

        image = np.zeros((height, width, 3))
        for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
            for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
                pixel = Point(x, y, 0)
                ray = Ray(*camera.coords, *(pixel.coords - camera.coords)).normalize()

                color = np.zeros((3))
                reflection = 1

                for k in range(0, max_depth):
                    # object intersection
                    nearest_object, min_distance, normal = self._nearest_intersected_object(ray)
                    if nearest_object is None:
                        break

                    max_illumination = np.zeros((3))  # nearest_object.getMaterial().ambient.arr

                    intersection = ray.pointAt(min_distance)

                    hasLight = False

                    # light intersection
                    for light in self.lights.values():
                        light_pos = light.pos

                        shifted_point = normal.pointAt(1e-5)
                        light_ray = Ray(*shifted_point.coords, *(light_pos.coords - shifted_point.coords)).normalize()

                        _, min_distance, _ = self._nearest_intersected_object(light_ray)
                        light_distance = np.linalg.norm(light_pos.coords - intersection.coords)

                        is_shadowed = min_distance < light_distance

                        if is_shadowed:
                            continue

                        hasLight = True

                        illumination = np.zeros((3))
                        illumination += nearest_object.getMaterial().ambient.arr * light.ambient.arr
                        illumination += nearest_object.getMaterial().diffuse.arr * light.diffuse.arr * np.dot(light_ray.vector, normal.vector) * light.intensity / light_distance**2

                        intersection_to_camera = Ray(*shifted_point.coords, *(-1*ray.vector)).normalize()
                        H = (light_ray.vector + intersection_to_camera.vector)/np.linalg.norm(light_ray.vector + intersection_to_camera.vector)
                        illumination += (nearest_object.getMaterial().specular.arr * light.specular.arr * np.dot(normal.vector, H) ** (nearest_object.getMaterial().specularity / 4)) * light.intensity / light_distance**2

                        max_illumination[illumination>max_illumination] = illumination[illumination>max_illumination]

                    if not hasLight:
                        max_illumination = nearest_object.getMaterial().ambient.arr

                    # Reflection
                    color += reflection * max_illumination
                    reflection *= nearest_object.getMaterial().reflection

                    # new ray
                    ray = _reflect(ray, normal)

                image[i, j] = np.clip(color, 0, 1)

            print("progress: %d/%d" % (i + 1, height))

        return image
