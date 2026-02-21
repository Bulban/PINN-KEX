from math import sqrt
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass


@dataclass
class Point:
    x: float
    y: float


class Rectangle:
    min_coord: Point
    height: int
    width: int
    max_coord: Point

    def __init__(self, min: Point, height: int, width: int) -> None:
        self.min_coord = min
        self.height = height
        self.width = width
        self.max_coord = Point(
            self.min_coord.x + self.width,
            self.min_coord.y + self.height,
        )
        self.center = Point(self.min_coord.x + width/2, self.min_coord.y + height/2)


@dataclass
class UShape:
    rectangles: list[Rectangle]


def create_u_shape(lower_left: Point, width: int = 5, height: int = 10) -> UShape:
    left_rectangle = Rectangle(lower_left, height + 10, width)
    mid_rectangle = Rectangle(Point(lower_left.x, lower_left.y), width, height + 2 * width)
    right_rectangle = Rectangle(
        Point(lower_left.x + width + height, lower_left.y), height + 10, width
    )
    u: UShape = UShape([left_rectangle, mid_rectangle, right_rectangle])
    return u


def create_coordinate_array(size: int = 100)  -> tuple[np.ndarray, np.ndarray]:
    x_array = np.linspace(0, size, size)
    y_array = np.linspace(0, size, size)
    xv, yv = np.meshgrid(x_array, y_array)
    return xv, yv


def distance(rect: Rectangle, point: Point) -> float:
    dx = max(rect.min_coord.x - point.x,0, point.x - rect.max_coord.x)
    dy = max(rect.min_coord.y - point.y,0, point.y - rect.max_coord.y)
    return sqrt(dx * dx + dy * dy)

def distance_from_rect(rect: Rectangle, point: Point) -> float:
    x_distance = abs(point.x - rect.center.x) - rect.width/2
    y_distance = abs(point.y - rect.center.y) - rect.height/2
    outside_distance = sqrt(max(x_distance, 0) * max(x_distance, 0) + max(y_distance, 0) * max(y_distance, 0))
    inside_distance = min(max(x_distance, y_distance), 0)
    return outside_distance + inside_distance

def in_ushape(u: UShape, point: Point) -> bool:
    for rect in u.rectangles:
        if rect.min_coord.x <= point.x and point.x <= rect.max_coord.x:
            if rect.min_coord.y <= point.y and point.y <= rect.max_coord.y:
                return True
    return False

def calculate_lse_distance(u: UShape, point: Point, tau: float = 1.0) -> float:
    #if in_ushape(u, point):
    #    return 0.0
    distances: list[float] = []
    for rect in u.rectangles:
        distances.append(distance_from_rect(rect, point))
    exp_sum = 0.0
    for dist in distances:
        exp_sum += np.exp(-dist / tau)
    return -np.log(exp_sum) * tau


def calculate_euclidian_distance(u: UShape, point: Point) -> float:
    distances: list[float] = []
    for rect in u.rectangles:
        distances.append(distance_from_rect(rect, point))

    return np.min(distances)


def main():
    size = 40
    xv, yv = create_coordinate_array(size)
    u = create_u_shape(Point(10, 10))
    for i in range(size):
        for j in range(size):
            xv[j, i] = calculate_lse_distance(u, Point(i, j))
    uv, vv = np.gradient(xv)
    fig, ax = plt.subplots()
    os.makedirs("results", exist_ok=True)
    np.save("../data/uv.npy", uv)
    np.save("../data/vv.npy", vv)
    np.save("../data/distance_field.npy", xv)
    imshow = ax.imshow(xv, origin="lower")
    fig.colorbar(imshow)
    ax.quiver( vv,uv, scale=50)
    rect = u.rectangles
    #ax.add_patch(
    #    patches.Rectangle(
    #        (rect[0].min_coord.x, rect[0].min_coord.y), rect[0].width, rect[0].height
    #    )
    #)
    #ax.add_patch(
    #    patches.Rectangle(
    #        (rect[1].min_coord.x, rect[1].min_coord.y), rect[1].width, rect[1].height
    #    )
    #)
    #ax.add_patch(
    #    patches.Rectangle(
    #        (rect[2].min_coord.x, rect[2].min_coord.y), rect[2].width, rect[2].height
    #    )
    #)
    plt.show()


if __name__ == "__main__":
    main()
