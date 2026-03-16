import re
from numpy.ctypeslib import as_array
from math import sqrt
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass
from skimage.measure import block_reduce
from skimage.transform import resize
from scipy.ndimage import gaussian_filter, uniform_filter
from include.a_star import a_star, find_retreat_turning_points


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
        self.center = Point(self.min_coord.x + width / 2, self.min_coord.y + height / 2)


@dataclass
class UShape:
    rectangles: list[Rectangle]


def create_u_shape(lower_left: Point, width: int = 5, height: int = 10) -> UShape:
    left_rectangle = Rectangle(lower_left, height + 10, width)
    mid_rectangle = Rectangle(
        Point(lower_left.x, lower_left.y), width, height + 2 * width
    )
    right_rectangle = Rectangle(
        Point(lower_left.x + width + height, lower_left.y), height + 10, width
    )
    u: UShape = UShape([left_rectangle, mid_rectangle, right_rectangle])
    return u


def create_coordinate_array(size: int = 100) -> tuple[np.ndarray, np.ndarray]:
    x_array = np.linspace(0, size, size)
    y_array = np.linspace(0, size, size)
    xv, yv = np.meshgrid(x_array, y_array)
    return xv, yv


def distance(rect: Rectangle, point: Point) -> float:
    dx = max(rect.min_coord.x - point.x, 0, point.x - rect.max_coord.x)
    dy = max(rect.min_coord.y - point.y, 0, point.y - rect.max_coord.y)
    return sqrt(dx * dx + dy * dy)


def distance_from_rect(rect: Rectangle, point: Point) -> float:
    x_distance = abs(point.x - rect.center.x) - rect.width / 2
    y_distance = abs(point.y - rect.center.y) - rect.height / 2
    outside_distance = sqrt(
        max(x_distance, 0) * max(x_distance, 0)
        + max(y_distance, 0) * max(y_distance, 0)
    )
    inside_distance = min(max(x_distance, y_distance), 0)
    return outside_distance + inside_distance


def in_ushape(u: UShape, point: Point) -> bool:
    for rect in u.rectangles:
        if rect.min_coord.x <= point.x and point.x <= rect.max_coord.x:
            if rect.min_coord.y <= point.y and point.y <= rect.max_coord.y:
                return True
    return False


def calculate_lse_distance(u: UShape, point: Point, tau: float = 1.0) -> float:
    # if in_ushape(u, point):
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


def create_occupancy_grid(sdf: np.ndarray) -> np.ndarray:
    return (sdf <= 0).astype(sdf.dtype)


def loss_function(sdf) -> np.ndarray:
    softplus_loss = np.log(1 + np.exp(-sdf))
    # blurred_sdf = gaussian_filter(sdf, sigma=100)
    # reduced_image = block_reduce(sdf, block_size=(5, 5), func=np.mean)
    # recized_image = resize(reduced_image, (40, 40))
    smoothed = uniform_filter(sdf, size=25)

    sdf_loss = 1 / (np.pow(sdf, 2) + 1e-1)
    # sdf_loss = -np.pow(sdf + 0.2 * smoothed, 1)
    smoothed_loss = 1 / (np.pow(smoothed, 1) + 1e-1)
    return 10 * smoothed_loss + sdf_loss + 10 * softplus_loss
    # return smoothed


def main():
    size = 40
    xv, yv = create_coordinate_array(size)
    u = create_u_shape(Point(10, 10))
    for i in range(size):
        for j in range(size):
            xv[j, i] = calculate_euclidian_distance(u, Point(i, j))
    uv, vv = np.gradient(xv)
    loss = loss_function(xv)
    uvloss, vvloss = np.gradient(loss)
    occupancy_grid: np.ndarray = create_occupancy_grid(xv)
    goal = (5, 5)
    start = (20, 20)
    a_star_path = a_star(start, goal, occupancy_grid)
    turning_points = np.array(find_retreat_turning_points(a_star_path, goal))
    a_star_array = np.array(a_star_path)
    fig, ax = plt.subplots()
    np.save("../../data/uv.npy", uv)
    np.save("../../data/vv.npy", vv)
    np.save("../../data/distance_field.npy", xv)
    np.save("../../data/occupancy_grid.npy", occupancy_grid)
    np.save("../../data/a_star_path.npy", a_star_array)
    np.save("../../data/turning_points.npy", turning_points)
    imshow = ax.imshow(loss, origin="lower")
    fig.colorbar(imshow)
    ax.quiver(vvloss, uvloss, scale=40)
    # ax.plot(a_star_array[:, 0], a_star_array[:, 1], color="r")
    # ax.scatter(turning_points[:, 0], turning_points[:, 1])
    # rect = u.rectangles
    # ax.add_patch(
    #    patches.Rectangle(
    #        (rect[0].min_coord.x, rect[0].min_coord.y), rect[0].width, rect[0].height
    #    )
    # )
    # ax.add_patch(
    #    patches.Rectangle(
    #        (rect[1].min_coord.x, rect[1].min_coord.y), rect[1].width, rect[1].height
    #    )
    # )
    # ax.add_patch(
    #    patches.Rectangle(
    #        (rect[2].min_coord.x, rect[2].min_coord.y), rect[2].width, rect[2].height
    #    )
    # )
    plt.savefig("../../data/plot")


if __name__ == "__main__":
    main()
