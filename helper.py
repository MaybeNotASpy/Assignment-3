from __future__ import annotations

from dataclasses import dataclass
import itertools

import deal
import numpy as np

Number = int | float | np.integer | np.floating


def take(iterable, n: int):
    li = list(itertools.islice(iterable, n))
    if len(li) != n:
        raise RuntimeError("too short iterable for take")
    return li


@deal.inv(lambda self: self.x >= 0 and self.y >= 0)
@dataclass
class Point:
    x: Number = 0
    y: Number = 0

    def __iter__(self):
        yield self.x
        yield self.y


@deal.post(lambda self: self.height > 0 and self.width > 0)
@dataclass
class Rectangle:
    """
    Attributes:
        height: The length of the vertical side.
        width: The length of the horizontal side.
        bottom_left: The bottom left corner of the rectangle. Defaults to None.
            Must be set to a Point to place the rectangle.
    """
    height: Number
    width: Number
    bottom_left: Point | None = None

    @deal.pre(lambda self, _: self.bottom_left is not None)
    def is_inside(self, point: Point) -> bool:
        """Checks if a point is inside the rectangle.

        Args:
            point: The point to check.

        Returns:
            bool: Whether the point is inside the rectangle.

        Raises:
            ValueError: If the rectangle has not been placed.
        """
        x, y = point
        x0, y0 = self.bottom_left
        x1 = x0 + self.width
        y1 = y0 + self.height
        return bool(x0 <= x <= x1 and y0 <= y <= y1)

    @deal.pre(
        lambda self, other: self.bottom_left is not None
        and other.bottom_left is not None
    )
    def bounds(self, other: Rectangle) -> bool:
        """
        Checks if the other rectangle is within the bounds of this rectangle.
        Args:
            other: The other rectangle.
        Returns:
            bool: Whether the other rectangle is within
                the bounds of this rectangle.
        """
        other_corners = other.get_corners()
        # Only need to check bottom left corner and top right corner
        return self.is_inside(other_corners[0]) \
            and self.is_inside(other_corners[3])

    @deal.pre(lambda self: self.bottom_left is not None)
    def get_center(self) -> Point:
        """Gets the center of the rectangle.

        Returns:
            Point: The center of the rectangle.
        """
        x, y = self.bottom_left
        return Point(x + self.width / 2, y + self.height / 2)

    @deal.pre(lambda self: self.bottom_left is not None)
    def get_corners(self) -> tuple[Point, Point, Point, Point]:
        """Gets the corners of the rectangle.
        The corners are returned in the following order:
        bottom_left, bottom_right, top_left, top_right.
        Returns:
            tuple[Point, Point, Point, Point]: The corners of the rectangle.
        """
        return (
            self.bottom_left,
            Point(self.bottom_left.x + self.width,
                  self.bottom_left.y),
            Point(self.bottom_left.x,
                  self.bottom_left.y + self.height),
            Point(self.bottom_left.x + self.width,
                  self.bottom_left.y + self.height),
        )

    def get_area(self) -> Number:
        """Gets the area of the rectangle.

        Returns:
            Number: The area of the rectangle.
        """
        return self.height * self.width

    def get_perimeter(self) -> Number:
        """Gets the perimeter of the rectangle.

        Returns:
            Number: The perimeter of the rectangle.
        """
        return 2 * (self.height + self.width)

    def get_diagonal(self) -> Number:
        """Gets the diagonal of the rectangle.

        Returns:
            Number: The diagonal of the rectangle.
        """
        return np.sqrt(self.height**2 + self.width**2)

    def get_flipped(self) -> Rectangle:
        """Gets a rectangle with the height and width flipped.

        Returns:
            Rectangle: The flipped rectangle.
        """
        return Rectangle(self.width, self.height, self.bottom_left)


def intersects(a: Rectangle, b: Rectangle) -> bool:
    """Checks if two rectangles intersect.

    Args:
        a: The first rectangle.
        b: The second rectangle.

    Returns:
        bool: Whether the two rectangles intersect.
    """
    a_corners = a.get_corners()
    b_corners = b.get_corners()
    for corner in a_corners:
        if b.is_inside(corner):
            return True
    for corner in b_corners:
        if a.is_inside(corner):
            return True
    return False


def check_rectangles_for_intersection(rects: list[Rectangle]) -> bool:
    """Checks if any of the rectangles in the list intersect.

    Args:
        rects: The list of rectangles to check.

    Returns:
        bool: Whether any of the rectangles intersect.
    """
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            if intersects(rects[i], rects[j]):
                return True
    return False


def get_cost_of_solution(solution: Solution) -> Number:
    """Calculates the cost of a solution.

    Args:
        solution: The solution to calculate the cost of.

    Returns:
        Number: The cost of the solution.
    """
    cost = 0
    for room in Room:
        width = solution.rooms[room].width
        height = solution.rooms[room].height
        if room == Room.Kitchen or room == Room.Bath:
            cost += width * height * 2
        else:
            cost += width * height
    return cost
