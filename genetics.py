from __future__ import annotations
from random import choice, shuffle
from re import S
from dataclasses import dataclass, field
import deal
from enum import Enum
import random
from functools import partial
from copy import deepcopy

from helper import Number, Point, Rectangle, check_rectangles_for_intersection, take


class Room(Enum):
    Living = 1
    Kitchen = 2
    Bath = 3
    Hall = 4
    Bed1 = 5
    Bed2 = 6
    Bed3 = 7


RoomsMinMax: dict[Room, tuple[int, int]] = {
    Room.Living: (9, 20),
    Room.Kitchen: (6, 18),
    Room.Bath: (5.5, 8.5),
    Room.Hall: (3.5, 6),
    Room.Bed1: (10, 16.4),
    Room.Bed2: (9, 16.4),
    Room.Bed3: (8.2, 16.4)
}

RoomsAreaMinMax: dict[Room, tuple[int, int]] = {
    Room.Living: (120, 300),
    Room.Kitchen: (50, 120),
    Room.Bath: (46.75, 46.75),
    Room.Hall: (19, 72),
    Room.Bed1: (100, 180),
    Room.Bed2: (100, 180),
    Room.Bed3: (100, 180)
}


def get_room_indices(room: Room, n: int) -> tuple[int, int]:
    if room == Room.Living:
        # Living Val (n bits) | Living Mult (1 bit) |
        # Size: n+1 bits
        return (0, n+1)
    elif room == Room.Kitchen:
        # Kitchen Val1 (n bits) | Kitchen Val2 (n bits)
        # Size: 2n bits
        return (n+1, 3*n+1)
    elif room == Room.Hall:
        # Hall Val (n bits)
        # Size: n bits
        return (3*n+1+1, 4*n+1+1)
    elif room == Room.Bed1:
        # Bed1 Val (n bits) | Bed1 Mult (1 bit) |
        # Size: n+1 bits
        return (4*n+1, 5*n+2)
    elif room == Room.Bed2:
        # Bed2 Val (n bits) | Bed2 Mult (1 bit) |
        # Size: n+1 bits
        return (5*n+2, 6*n+3)
    elif room == Room.Bed3:
        # Bed3 Val (n bits) | Bed3 Mult (1 bit) |
        # Size: n+1 bits
        return (6*n+3, 7*n+4)


def room_is_valid(room: Room, h: float, w: float) -> bool:
    """Check if a room is valid."""
    min_side, max_side = RoomsMinMax[room]
    min_area, max_area = RoomsAreaMinMax[room]
    if room == Room.Kitchen:
        if not ((min_side <= h) and (h <= max_side)
                and (min_side <= w) and (w <= max_side)
                and (min_area <= h * w) and (h * w <= max_area)):
            return False
    elif room == Room.Hall:
        if not ((min_side <= h) and (h <= max_side)
                and (min_area <= h * w) and (h * w <= max_area)):
            return False
    else:
        if not ((min_side <= h) and (h <= max_side)
                and (min_side <= w) and (w <= max_side)
                and (min_area <= h * w) and (h * w <= max_area)):
            return False
    return True


@deal.post(lambda self: len(self.rooms) == len(Room))
@deal.post(lambda self: len(self.order) == len(Room))
@deal.post(lambda self: set(self.order) == self.order)
class Solution():
    rooms: dict[str, Rectangle]
    order: set[Room] = set()

    def __init__(self):
        self.rooms = {
            Room.Living: Rectangle(1, 1),
            Room.Kitchen: Rectangle(1, 1),
            Room.Bath: Rectangle(5.5, 8.5),
            Room.Hall: Rectangle(1, 1),
            Room.Bed1: Rectangle(1, 1),
            Room.Bed2: Rectangle(1, 1),
            Room.Bed3: Rectangle(1, 1),
        }
        self.order = set([room for room in Room])

    def __str__(self):
        string1 = "\n".join([f"{room}: {self.rooms[room]}" for room in self.rooms])
        string2 = "\n".join([f"{room}" for room in self.order])
        return string1 + "\n" + string2
    
    def is_valid(self) -> bool:
        """Check if the solution is valid."""
        # Check if each room fits within their bounds
        for room in Room: 
            h = self.rooms[room].height
            w = self.rooms[room].width
            if not room_is_valid(room, h, w):
                return False
        return True


@deal.inv(lambda self: len(self.chromosome) == 7 * self.val_size + 25 or len(self.chromosome) == 0)
class Genome():
    """
    Class to keep track of the genome of an individual.
    Genome is stored as a binary string.
    Methods are provided to mutate, crossover, and decode the genome.

    Attributes:
        mutation_rate (float): The probability of a mutation.
        crossover_rate (float): The probability of a crossover.
        genome: The binary string representing the genome.
        val_size (int): The number of bits used to represent a value.
    """
    mutation_rate = 0.0
    crossover_rate = 0.0
    """
    Chromosome follows the format (given n as the number of bits for each number value):
    Living Val (n bits) | Living Mult (1 bit) |
    Kitchen Val1 (n bits) | Kitchen Val2 (n bits)
    Hall Val (n bits) |
    Bed1 Val (n bits) | Bed1 Mult (1 bit) |
    Bed2 Val (n bits) | Bed2 Mult (1 bit) |
    Bed3 Val (n bits) | Bed3 Mult (1 bit) |
    Set of IDs (3 bits each, 21 bits total)

    Total size: 7n + 25 bits
    Example (n=10): 95 bits
    """
    
    chromosome = ""
    val_size = 10
    
    @deal.pure
    def get_length(self):
        return 7 * Genome.val_size + 25

    @deal.pre(lambda _: 0 <= _.mutation_rate <= 1)
    @deal.pre(lambda _: 0 <= _.crossover_rate <= 1)
    def __init__(self,
                 val_size: int = 10,
                 mutation_rate: float = 0.01,
                 crossover_rate: float = 0.7):
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.val_size = val_size
        self.chromosome: str = self.generate_genome()
        assert len(self.chromosome) == 7 * self.val_size + 25

    @deal.pure
    def bits_to_num(self, bits: str, min_val: float, max_val: float) -> int:
        """Convert a string of bits to a number.
        Result will be in the range [min_val, max_val].
        Follows the formula: min_val + (max_val - min_val) * (bits / (2 ** len(bits) - 1))
        Args:
            bits (str): The string of bits to convert.
            min_val (min_val): The minimum value.
            max_val (max_val): The maximum value.
        Returns:
            int: The number value of the bits.
        """
        num_as_int = int("0b" + bits, 2)
        return min_val + (max_val - min_val) * (num_as_int / (2 ** self.val_size - 1))

    @deal.pure
    def num_to_bits(self, num: float, min_val: float, max_val: float) -> str:
        """Convert a number to a string of bits.
        Follows the formula: (num - min_val) / (max_val - min_val) * (2 ** len(bits))

        Args:
            num (float): The number to convert.
            min_val (min_val): The minimum value.
            max_val (max_val): The maximum value.
        Returns:
            str: The string of bits representing the number.
        """
        interval = max_val - min_val
        quantum = interval / (2 ** self.val_size - 1)
        num_as_int = int((num - min_val) / quantum)
        bits = bin(num_as_int)[2:]
        res = "0" * (self.val_size - len(bits)) + bits
        return res

    def generate_random_room_str(self, room: Room) -> str:
        minSide, maxSide = RoomsMinMax[room]
        h, w = 0, 0
        while not room_is_valid(room, h, w):
            # Step 1: Randomly select a height
            h = random.uniform(minSide, maxSide)

            # Step 2: Calculate a valid width
            w = 0
            if room == Room.Kitchen:
                w = random.uniform(minSide, maxSide)
            elif room == Room.Hall:
                w = 5.5
                h = random.uniform(minSide, maxSide)
            # Randomly choose between 1.5h and 2/3h as the width
            else:
                if random.random() < 0.5:
                    w = 1.5 * h
                else:
                    w = 2/3 * h
        # End while
        # Convert the height and width to binary
        if room == Room.Kitchen:
            bits = self.num_to_bits(h, minSide, maxSide) + self.num_to_bits(w, minSide, maxSide)
            assert len(bits) == 2 * self.val_size
        elif room == Room.Hall:
            bits = self.num_to_bits(h, minSide, maxSide)
            assert len(bits) == self.val_size
        else:
            bits = self.num_to_bits(h, minSide, maxSide)
            if w == 1.5 * h:
                bits += "1"
            else:
                bits += "0"
            assert len(bits) == self.val_size + 1
        return bits

    def generate_genome(self) -> str:
        """Generate a random genome."""
        self.chromosome = ""
        room_data = ""
        for room in Room:
            if room == Room.Bath:
                continue
            room_data += self.generate_random_room_str(room)
        
        # Generate the IDs
        IDs = [bin(i)[2:].zfill(3) for i in range(1, 8)]
        shuffle(IDs)
        ID_data = "".join(IDs)
        self.chromosome = room_data + ID_data
        return self.chromosome

    def mutate(self) -> None:
        """Mutate the genome with probability mutation_rate."""
        self.mutate_rooms()
        self.mutate_order()

    def mutate_rooms(self) -> None:
        """Mutate the room values."""
        # Mutate here
        for i in range(0, 7 * self.val_size + 4):
            if random.random() < self.mutation_rate:
                chromosome = list(self.chromosome)
                chromosome[i] = "1" if chromosome[i] == "0" else "0"
                self.chromosome = "".join(chromosome)
        self.repair_genome()

    def repair_genome(self) -> None:
        """Repair the genome."""
        solution = self.decode()
        while not solution.is_valid():
            for room in Room:
                if room == Room.Bath:
                    continue
                room_min, room_max = RoomsMinMax[room]
                room_area_min, room_area_max = RoomsAreaMinMax[room]
                room_start, room_end = get_room_indices(room, self.val_size)
                data = self.chromosome[room_start:room_end]
                h, w = 0, 0
                if room == Room.Kitchen:
                    # Kitchen Val1 (n bits) | Kitchen Val2 (n bits)
                    h = self.bits_to_num(data[:self.val_size], room_min, room_max)
                    w = self.bits_to_num(data[self.val_size:], room_min, room_max)
                    A = h * w
                    quantum = (room_area_max - room_area_min) / (2 ** self.val_size - 1)
                    while not room_is_valid(room, h, w):
                        if A < room_area_min:
                            w += quantum
                        elif A > room_area_max:
                            w -= quantum
                    data = self.num_to_bits(h, room_min, room_max) + self.num_to_bits(w, room_min, room_max)
                    self.chromosome = self.chromosome[:room_start] + data + self.chromosome[room_end:]
                elif room == Room.Hall:
                    # Hall Val (n bits)
                    h = self.bits_to_num(data, room_min, room_max)
                    w = 5.5
                    if h < room_min:
                        h = room_min
                    elif h > room_max:
                        h = room_max
                    data = self.num_to_bits(h, room_min, room_max)
                    self.chromosome = self.chromosome[:room_start] + data + self.chromosome[room_end:]
                else:
                    # Living Val (n bits) | Living Mult (1 bit) |
                    # Bed1 Val (n bits) | Bed1 Mult (1 bit) |
                    # Bed2 Val (n bits) | Bed2 Mult (1 bit) |
                    # Bed3 Val (n bits) | Bed3 Mult (1 bit) |
                    h = self.bits_to_num(data[:-1], room_min, room_max)
                    mult = data[-1]
                    if mult == "1":
                        w = 1.5 * h
                        A = h * w
                        # If the area is too large, set the multiplier to 2/3
                        if A > room_area_max:
                            mult = "0"
                            w = 2/3 * h
                    else:
                        w = 2/3 * h
                        A = h * w
                        # If the area is too small, set the multiplier to 1.5
                        if A < room_area_min:
                            mult = "1"
                            w = 1.5 * h
                    # If the area is still too invalid, decrease the height
                    # Decrease by quantum
                    quantum = (room_area_max - room_area_min) / (2 ** self.val_size - 1)
                    while not room_is_valid(room, h, w):
                        h -= quantum
                        if h < room_min:
                            h = room_min
                            mult = "1"
                        w = 1.5 * h if mult == "1" else 2/3 * h
                        A = h * w
                        if A < room_area_min:
                            mult = "1"
                            w = 1.5 * h
                        elif A > room_area_max:
                            mult = "0"
                            w = 2/3 * h
                        if h < room_min:
                            h = room_min
                            mult = "1"
                            w = 1.5 * h

                    data = self.num_to_bits(h, room_min, room_max) + mult
                    self.chromosome = self.chromosome[:room_start] + data + self.chromosome[room_end:]
                # End else
            # End for
            solution = self.decode()
        # End while
    # End def

    def mutate_order(self) -> None:
        """Mutate the room order.
        Uses a swap mutation."""
        if random.random() < self.mutation_rate:
            # Get the list of room IDs in binary
            room_ids = self.chromosome[7 * self.val_size + 4:]
            rooms = [room_ids[i:i+3] for i in range(0, len(room_ids), 3)]
            # Swap two random rooms
            i, j = random.sample(range(len(rooms)), 2)
            rooms[i], rooms[j] = rooms[j], rooms[i]
            # Update the chromosome
            room_ids = "".join(rooms)
            self.chromosome = self.chromosome[:7 * self.val_size + 4] + room_ids

    @deal.pre(lambda self, other: len(self) == len(other))
    def crossover(self, other: Genome) -> tuple(Genome, Genome):
        """Crossover this genome with another genome. Use 2 point crossover."""
        # Crossover room values first
        child_1_data = ""
        child_2_data = ""
        room_data_1 = self.chromosome[:7 * self.val_size + 4]
        room_data_2 = other.chromosome[:7 * self.val_size + 4]
        # Get the crossover points
        i, j = random.sample(range(len(room_data_1)), 2)
        if i > j:
            i, j = j, i
        # Swap the room data
        child_1_data = room_data_1[:i] + room_data_2[i:j] + room_data_1[j:]
        child_2_data = room_data_2[:i] + room_data_1[i:j] + room_data_2[j:]

        # Crossover room order. Use PMX crossover
        room_ids_1 = self.chromosome[7 * self.val_size + 4:]
        room_ids_2 = other.chromosome[7 * self.val_size + 4:]

        # Convert binary to decimal
        room_ids_1 = [int(room_ids_1[i:i+3], 2) for i in range(0, len(room_ids_1), 3)]
        room_ids_2 = [int(room_ids_2[i:i+3], 2) for i in range(0, len(room_ids_2), 3)]
        
        # Get the crossover points
        i, j = random.sample(range(len(room_ids_1)), 2)
        if i > j:
            i, j = j, i
        # Copy the selected segments
        child_1_ids = [-1] * len(room_ids_1)
        child_2_ids = [-1] * len(room_ids_2)
        child_1_ids[i:j] = room_ids_1[i:j]
        child_2_ids[i:j] = room_ids_2[i:j]

        # Fill in the remaining values by exchanging values between parents
        for k in range(len(child_1_ids)):
            if i <= k < j:
                continue
            # Get the value from the other parent
            val = room_ids_2[k]
            # Check if the value is already in the child
            if val in child_1_ids:
                # Get the index of the value in the child
                idx = child_1_ids.index(val)
                # Get the value from the other parent
                val = room_ids_2[idx]
                # Check if the value is already in the child
                if val in child_1_ids:
                    # Get the index of the value in the child
                    idx = child_1_ids.index(val)
                    # Get the value from the other parent
                    val = room_ids_2[idx]
            child_1_ids[k] = val
        
        # Repeat for the other child
        for k in range(len(child_2_ids)):
            if i <= k < j:
                continue
            # Get the value from the other parent
            val = room_ids_1[k]
            # Check if the value is already in the child
            if val in child_2_ids:
                # Get the index of the value in the child
                idx = child_2_ids.index(val)
                # Get the value from the other parent
                val = room_ids_1[idx]
                # Check if the value is already in the child
                if val in child_2_ids:
                    # Get the index of the value in the child
                    idx = child_2_ids.index(val)
                    # Get the value from the other parent
                    val = room_ids_1[idx]
            child_2_ids[k] = val

        
        # Fill in the remaining values by exchanging values between parents
        for k in range(len(room_ids_1)):
            if child_1_ids[i] == -1:
                child_1_ids[i] = room_ids_2[i]
            if child_2_ids[i] == -1:
                child_2_ids[i] = room_ids_1[i]

        # Convert the IDs back to binary
        child_1_ids = [bin(i)[2:].zfill(3) for i in child_1_ids]
        child_2_ids = [bin(i)[2:].zfill(3) for i in child_2_ids]

        # Create the children
        child_1 = deepcopy(self)
        child_2 = deepcopy(other)
        child_1_data += "".join(child_1_ids)
        assert len(child_1_data) == len(child_1)
        child_1.chromosome = child_1_data
        child_2_data += "".join(child_2_ids)
        assert len(child_2_data) == len(child_2)
        child_2.chromosome = child_2_data

        # Repair the children
        child_1.repair_genome()
        child_2.repair_genome()

        return (child_1, child_2)



    def decode(self) -> Solution:
        """Decode the genome into a Solution."""
        solution = Solution()

        for room in Room:
            if room == Room.Bath:
                continue
            s, e = get_room_indices(room, self.val_size)
            data = self.chromosome[s:e]
            h, w = 0, 0
            if room == Room.Kitchen:
                # Kitchen Val1 (n bits) | Kitchen Val2 (n bits)
                h = self.bits_to_num(data[:self.val_size], *RoomsMinMax[room])
                w = self.bits_to_num(data[self.val_size:], *RoomsMinMax[room])
                solution.rooms[room] = Rectangle(h, w)
            elif room == Room.Hall:
                # Hall Val (n bits)
                h = self.bits_to_num(data, *RoomsMinMax[room])
                solution.rooms[room] = Rectangle(5.5, h)
            else:
                # Living Val (n bits) | Living Mult (1 bit) |
                # Bed1 Val (n bits) | Bed1 Mult (1 bit) |
                # Bed2 Val (n bits) | Bed2 Mult (1 bit) |
                # Bed3 Val (n bits) | Bed3 Mult (1 bit) |
                h = self.bits_to_num(data[:-1], *RoomsMinMax[room])
                mult = data[-1]
                if mult == "1":
                    w = 1.5 * h
                else:
                    w = 2/3 * h
                solution.rooms[room] = Rectangle(h, w)
        
        data = self.chromosome[7 * self.val_size + 4:]
        room_list = [data[i:i+3] for i in range(0, len(data), 3)]
        for room_id in room_list:
            room_id = "".join(room_id)
            room_id = int(room_id, 2)
            solution.order.add(Room(room_id))

        return solution

    @deal.pure
    def __str__(self):
        return self.chromosome

    @deal.pure
    def __len__(self):
        return len(self.chromosome)
    
    @deal.pure
    def print_genome(self):
        print("Genome: ", self.chromosome)
        bit_iter = iter(self.chromosome)
        print("Living: ", "".join(take(bit_iter, self.val_size)), " ".join(take(bit_iter, 1)))
        print("Kitchen: ", "".join(take(bit_iter, self.val_size)), " " + "".join(take(bit_iter, self.val_size)))
        print("Hall: ", "".join(take(bit_iter, self.val_size)))
        print("Bed1: ", "".join(take(bit_iter, self.val_size)), " ".join(take(bit_iter, 1)))
        print("Bed2: ", "".join(take(bit_iter, self.val_size)), " ".join(take(bit_iter, 1)))
        print("Bed3: ", "".join(take(bit_iter, self.val_size)), " ".join(take(bit_iter, 1)))

        print("IDs: ", end="")
        for _ in range(7):
            print("".join(take(bit_iter, 3)), end=" ")
        print()
