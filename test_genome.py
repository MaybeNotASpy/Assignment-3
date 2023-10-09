import unittest
from genetics import Genome, Solution, Room
import random
from copy import deepcopy


class TestGenome(unittest.TestCase):
    def setUp(self):
        self.genome = Genome()
        self.genome.generate_genome()

    def test_mutate(self):
        """Test mutate."""
        for _ in range(100):
            self.genome.mutate()
            solution = self.genome.decode()
            self.assertTrue(solution.is_valid())

    def test_crossover(self):
        """Test crossover."""
        for _ in range(100):
            other = Genome()
            other.generate_genome()
            new_genome1, new_genome2 = self.genome.crossover(other)
            self.assertEqual(len(new_genome1), len(self.genome))
            self.assertEqual(len(new_genome2), len(other))
            self.assertEqual(len(new_genome1), len(new_genome2))

    def test_decode(self):
        """Test decode."""
        decoded: Solution = self.genome.decode()

    def test_num_to_bits(self):
        """Test num_to_bits."""
        min_val = -10
        max_val = 10
        for _ in range(100):
            orig_num = random.uniform(min_val, max_val)
            bits = self.genome.num_to_bits(orig_num, min_val, max_val)
            self.assertEqual(len(bits), self.genome.val_size)
        self.assertEqual(self.genome.num_to_bits(min_val, min_val, max_val), "0" * self.genome.val_size)
        self.assertEqual(self.genome.num_to_bits(max_val, min_val, max_val), "1" * self.genome.val_size)
    
    def test_bits_to_num(self):
        """Test bits_to_num."""
        min_val = -10
        max_val = 10
        self.assertEqual(self.genome.bits_to_num("0" * self.genome.val_size, min_val, max_val), min_val)
        self.assertEqual(self.genome.bits_to_num("1" * self.genome.val_size, min_val, max_val), max_val)
        for _ in range(100):
            orig_num = random.uniform(min_val, max_val)
            bits = self.genome.num_to_bits(orig_num, min_val, max_val)
            num = self.genome.bits_to_num(bits, min_val, max_val)
            self.assertLessEqual(abs(orig_num - num), (max_val - min_val) / (2 ** self.genome.val_size - 1))
        
    def test_num_to_bits_and_back(self):
        """Test num_to_bits and bits_to_num."""
        min_val = -10
        max_val = 10
        for _ in range(100):
            orig_num = random.uniform(min_val, max_val)
            bits = self.genome.num_to_bits(orig_num, min_val, max_val)
            self.assertEqual(len(bits), self.genome.val_size)
            num = self.genome.bits_to_num(bits, min_val, max_val)
            self.assertLessEqual(abs(orig_num - num), (max_val - min_val) / (2 ** self.genome.val_size - 1))

    def test_generate_random_room_str(self):
        """Test generate_random_room_str."""
        room = Room.Living
        room_str = self.genome.generate_random_room_str(room)
        self.assertEqual(len(room_str), self.genome.val_size + 1)
        room = Room.Kitchen
        room_str = self.genome.generate_random_room_str(room)
        self.assertEqual(len(room_str), self.genome.val_size * 2)
        room = Room.Hall
        room_str = self.genome.generate_random_room_str(room)
        self.assertEqual(len(room_str), self.genome.val_size)
        room = Room.Bed1
        room_str = self.genome.generate_random_room_str(room)
        self.assertEqual(len(room_str), self.genome.val_size + 1)
        room = Room.Bed2
        room_str = self.genome.generate_random_room_str(room)
        self.assertEqual(len(room_str), self.genome.val_size + 1)
        room = Room.Bed3
        room_str = self.genome.generate_random_room_str(room)
        self.assertEqual(len(room_str), self.genome.val_size + 1)

    def test_generate_genome(self):
        """Test generate_genome."""
        self.genome.generate_genome()
        self.assertEqual(len(self.genome), self.genome.get_length())
