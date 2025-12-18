#!/usr/bin/env python3
"""
Unit tests for get_effective_nuclear_charge function using Slater's rules.
"""

import unittest
from mendeleev import element
from research_inertia import get_effective_nuclear_charge


class TestSlaterRules(unittest.TestCase):
    """Test cases for Slater's rules implementation."""

    def test_nitrogen(self):
        """
        Test Nitrogen (Z=7).
        Configuration: 1s² 2s² 2p³
        Expected: S = (4 * 0.35) + (2 * 0.85) = 3.1
        Expected: Z_eff = 7 - 3.1 = 3.9
        """
        N = element(7)
        Z_eff = get_effective_nuclear_charge(N)
        self.assertAlmostEqual(Z_eff, 3.9, places=2,
                               msg=f"Nitrogen Z_eff should be 3.9, got {Z_eff}")

    def test_zinc(self):
        """
        Test Zinc (Z=30).
        Configuration: [Ar] 3d¹⁰ 4s²
        Expected: S = (1 * 0.35) + (18 * 0.85) + (10 * 1.00) = 25.65
        Expected: Z_eff = 30 - 25.65 = 4.35
        """
        Zn = element(30)
        Z_eff = get_effective_nuclear_charge(Zn)
        self.assertAlmostEqual(Z_eff, 4.35, places=2,
                               msg=f"Zinc Z_eff should be 4.35, got {Z_eff}")

    def test_hydrogen(self):
        """
        Test Hydrogen (Z=1).
        Configuration: 1s¹
        Expected: S = 0 (no other electrons to shield)
        Expected: Z_eff = 1 - 0 = 1.0
        """
        H = element(1)
        Z_eff = get_effective_nuclear_charge(H)
        self.assertAlmostEqual(Z_eff, 1.0, places=2,
                               msg=f"Hydrogen Z_eff should be 1.0, got {Z_eff}")

    def test_helium(self):
        """
        Test Helium (Z=2).
        Configuration: 1s²
        Expected: S = 1 * 0.30 = 0.30 (one other 1s electron)
        Expected: Z_eff = 2 - 0.30 = 1.70
        """
        He = element(2)
        Z_eff = get_effective_nuclear_charge(He)
        self.assertAlmostEqual(Z_eff, 1.70, places=2,
                               msg=f"Helium Z_eff should be 1.70, got {Z_eff}")

    def test_oxygen(self):
        """
        Test Oxygen (Z=8).
        Configuration: 1s² 2s² 2p⁴
        For outermost 2p electron:
        Expected: S = (5 * 0.35) + (2 * 0.85) = 1.75 + 1.70 = 3.45
        Expected: Z_eff = 8 - 3.45 = 4.55
        """
        O = element(8)
        Z_eff = get_effective_nuclear_charge(O)
        expected_S = (5 * 0.35) + (2 * 0.85)  # 3.45
        expected_Z_eff = 8 - expected_S  # 4.55
        self.assertAlmostEqual(Z_eff, expected_Z_eff, places=2,
                               msg=f"Oxygen Z_eff should be {expected_Z_eff:.2f}, got {Z_eff}")

    def test_sodium(self):
        """
        Test Sodium (Z=11).
        Configuration: [Ne] 3s¹
        For outermost 3s electron:
        Expected: S = (0 * 0.35) + (8 * 0.85) + (2 * 1.00) = 0 + 6.8 + 2 = 8.8
        Expected: Z_eff = 11 - 8.8 = 2.2
        """
        Na = element(11)
        Z_eff = get_effective_nuclear_charge(Na)
        expected_S = (0 * 0.35) + (8 * 0.85) + (2 * 1.00)  # 8.8
        expected_Z_eff = 11 - expected_S  # 2.2
        self.assertAlmostEqual(Z_eff, expected_Z_eff, places=2,
                               msg=f"Sodium Z_eff should be {expected_Z_eff:.2f}, got {Z_eff}")

    def test_chlorine(self):
        """
        Test Chlorine (Z=17).
        Configuration: [Ne] 3s² 3p⁵
        For outermost 3p electron:
        Expected: S = (6 * 0.35) + (8 * 0.85) + (2 * 1.00) = 2.1 + 6.8 + 2 = 10.9
        Expected: Z_eff = 17 - 10.9 = 6.1
        """
        Cl = element(17)
        Z_eff = get_effective_nuclear_charge(Cl)
        expected_S = (6 * 0.35) + (8 * 0.85) + (2 * 1.00)  # 10.9
        expected_Z_eff = 17 - expected_S  # 6.1
        self.assertAlmostEqual(Z_eff, expected_Z_eff, places=2,
                               msg=f"Chlorine Z_eff should be {expected_Z_eff:.2f}, got {Z_eff}")

    def test_iron(self):
        """
        Test Iron (Z=26).
        Configuration: [Ar] 3d⁶ 4s²
        For outermost 4s electron:
        Expected: S = (1 * 0.35) + (14 * 0.85) + (8 * 1.00) + (2 * 1.00)
                    = 0.35 + 11.9 + 8 + 2 = 22.25
        Expected: Z_eff = 26 - 22.25 = 3.75
        """
        Fe = element(26)
        Z_eff = get_effective_nuclear_charge(Fe)
        # For 4s: same shell (1 other 4s), n-1 (3s² 3p⁶ 3d⁶ = 14), n-2 (2s² 2p⁶ = 8), n-3 (1s² = 2)
        expected_S = (1 * 0.35) + (14 * 0.85) + (8 * 1.00) + (2 * 1.00)  # 22.25
        expected_Z_eff = 26 - expected_S  # 3.75
        self.assertAlmostEqual(Z_eff, expected_Z_eff, places=2,
                               msg=f"Iron Z_eff should be {expected_Z_eff:.2f}, got {Z_eff}")

    def test_scandium_d_orbital(self):
        """
        Test Scandium (Z=21) - testing d orbital rules.
        Configuration: [Ar] 3d¹ 4s²
        For outermost 4s electron:
        Expected: S = (1 * 0.35) + (9 * 0.85) + (8 * 1.00) + (2 * 1.00)
                    = 0.35 + 7.65 + 8 + 2 = 18.0
        Expected: Z_eff = 21 - 18.0 = 3.0
        """
        Sc = element(21)
        Z_eff = get_effective_nuclear_charge(Sc)
        # For 4s: same shell (1 other 4s), n-1 (3s² 3p⁶ 3d¹ = 9), n-2 (2s² 2p⁶ = 8), n-3 (1s² = 2)
        expected_S = (1 * 0.35) + (9 * 0.85) + (8 * 1.00) + (2 * 1.00)  # 18.0
        expected_Z_eff = 21 - expected_S  # 3.0
        self.assertAlmostEqual(Z_eff, expected_Z_eff, places=2,
                               msg=f"Scandium Z_eff should be {expected_Z_eff:.2f}, got {Z_eff}")

    def test_argon(self):
        """
        Test Argon (Z=18).
        Configuration: [Ne] 3s² 3p⁶
        For outermost 3p electron:
        Expected: S = (7 * 0.35) + (8 * 0.85) + (2 * 1.00) = 2.45 + 6.8 + 2 = 11.25
        Expected: Z_eff = 18 - 11.25 = 6.75
        """
        Ar = element(18)
        Z_eff = get_effective_nuclear_charge(Ar)
        expected_S = (7 * 0.35) + (8 * 0.85) + (2 * 1.00)  # 11.25
        expected_Z_eff = 18 - expected_S  # 6.75
        self.assertAlmostEqual(Z_eff, expected_Z_eff, places=2,
                               msg=f"Argon Z_eff should be {expected_Z_eff:.2f}, got {Z_eff}")

    def test_z_eff_positive(self):
        """Test that Z_eff is always positive for all elements."""
        for atomic_number in range(1, 100):
            elem = element(atomic_number)
            Z_eff = get_effective_nuclear_charge(elem)
            self.assertGreater(Z_eff, 0,
                               msg=f"Element {elem.symbol} (Z={atomic_number}) has non-positive Z_eff: {Z_eff}")

    def test_z_eff_less_than_z(self):
        """Test that Z_eff is always less than Z (shielding reduces effective charge)."""
        for atomic_number in range(2, 100):  # Skip hydrogen which has Z_eff = Z
            elem = element(atomic_number)
            Z_eff = get_effective_nuclear_charge(elem)
            self.assertLess(Z_eff, atomic_number,
                            msg=f"Element {elem.symbol} (Z={atomic_number}) has Z_eff >= Z: {Z_eff}")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
