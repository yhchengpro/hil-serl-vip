"""Python SDK for arm kinematics and dynamics"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pinocchio as pin
from pinocchio.utils import zero
from xacrodoc import XacroDoc
from dataclasses import dataclass


def _cos(q: float) -> float:
    """
    Calculate the cosine of an angle in radians.

    Args:
        q (float): angle in radians

    Returns:
        float: cosine of the angle
    """
    while q < -np.pi:
        q += 2 * np.pi
    while q > np.pi:
        q -= 2 * np.pi
    if q < 0:
        return _cos(-q)
    if q > np.pi / 2:
        return -_cos(np.pi - q)
    if abs(q) < 1e-6:
        return 1
    if abs(q - np.pi / 2) < 1e-6:
        return 0
    return np.cos(q)


def _sin(q: float) -> float:
    """
    Calculate the sine of an angle in radians.

    Args:
        q (float): angle in radians

    Returns:
        float: sine of the angle
    """
    while q < -np.pi:
        q += 2 * np.pi
    while q > np.pi:
        q -= 2 * np.pi
    if q < 0:
        return -_sin(-q)
    if q > np.pi / 2:
        return _sin(np.pi - q)
    if abs(q) < 1e-6:
        return 0
    if abs(q - np.pi / 2) < 1e-6:
        return 1
    return np.sin(q)


@dataclass
class ArmDH:
    """Arm Denavit-Hartenberg parameters class"""

    alpha: np.array  # alpha[i]
    a: np.array  # a[i]
    d: np.array  # d[i+1]
    theta_offset: np.array  # theta_offset[i+1]
    end_convert_matrix: np.array
    joints_limit: np.array

    def adjacent_transform(self, q: float, index: int) -> np.array:
        """
        Get the transformation matrix of the i-th joint. (1-7)

        Args:
            q (float): joint angle
            index (int): joint index

        Returns:
            np.array: transformation matrix
        """
        # fmt: off
        # pylint: disable=line-too-long
        res =  np.array([
            [                            _cos(q + self.theta_offset[index-1]),                            -_sin(q + self.theta_offset[index-1]),                          0,                              self.a[index-1]],
            [_sin(q + self.theta_offset[index-1]) * _cos(self.alpha[index-1]), _cos(q + self.theta_offset[index-1]) * _cos(self.alpha[index-1]), -_sin(self.alpha[index-1]), -self.d[index-1] * _sin(self.alpha[index-1])],
            [_sin(q + self.theta_offset[index-1]) * _sin(self.alpha[index-1]), _cos(q + self.theta_offset[index-1]) * _sin(self.alpha[index-1]),  _cos(self.alpha[index-1]),  self.d[index-1] * _cos(self.alpha[index-1])],
            [                                                               0,                                                                0,                          0,                                            1]
        ])
        # pylint: enable=line-too-long
        # fmt: on
        # print("adjacent_transform", index)
        # print(res)
        return res

    def valid_joints(self, q: np.array) -> bool:
        """
        Check if the joint angles are within the limits.

        Args:
            q (np.array): joint angles

        Returns:
            bool: True if valid, False otherwise
        """
        for i in range(len(q)):
            if q[i] < self.joints_limit[i][0] or q[i] > self.joints_limit[i][1]:
                return False
        return True

    def limit_joints(self, q: np.array) -> np.array:
        """
        Limit the joint angles to the limits.

        Args:
            q (np.array): joint angles

        Returns:
            np.array: limited joint angles
        """
        for i in range(len(q)):
            while q[i] < self.joints_limit[i][0]:
                q[i] += 2 * np.pi
            while q[i] > self.joints_limit[i][1]:
                q[i] -= 2 * np.pi
        # print("q", q)
        if not self.valid_joints(q):
            # print("joints cannot be limited")
            # print(q)
            return None
        return q


# fmt: off
# pylint: disable=line-too-long
DEFAULT_DH_MAP = {
    "play_short": {
        "none": ArmDH(
            alpha        = np.array([     0,                   np.pi / 2,                               0, -np.pi / 2, np.pi / 2, -np.pi / 2,         0]),
            a            = np.array([     0,                           0,                        0.270092,          0,         0,          0,         0]),
            d            = np.array([0.1127,                           0,                               0,   0.290149,         0,          0, 0.0864995]),
            theta_offset = np.array([     0, np.pi - 21.93 / 180 * np.pi, np.pi / 2 + 21.93 / 180 * np.pi, -np.pi / 2,         0,          0,         0]),
            end_convert_matrix = np.array([
                [ 0, -1,  0, 0],
                [ 0,  0, -1, 0],
                [ 1,  0,  0, 0],
                [ 0,  0,  0, 1]
            ]),
            joints_limit= np.array([
                [-3.151, 2.080],  # joint 1
                [-2.963, 0.181],  # joint 2
                [-0.094, 3.161],  # joint 3
                [-3.012, 3.012],  # joint 4
                [-1.859, 1.859],  # joint 5
                [-3.017, 3.017],  # joint 6
            ]
            )
        ),
        "G2": ArmDH(
            alpha        = np.array([     0,                   np.pi / 2,                               0, -np.pi / 2, np.pi / 2, -np.pi / 2,         0]),
            a            = np.array([     0,                           0,                        0.270092,          0,         0,          0,         0]),
            d            = np.array([0.1127,                           0,                               0,   0.290149,         0,          0, 0.2466995]),
            theta_offset = np.array([     0, np.pi - 21.93 / 180 * np.pi, np.pi / 2 + 21.93 / 180 * np.pi, -np.pi / 2,         0,          0,         0]),
            end_convert_matrix = np.array([
                [ 0, -1,  0, 0],
                [ 0,  0, -1, 0],
                [ 1,  0,  0, 0],
                [ 0,  0,  0, 1]
            ]),
            joints_limit= np.array([
                [-3.151, 2.080],  # joint 1
                [-2.963, 0.181],  # joint 2
                [-0.094, 3.161],  # joint 3
                [-3.012, 3.012],  # joint 4
                [-1.859, 1.859],  # joint 5
                [-3.017, 3.017],  # joint 6
            ]
            )
        ),
        "E2B": ArmDH(
            alpha        = np.array([     0,                   np.pi / 2,                               0, -np.pi / 2, np.pi / 2, -np.pi / 2,         0]),
            a            = np.array([     0,                           0,                        0.270092,          0,         0,          0,         0]),
            d            = np.array([0.1127,                           0,                               0,   0.290149,         0,          0, 0.1488995]),
            theta_offset = np.array([     0, np.pi - 21.93 / 180 * np.pi, np.pi / 2 + 21.93 / 180 * np.pi, -np.pi / 2,         0,          0,         0]),
            end_convert_matrix = np.array([
                [ 0, -1,  0, 0],
                [ 0,  0, -1, 0],
                [ 1,  0,  0, 0],
                [ 0,  0,  0, 1]
            ]),
            joints_limit= np.array([
                [-3.151, 2.080],  # joint 1
                [-2.963, 0.181],  # joint 2
                [-0.094, 3.161],  # joint 3
                [-3.012, 3.012],  # joint 4
                [-1.859, 1.859],  # joint 5
                [-3.017, 3.017],  # joint 6
            ]
            )
        ),
        "INSFTP56L": ArmDH(
            alpha        = np.array([     0,                   np.pi / 2,                               0, -np.pi / 2, np.pi / 2, -np.pi / 2, 0]),
            a            = np.array([     0,                           0,                        0.270092,          0,         0,          0, 0]),
            d            = np.array([0.1127,                           0,                               0,   0.290149,         0,          0, 0]),
            theta_offset = np.array([     0, np.pi - 21.93 / 180 * np.pi, np.pi / 2 + 21.93 / 180 * np.pi, -np.pi / 2,         0,          0, 0]),
            end_convert_matrix = np.array([
                [ 0.8414710, -0.5403023,  0,      0.08],
                [         0,          0, -1, -0.038225],
                [ 0.5403023,  0.8414710,  0, 0.2814995],
                [         0,          0,  0,         1]
            ]),
            joints_limit= np.array([
                [-3.151, 2.080],  # joint 1
                [-2.963, 0.181],  # joint 2
                [-0.094, 3.161],  # joint 3
                [-3.012, 3.012],  # joint 4
                [-1.859, 1.859],  # joint 5
                [-3.017, 3.017],  # joint 6
            ]
            )
        ),
        "INSFTP56R": ArmDH(
            alpha        = np.array([     0,                   np.pi / 2,                               0, -np.pi / 2, np.pi / 2, -np.pi / 2, 0]),
            a            = np.array([     0,                           0,                        0.270092,          0,         0,          0, 0]),
            d            = np.array([0.1127,                           0,                               0,   0.290149,         0,          0, 0]),
            theta_offset = np.array([     0, np.pi - 21.93 / 180 * np.pi, np.pi / 2 + 21.93 / 180 * np.pi, -np.pi / 2,         0,          0, 0]),
            end_convert_matrix = np.array([
                [ -0.8414710, -0.5403023,  0,     -0.08],
                [          0,          0, -1, -0.038225],
                [  0.5403023, -0.8414710,  0, 0.2814995],
                [          0,          0,  0,         1]
            ]),
            joints_limit= np.array([
                [-3.151, 2.080],  # joint 1
                [-2.963, 0.181],  # joint 2
                [-0.094, 3.161],  # joint 3
                [-3.012, 3.012],  # joint 4
                [-1.859, 1.859],  # joint 5
                [-3.017, 3.017],  # joint 6
            ]
            )
        ),
    },
    "play_long": {
        "none": ArmDH(
            alpha        = np.array([     0,                   np.pi / 2,                               0, -np.pi / 2, np.pi / 2, -np.pi / 2,         0]),
            a            = np.array([     0,                           0,                        0.308540,          0,         0,          0,         0]),
            d            = np.array([0.1127,                           0,                               0,   0.290149,         0,          0, 0.0864995]),
            theta_offset = np.array([     0, np.pi - 21.93 / 180 * np.pi, np.pi / 2 + 21.93 / 180 * np.pi, -np.pi / 2,         0,          0,         0]),
            end_convert_matrix = np.array([
                [ 0, -1,  0, 0],
                [ 0,  0, -1, 0],
                [ 1,  0,  0, 0],
                [ 0,  0,  0, 1]
            ]),
            joints_limit= np.array([
                [-3.151, 2.080],  # joint 1
                [-2.963, 0.181],  # joint 2
                [-0.094, 3.161],  # joint 3
                [-3.012, 3.012],  # joint 4
                [-1.859, 1.859],  # joint 5
                [-3.017, 3.017],  # joint 6
            ]
            )
        ),
        "G2": ArmDH(
            alpha        = np.array([     0,                   np.pi / 2,                               0, -np.pi / 2, np.pi / 2, -np.pi / 2,         0]),
            a            = np.array([     0,                           0,                        0.308540,          0,         0,          0,         0]),
            d            = np.array([0.1127,                           0,                               0,   0.290149,         0,          0, 0.2466995]),
            theta_offset = np.array([     0, np.pi - 21.93 / 180 * np.pi, np.pi / 2 + 21.93 / 180 * np.pi, -np.pi / 2,         0,          0,         0]),
            end_convert_matrix = np.array([
                [ 0, -1,  0, 0],
                [ 0,  0, -1, 0],
                [ 1,  0,  0, 0],
                [ 0,  0,  0, 1]
            ]),
            joints_limit= np.array([
                [-3.151, 2.080],  # joint 1
                [-2.963, 0.181],  # joint 2
                [-0.094, 3.161],  # joint 3
                [-3.012, 3.012],  # joint 4
                [-1.859, 1.859],  # joint 5
                [-3.017, 3.017],  # joint 6
            ]
            )
        ),
        "E2B": ArmDH(
            alpha        = np.array([     0,                   np.pi / 2,                               0, -np.pi / 2, np.pi / 2, -np.pi / 2,         0]),
            a            = np.array([     0,                           0,                        0.308540,          0,         0,          0,         0]),
            d            = np.array([0.1127,                           0,                               0,   0.290149,         0,          0, 0.1488995]),
            theta_offset = np.array([     0, np.pi - 21.93 / 180 * np.pi, np.pi / 2 + 21.93 / 180 * np.pi, -np.pi / 2,         0,          0,         0]),
            end_convert_matrix = np.array([
                [ 0, -1,  0, 0],
                [ 0,  0, -1, 0],
                [ 1,  0,  0, 0],
                [ 0,  0,  0, 1]
            ]),
            joints_limit= np.array([
                [-3.151, 2.080],  # joint 1
                [-2.963, 0.181],  # joint 2
                [-0.094, 3.161],  # joint 3
                [-3.012, 3.012],  # joint 4
                [-1.859, 1.859],  # joint 5
                [-3.017, 3.017],  # joint 6
            ]
            )
        ),
    },
    "play_pro": {
        "none": ArmDH(
            alpha        = np.array([     0,                   np.pi / 2,                               0, -np.pi / 2, np.pi / 2, -np.pi / 2,         0]),
            a            = np.array([     0,                           0,                        0.270092,          0,         0,          0,         0]),
            d            = np.array([0.1127,                           0,                               0,   0.290149,         0,          0, 0.0864995]),
            theta_offset = np.array([     0, np.pi - 21.93 / 180 * np.pi, np.pi / 2 + 21.93 / 180 * np.pi, -np.pi / 2,         0,          0,         0]),
            end_convert_matrix = np.array([
                [ 0, -1,  0, 0],
                [ 0,  0, -1, 0],
                [ 1,  0,  0, 0],
                [ 0,  0,  0, 1]
            ]),
            joints_limit= np.array([
                [-2.740, 2.740],  # joint 1
                [-2.963, 0.181],  # joint 2
                [-0.094, 3.161],  # joint 3
                [-3.012, 3.012],  # joint 4
                [-1.859, 1.859],  # joint 5
                [-3.017, 3.017],  # joint 6
            ]
            )
        ),
        "G2": ArmDH(
            alpha        = np.array([     0,                   np.pi / 2,                               0, -np.pi / 2, np.pi / 2, -np.pi / 2,         0]),
            a            = np.array([     0,                           0,                        0.270092,          0,         0,          0,         0]),
            d            = np.array([0.1127,                           0,                               0,   0.290149,         0,          0, 0.2466995]),
            theta_offset = np.array([     0, np.pi - 21.93 / 180 * np.pi, np.pi / 2 + 21.93 / 180 * np.pi, -np.pi / 2,         0,          0,         0]),
            end_convert_matrix = np.array([
                [ 0, -1,  0, 0],
                [ 0,  0, -1, 0],
                [ 1,  0,  0, 0],
                [ 0,  0,  0, 1]
            ]),
            joints_limit= np.array([
                [-2.740, 2.740],  # joint 1
                [-2.963, 0.181],  # joint 2
                [-0.094, 3.161],  # joint 3
                [-3.012, 3.012],  # joint 4
                [-1.859, 1.859],  # joint 5
                [-3.017, 3.017],  # joint 6
            ]
            )
        ),
        "E2B": ArmDH(
            alpha        = np.array([     0,                   np.pi / 2,                               0, -np.pi / 2, np.pi / 2, -np.pi / 2,         0]),
            a            = np.array([     0,                           0,                        0.270092,          0,         0,          0,         0]),
            d            = np.array([0.1127,                           0,                               0,   0.290149,         0,          0, 0.1488995]),
            theta_offset = np.array([     0, np.pi - 21.93 / 180 * np.pi, np.pi / 2 + 21.93 / 180 * np.pi, -np.pi / 2,         0,          0,         0]),
            end_convert_matrix = np.array([
                [ 0, -1,  0, 0],
                [ 0,  0, -1, 0],
                [ 1,  0,  0, 0],
                [ 0,  0,  0, 1]
            ]),
            joints_limit= np.array([
                [-2.740, 2.740],  # joint 1
                [-2.963, 0.181],  # joint 2
                [-0.094, 3.161],  # joint 3
                [-3.012, 3.012],  # joint 4
                [-1.859, 1.859],  # joint 5
                [-3.017, 3.017],  # joint 6
            ]
            )
        ),
        "INSFTP56L": ArmDH(
            alpha        = np.array([     0,                   np.pi / 2,                               0, -np.pi / 2, np.pi / 2, -np.pi / 2, 0]),
            a            = np.array([     0,                           0,                        0.270092,          0,         0,          0, 0]),
            d            = np.array([0.1127,                           0,                               0,   0.290149,         0,          0, 0]),
            theta_offset = np.array([     0, np.pi - 21.93 / 180 * np.pi, np.pi / 2 + 21.93 / 180 * np.pi, -np.pi / 2,         0,          0, 0]),
            end_convert_matrix = np.array([
                [ 0.8414710, -0.5403023,  0,      0.08],
                [         0,          0, -1, -0.038225],
                [ 0.5403023,  0.8414710,  0, 0.2814995],
                [         0,          0,  0,         1]
            ]),
            joints_limit= np.array([
                [-2.740, 2.740],  # joint 1
                [-2.963, 0.181],  # joint 2
                [-0.094, 3.161],  # joint 3
                [-3.012, 3.012],  # joint 4
                [-1.859, 1.859],  # joint 5
                [-3.017, 3.017],  # joint 6
            ]
            )
        ),
        "INSFTP56R": ArmDH(
            alpha        = np.array([     0,                   np.pi / 2,                               0, -np.pi / 2, np.pi / 2, -np.pi / 2, 0]),
            a            = np.array([     0,                           0,                        0.270092,          0,         0,          0, 0]),
            d            = np.array([0.1127,                           0,                               0,   0.290149,         0,          0, 0]),
            theta_offset = np.array([     0, np.pi - 21.93 / 180 * np.pi, np.pi / 2 + 21.93 / 180 * np.pi, -np.pi / 2,         0,          0, 0]),
            end_convert_matrix = np.array([
                [ -0.8414710, -0.5403023,  0,     -0.08],
                [          0,          0, -1, -0.038225],
                [  0.5403023, -0.8414710,  0, 0.2814995],
                [          0,          0,  0,         1]
            ]),
            joints_limit= np.array([
                [-2.740, 2.740],  # joint 1
                [-2.963, 0.181],  # joint 2
                [-0.094, 3.161],  # joint 3
                [-3.012, 3.012],  # joint 4
                [-1.859, 1.859],  # joint 5
                [-3.017, 3.017],  # joint 6
            ]
            )
        ),
    },
    "play_lite": {
        "none": ArmDH(
            alpha        = np.array([     0,                   np.pi / 2,                               0, -np.pi / 2, np.pi / 2, -np.pi / 2,         0]),
            a            = np.array([     0,                           0,                        0.270092,          0,         0,          0,         0]),
            d            = np.array([0.1127,                           0,                               0,   0.290149,         0,          0, 0.0864995]),
            theta_offset = np.array([     0, np.pi - 21.93 / 180 * np.pi, np.pi / 2 + 21.93 / 180 * np.pi, -np.pi / 2,         0,          0,         0]),
            end_convert_matrix = np.array([
                [ 0, -1,  0, 0],
                [ 0,  0, -1, 0],
                [ 1,  0,  0, 0],
                [ 0,  0,  0, 1]
            ]),
            joints_limit= np.array([
                [-2.740, 2.740],  # joint 1
                [-2.963, 0.181],  # joint 2
                [-0.094, 3.161],  # joint 3
                [-3.012, 3.012],  # joint 4
                [-1.859, 1.859],  # joint 5
                [-3.017, 3.017],  # joint 6
            ]
            )
        ),
        "G2": ArmDH(
            alpha        = np.array([     0,                   np.pi / 2,                               0, -np.pi / 2, np.pi / 2, -np.pi / 2,         0]),
            a            = np.array([     0,                           0,                        0.270092,          0,         0,          0,         0]),
            d            = np.array([0.1127,                           0,                               0,   0.290149,         0,          0, 0.2466995]),
            theta_offset = np.array([     0, np.pi - 21.93 / 180 * np.pi, np.pi / 2 + 21.93 / 180 * np.pi, -np.pi / 2,         0,          0,         0]),
            end_convert_matrix = np.array([
                [ 0, -1,  0, 0],
                [ 0,  0, -1, 0],
                [ 1,  0,  0, 0],
                [ 0,  0,  0, 1]
            ]),
            joints_limit= np.array([
                [-2.740, 2.740],  # joint 1
                [-2.963, 0.181],  # joint 2
                [-0.094, 3.161],  # joint 3
                [-3.012, 3.012],  # joint 4
                [-1.859, 1.859],  # joint 5
                [-3.017, 3.017],  # joint 6
            ]
            )
        ),
        "E2B": ArmDH(
            alpha        = np.array([     0,                   np.pi / 2,                               0, -np.pi / 2, np.pi / 2, -np.pi / 2,         0]),
            a            = np.array([     0,                           0,                        0.270092,          0,         0,          0,         0]),
            d            = np.array([0.1127,                           0,                               0,   0.290149,         0,          0, 0.1488995]),
            theta_offset = np.array([     0, np.pi - 21.93 / 180 * np.pi, np.pi / 2 + 21.93 / 180 * np.pi, -np.pi / 2,         0,          0,         0]),
            end_convert_matrix = np.array([
                [ 0, -1,  0, 0],
                [ 0,  0, -1, 0],
                [ 1,  0,  0, 0],
                [ 0,  0,  0, 1]
            ]),
            joints_limit= np.array([
                [-2.740, 2.740],  # joint 1
                [-2.963, 0.181],  # joint 2
                [-0.094, 3.161],  # joint 3
                [-3.012, 3.012],  # joint 4
                [-1.859, 1.859],  # joint 5
                [-3.017, 3.017],  # joint 6
            ]
            )
        ),
        "INSFTP56L": ArmDH(
            alpha        = np.array([     0,                   np.pi / 2,                               0, -np.pi / 2, np.pi / 2, -np.pi / 2, 0]),
            a            = np.array([     0,                           0,                        0.270092,          0,         0,          0, 0]),
            d            = np.array([0.1127,                           0,                               0,   0.290149,         0,          0, 0]),
            theta_offset = np.array([     0, np.pi - 21.93 / 180 * np.pi, np.pi / 2 + 21.93 / 180 * np.pi, -np.pi / 2,         0,          0, 0]),
            end_convert_matrix = np.array([
                [ 0.8414710, -0.5403023,  0,      0.08],
                [         0,          0, -1, -0.038225],
                [ 0.5403023,  0.8414710,  0, 0.2814995],
                [         0,          0,  0,         1]
            ]),
            joints_limit= np.array([
                [-2.740, 2.740],  # joint 1
                [-2.963, 0.181],  # joint 2
                [-0.094, 3.161],  # joint 3
                [-3.012, 3.012],  # joint 4
                [-1.859, 1.859],  # joint 5
                [-3.017, 3.017],  # joint 6
            ]
            )
        ),
        "INSFTP56R": ArmDH(
            alpha        = np.array([     0,                   np.pi / 2,                               0, -np.pi / 2, np.pi / 2, -np.pi / 2, 0]),
            a            = np.array([     0,                           0,                        0.270092,          0,         0,          0, 0]),
            d            = np.array([0.1127,                           0,                               0,   0.290149,         0,          0, 0]),
            theta_offset = np.array([     0, np.pi - 21.93 / 180 * np.pi, np.pi / 2 + 21.93 / 180 * np.pi, -np.pi / 2,         0,          0, 0]),
            end_convert_matrix = np.array([
                [ -0.8414710, -0.5403023,  0,     -0.08],
                [          0,          0, -1, -0.038225],
                [  0.5403023, -0.8414710,  0, 0.2814995],
                [          0,          0,  0,         1]
            ]),
            joints_limit= np.array([
                [-2.740, 2.740],  # joint 1
                [-2.963, 0.181],  # joint 2
                [-0.094, 3.161],  # joint 3
                [-3.012, 3.012],  # joint 4
                [-1.859, 1.859],  # joint 5
                [-3.017, 3.017],  # joint 6
            ]
            )
        ),
    },
}
# pylint: enable=line-too-long
# fmt: on


class ArmKdl:
    """Arm kinematics and dynamics class"""

    def __init__(self, arm_type: str = "play_short", eef_type: str = "G2"):
        if eef_type == "old_G2":
            eef_type = "G2"
        self.arm_type = arm_type
        self.eef_type = eef_type
        self.dh = DEFAULT_DH_MAP[arm_type][eef_type]
        doc = XacroDoc.from_file(
            os.path.join(
                str(Path(__file__).resolve().parent),
                "./models/urdf/airbot_" + arm_type + ".xacro",
            ),
            subargs={"eef_type": eef_type, "disable_ros2_control": "true"},
        )

        urdf_str = doc.to_urdf_string()

        self._model = pin.buildModelFromXML(urdf_str)
        print(self._model)
        self._data = self._model.createData()
        self.size = self._model.nv

    def inverse_dynamics(
        self, q: np.ndarray, v: np.ndarray, a: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the inverse dynamics of the arm.

        Args:
            q (np.ndarray): current joint positions
            v (np.ndarray): current joint velocities
            a (np.ndarray): current joint accelerations

        Returns:
            np.ndarray: joint torques
        """
        if len(q) < self.size:
            q = np.append(q, zero(self.size - len(q)))
        if len(v) < self.size:
            v = np.append(v, zero(self.size - len(v)))
        if len(a) < self.size:
            a = np.append(a, zero(self.size - len(a)))
        return pin.rnea(self._model, self._data, q, v, a)[0:6]

    def jacobian(self, q: np.array) -> np.array:
        """
        Calculate the Jacobian matrix of the arm.

        Args:
            q (np.array): current joint positions (6 elements)

        Returns:
            np.array: 6x6 Jacobian matrix [J_v; J_w] where J_v is linear velocity
                    and J_w is angular velocity
        """
        assert len(q) == 6, "q must be 6 elements"

        J = np.zeros((6, 7))
        T = [np.eye(4)]  # T[0] is base frame

        # Calculate T0_1 to T0_6
        for i in range(6):
            T_i = np.dot(T[-1], self.dh.adjacent_transform(q[i], i + 1))
            T.append(T_i)

        T_7 = np.dot(T[-1], self.dh.adjacent_transform(0, 7))
        T_7 = np.dot(T_7, self.dh.end_convert_matrix)
        T.append(T_7)

        o_n = T_7[0:3, 3]

        # Calculate Jacobian columns for each joint
        for i in range(7):
            z_i = T[i + 1][0:3, 2]  # z-axis
            o_i = T[i + 1][0:3, 3]  # origin position

            # Linear velocity part: J_v = z_i × (o_n - o_i)
            J[0:3, i] = np.cross(z_i, o_n - o_i)

            # Angular velocity part: J_w = z_i
            J[3:6, i] = z_i

        return np.round(J[:, :6], decimals=9)

    def forward_kinematics(self, q: np.array) -> np.array:
        """
        Calculate the forward kinematics of the arm.

        Args:
            q (np.array): current joint positions

        Returns:
            np.array: end effector position
        """
        assert len(q) == 6, "q must be 6 elements"

        res = np.eye(4)
        for i in range(6):
            res = np.dot(res, self.dh.adjacent_transform(q[i], i + 1))
            # print(i+1)
            # print(res)
        res = np.dot(res, self.dh.adjacent_transform(0, 7))
        # print(7)
        # print(res)
        res = np.dot(res, self.dh.end_convert_matrix)
        # print("res")
        return np.round(res, decimals=9)

    def inverse_kinematics(self, pose: np.array, ref_pos: np.array = None) -> np.array:
        """
        Calculate the inverse kinematics of the arm.

        Args:
            pose (np.array): end effector pose
            ref_pos (np.array): reference position

        Returns:
            np.array: joint angles
        """
        assert len(pose) == 4 and len(pose[0]) == 4, "pose must be 4x4 matrix"
        if ref_pos is not None:
            assert len(ref_pos) == 6, "ref_pos must be 6 elements"

        res = []

        arm_pose = self._cut_end(pose)

        # print("arm_pose")
        # print(arm_pose)

        wrist_pos = arm_pose[0:3, 3]
        s3 = -(
            wrist_pos[0] ** 2
            + wrist_pos[1] ** 2
            + (wrist_pos[2] - self.dh.d[0]) ** 2
            - self.dh.d[3] ** 2
            - self.dh.a[2] ** 2
        ) / (2 * self.dh.d[3] * self.dh.a[2])

        if s3 < -1 or s3 > 1:
            return None

        for i1 in [1, -1]:
            theta1 = (
                np.arctan2(i1 * wrist_pos[1], i1 * wrist_pos[0])
                - self.dh.theta_offset[0]
            )
            for i2 in [1, -1]:
                c3 = i2 * np.sqrt(1 - s3**2)
                theta3 = np.arctan2(s3, c3) - self.dh.theta_offset[2]

                k1 = self.dh.a[2] - self.dh.d[3] * s3
                k2 = self.dh.d[3] * c3
                k3 = np.sqrt(wrist_pos[0] ** 2 + wrist_pos[1] ** 2)
                k4 = wrist_pos[2] - self.dh.d[0]

                theta2 = (
                    np.arctan2(-i1 * k2 * k3 + k1 * k4, i1 * k1 * k3 + k2 * k4)
                    - self.dh.theta_offset[1]
                )

                t_3_0 = np.dot(
                    self.dh.adjacent_transform(theta1, 1),
                    np.dot(
                        self.dh.adjacent_transform(theta2, 2),
                        self.dh.adjacent_transform(theta3, 3),
                    ),
                )
                t_6_3 = np.dot(
                    np.linalg.inv(t_3_0),
                    arm_pose,
                )
                # print("t_6_3")
                # print(t_6_3)
                for i5 in [1, -1]:
                    theta5 = (
                        np.arctan2(
                            i5 * np.sqrt(t_6_3[1, 0] ** 2 + t_6_3[1, 1] ** 2),
                            t_6_3[1, 2],
                        )
                        - self.dh.theta_offset[4]
                    )
                    if abs(theta5) > 1e-3:
                        theta4 = (
                            np.arctan2(i5 * t_6_3[2, 2], -i5 * t_6_3[0, 2])
                            - self.dh.theta_offset[3]
                        )
                        theta6 = (
                            np.arctan2(-i5 * t_6_3[1, 1], i5 * t_6_3[1, 0])
                            - self.dh.theta_offset[5]
                        )
                    else:
                        # if theta5 is too small, which means that motor 3, 4, 5, 6 are on the same line, calculate the total rotation angle of motor 4 and 6 through the rotation of t_6_3
                        rot_angle = np.arctan2(-t_6_3[2, 0], t_6_3[0, 0])
                        if ref_pos is not None:
                            rot_angle = rot_angle - (
                                ref_pos[3]
                                + self.dh.theta_offset[3]
                                + ref_pos[5]
                                + self.dh.theta_offset[5]
                            )
                            min_angle = (
                                self.dh.joints_limit[3][0]
                                - ref_pos[3]
                                + self.dh.joints_limit[5][0]
                                - ref_pos[5]
                            )
                            max_angle = (
                                self.dh.joints_limit[3][1]
                                - ref_pos[3]
                                + self.dh.joints_limit[5][1]
                                - ref_pos[5]
                            )
                            while rot_angle < min_angle:
                                rot_angle += 2 * np.pi
                            while rot_angle > max_angle:
                                rot_angle -= 2 * np.pi
                            while True:
                                if rot_angle + 2 * np.pi < max_angle and abs(
                                    rot_angle + 2 * np.pi
                                ) < abs(rot_angle):
                                    rot_angle += 2 * np.pi
                                elif rot_angle - 2 * np.pi > min_angle and abs(
                                    rot_angle - 2 * np.pi
                                ) < abs(rot_angle):
                                    rot_angle -= 2 * np.pi
                                else:
                                    break
                            if rot_angle / 2 + ref_pos[3] > self.dh.joints_limit[3][1]:
                                theta4 = self.dh.joints_limit[3][1]
                                theta6 = (
                                    ref_pos[3]
                                    + ref_pos[5]
                                    + rot_angle
                                    - self.dh.joints_limit[3][1]
                                )
                            elif (
                                rot_angle / 2 + ref_pos[3] < self.dh.joints_limit[3][0]
                            ):
                                theta4 = self.dh.joints_limit[3][0]
                                theta6 = (
                                    ref_pos[3]
                                    + ref_pos[5]
                                    + rot_angle
                                    - self.dh.joints_limit[3][0]
                                )
                            elif (
                                rot_angle / 2 + ref_pos[5] > self.dh.joints_limit[5][1]
                            ):
                                theta6 = self.dh.joints_limit[5][1]
                                theta4 = (
                                    ref_pos[3]
                                    + ref_pos[5]
                                    + rot_angle
                                    - self.dh.joints_limit[5][1]
                                )
                            elif (
                                rot_angle / 2 + ref_pos[5] < self.dh.joints_limit[5][0]
                            ):
                                theta6 = self.dh.joints_limit[5][0]
                                theta4 = (
                                    ref_pos[3]
                                    + ref_pos[5]
                                    + rot_angle
                                    - self.dh.joints_limit[5][0]
                                )
                            else:
                                theta4 = rot_angle / 2 + ref_pos[3]
                                theta6 = rot_angle / 2 + ref_pos[5]
                        else:
                            theta4 = rot_angle / 2 - self.dh.theta_offset[3]
                            theta6 = rot_angle / 2 - self.dh.theta_offset[5]

                    # print("i5 ", i5, "t_6_3[1,1]", t_6_3[1, 1], "t_6_3[1,0]", t_6_3[1, 0])
                    # print(-i5 * t_6_3[1, 1], i5 * t_6_3[1, 0], np.arctan2(-i5 * t_6_3[1, 1], i5 * t_6_3[1, 0]), self.dh.theta_offset[5])
                    # print("theta6", theta6)
                    # print("theta1,2,3,4,5,6")
                    # print(theta1, theta2, theta3, theta4, theta5, theta6)
                    joints = self.dh.limit_joints(
                        np.array(
                            [
                                theta1,
                                theta2,
                                theta3,
                                theta4,
                                theta5,
                                theta6,
                            ]
                        )
                    )
                    if joints is not None:
                        res.append(joints)
        if ref_pos is not None and len(res):
            min_bias = float("inf")
            for i in range(len(res)):
                bias = sum(abs(res[i][j] - ref_pos[j]) for j in range(len(ref_pos)))
                if bias < min_bias:
                    min_bias = bias
                    best_joints = res[i]
            res = [best_joints]
        return res

    def _cut_end(self, pose: np.array) -> np.array:
        """
        Cut the end effector position to joint4(,5, 6).

        Args:
            pose (np.array): end effector pose

        Returns:
            np.array: joint4(,5, 6) pose
        """
        assert len(pose) == 4 and len(pose[0]) == 4, "pose must be 4x4 matrix"

        pose = np.dot(pose, np.linalg.inv(self.dh.end_convert_matrix))

        pos = pose[0:3, 3]
        rot = pose[0:3, 0:3]

        res_pos = pos - rot[:, 2] * self.dh.d[6]
        res_rot = rot
        res = np.eye(4)
        res[0:3, 0:3] = res_rot
        res[0:3, 3] = res_pos
        res[3, 3] = 1.0

        return res


def validate_jacobian(arm_kdl, q, epsilon=1e-4):
    """完整验证几何雅可比矩阵"""
    J_geo = np.round(arm_kdl.jacobian(q), 3)
    # 验证线速度部分（前3行）
    J_v_numerical = np.zeros((3, len(q)))

    # 验证角速度部分（后3行）- 这部分更复杂
    J_w_numerical = np.zeros((3, len(q)))

    # 获取当前位姿
    T_current = arm_kdl.forward_kinematics(q)
    p_current = T_current[:3, 3]
    R_current = T_current[:3, :3]

    for i in range(len(q)):
        dq = np.zeros_like(q)
        dq[i] = epsilon

        # 计算位置变化 - 线速度验证
        T_plus = arm_kdl.forward_kinematics(q + dq)
        T_minus = arm_kdl.forward_kinematics(q - dq)

        p_plus = T_plus[:3, 3]
        p_minus = T_minus[:3, 3]
        J_v_numerical[:, i] = (p_plus - p_minus) / (2 * epsilon)

        # 计算角速度 - 使用旋转矩阵的变化
        R_plus = T_plus[:3, :3]
        R_minus = T_minus[:3, :3]

        # 角速度的近似：ω ≈ (R_plus - R_minus) * R_current^T / (2*epsilon)的反对称部分
        dR = (R_plus - R_minus) / (2 * epsilon)
        omega_skew = dR @ R_current.T

        # 提取反对称矩阵的向量形式
        J_w_numerical[0, i] = omega_skew[2, 1]  # ω_x
        J_w_numerical[1, i] = omega_skew[0, 2]  # ω_y
        J_w_numerical[2, i] = omega_skew[1, 0]  # ω_z

    print("数值解jacobian：\n", np.round(np.concatenate([J_v_numerical, J_w_numerical]), 3))

    # 比较结果
    print("=== 线速度部分验证 ===")
    print("数值计算:", J_v_numerical)
    print("解析计算:", J_geo[:3, :])
    print("最大误差:", np.max(np.abs(J_geo[:3, :] - J_v_numerical)))

    print("\n=== 角速度部分验证 ===")
    print("数值计算:", np.round(J_w_numerical, 3))
    print("解析计算:", J_geo[3:, :])
    print("最大误差:", np.max(np.abs(J_geo[3:, :] - J_w_numerical)))

    return (
        np.max(np.abs(J_geo[:3, :] - J_v_numerical)) < 1e-4
        and np.max(np.abs(J_geo[3:, :] - J_w_numerical)) < 1e-4
    )


def main():
    """
    Example usage of the ArmKdl class.
    """
    arm_kdl = ArmKdl(eef_type="none")
    # print(
    #     arm_kdl.inverse_dynamics(
    #         np.array([0.0, 0.171, -0.087, -1.571, 0.105, 1.571]), zero(6), zero(6)
    #     )
    # )
    # print(
    #     arm_kdl.inverse_dynamics(
    #         np.array([0.0, 0.0, 0.0, 0.0, -1.57, 0.0]), zero(6), zero(6)
    #     )
    # )
    end_pose = arm_kdl.forward_kinematics(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
    print("end_pose")
    print(end_pose)
    # 0.250000 0.200000 0.450000 0.000000 0.000000 -0.000000 1.000000
    # end_pose = np.array(
    #     [
    #         [0.0000000, 0.0000000, 1.0000000, 0.31657160038479626],
    #         [0.9999983, 0.0018234, -0.0000000, 0.07548884871169535],
    #         [-0.0018234, 0.9999983, -0.0000000, 0.0],
    #         [0.0, 0.0, 0.0, 1.0],
    #     ]
    # )
    
    def construct_homogeneous_matrix(tcp_pose):
        """
        Construct the homogeneous transformation matrix from given pose.
        args: tcp_pose: (x, y, z, qx, qy, qz, qw)
        """
        from scipy.spatial.transform import Rotation as R
        rotation = R.from_quat(tcp_pose[3:]).as_matrix()
        translation = np.array(tcp_pose[:3])
        T = np.zeros((4, 4))
        T[:3, :3] = rotation
        T[:3, 3] = translation
        T[3, 3] = 1
        return T

    joints = arm_kdl.inverse_kinematics(
        construct_homogeneous_matrix(np.array([0.0, 0.0, 0.03149, 0.0, 0.0, 0.0, 1.0])),
        ref_pos=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        # ref_pos=np.array(
        #     [
        #         0.016594186425209045,
        #         -0.579270601272583,
        #         1.1369879245758057,
        #         -1.575303316116333,
        #         1.71644926071167,
        #         1.5714884996414185,
        #     ]
        # ),
    )
    print("joints")
    print(joints)
    # for j in joints:
    #     print(j)
    #     print("forward_kinematics")
    #     print(arm_kdl.forward_kinematics(j))
    #     print()
    #     print()
    #     print()
    jacobian = np.round(arm_kdl.jacobian(joints[0]), 3)
    print("jacobian")
    print(jacobian)
    validate_jacobian(arm_kdl, joints[0])


if __name__ == "__main__":
    main()
