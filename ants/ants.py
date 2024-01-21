from networkx import center
import torch as th
import torch.nn.functional as F
from torch import Tensor
import pygame
from pygame import Surface

from ants.pytorch_utils import gaussian_blur_kernel, random_disk


class Ants:
    def __init__(self, screen_size: Tensor, resolution: int, ants_count: int, dtype=th.float32, device="cpu"):
        self.dtype = dtype
        self.device = device
        self.resolution = resolution
        self.texture_size = th.ceil(screen_size / resolution).to(device)
        self.set_random_pos_speed(ants_count)

        self.min_color = th.tensor((0, 0, 0), dtype=dtype).to(device)
        self.max_color = th.tensor((255, 255, 255), dtype=dtype).to(device)
        self.zero_2D = th.tensor((0, 0), dtype=dtype).to(device)
        self.one_1D = th.tensor(1, dtype=dtype).to(device)

        self.diffusion_rate = 0.26
        self.diffusion_kernel = gaussian_blur_kernel(3, self.diffusion_rate, dtype=self.dtype).to(device)
        self.trail_power = 0.05
        self.decay_factor = 0.99
        self.distance_detection = 16
        self.angle_detection = th.tensor([-th.pi / 8, 0, th.pi / 8], dtype=th.float32).to(device)
        self.area_detection = 1
        area_detection_1D = th.arange(-self.area_detection // 2 + 1, self.area_detection // 2 + 1, dtype=th.int)
        self.area_detection_indices = th.stack(th.meshgrid(area_detection_1D, area_detection_1D), dim=2)
        self.area_detection_indices = self.area_detection_indices.view(-1, 2).to(device)
        self.turn_speed = 0.1

    def set_random_pos_speed(self, ants_count: int):
        self.pos = random_disk(ants_count, self.device) * self.texture_size[1] / 2 + self.texture_size / 2
        self.angle = th.rand(ants_count, dtype=self.dtype).to(self.device) * (2 * th.pi)
        self.speed = 0.3
        self.trail_map = th.zeros(
            (int(self.texture_size[0].item()), int(self.texture_size[1].item())), dtype=self.dtype
        ).to(self.device)

    def get_direction(self) -> Tensor:
        return th.stack([th.cos(self.angle), th.sin(self.angle)], dim=1)

    def move(self):
        self.pos += self.speed * self.get_direction()
        x, y = self.pos[:, 0], self.pos[:, 1]
        self.angle = th.where((x < 0) | (x > self.texture_size[0] - 1), th.pi - self.angle, self.angle)
        self.angle = th.where((y < 0) | (y > self.texture_size[1] - 1), -self.angle, self.angle)
        self.pos = th.clamp(self.pos, self.zero_2D, self.texture_size - 1)
        ants_coords = self.pos.to(dtype=th.int)
        self.trail_map[ants_coords[:, 0], ants_coords[:, 1]] += self.trail_power
        self.trail_map = th.minimum(self.trail_map, self.one_1D)

    def diffuse(self):
        trail_map = self.trail_map.view((1, 1, self.trail_map.shape[0], self.trail_map.shape[1]))
        padding = (self.diffusion_kernel.size(2) - 1) // 2
        trail_map = F.conv2d(trail_map, self.diffusion_kernel, stride=1, padding=padding)
        self.trail_map = trail_map.view((self.trail_map.shape[0], self.trail_map.shape[1]))
        self.trail_map *= self.decay_factor

    def get_offset_direction(self, offset_angle: Tensor) -> Tensor:
        """
        offset: 1D tensor
        return (offset_size, ants_size, 2)
        """
        angle = th.stack([self.angle] * offset_angle.shape[0], dim=1)  # (ants_size, offset_size)
        angle = angle + offset_angle  # (ants_size, offset_size)
        angle = angle.T  # (offset_size, ants_size)
        return th.stack([th.cos(angle), th.sin(angle)], dim=2)  # (offset_size, ants_size, 2)

    def look(self) -> Tensor:
        offset_direction = self.get_offset_direction(self.angle_detection)  # (offset_size, ants_size, 2)
        center_detection = (self.pos + self.distance_detection * offset_direction).to(
            dtype=th.int
        )  # (offset_size, ants_size, 2)
        center_detection = center_detection.repeat(
            self.area_detection_indices.shape[0], 1, 1, 1
        )  # (area_detection * area_detection, offset_size, ants_size, 2)
        center_detection = center_detection.permute(
            1, 2, 0, 3
        )  # (offset_size, ants_size, area_detection * area_detection, 2)

        padding = self.distance_detection + self.area_detection // 2
        detection_indices = (
            center_detection + self.area_detection_indices + padding
        )  # (offset_size, ants_size, area_detection * area_detection, 2)
        padded_trail_map = th.nn.functional.pad(self.trail_map, (padding, padding, padding, padding), value=0)
        zone_detection_trail = padded_trail_map[
            detection_indices[:, :, :, 0], detection_indices[:, :, :, 1]
        ]  # (offset_size, ants_size, area_detection * area_detection)
        total_detection_trail = zone_detection_trail.sum(dim=2)  # (offset_size, ants_size)

        return total_detection_trail

    def turn(self):
        total_detection_trail = self.look()  # (offset_size, ants_size)
        ab = total_detection_trail[0] < total_detection_trail[1]
        bc = total_detection_trail[1] < total_detection_trail[2]
        self.angle = th.where(~ab & ~bc, self.angle, self.angle + self.turn_speed)  # 0 est le plus grand
        self.angle = th.where(ab & bc, self.angle, self.angle - self.turn_speed)  # 2 est le plus grand

    def render(self) -> Surface:
        texture = (
            th.einsum("mn, o -> mno", 1 - self.trail_map, self.min_color)
            + th.einsum("mn, o -> mno", self.trail_map, self.max_color)
        ).to(dtype=th.int8)
        surface = pygame.surfarray.make_surface(texture.cpu().numpy())
        surface = pygame.transform.scale(
            surface, (self.resolution * surface.get_width(), self.resolution * surface.get_height())
        )
        return surface
