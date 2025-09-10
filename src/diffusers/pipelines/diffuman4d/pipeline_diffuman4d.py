# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Callable, Dict, List, Optional, Union
from copy import deepcopy
from tqdm import tqdm

import torch

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import (
    FromSingleFileMixin,
    IPAdapterMixin,
    StableDiffusionLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.models import AutoencoderKL
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin

from ...models.unets.unet_multiview_condition import UNetMultiviewConditionModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import Diffuman4DPipeline
        ```
"""


def encode_vae(vae, pixel_values, batch_size=8):
    # Split the latents into batches to avoid OOM
    pixel_values_batches = pixel_values.split(batch_size)
    latents = []
    for pixel_values in pixel_values_batches:
        out = vae.encode(pixel_values).latent_dist.sample()
        latents.append(out)
    latents = torch.cat(latents, dim=0)
    latents = latents * vae.config.scaling_factor
    return latents


def decode_vae(vae, latents, generator=None, batch_size=8):
    # Split the latents into batches to avoid OOM
    latents_batches = latents.split(batch_size)
    # Decode the latents in batches
    images = []
    for latents in latents_batches:
        out = vae.decode(
            latents / vae.config.scaling_factor,
            return_dict=False,
            generator=generator,
        )[0]
        images.append(out)
    images = torch.cat(images, dim=0)
    return images


def encode_image_vae(vae, images, image_latents=None, dtype=None, device=None):
    dtype = vae.dtype if dtype is None else dtype
    device = vae.device if device is None else device

    if image_latents is None:
        if images is None:
            return None
        images = images.to(dtype=dtype, device=device)
        image_latents = encode_vae(vae, images)
    else:
        image_latents = image_latents.to(dtype=dtype, device=device)

    return image_latents


def encode_image_resizing(images, image_latents=None, shape=None, mode="bilinear", dtype=None, device=None):
    if image_latents is None:
        if images is None:
            return None
        image_latents = torch.nn.functional.interpolate(images, size=shape, mode=mode)

    if dtype is not None:
        image_latents = image_latents.to(dtype=dtype)
    if device is not None:
        image_latents = image_latents.to(device=device)
    return image_latents


def get_negative_latents(latents, color="black"):
    negative_latents = torch.ones_like(latents)
    if color == "black":
        negative_latents = -1.0 * negative_latents
    elif color == "white":
        negative_latents = negative_latents
    elif color == "grey" or color == "random":
        negative_latents = 0.0 * negative_latents
    else:
        raise ValueError(f"color: {color} not supported.")
    return negative_latents


class Diffuman4DPipeline(
    DiffusionPipeline,
    StableDiffusionMixin,
    TextualInversionLoaderMixin,
    StableDiffusionLoraLoaderMixin,
    IPAdapterMixin,
    FromSingleFileMixin,
):
    r"""
    Pipeline for Diffuman4D.
    """

    model_cpu_offload_seq = "unet->vae"

    def __init__(
        self,
        vae: AutoencoderKL,
        unet: UNetMultiviewConditionModel,
        scheduler: KarrasDiffusionSchedulers,
    ):
        super().__init__()

        self.register_modules(vae=vae, unet=unet, scheduler=scheduler)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    @property
    def guidance_scale(self):
        return self._guidance_scale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_all_latents(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        plucker_embeds: Optional[torch.Tensor] = None,
        skeletons: Optional[torch.Tensor] = None,
        cond_masks: Optional[torch.Tensor] = None,
        pixel_values_latents: Optional[torch.Tensor] = None,
        plucker_embeds_latents: Optional[torch.Tensor] = None,
        skeletons_latents: Optional[torch.Tensor] = None,
        cond_masks_latents: Optional[torch.Tensor] = None,
        latents: Optional[torch.Tensor] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ):
        # image
        dtype, device = self.vae.dtype, self._execution_device
        pixel_values_latents = encode_image_vae(
            self.vae,
            images=pixel_values,
            image_latents=pixel_values_latents,
            dtype=dtype,
            device=device,
        )
        num_frames, latent_dim, height, width = pixel_values_latents.shape

        # plucker
        plucker_embeds_latents = encode_image_resizing(
            images=plucker_embeds,
            image_latents=plucker_embeds_latents,
            shape=(height, width),
            mode="bilinear",
            dtype=dtype,
            device=device,
        )

        # skeletons
        if skeletons_latents is not None:
            skeletons_latents = skeletons_latents.to(dtype=dtype, device=device)
        elif self.unet.config.enable_pose_encoder:
            skeletons_latents = skeletons.to(dtype=dtype, device=device)
        else:
            skeletons_latents = encode_image_vae(
                self.vae,
                images=skeletons,
                image_latents=skeletons_latents,
                dtype=dtype,
                device=device,
            )

        # conditional masks
        cond_masks_latents = encode_image_resizing(
            images=cond_masks,
            image_latents=cond_masks_latents,
            shape=(height, width),
            mode="nearest",
            dtype=dtype,
            device=device,
        )

        # latents
        latents = self.prepare_latents(
            num_frames,
            latent_dim,
            height * self.vae_scale_factor,
            width * self.vae_scale_factor,
            dtype,
            device,
            generator,
            latents,
        )

        return pixel_values_latents, plucker_embeds_latents, skeletons_latents, cond_masks_latents, latents

    def parepare_schedulers(self, num_inference_steps: int, num_frames: int):
        # create a scheduler for each latent
        device = self._execution_device
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        schedulers = [deepcopy(self.scheduler) for _ in range(num_frames)]
        timesteps = self.scheduler.timesteps
        return schedulers, timesteps

    def get_timestep(self, timesteps, timestep_indices, is_cond):
        # timestep of conditional latents === 0
        timestep_indices[is_cond] = 0
        timestep = timesteps[timestep_indices]
        timestep[is_cond] = 0
        return timestep

    def post_process(self, latents, output_type="pt", generator=None):
        images = decode_vae(self.vae, latents, generator=generator)
        images = self.image_processor.postprocess(
            images, output_type=output_type, do_denormalize=[True] * images.shape[0]
        )
        return images

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        plucker_embeds: Optional[torch.Tensor] = None,
        skeletons: Optional[torch.Tensor] = None,
        cond_masks: Optional[torch.Tensor] = None,
        pixel_values_latents: Optional[torch.Tensor] = None,
        plucker_embeds_latents: Optional[torch.Tensor] = None,
        skeletons_latents: Optional[torch.Tensor] = None,
        cond_masks_latents: Optional[torch.Tensor] = None,
        latents: Optional[torch.Tensor] = None,
        domains: List[str] = None,
        num_inference_steps: int = 1,
        schedulers: Optional[List[object]] = None,
        timesteps: Optional[List[int]] = None,
        timestep_indices: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "latent",
    ):
        r"""
        Denoise a sample sequence for num_inference_steps starting from timestep_indices.

        Examples:
        """

        # Deafult args
        dtype = self.vae.dtype
        device = self._execution_device

        if pixel_values is not None:
            num_frames, _, height, width = pixel_values.shape
        else:
            num_frames, _, height, width = pixel_values_latents.shape
            height *= self.vae_scale_factor
            width *= self.vae_scale_factor

        self._guidance_scale = guidance_scale
        self._interrupt = False

        # Prepare conditions and latents
        pixel_values_latents, plucker_embeds_latents, skeletons_latents, cond_masks_latents, latents = (
            self.prepare_all_latents(
                pixel_values=pixel_values,
                plucker_embeds=plucker_embeds,
                skeletons=skeletons,
                cond_masks=cond_masks,
                pixel_values_latents=pixel_values_latents,
                plucker_embeds_latents=plucker_embeds_latents,
                skeletons_latents=skeletons_latents,
                cond_masks_latents=cond_masks_latents,
                latents=latents,
                generator=generator,
            )
        )

        is_cond = cond_masks_latents[:, 0, 0, 0] == 0

        # Concatenate the unconditional and conditional embeddings into a single batch
        if self.do_classifier_free_guidance:
            negative_pixel_values_latents = get_negative_latents(pixel_values_latents, color="white")
            if plucker_embeds_latents is not None:
                negative_plucker_embeds_latents = get_negative_latents(plucker_embeds_latents, color="grey")
                plucker_embeds_latents = torch.cat([negative_plucker_embeds_latents, plucker_embeds_latents])
            if skeletons_latents is not None:
                negative_skeletons_latents = get_negative_latents(skeletons_latents, color="black")
                skeletons_latents = torch.cat([negative_skeletons_latents, skeletons_latents])
            cond_masks_latents = torch.cat([cond_masks_latents] * 2)
            domains = domains * 2

        # Prepare timesteps
        # denoise num_inference_steps starting from timestep_indices
        if schedulers is None:
            schedulers, timesteps = self.parepare_schedulers(num_inference_steps, num_frames)
            timestep_indices = torch.zeros(num_frames)
        timestep_indices = timestep_indices.to(device=device)

        # Denoising loop
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for _ in range(num_inference_steps):
                if self.interrupt:
                    continue

                timestep = self.get_timestep(timesteps, timestep_indices, is_cond)

                latent_model_input = latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep)

                # Replace the noise latents with the conditional image latents
                latent_model_input[is_cond, ...] = pixel_values_latents[is_cond, ...]

                # expand the latents if we are doing classifier free guidance
                if self.do_classifier_free_guidance:
                    timestep = torch.cat([timestep] * 2)
                    negative_latent_model_input = latent_model_input.clone()
                    negative_latent_model_input[is_cond, ...] = negative_pixel_values_latents[is_cond, ...]
                    latent_model_input = torch.cat([negative_latent_model_input, latent_model_input])

                # Concat the latents along the channel dimension
                latent_model_input = [latent_model_input]
                if plucker_embeds_latents is not None:
                    latent_model_input.append(plucker_embeds_latents)
                if skeletons_latents is not None and not self.unet.config.enable_pose_encoder:
                    latent_model_input.append(skeletons_latents)
                latent_model_input.append(cond_masks_latents)
                latent_model_input = torch.cat(latent_model_input, dim=1)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    timestep=timestep,
                    skeletons=skeletons_latents,
                    domains=domains,
                    num_frames=num_frames,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                new_latents = []
                for j in range(len(noise_pred)):
                    t = timestep[j].item()
                    noise = noise_pred[j].unsqueeze(0)
                    latent = latents[j].unsqueeze(0)
                    # only denoise target latents
                    if not is_cond[j]:
                        latent = schedulers[j].step(noise, t, latent, return_dict=False)[0]
                    new_latents.append(latent.to(dtype=dtype))
                latents = torch.cat(new_latents)
                timestep_indices[~is_cond] += 1

                progress_bar.update()

        # Postprocess the denoised latents
        if output_type != "latent":
            # output_type: pt, np, pil
            images = self.post_process(latents, output_type, generator)
        else:
            images = latents

        # Offload all models
        self.maybe_free_model_hooks()

        return images

    def sliding_iterative_denoise(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        plucker_embeds: Optional[torch.Tensor] = None,
        skeletons: Optional[torch.Tensor] = None,
        cond_masks: Optional[torch.Tensor] = None,
        latents: Optional[torch.Tensor] = None,
        domain: str = "spatial",  # "spatial" or "temporal"
        timestep_indices: Optional[torch.Tensor] = None,
        # sliding denoising args
        window_size: int = 12,
        sliding_stride: int = 1,
        sliding_shift: int = 0,
        bidirectional: bool = True,
        num_denoising_steps: int = 1,
        alternation_rounds: int = 3,
        guidance_scale: float = 2.0,
        tqdm: Callable = tqdm,
    ):
        """
        Denoise a spatial or temporal sample sequence with sliding iterative denoising scheme.
        """
        dtype, device = self.vae.dtype, self._execution_device

        if (window_size * num_denoising_steps) % sliding_stride != 0:
            raise ValueError(
                f"The window size ({window_size}) * num denoising steps ({num_denoising_steps}) "
                f"should be divisible by the sliding stride ({sliding_stride})"
            )
        num_denoising_steps_peralt = window_size * num_denoising_steps // sliding_stride
        if bidirectional:
            num_denoising_steps_peralt *= 2
        # num_inference_steps is the total number of denoising steps for each sample
        num_inference_steps = num_denoising_steps_peralt * alternation_rounds

        timestep_indices = timestep_indices.to(device=device)
        target_indices = torch.where(cond_masks[:, 0, 0, 0] != 0.0)[0].to(device=device)
        input_indices = torch.where(cond_masks[:, 0, 0, 0] == 0.0)[0].to(device=device)
        target_timestep_indices = timestep_indices[target_indices]
        input_timestep_indices = timestep_indices[input_indices]
        timestep_id_end = target_timestep_indices[0].item() + num_denoising_steps_peralt
        if (target_timestep_indices != target_timestep_indices[0]).any():
            raise ValueError(
                f"The timestep indices should be the same for all target samples, timestep_indices = {timestep_indices}"
            )
        if (input_timestep_indices != 0).any():
            raise ValueError(
                f"The timestep indices should be 0 for all input samples, timestep_indices = {timestep_indices}"
            )

        # prepare input latents
        pixel_values_latents, plucker_embeds_latents, skeletons_latents, cond_masks_latents, latents = (
            self.prepare_all_latents(
                pixel_values=pixel_values,
                plucker_embeds=plucker_embeds,
                skeletons=skeletons,
                cond_masks=cond_masks,
                latents=latents,
            )
        )

        # prepare schedulers
        schedulers, timesteps = self.parepare_schedulers(num_inference_steps, len(latents))

        # prepare sliding windows
        target_windows, input_windows = [], []
        directions = (-1, 1) if bidirectional else (-1,)
        for direction in directions:
            for shift in range(sliding_shift, sliding_shift + len(target_indices), sliding_stride):
                target_window = target_indices.roll(shifts=shift * direction)
                target_window = target_window[:window_size]
                target_windows.append(target_window)

                if domain == "spatial":
                    # num_frames = window_size
                    input_window = input_indices
                elif domain == "temporal":
                    # num_frames = 2 * window_size
                    input_window = target_window - len(input_indices)
                input_windows.append(input_window)

        # sliding iterative denoising
        for target_window, input_window in tqdm(zip(target_windows, input_windows), total=len(target_windows)):
            window = torch.cat([input_window, target_window])
            get_slice = lambda x: x[window] if x is not None else None

            # few-step denoising for each window
            latents_window = self(
                pixel_values_latents=get_slice(pixel_values_latents),
                plucker_embeds_latents=get_slice(plucker_embeds_latents),
                skeletons_latents=get_slice(skeletons_latents),
                cond_masks_latents=get_slice(cond_masks_latents),
                latents=get_slice(latents),
                domains=[domain],
                guidance_scale=guidance_scale,
                num_inference_steps=num_denoising_steps,
                schedulers=[schedulers[i] for i in window],
                timesteps=timesteps,
                timestep_indices=timestep_indices[window],
                output_type="latent",
            )

            # update latents and timesteps
            timestep_indices[target_window] += num_denoising_steps
            latents[window] = latents_window

        # sanity check
        if (timestep_indices[target_indices] != timestep_id_end).any():
            raise ValueError(
                f"The denoised timesteps of target samples mismatch the config, timestep_indices = {timestep_indices}"
            )
        if (timestep_indices[input_indices] != 0).any():
            raise ValueError(f"Timesteps of input samples have changed, timestep_indices = {timestep_indices}")

        images = self.post_process(latents, output_type="pt")
        return {
            "images": images,
            "latents": latents,
            "timestep_indices": timestep_indices,
            "fully_denoised": timestep_indices == num_inference_steps,
        }
