import cv2
import numpy as np
import torch
from captum.attr import IntegratedGradients


def min_max_normalize(x):
    x = np.asarray(x, dtype=np.float32)
    x = x - np.min(x)
    return x / (np.max(x) + 1e-8)


class CaptumModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)

        if isinstance(output, tuple):
            output = output[0]

        if output.ndim == 1:
            output = output.unsqueeze(1)

        return output


class IntegratedGradientsSmoothGradXAI:
    def __init__(
        self,
        model,
        device=None,
        nt_samples=64,
        stdevs=0.10,
        n_steps=80,
        nt_type="smoothgrad_sq",
        attribution_mode="abs",
        percentile_clip=(2, 98),
        internal_batch_size=4,
        nt_batch_size=1,
        clear_cuda_cache=False,
        blur=True,
        blur_ksize=15,
        alpha=0.30,
        colormap=cv2.COLORMAP_JET,
    ):
        self.original_model = model
        self.original_model.eval()

        if device is None:
            device = next(model.parameters()).device

        self.device = device

        self.model = CaptumModelWrapper(model).to(device)
        self.model.eval()

        self.nt_samples = int(nt_samples)
        self.stdevs = float(stdevs)
        self.n_steps = int(n_steps)
        self.nt_type = nt_type
        self.internal_batch_size = internal_batch_size
        self.nt_batch_size = max(1, int(nt_batch_size))
        self.clear_cuda_cache = clear_cuda_cache

        self.attribution_mode = attribution_mode
        self.percentile_clip = percentile_clip

        self.blur = blur
        self.blur_ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1

        self.alpha = alpha
        self.colormap = colormap

        self.ig = IntegratedGradients(self.model)

    def _get_target_class(self, input_tensor):
        with torch.no_grad():
            output = self.model(input_tensor)

            if output.shape[-1] == 1:
                return 0

            return int(output.argmax(dim=1).item())

    def _normalize_heatmap(self, heatmap):
        heatmap = np.asarray(heatmap, dtype=np.float32)

        if self.percentile_clip is not None:
            low, high = np.percentile(heatmap, self.percentile_clip)
            heatmap = np.clip(heatmap, low, high)

        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / (heatmap.max() + 1e-8)

        return heatmap.astype(np.float32)

    def _postprocess_attribution(self, attribution):
        if attribution.ndim == 4:
            attribution = attribution[0]

        attribution = attribution.detach().float().cpu()

        if self.attribution_mode == "abs":
            heatmap = attribution.abs().sum(dim=0).numpy()

        elif self.attribution_mode == "positive":
            heatmap = attribution.clamp(min=0).sum(dim=0).numpy()

        elif self.attribution_mode == "signed":
            heatmap = attribution.sum(dim=0).numpy()
            heatmap = np.maximum(heatmap, 0)

        else:
            raise ValueError(
                "attribution_mode must be one of: 'abs', 'positive', 'signed'"
            )

        heatmap = self._normalize_heatmap(heatmap)

        if self.blur:
            heatmap = cv2.GaussianBlur(
                heatmap.astype(np.float32),
                (self.blur_ksize, self.blur_ksize),
                sigmaX=0,
            )

            heatmap = self._normalize_heatmap(heatmap)

        return heatmap.astype(np.float32)

    def _accumulate_smoothig_streaming(
        self,
        input_tensor,
        baseline,
        target_class,
        internal_batch_size,
    ):
        """
        Memory-efficient replacement for Captum NoiseTunnel.

        Instead of materializing all nt_samples noisy inputs together, this streams
        small noise chunks and accumulates only a CxHxW attribution tensor on CPU.
        Peak GPU memory is controlled mainly by internal_batch_size and nt_batch_size.
        """
        if input_tensor.shape[0] != 1:
            raise ValueError(
                "IntegratedGradientsSmoothGradXAI expects batch size 1 for heatmap generation."
            )

        if self.nt_type not in {"smoothgrad", "smoothgrad_sq", "vargrad"}:
            raise ValueError("nt_type must be one of: smoothgrad, smoothgrad_sq, vargrad")

        sum_attr_cpu = None
        sum_sq_attr_cpu = None
        done = 0

        while done < self.nt_samples:
            chunk_size = min(self.nt_batch_size, self.nt_samples - done)

            noise_shape = (chunk_size,) + tuple(input_tensor.shape[1:])
            noise = torch.randn(
                noise_shape,
                device=self.device,
                dtype=input_tensor.dtype,
            ) * self.stdevs

            noisy_input = input_tensor.expand(chunk_size, -1, -1, -1) + noise
            noisy_input = noisy_input.detach().requires_grad_(True)

            chunk_baseline = baseline.expand_as(noisy_input)

            with torch.enable_grad():
                attr = self.ig.attribute(
                    noisy_input,
                    baselines=chunk_baseline,
                    target=target_class,
                    n_steps=self.n_steps,
                    internal_batch_size=internal_batch_size,
                )

            attr_cpu = attr.detach().float().cpu()

            if self.nt_type in {"smoothgrad", "vargrad"}:
                chunk_sum = attr_cpu.sum(dim=0)
                if sum_attr_cpu is None:
                    sum_attr_cpu = chunk_sum
                else:
                    sum_attr_cpu.add_(chunk_sum)

            if self.nt_type in {"smoothgrad_sq", "vargrad"}:
                chunk_sum_sq = attr_cpu.pow(2).sum(dim=0)
                if sum_sq_attr_cpu is None:
                    sum_sq_attr_cpu = chunk_sum_sq
                else:
                    sum_sq_attr_cpu.add_(chunk_sum_sq)

            done += chunk_size

            del noise, noisy_input, chunk_baseline, attr, attr_cpu

            if self.clear_cuda_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()

        if self.nt_type == "smoothgrad":
            return sum_attr_cpu / float(self.nt_samples)

        if self.nt_type == "smoothgrad_sq":
            return sum_sq_attr_cpu / float(self.nt_samples)

        mean_attr = sum_attr_cpu / float(self.nt_samples)
        mean_sq_attr = sum_sq_attr_cpu / float(self.nt_samples)
        return (mean_sq_attr - mean_attr.pow(2)).clamp_min_(0.0)

    def generate_heatmap(
        self,
        input_tensor,
        target_class=None,
        baseline=None,
        internal_batch_size=None,
    ):
        input_tensor = input_tensor.to(self.device)
        input_tensor = input_tensor.clone().detach().requires_grad_(True)

        if internal_batch_size is None:
            internal_batch_size = self.internal_batch_size

        if internal_batch_size is None:
            internal_batch_size = 4

        if target_class is None:
            target_class = self._get_target_class(input_tensor)

        if baseline is None:
            baseline = torch.zeros_like(input_tensor, device=self.device)
        else:
            baseline = baseline.to(self.device).detach()

        attribution = self._accumulate_smoothig_streaming(
            input_tensor=input_tensor,
            baseline=baseline,
            target_class=target_class,
            internal_batch_size=internal_batch_size,
        )

        heatmap = self._postprocess_attribution(attribution)

        del input_tensor, baseline, attribution

        if self.clear_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return heatmap, target_class

    def overlay(self, image_rgb, heatmap):
        if image_rgb.max() <= 1.0:
            image_rgb = (image_rgb * 255).astype(np.uint8)
        else:
            image_rgb = image_rgb.astype(np.uint8)

        h, w = image_rgb.shape[:2]

        if heatmap.shape[:2] != (h, w):
            heatmap = cv2.resize(
                heatmap,
                (w, h),
                interpolation=cv2.INTER_CUBIC,
            )

        heatmap = min_max_normalize(heatmap)
        heatmap_uint8 = np.uint8(255 * heatmap)

        heatmap_color = cv2.applyColorMap(heatmap_uint8, self.colormap)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(
            image_rgb,
            1.0 - self.alpha,
            heatmap_color,
            self.alpha,
            0,
        )

        return overlay
