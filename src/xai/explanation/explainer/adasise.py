import cv2
import numpy as np
import torch
import torch.nn.functional as F
from collections import OrderedDict


def _last_stage_module(stage):
    if hasattr(stage, "__len__") and len(stage) > 0:
        return stage[-1]
    return stage


def select_adasise_target_layers(model, mode="lesion"):
    if mode == "auto":
        return None

    encoder = getattr(model, "encoder", None)

    if encoder is None or not hasattr(encoder, "blocks"):
        if encoder is not None and hasattr(encoder, "conv_head"):
            return [("encoder.conv_head", encoder.conv_head)]
        raise ValueError("Could not select AdaSISE target layers for this model.")

    blocks = list(encoder.blocks)
    n = len(blocks)

    if mode == "lesion":
        idxs = [i for i in [1, 2, 3] if i < n]
    elif mode == "mid":
        idxs = [i for i in [2, 3, 4] if i < n]
    elif mode == "late":
        idxs = list(range(max(0, n - 3), n))
    elif mode == "all":
        idxs = list(range(1, n))
    elif mode == "semantic":
        idxs = list(range(max(0, n - 3), n))
    else:
        raise ValueError(f"Unknown target layer mode: {mode}")

    layers = [
        (f"encoder.blocks.{idx}", _last_stage_module(blocks[idx]))
        for idx in idxs
    ]

    if mode == "semantic":
        if hasattr(encoder, "conv_head"):
            layers.append(("encoder.conv_head", encoder.conv_head))

        if getattr(model, "use_cbam", False) and hasattr(model, "cbam"):
            layers.append(("cbam", model.cbam))

    return layers


class AdaSISE:
    def __init__(
        self,
        model,
        input_size,
        target_layers=None,
        gpu_batch=32,
        device=None,
        score_mode="raw",
        gradient_score_mode="raw",
        percentile_clip=(2, 98),
        alpha=0.30,
        colormap=cv2.COLORMAP_JET,
        otsu_bins=256,
        mask_power=2.0,
        max_mask_area_ratio=0.20,
        min_mask_area_ratio=0.0005,
        otsu_relax_factor=1.0,
        min_selected_channels=None,
        max_selected_channels=None,
        eps=1e-8,
    ):
        self.model = model
        self.model.eval()

        if device is None:
            device = next(model.parameters()).device

        self.device = device
        self.input_size = input_size
        self.gpu_batch = gpu_batch
        self.score_mode = score_mode
        self.gradient_score_mode = gradient_score_mode
        self.percentile_clip = percentile_clip
        self.alpha = alpha
        self.colormap = colormap
        self.otsu_bins = otsu_bins
        self.mask_power = mask_power
        self.max_mask_area_ratio = max_mask_area_ratio
        self.min_mask_area_ratio = min_mask_area_ratio
        self.otsu_relax_factor = otsu_relax_factor
        self.min_selected_channels = min_selected_channels
        self.max_selected_channels = max_selected_channels
        self.eps = eps

        self.activations = OrderedDict()
        self.gradients = OrderedDict()
        self.handles = []
        self.capture = False

        if target_layers is None:
            target_layers = select_adasise_target_layers(model, mode="lesion")

        self.target_layers = self._standardize_target_layers(target_layers)
        self._register_hooks()

    def _standardize_target_layers(self, target_layers):
        if isinstance(target_layers, OrderedDict):
            return target_layers

        standardized = OrderedDict()

        for idx, item in enumerate(target_layers):
            if isinstance(item, tuple):
                name, module = item
            else:
                name, module = f"layer_{idx}", item
            standardized[name] = module

        if len(standardized) == 0:
            raise ValueError("No target layers were provided or inferred.")

        return standardized

    def _register_hooks(self):
        for name, module in self.target_layers.items():
            handle = module.register_forward_hook(self._make_forward_hook(name))
            self.handles.append(handle)

    def _make_forward_hook(self, name):
        def hook(module, inputs, output):
            if not self.capture:
                return

            if isinstance(output, (tuple, list)):
                output = output[0]

            if not torch.is_tensor(output):
                return

            self.activations[name] = output

            if output.requires_grad:
                def grad_hook(grad):
                    self.gradients[name] = grad.detach()

                output.register_hook(grad_hook)

        return hook

    def _clean_output(self, output):
        if isinstance(output, tuple):
            output = output[0]

        if output.ndim == 1:
            output = output.unsqueeze(0)

        return output

    def _model_forward(self, x):
        return self._clean_output(self.model(x))

    def _score_for_target(self, output, target_class, mode):
        output = self._clean_output(output)

        if output.shape[-1] == 1:
            return output.view(-1)

        if mode == "softmax":
            return torch.softmax(output, dim=1)[:, target_class]

        if mode == "sigmoid":
            return torch.sigmoid(output)[:, target_class]

        if mode == "raw":
            return output[:, target_class]

        raise ValueError(f"Unsupported score mode: {mode}")

    def _get_target_class_from_output(self, output):
        output = self._clean_output(output)

        if output.shape[-1] == 1:
            return 0

        if self.score_mode == "sigmoid":
            scores = torch.sigmoid(output)
        elif self.score_mode == "softmax":
            scores = torch.softmax(output, dim=1)
        else:
            scores = output

        return int(scores.argmax(dim=1).item())

    def _normalize_np(self, arr):
        arr = np.asarray(arr, dtype=np.float32)

        if self.percentile_clip is not None:
            low, high = np.percentile(arr, self.percentile_clip)
            arr = np.clip(arr, low, high)

        arr = arr - arr.min()
        arr = arr / (arr.max() + self.eps)

        return arr.astype(np.float32)

    def _normalize_torch_per_mask(self, masks):
        flat = masks.flatten(1)
        mn = flat.min(dim=1).values.view(-1, 1, 1)
        mx = flat.max(dim=1).values.view(-1, 1, 1)
        return (masks - mn) / (mx - mn + self.eps)

    def _otsu_threshold_1d(self, values):
        values = np.asarray(values, dtype=np.float32)
        values = values[np.isfinite(values)]

        if values.size == 0:
            return 1.0

        if values.size == 1:
            return float(values[0])

        if np.allclose(values.max(), values.min()):
            return float(values.min())

        hist, bin_edges = np.histogram(
            values,
            bins=self.otsu_bins,
            range=(0.0, 1.0),
        )

        hist = hist.astype(np.float64)
        prob = hist / (hist.sum() + self.eps)

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        omega = np.cumsum(prob)
        mu = np.cumsum(prob * bin_centers)
        mu_t = mu[-1]

        sigma_b = (mu_t * omega - mu) ** 2 / (
            omega * (1.0 - omega) + self.eps
        )

        sigma_b[(omega <= 0) | (omega >= 1)] = -1

        idx = int(np.argmax(sigma_b))
        return float(bin_centers[idx])

    def _otsu_binary_mask_2d(self, heatmap):
        heatmap = self._normalize_np(heatmap)
        threshold = self._otsu_threshold_1d(heatmap.reshape(-1))
        return (heatmap >= threshold).astype(np.float32)

    def _extract_activations_and_gradients(self, input_tensor, target_class=None):
        self.activations.clear()
        self.gradients.clear()

        self.model.zero_grad(set_to_none=True)
        self.capture = True

        with torch.enable_grad():
            output = self._model_forward(input_tensor)

            if target_class is None:
                target_class = self._get_target_class_from_output(output)

            target_score = self._score_for_target(
                output,
                target_class,
                mode=self.gradient_score_mode,
            ).sum()

            target_score.backward()

        self.capture = False

        return target_class

    def _select_feature_maps_adasise(self, activation, gradient):
        acts = activation.detach()[0]
        grads = gradient.detach()[0]

        avg_grads = grads.mean(dim=(1, 2))
        positive = avg_grads > 0

        if positive.sum() == 0:
            return None, None, None, None

        pos_indices = torch.where(positive)[0]
        pos_scores = avg_grads[pos_indices]

        max_score = pos_scores.max().clamp_min(self.eps)
        norm_scores = pos_scores / max_score

        threshold = self._otsu_threshold_1d(
            norm_scores.detach().cpu().numpy()
        )

        relaxed_threshold = threshold * self.otsu_relax_factor
        keep = norm_scores >= relaxed_threshold

        if self.min_selected_channels is not None:
            if keep.sum() < self.min_selected_channels:
                k = min(self.min_selected_channels, len(norm_scores))
                topk_indices = torch.topk(norm_scores, k=k).indices
                keep = torch.zeros_like(norm_scores, dtype=torch.bool)
                keep[topk_indices] = True

        if self.max_selected_channels is not None:
            if keep.sum() > self.max_selected_channels:
                kept_scores = norm_scores[keep]
                kept_positions = torch.where(keep)[0]

                k = min(self.max_selected_channels, len(kept_scores))
                topk_local = torch.topk(kept_scores, k=k).indices
                topk_positions = kept_positions[topk_local]

                new_keep = torch.zeros_like(norm_scores, dtype=torch.bool)
                new_keep[topk_positions] = True
                keep = new_keep

        if keep.sum() == 0:
            keep[torch.argmax(norm_scores)] = True

        selected_indices = pos_indices[keep]
        selected_scores = norm_scores[keep]
        selected_acts = acts[selected_indices]

        return selected_acts, selected_indices, selected_scores, threshold

    def _activation_to_masks(self, selected_acts):
        masks = selected_acts.unsqueeze(1)

        masks = F.interpolate(
            masks,
            size=self.input_size,
            mode="bilinear",
            align_corners=False,
        )

        masks = masks.squeeze(1)
        masks = torch.relu(masks)
        masks = self._normalize_torch_per_mask(masks)

        if self.mask_power is not None and self.mask_power != 1.0:
            masks = masks.pow(self.mask_power)

        area_ratio = (masks > 0.25).float().flatten(1).mean(dim=1)

        keep_area = (
            (area_ratio >= self.min_mask_area_ratio)
            & (area_ratio <= self.max_mask_area_ratio)
        )

        if keep_area.any():
            masks = masks[keep_area]
        else:
            order = torch.argsort(area_ratio)
            keep_n = min(8, len(order))
            masks = masks[order[:keep_n]]

        flat_sum = masks.flatten(1).sum(dim=1)
        keep_nonempty = flat_sum > self.eps
        masks = masks[keep_nonempty]

        return masks

    def _score_masks(self, input_tensor, masks, target_class, baseline=None):
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        else:
            baseline = baseline.to(
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            )

        num_masks = masks.shape[0]

        if num_masks == 0:
            return None

        layer_map = torch.zeros(
            self.input_size,
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )

        with torch.no_grad():
            for start in range(0, num_masks, self.gpu_batch):
                end = min(start + self.gpu_batch, num_masks)

                mask_batch = masks[start:end].to(
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                )

                mask_batch_4d = mask_batch.unsqueeze(1)

                masked_input = (
                    input_tensor * mask_batch_4d
                    + baseline * (1.0 - mask_batch_4d)
                )

                output = self._model_forward(masked_input)

                scores = self._score_for_target(
                    output,
                    target_class,
                    mode=self.score_mode,
                )

                areas = mask_batch.flatten(1).sum(dim=1).clamp_min(self.eps)
                weights = scores / areas

                layer_map += torch.sum(
                    weights.view(-1, 1, 1) * mask_batch,
                    dim=0,
                )

        layer_map = layer_map.detach().cpu().numpy()
        layer_map = self._normalize_np(layer_map)

        return layer_map

    def _fuse_layer_maps(self, layer_maps):
        valid_maps = [
            self._normalize_np(m)
            for m in layer_maps
            if m is not None and np.isfinite(m).all()
        ]

        if len(valid_maps) == 0:
            return np.zeros(self.input_size, dtype=np.float32)

        if len(valid_maps) == 1:
            return valid_maps[0]

        stack = np.stack(valid_maps, axis=0)

        mean_map = stack.mean(axis=0)
        max_map = stack.max(axis=0)

        fused = 0.50 * mean_map + 0.50 * max_map
        fused = self._normalize_np(fused)

        return fused.astype(np.float32)

    def generate_heatmap(
        self,
        input_tensor,
        target_class=None,
        baseline=None,
        return_debug=False,
    ):
        if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
            raise ValueError("input_tensor must have shape [1, C, H, W].")

        input_tensor = input_tensor.to(self.device)

        target_class = self._extract_activations_and_gradients(
            input_tensor=input_tensor,
            target_class=target_class,
        )

        layer_maps = []

        debug = {
            "target_class": target_class,
            "layers": {},
        }

        for name in self.target_layers.keys():
            if name not in self.activations or name not in self.gradients:
                continue

            activation = self.activations[name]
            gradient = self.gradients[name]

            if activation.ndim != 4 or gradient.ndim != 4:
                continue

            selected_acts, selected_indices, selected_scores, threshold = (
                self._select_feature_maps_adasise(activation, gradient)
            )

            if selected_acts is None:
                debug["layers"][name] = {
                    "num_available": int(activation.shape[1]),
                    "num_selected": 0,
                    "num_masks_after_area_filter": 0,
                    "otsu_threshold": None,
                }
                continue

            masks = self._activation_to_masks(selected_acts)

            layer_map = self._score_masks(
                input_tensor=input_tensor,
                masks=masks,
                target_class=target_class,
                baseline=baseline,
            )

            if layer_map is not None:
                layer_maps.append(layer_map)

            debug["layers"][name] = {
                "num_available": int(activation.shape[1]),
                "num_selected": int(
                    0 if selected_indices is None else len(selected_indices)
                ),
                "num_masks_after_area_filter": int(masks.shape[0]),
                "otsu_threshold": None if threshold is None else float(threshold),
            }

        heatmap = self._fuse_layer_maps(layer_maps)

        self.activations.clear()
        self.gradients.clear()

        if return_debug:
            debug["num_layer_maps"] = len(layer_maps)
            return heatmap, target_class, debug

        return heatmap, target_class

    def overlay(self, image_rgb, heatmap, alpha=None, min_visible=1e-6):
        """
        Direct heatmap overlay.

        This overlays the same heatmap colors on the original image.
        Zero heatmap pixels are transparent.
        """

        if alpha is None:
            alpha = self.alpha

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

        heatmap = self._normalize_np(heatmap)

        heatmap_uint8 = np.uint8(255 * heatmap)

        heatmap_color = cv2.applyColorMap(heatmap_uint8, self.colormap)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        support = heatmap > min_visible

        overlay = image_rgb.astype(np.float32).copy()

        overlay[support] = (
            (1.0 - alpha) * image_rgb.astype(np.float32)[support]
            + alpha * heatmap_color.astype(np.float32)[support]
        )

        return np.clip(overlay, 0, 255).astype(np.uint8)

    def remove_hook(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()