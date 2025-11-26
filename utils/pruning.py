import torch
import torch.nn as nn

class PruningManager:
    def __init__(self, model):
        self.model = model
        self.initial_state = None
        self.masks = {}

    # save θ₀
    def save_initial_weights(self):
        self.initial_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        print("Saved initial weights")

    # create all-ones masks
    def initialize_masks(self):
        modules = dict(self.model.named_modules())
        self.masks = {}

        for name, p in self.model.named_parameters():
            if "weight" in name:
                module_name, _, p_short = name.rpartition(".")
                module = modules[module_name] if module_name else self.model

                mask = torch.ones_like(p.data)
                buf = f"{p_short}_mask"

                if hasattr(module, buf):
                    setattr(module, buf, mask)
                else:
                    module.register_buffer(buf, mask)

                self.masks[name] = buf

        print(f"Initialized masks for {len(self.masks)} tensors")

    # zero out pruned weights
    def apply_masks(self):
        modules = dict(self.model.named_modules())
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if name in self.masks:
                    module_name, _, p_short = name.rpartition(".")
                    module = modules[module_name] if module_name else self.model
                    mask = getattr(module, self.masks[name])
                    p.data.mul_(mask)

    # magnitude pruning
    def prune_by_magnitude(self, pruning_rate, layer_wise=True):
        if not self.masks:
            self.initialize_masks()

        modules = dict(self.model.named_modules())

        with torch.no_grad():
            if layer_wise:
                print(f"Pruning {pruning_rate*100:.1f}% per layer")

                for name, p in self.model.named_parameters():
                    if name in self.masks:
                        module_name, _, p_short = name.rpartition(".")
                        module = modules[module_name] if module_name else self.model
                        mask = getattr(module, self.masks[name])

                        active = (mask == 1)
                        w = p.data[active]

                        if w.numel() == 0:
                            continue

                        k = int(w.numel() * pruning_rate)
                        if k > 0:
                            thr = torch.kthvalue(w.abs(), k)[0]
                            new_mask = (p.data.abs() > thr).float()
                            combined = torch.min(mask, new_mask)
                            setattr(module, self.masks[name], combined)

            else:
                print(f"Global pruning {pruning_rate*100:.1f}%")

                all_w = []
                for name, p in self.model.named_parameters():
                    if name in self.masks:
                        module_name, _, p_short = name.rpartition(".")
                        module = modules[module_name] if module_name else self.model
                        mask = getattr(module, self.masks[name])
                        all_w.append(p.data[mask == 1].abs().flatten())

                if len(all_w) > 0:
                    all_w = torch.cat(all_w)
                    k = int(all_w.numel() * pruning_rate)

                    if 0 < k < all_w.numel():
                        thr = torch.kthvalue(all_w, k)[0]

                        for name, p in self.model.named_parameters():
                            if name in self.masks:
                                module_name, _, p_short = name.rpartition(".")
                                module = modules[module_name] if module_name else self.model
                                mask = getattr(module, self.masks[name])
                                new_mask = (p.data.abs() > thr).float()
                                combined = torch.min(mask, new_mask)
                                setattr(module, self.masks[name], combined)

        self.apply_masks()
        sp = self.get_sparsity()
        print(f"Sparsity now: {sp:.2f}%")
        return sp

    # random pruning
    def prune_random(self, pruning_rate, layer_wise=True):
        if not self.masks:
            self.initialize_masks()

        modules = dict(self.model.named_modules())
        print(f"Random pruning {pruning_rate*100:.1f}%")

        with torch.no_grad():
            if layer_wise:
                for name, p in self.model.named_parameters():
                    if name in self.masks:
                        module_name, _, p_short = name.rpartition(".")
                        module = modules[module_name] if module_name else self.model
                        mask = getattr(module, self.masks[name])

                        flat = mask.flatten()
                        active = (flat == 1).nonzero().flatten()
                        if active.numel() == 0:
                            continue

                        k = int(active.numel() * pruning_rate)
                        if k == 0:
                            continue

                        prune_idx = active[torch.randperm(active.numel(), device=p.device)[:k]]
                        flat[prune_idx] = 0
                        combined = torch.min(mask, flat.reshape(mask.shape))
                        setattr(module, self.masks[name], combined)

            else:
                active_map = {}
                total_active = 0

                for name, p in self.model.named_parameters():
                    if name in self.masks:
                        module_name, _, p_short = name.rpartition(".")
                        module = modules[module_name] if module_name else self.model
                        mask = getattr(module, self.masks[name])
                        flat = mask.flatten()
                        idx = (flat == 1).nonzero().flatten()

                        if idx.numel() > 0:
                            active_map[name] = (module, flat, idx)
                            total_active += idx.numel()

                k = int(total_active * pruning_rate)
                if k > 0:
                    perm = torch.randperm(total_active, device=next(self.model.parameters()).device)
                    chosen = perm[:k]

                    names = list(active_map.keys())
                    counts = [active_map[n][2].numel() for n in names]

                    cum = [0]
                    for c in counts:
                        cum.append(cum[-1] + c)

                    for pos in chosen.tolist():
                        for i in range(len(names)):
                            if cum[i] <= pos < cum[i+1]:
                                local = pos - cum[i]
                                module, flat_mask, idx = active_map[names[i]]
                                flat_mask[idx[local]] = 0
                                setattr(module, self.masks[names[i]], flat_mask.reshape(mask.shape))
                                break

        self.apply_masks()
        sp = self.get_sparsity()
        print(f"Sparsity now: {sp:.2f}%")
        return sp

    # reset θ → θ₀ but keep masks
    def reset_to_initial_weights(self):
        if self.initial_state is None:
            raise ValueError("Call save_initial_weights() first.")

        device = next(self.model.parameters()).device
        state = {k: v.to(device) for k, v in self.initial_state.items()}

        self.model.load_state_dict(state, strict=False)
        self.apply_masks()

        print("Reset weights to initial values")

    # % of pruned weights
    def get_sparsity(self):
        modules = dict(self.model.named_modules())
        total, pruned = 0, 0

        for name, buf in self.masks.items():
            module_name, _, p_short = name.rpartition(".")
            module = modules[module_name] if module_name else self.model
            mask = getattr(module, buf)
            total += mask.numel()
            pruned += (mask == 0).sum().item()

        return 100.0 * pruned / total if total > 0 else 0.0

    def get_remaining_percentage(self):
        return 100 - self.get_sparsity()
