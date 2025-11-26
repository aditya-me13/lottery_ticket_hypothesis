import torch
import torch.nn as nn
import torch.optim as optim
import time

class Trainer:
    def __init__(self, model, device, train_loader, val_loader, test_loader):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.criterion = nn.CrossEntropyLoss()

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "test_acc": [],
            "iteration": [],
        }

    # one full pass over train set (used to keep epoch stats)
    def train_epoch(self, optimizer, pruner=None):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0

        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()
            out = self.model(data)
            loss = self.criterion(out, target)
            loss.backward()
            optimizer.step()

            if pruner is not None:
                pruner.apply_masks()  # keep pruned weights at zero

            running_loss += loss.item()
            _, pred = out.max(1)
            total += target.size(0)
            correct += pred.eq(target).sum().item()

        return running_loss / len(self.train_loader), 100.0 * correct / total

    def evaluate(self, data_loader, desc="eval"):
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                out = self.model(data)
                loss = self.criterion(out, target)

                running_loss += loss.item()
                _, pred = out.max(1)
                total += target.size(0)
                correct += pred.eq(target).sum().item()

        avg_loss = running_loss / len(data_loader)
        acc = 100.0 * correct / total
        return avg_loss, acc

    def train(self, num_iterations, learning_rate=0.0012, eval_every=100, pruner=None):
        """
        Train for a fixed number of iterations. Validates every epoch end.
        """
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        it_per_epoch = len(self.train_loader)
        # ceil division for display only
        num_epochs = (num_iterations + it_per_epoch - 1) // it_per_epoch

        print("\n" + "=" * 60)
        print("Training Configuration:")
        print(f"  Total iterations: {num_iterations:,}")
        print(f"  Epochs (approx): {num_epochs}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Optimizer: Adam")
        print(f"  Device: {self.device}")
        # model must expose these (your models do)
        print(f"  Parameters: {self.model.count_parameters():,}")
        print(f"  Non-zero parameters: {self.model.count_nonzero_parameters():,}")
        print(f"  Sparsity: {self.model.get_sparsity():.2f}%")
        print("=" * 60 + "\n")

        iteration = 0
        best_val_loss = float("inf")
        best_val_iteration = 0

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # epoch stats
            self.model.train()
            epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0

            for batch_idx, (data, target) in enumerate(self.train_loader):
                if iteration >= num_iterations:
                    break

                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                out = self.model(data)
                loss = self.criterion(out, target)
                loss.backward()
                optimizer.step()

                if pruner is not None:
                    pruner.apply_masks()

                # accumulate epoch stats
                epoch_loss += loss.item()
                _, pred = out.max(1)
                epoch_total += target.size(0)
                epoch_correct += pred.eq(target).sum().item()

                iteration += 1

            # epoch-end metrics
            if epoch_total > 0:
                ep_loss = epoch_loss / (batch_idx + 1)
                ep_acc = 100.0 * epoch_correct / epoch_total
            else:
                ep_loss, ep_acc = 0.0, 0.0

            # store epoch train metrics
            self.history["train_loss"].append(ep_loss)
            self.history["train_acc"].append(ep_acc)

            # always do a val pass at epoch end
            val_loss, val_acc = self.evaluate(self.val_loader, desc="val")
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["iteration"].append(iteration)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_iteration = iteration

            print(f"  Train Loss: {ep_loss:.4f} | Train Acc: {ep_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:  {val_acc:.2f}%")
            print(f"  Best Val Loss: {best_val_loss:.4f} @ iter {best_val_iteration}")

            if iteration >= num_iterations:
                break

        # final test eval
        test_loss, test_acc = self.evaluate(self.test_loader, desc="test")
        self.history["test_acc"].append(test_acc)

        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"  Final Test Accuracy: {test_acc:.2f}%")
        print(f"  Best Validation Loss: {best_val_loss:.4f} @ iter {best_val_iteration}")
        print("=" * 60 + "\n")

        return {
            "best_val_loss": best_val_loss,
            "best_val_iteration": best_val_iteration,
            "final_test_acc": test_acc,
            "history": self.history,
        }

    def get_early_stopping_iteration(self):
        # iteration with minimum val loss
        min_idx = self.history["val_loss"].index(min(self.history["val_loss"]))
        return self.history["iteration"][min_idx]
