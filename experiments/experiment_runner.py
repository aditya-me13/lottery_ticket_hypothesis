import json
import time
from datetime import datetime
from pathlib import Path

import torch


# project imports
from utils.data_loader import get_dataloaders
from utils.trainer import Trainer
from utils.pruning import PruningManager
from models.lenet_fc import LeNet300100
from models.lenet_conv import LeNet5
from models.conv6 import Conv6  # ensure you placed Conv6 here

MODEL_REGISTRY = {
    "lenet_fc": LeNet300100,
    "lenet_conv": LeNet5,
    "conv6": Conv6,
}

class ExperimentRunner:
    """
    Unified IMP experiment runner.
    """

    def __init__(
        self,
        dataset="mnist",                 # mnist | fashion | cifar10
        model_name="lenet_fc",           # lenet_fc | lenet_conv | conv6
        pruning_type="magnitude",        # magnitude | random
        pruning_scope="layerwise",       # layerwise | global
        reinit_method="rewind",          # rewind | none | random
        pruning_rate=0.3,
        num_rounds=16,
        iterations=40000,
        learning_rate=0.0012,
        batch_size=60,
        device=None,
        results_dir="results",
        verbose=True,
    ):
        self.dataset = dataset
        self.model_name = model_name
        self.Model = MODEL_REGISTRY[model_name]
        self.pruning_type = pruning_type
        self.layer_wise = (pruning_scope == "layerwise")
        self.reinit_method = reinit_method
        self.pruning_rate = pruning_rate
        self.num_rounds = num_rounds
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.results_dir = results_dir
        self.verbose = verbose

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # data
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            dataset=self.dataset,
            batch_size=self.batch_size,
        )

        # model + pruning
        self.model = self.Model().to(self.device)
        self.pruner = PruningManager(self.model)
        self.pruner.save_initial_weights()
        self.pruner.initialize_masks()

        # results
        self.results = {
            "config": {
                "dataset": self.dataset,
                "model": self.model_name,
                "pruning_type": self.pruning_type,
                "pruning_scope": pruning_scope,
                "reinit_method": self.reinit_method,
                "pruning_rate": self.pruning_rate,
                "num_rounds": self.num_rounds,
                "iterations": self.iterations,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "device": str(self.device),
            },
            "rounds": [],
        }

        if self.verbose:
            print("\n" + "=" * 70)
            print("EXPERIMENT")
            print("=" * 70)
            for k, v in self.results["config"].items():
                print(f"  {k}: {v}")
            print("=" * 70 + "\n")

    def _train_one_round(self):
        trainer = Trainer(self.model, self.device, self.train_loader, self.val_loader, self.test_loader)
        out = trainer.train(
            num_iterations=self.iterations,
            learning_rate=self.learning_rate,
            pruner=self.pruner,
        )
        early_iter = trainer.get_early_stopping_iteration()
        idx = trainer.history["iteration"].index(early_iter)
        early_val_acc = trainer.history["val_acc"][idx]
        return out, early_iter, early_val_acc, trainer.history

    def _do_prune(self):
        if self.pruning_type == "magnitude":
            return self.pruner.prune_by_magnitude(self.pruning_rate, layer_wise=self.layer_wise)
        elif self.pruning_type == "random":
            return self.pruner.prune_random(self.pruning_rate, layer_wise=self.layer_wise)
        else:
            raise ValueError(f"Unknown pruning_type: {self.pruning_type}")

    def _do_reinit(self):
        if self.reinit_method == "rewind":
            self.pruner.reset_to_initial_weights()
        elif self.reinit_method == "random":
            self.pruner.random_reinitialize_survivors()
        elif self.reinit_method == "none":
            pass
        else:
            raise ValueError(f"Unknown reinit_method: {self.reinit_method}")

    def run(self):
        for r in range(1, self.num_rounds + 1):
            print("\n" + "=" * 70)
            print(f"ROUND {r}/{self.num_rounds}")
            print("=" * 70)

            t0 = time.time()

            # pre-train sparsity snapshot
            sp_before = self.pruner.get_sparsity()
            rem_before = 100.0 - sp_before
            print(f"Start sparsity: {sp_before:.2f}%  ({rem_before:.2f}% remaining)")

            # train
            out, early_iter, early_val_acc, history = self._train_one_round()

            round_rec = {
                "round": r,
                "sparsity_before": sp_before,
                "remaining_before": rem_before,
                "best_val_loss": out["best_val_loss"],
                "best_val_iteration": out["best_val_iteration"],
                "early_stop_iteration": early_iter,
                "early_stop_val_acc": early_val_acc,
                "final_test_acc": out["final_test_acc"],
                "training_time_sec": time.time() - t0,
                "history": history,
            }

            print(f"  Early stop at {early_iter} | Validation at ES {early_val_acc:.2f}% | Test at ES {out['final_test_acc']:.2f}%")

            # skip pruning after last round
            if r < self.num_rounds:
                print(f"\nPrune {self.pruning_rate*100:.1f}% ({self.pruning_type}, {'layerwise' if self.layer_wise else 'global'})")
                sp_after = self._do_prune()
                rem_after = 100.0 - sp_after
                round_rec["sparsity_after"] = sp_after
                round_rec["remaining_after"] = rem_after
                print(f"Sparsity now: {sp_after:.2f}%  ({rem_after:.2f}% remaining)")

                print(f"Reinit: {self.reinit_method}")
                self._do_reinit()

            self.results["rounds"].append(round_rec)

        self._print_summary()
        return self.results

    def _print_summary(self):
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"{'Round':<8} {'Remaining%':<12} {'Early-Stop':<12} {'Val Acc%':<10} {'Test Acc%':<10}")
        print("-" * 70)
        for r in self.results["rounds"]:
            print(f"{r['round']:<8}"
                f"{r['remaining_before']:>12.2f}"
                f"{r['early_stop_iteration']:>12}"
                f"{r['early_stop_val_acc']:>12.2f}"
                f"{r['final_test_acc']:>12.2f}")
            
        print("=" * 70 + "\n")

    def save_results(self, filename=None):
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)

        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            pr_scope = "layer" if self.layer_wise else "global"
            pr_rate = int(self.pruning_rate * 100)
            filename = (
                f"{self.dataset}_{self.model_name}_{self.pruning_type}_{pr_scope}_"
                f"{self.reinit_method}_pr{pr_rate}_r{self.num_rounds}_it{self.iterations}_lr{self.learning_rate}_{ts}.json"
            )
        path = Path(self.results_dir) / filename

        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"Saved: {path}")
        return str(path)
