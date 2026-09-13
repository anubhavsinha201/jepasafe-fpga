"""UCI HAR baseline classifier — 1D CNN + LSTM, PyTorch.

    Conv1D(64,k=5) -> Conv1D(64,k=5) -> MaxPool -> LSTM(64) -> Dense(64) -> Dense(6)

The convolutions learn short local motion primitives (a heel strike, a turn)
and hand the LSTM a shorter, richer sequence to reason over. That is what buys
the accuracy over the plain LSTM this script used to run.

This is a PC-side accuracy benchmark, NOT a deployment candidate: hls4ml's LSTM
support is thin and this model is far past the Tang Nano 9K's 8,640 LUT4 / 20
DSP budget. The deployable architecture lives in `model.py` (Conv1D + attention,
no recurrence).

Data is read straight from the UCI HAR raw inertial signals — no TensorFlow.

    python src/jepasafe/demopart.py                   # finds the dataset itself
    python src/jepasafe/demopart.py --model lstm      # the old plain-LSTM baseline

The dataset folder is located automatically (see DATASET_CANDIDATES); use
--root to point somewhere else. It must contain train/ and test/, each with an
`Inertial Signals/` subfolder.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def _start_dirs() -> list[Path]:
    """Places to begin an upward search.

    `__file__` is missing (or misleading) inside Jupyter, so the current working
    directory has to be a first-class starting point — otherwise every path in
    this module silently resolves somewhere else when run from a notebook.
    """
    dirs = [Path.cwd().resolve()]
    try:
        dirs.append(Path(__file__).resolve().parent)
    except NameError:  # notebook / REPL
        pass
    return dirs


def _walk_up(start: Path):
    return [start, *start.parents]


def find_repo_root() -> Path:
    """Nearest ancestor holding CLAUDE.md — the repo root marker."""
    for start in _start_dirs():
        for directory in _walk_up(start):
            if (directory / "CLAUDE.md").is_file():
                return directory
    return Path.cwd().resolve()


REPO_ROOT = find_repo_root()

# The dataset folder gets named inconsistently depending on where it came from
# ("UCI-HAR Dataset" from some mirrors, "UCI HAR Dataset" from the UCI zip), and
# it may sit at the repo root or under data/raw/.
DATASET_DIR_NAMES = ["UCI-HAR Dataset", "UCI HAR Dataset"]
DATASET_SUBPATHS = [Path("."), Path("data") / "raw"]


def _dataset_candidates() -> list[Path]:
    seen, out = set(), []
    roots = [REPO_ROOT, *_start_dirs()]
    for root in roots:
        for directory in _walk_up(root.resolve()):
            for sub in DATASET_SUBPATHS:
                for name in DATASET_DIR_NAMES:
                    candidate = (directory / sub / name).resolve()
                    if candidate not in seen:
                        seen.add(candidate)
                        out.append(candidate)
    return out


def is_dataset_root(path: Path) -> bool:
    return (path / "train" / "Inertial Signals").is_dir()


def find_dataset_root() -> Path:
    """First candidate that actually contains the inertial signals."""
    for candidate in _dataset_candidates():
        if is_dataset_root(candidate):
            return candidate
    return REPO_ROOT / DATASET_DIR_NAMES[0]  # for the error message in load_split


DEFAULT_ROOT = find_dataset_root()

# Parsed .npy cache. Lives under data/processed/ because .gitignore already
# excludes that tree — the arrays are ~46 MB.
CACHE_DIR = REPO_ROOT / "data" / "processed" / "uci_har"

# The 9 channels, in the conventional order. This fixes the meaning of the
# last input axis — keep it stable or trained weights stop matching the data.
CHANNELS = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]

WINDOW = 128
N_CHANNELS = len(CHANNELS)
N_CLASSES = 6

CLASS_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]


# ------------------------------------------------------------------ data


def load_split(root: Path, split: str, use_cache: bool = True):
    """Read one split into (X, y) with X shaped (n, 128, 9) and y in 0..5.

    Parsing 9 text files per split takes ~30 s, so the result is cached as
    .npy on first run.
    """
    x_cache = CACHE_DIR / f"X_{split}.npy"
    y_cache = CACHE_DIR / f"y_{split}.npy"

    if use_cache and x_cache.exists() and y_cache.exists():
        return np.load(x_cache), np.load(y_cache)

    signals_dir = root / split / "Inertial Signals"
    if not signals_dir.is_dir():
        looked = "\n  ".join(str(c) for c in _dataset_candidates()[:8])
        raise FileNotFoundError(
            f"{signals_dir} not found.\nSearched (first few):\n  {looked}\n"
            "Pass root= / --root pointing at the folder that contains train/ and test/."
        )

    # Each file is (n_windows, 128); stacking on the last axis gives (n, 128, 9)
    channels = [
        np.loadtxt(signals_dir / f"{name}_{split}.txt", dtype=np.float32) for name in CHANNELS
    ]
    x = np.stack(channels, axis=-1)

    # Labels ship as 1..6; the loss expects 0..5
    y = np.loadtxt(root / split / f"y_{split}.txt", dtype=np.int64) - 1

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(x_cache, x)
        np.save(y_cache, y)

    return x, y


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ----------------------------------------------------------------- models


def init_lstm(lstm: nn.LSTM) -> None:
    """Xavier input kernel, orthogonal recurrent kernel, forget-gate bias = 1.

    PyTorch defaults every LSTM weight to uniform(±1/√hidden). The orthogonal
    recurrent kernel keeps gradients from vanishing over 60+ timesteps, and the
    unit forget bias stops the cell from erasing its state early in training —
    both are standard, and both are what Keras does by default.

    Gate order is (input, forget, cell, output), so the forget slice is the
    second quarter. PyTorch's two bias vectors simply sum, so setting the 1.0
    on bias_ih alone is enough.
    """
    hidden = lstm.hidden_size
    for name, param in lstm.named_parameters():
        if name.startswith("weight_ih"):
            nn.init.xavier_uniform_(param)
        elif name.startswith("weight_hh"):
            nn.init.orthogonal_(param)
        elif name.startswith("bias"):
            nn.init.zeros_(param)
    with torch.no_grad():
        lstm.bias_ih_l0[hidden : 2 * hidden].fill_(1.0)


class CNNLSTMClassifier(nn.Module):
    """1D CNN front end + LSTM.

    Shapes, for (batch, 128, 9) in:

        permute      -> (B,  9, 128)   Conv1d wants channels-first
        conv1 k=5    -> (B, 64, 124)
        conv2 k=5    -> (B, 64, 120)
        maxpool 2    -> (B, 64,  60)
        permute      -> (B, 60,  64)   LSTM wants (batch, time, feature)
        lstm         -> (B, 64)        final hidden state only
        head         -> (B,  6)        logits

    Convs carry no bias — BatchNorm's shift parameter makes it redundant.
    """

    def __init__(self, n_channels=N_CHANNELS, filters=64, hidden=64, dense=64,
                 n_classes=N_CLASSES, dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(n_channels, filters, kernel_size=5, bias=False),
            nn.BatchNorm1d(filters),
            nn.ReLU(inplace=True),
            nn.Conv1d(filters, filters, kernel_size=5, bias=False),
            nn.BatchNorm1d(filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(filters, hidden, batch_first=True)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, dense),
            nn.ReLU(inplace=True),
            nn.Linear(dense, n_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        init_lstm(self.lstm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x.transpose(1, 2))
        _, (h_n, _) = self.lstm(x.transpose(1, 2))
        return self.head(h_n[-1])

    @torch.no_grad()
    def probabilities(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self(x), dim=1)


class LSTMClassifier(nn.Module):
    """The previous plain-LSTM baseline, kept for a like-for-like comparison.

    Reachable with `--model lstm`.
    """

    def __init__(self, n_channels=N_CHANNELS, hidden=64, dense=64, n_classes=N_CLASSES):
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden, batch_first=True)
        self.fc1 = nn.Linear(hidden, dense)
        self.fc2 = nn.Linear(dense, n_classes)

        init_lstm(self.lstm)
        for layer in (self.fc1, self.fc2):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.fc2(torch.relu(self.fc1(h_n[-1])))

    @torch.no_grad()
    def probabilities(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self(x), dim=1)


MODELS = {"cnnlstm": CNNLSTMClassifier, "lstm": LSTMClassifier}


# -------------------------------------------------------------- train/eval

# Logits, not softmax: CrossEntropyLoss applies log_softmax internally, which
# is the same function as softmax + sparse_categorical_crossentropy but
# numerically stabler. argmax over logits == argmax over probabilities.


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass. Passing an optimizer makes it a training pass."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    correct = 0
    seen = 0

    with torch.set_grad_enabled(training):
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * y_batch.size(0)
            correct += (logits.argmax(dim=1) == y_batch).sum().item()
            seen += y_batch.size(0)

    return total_loss / seen, correct / seen


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for x_batch, y_batch in loader:
        logits = model(x_batch.to(device))
        y_pred.append(logits.argmax(dim=1).cpu().numpy())
        y_true.append(y_batch.numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def run(
    root: Path | str | None = None,
    model: str = "cnnlstm",
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    dropout: float = 0.3,
    seed: int = 42,
    use_cache: bool = True,
    restore_best: bool = True,
    save: Path | str | None = None,
    verbose: bool = True,
) -> dict:
    """Train and evaluate. This is the entry point to call from a notebook.

        from jepasafe.demopart import run
        result = run(epochs=30)
        result["confusion_matrix"]

    Returns a dict with the trained model, per-epoch history, predictions and
    metrics, so the caller can plot or compare without re-running anything.
    """
    root = Path(root) if root is not None else DEFAULT_ROOT

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"device: {device}")
        print(f"dataset: {root}")

    x_train, y_train = load_split(root, "train", use_cache=use_cache)
    x_test, y_test = load_split(root, "test", use_cache=use_cache)
    if verbose:
        print(f"train: {x_train.shape}  test: {x_test.shape}")

    train_loader = make_loader(x_train, y_train, batch_size, shuffle=True)
    test_loader = make_loader(x_test, y_test, batch_size, shuffle=False)

    if model not in MODELS:
        raise ValueError(f"unknown model {model!r} — choose from {sorted(MODELS)}")
    net = (CNNLSTMClassifier(dropout=dropout) if model == "cnnlstm" else LSTMClassifier()).to(device)
    if verbose:
        print(f"model: {model}  ({sum(p.numel() for p in net.parameters()):,} params)")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-7)

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    best_acc = 0.0
    best_epoch = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(net, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(net, test_loader, criterion, device)

        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        marker = ""
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            best_state = copy.deepcopy(net.state_dict())
            marker = "  *"

        if verbose:
            print(
                f"Epoch {epoch:2d}/{epochs}  {time.time() - start:5.1f}s"
                f" - loss: {train_loss:.4f} - accuracy: {train_acc:.4f}"
                f" - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}{marker}"
            )

    # Dropout slows overfitting but doesn't stop it — the last epoch is rarely
    # the best one. Report the best unless asked otherwise.
    if best_state is not None and restore_best:
        net.load_state_dict(best_state)
        if verbose:
            print(f"restored best epoch {best_epoch} (val_accuracy {best_acc:.4f})")

    y_true, y_pred = collect_predictions(net, test_loader, device)

    result = {
        "model": net,
        "history": history,
        "y_true": y_true,
        "y_pred": y_pred,
        "accuracy": accuracy_score(y_true, y_pred) * 100,
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1": f1_score(y_true, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "best_epoch": best_epoch,
    }

    if verbose:
        print(f"Accuracy : {result['accuracy']:.2f}%")
        print(f"Precision: {result['precision']:.4f}")
        print(f"Recall   : {result['recall']:.4f}")
        print(f"F1 Score : {result['f1']:.4f}")
        print("Confusion Matrix:")
        print(result["confusion_matrix"])
        print("(row/col order: " + ", ".join(CLASS_NAMES) + ")")

    if save:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), save)
        if verbose:
            print(f"saved -> {save}")

    return result


def in_notebook() -> bool:
    """True under Jupyter/IPython, where sys.argv belongs to the kernel."""
    try:
        from IPython import get_ipython

        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--model", choices=sorted(MODELS), default="cnnlstm")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cache", action="store_true", help="always re-parse the .txt files")
    parser.add_argument(
        "--no-restore-best",
        action="store_true",
        help="report the last epoch instead of the best-val-accuracy one",
    )
    parser.add_argument("--save", type=Path, default=None, help="path to save the trained weights")

    # Under Jupyter sys.argv is the kernel's own (-f kernel.json), which
    # argparse rejects with SystemExit(2). Fall back to defaults there.
    if argv is None and in_notebook():
        argv = []
    args = parser.parse_args(argv)

    return run(
        root=args.root,
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout=args.dropout,
        seed=args.seed,
        use_cache=not args.no_cache,
        restore_best=not args.no_restore_best,
        save=args.save,
    )


if __name__ == "__main__":
    main()
