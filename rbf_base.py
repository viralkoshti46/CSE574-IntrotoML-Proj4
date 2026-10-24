__Course__ = "CSE 574 - Intro To Machine Learning"
__author__ = "Viral Koshti"
__email__ = "viraljag@buffalo.edu"
__UBPersonNumber__ = "50761354"

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from common import (
    RESULTS_DIR,
    dataset,
    read_images,
    read_labels,
    read_centers,
    rbf_features,
    distances_squared,
    save_rows,
)

EPOCHS = 12
LEARNING_RATE = 0.2  # Required initial learning rate
DECAY = 1.0  # 1.0 means no learning-rate decay
WIDTH_SCALE_BY_K = {10: 1.0, 100: 4.0, 500: 8.0}  # Final selected settings
SEED = 42


def softmax(values):
    values = values - np.max(values)
    exp_values = np.exp(values)
    return exp_values / np.sum(exp_values)


def evaluate(features, labels, weights, biases):
    # Compute predictions for a dataset using matrix operations.
    scores = features @ weights + biases
    scores -= np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    predicted = np.argmax(probabilities, axis=1)
    errors = int(np.count_nonzero(predicted != labels))
    accuracy = 100.0 * (len(labels) - errors) / len(labels)
    targets = np.eye(10, dtype=np.float32)[labels]
    mse = float(np.mean((targets - probabilities) ** 2))
    return errors, accuracy, mse


def activation_summary(name, features):
    # Helps diagnose Gaussian widths that make nearly all responses zero.
    print(f"{name} RBF mean: {float(np.mean(features)):.8f}")
    print(f"{name} RBF maximum: {float(np.max(features)):.8f}")
    print(
        f"{name} mean strongest response: "
        f"{float(np.mean(np.max(features, axis=1))):.8f}"
    )
    print(f"{name} fraction below 0.001: {float(np.mean(features < 0.001)):.2%}")


def save_plot(history, key, ylabel, title, filename):
    epochs = [item["epoch"] for item in history]
    values = [item[key] for item in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, values, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=180)
    plt.close()


def run(k):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    centers = read_centers(k)
    x_train = read_images(dataset("train-images.idx3-ubyte"))
    y_train = read_labels(dataset("train-labels.idx1-ubyte"))
    x_test = read_images(dataset("t10k-images.idx3-ubyte"))
    y_test = read_labels(dataset("t10k-labels.idx1-ubyte"))

    if len(x_train) != len(y_train) or len(x_test) != len(y_test):
        raise ValueError("The number of images and labels must match.")
    if len(x_test) != 10000:
        raise ValueError("The full MNIST test set must contain 10,000 images.")
    if k not in WIDTH_SCALE_BY_K:
        raise ValueError(
            f"Unsupported K={k}; expected one of {tuple(WIDTH_SCALE_BY_K)}."
        )
    width_scale = WIDTH_SCALE_BY_K[k]
    if width_scale <= 0 or LEARNING_RATE <= 0 or EPOCHS <= 0 or not 0 < DECAY <= 1:
        raise ValueError("Check EPOCHS, LEARNING_RATE, DECAY and width scale.")

    # Lecture 13 - shared Gaussian width, followed by an optional experiment scale.
    d_max = float(np.sqrt(np.max(distances_squared(centers, centers))))
    lecture_sigma = d_max / np.sqrt(2.0 * k)
    sigma = lecture_sigma * width_scale
    variance = sigma**2
    print(f"K={k}, d_max={d_max:.6f}")
    print(f"Lecture sigma={lecture_sigma:.6f}; width scale={width_scale:.2f}")
    print(f"Used sigma={sigma:.6f}; used variance={variance:.6f}")

    # Each image becomes K Gaussian RBF outputs.
    phi_train = rbf_features(x_train, centers, variance)
    phi_test = rbf_features(x_test, centers, variance)
    activation_summary("Training", phi_train)

    rng = np.random.default_rng(SEED)
    weights = rng.normal(0, 0.01, size=(k, 10)).astype(np.float32)
    biases = np.zeros(10, dtype=np.float32)
    history = []

    for epoch in range(1, EPOCHS + 1):
        rate = LEARNING_RATE * DECAY ** (epoch - 1)

        # Online backpropagation: present exactly one training image per update.
        for index in rng.permutation(len(y_train)):
            activation = phi_train[index]
            probabilities = softmax(activation @ weights + biases)
            error = probabilities.copy()
            error[y_train[index]] -= 1.0

            # Update the 10-output layer; RBF centers are not changed.
            weights -= rate * np.outer(activation, error)
            biases -= rate * error

        train_errors, train_accuracy, train_mse = evaluate(
            phi_train, y_train, weights, biases
        )
        test_errors, test_accuracy, test_mse = evaluate(
            phi_test, y_test, weights, biases
        )

        history.append(
            {
                "epoch": epoch,
                "learning_rate": rate,
                "train_mse": train_mse,
                "train_accuracy": train_accuracy,
                "train_errors": train_errors,
                "test_accuracy": test_accuracy,
                "test_errors": test_errors,
                "test_mse": test_mse,
            }
        )
        print(
            f"K={k}, epoch={epoch:02d}: train MSE={train_mse:.6f}, "
            f"test accuracy={test_accuracy:.2f}%, test errors={test_errors}",
            flush=True,
        )

    # Use a suffix for experimental widths to avoid overwriting baseline figures.
    suffix = "" if width_scale == 1.0 else f"_width{width_scale:g}"
    save_rows(RESULTS_DIR / f"RBF_{k}{suffix}_metrics.csv", list(history[0]), history)
    np.savez_compressed(
        RESULTS_DIR / f"RBF_{k}{suffix}_weights.npz", weights=weights, biases=biases
    )
    (RESULTS_DIR / f"RBF_{k}{suffix}_variance.txt").write_text(
        f"K={k}\nd_max={d_max:.9f}\nlecture_sigma={lecture_sigma:.9f}\n"
        f"width_scale={width_scale}\nsigma_used={sigma:.9f}\n"
        f"variance_used={variance:.9f}\nlearning_rate={LEARNING_RATE}\n"
        f"decay={DECAY}\nepochs={EPOCHS}\n",
        encoding="utf-8",
    )

    save_plot(
        history,
        "train_mse",
        "Training MSE",
        f"RBF K={k}: Training MSE (width x{width_scale:g})",
        RESULTS_DIR / f"TrainMSE_{k}{suffix}.png",
    )
    save_plot(
        history,
        "test_accuracy",
        "Test accuracy (%)",
        f"RBF K={k}: Test Accuracy (width x{width_scale:g})",
        RESULTS_DIR / f"Accuracy_{k}{suffix}.png",
    )
    print(f"Saved results in {RESULTS_DIR} (suffix: {suffix or 'baseline'})")
