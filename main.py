import argparse
import numpy as np
import logging
from typing import List, Dict
from spcarobustness.utils.logging_config import configure_logging
from spcarobustness.models.components import FixedLinear, FixedScaler, ClassifierNN, PipelineModel, ImageFlattenWrapper
from spcarobustness.train.trainer import train_pipeline_model
from spcarobustness.attacks.factory import make_attack
from spcarobustness.eval.metrics import evaluate_robustness
from spcarobustness.eval.plots import plot_benchmark, show_adversarial_samples_mnist, show_adversarial_samples_cifar_binary
from spcarobustness.utils.io import model_path, save_classifier, load_classifier

import torch
from sklearn.decomposition import PCA, SparsePCA, MiniBatchSparsePCA
from sklearn.preprocessing import StandardScaler


def _build_pipeline(transformer, X_train, y_train, device, num_classes, num_epochs, input_dim, image_input_shape=None):
    transformer.fit(X_train)
    components = transformer.components_
    if hasattr(transformer, "mean_"):
        mean_val = transformer.mean_
    else:
        import numpy as np
        mean_val = np.zeros(input_dim, dtype=np.float32)

    import numpy as np
    X_train_proj = (X_train - mean_val) @ components.T
    scaler = StandardScaler()
    scaler.fit(X_train_proj)

    n_comp = components.shape[0]
    import torch as _torch
    weight = _torch.tensor(components, dtype=_torch.float32)
    bias = -_torch.tensor(mean_val, dtype=_torch.float32) @ _torch.tensor(components.T, dtype=_torch.float32)

    fixed_transform = FixedLinear(weight, bias)
    fixed_scaler = FixedScaler(scaler.mean_, scaler.scale_)
    classifier_nn = ClassifierNN(input_dim=n_comp, num_classes=num_classes)
    model = PipelineModel(fixed_transform, fixed_scaler, classifier_nn).to(device)

    model, optimizer, criterion = train_pipeline_model(
        model, X_train, y_train, device, num_epochs=num_epochs, verbose=True
    )

    from art.estimators.classification import PyTorchClassifier
    # If attacks expect image-shaped inputs (e.g., SquareAttack), we can wrap the model to accept images
    # while still operating on flattened vectors internally. When image_input_shape is provided,
    # we use ImageFlattenWrapper and set input_shape accordingly.
    wrapped_model = model
    input_shape = (input_dim,)
    if image_input_shape is not None:
        wrapped_model = ImageFlattenWrapper(model)
        input_shape = image_input_shape

    art_classifier = PyTorchClassifier(
        model=wrapped_model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=input_shape,
        nb_classes=num_classes,
        clip_values=(0, 1),
    )
    return art_classifier, components, mean_val, scaler


def run_experiment(args):
    logger = configure_logging(logging.INFO)
    attacks = []
    if args.attacks:
        if len(args.attacks) == 1 and args.attacks[0].upper() == "ALL":
            attacks = ["FGSM", "PGD", "MIM", "SQUARE"]
        else:
            attacks = [a.upper() for a in args.attacks]
    else:
        attacks = [args.attack.upper()]
    logger.info(f"Attacks: {attacks}")
    logger.info(f"Norm: {args.norm}")
    logger.info(f"n_components_list: {args.n_components}")
    logger.info(f"eps_list: {args.eps_start}..{args.eps_end} step {args.eps_step}")
    logger.info(f"n_samples: {args.n_samples}")

    if args.dataset == "mnist":
        from spcarobustness.data.mnist import load_mnist
        X_train, X_test, y_train, y_test, n_total = load_mnist(n_samples=args.n_samples)
        num_classes = 10
        input_dim = 784
        image_shape = (1, 28, 28)
        prefix = "mnist_"
    elif args.dataset == "cifar-binary":
        from spcarobustness.data.cifar import load_cifar10_binary_airplane_frog
        X_train, X_test, y_train, y_test, n_total = load_cifar10_binary_airplane_frog(n_samples=args.n_samples)
        num_classes = 2
        input_dim = 3072
        image_shape = (3, 32, 32)
        prefix = "cifar10_binary_airplane_frog_"
    else:
        raise ValueError("Unknown dataset")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    classifiers_pca: Dict[int, object] = {}
    classifiers_spca: Dict[int, object] = {}
    cached_meta: Dict[str, Dict] = {}

    for n_comp in args.n_components:
        if args.dataset.startswith("cifar"):
            pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=29)
        else:
            pca = PCA(n_components=n_comp)
        spca = None
        if args.spca_mode == "minibatch":
            spca = MiniBatchSparsePCA(n_components=n_comp, alpha=1, max_iter=200, batch_size=256, random_state=29)
        else:
            spca = SparsePCA(n_components=n_comp, random_state=29, max_iter=200, alpha=1)

        # Try load PCA
        pca_path = model_path(args.models_dir, args.dataset, "pca", n_comp, num_classes, input_dim)
        loaded = load_classifier(pca_path, device)
        if loaded and args.save_models:
            classifiers_pca[n_comp], _ = loaded
            logger.info(f"Loaded PCA model from {pca_path}")
        else:
            clf, comps, mean_v, scaler = _build_pipeline(pca, X_train, y_train, device, num_classes, args.epochs, input_dim, image_input_shape=image_shape)
            classifiers_pca[n_comp] = clf
            if args.save_models:
                save_classifier(
                    pca_path,
                    algo="pca",
                    n_components=n_comp,
                    input_dim=input_dim,
                    num_classes=num_classes,
                    components=comps,
                    mean_vec=mean_v,
                    scaler_mean=scaler.mean_,
                    scaler_scale=scaler.scale_,
                    classifier_state=clf.model.classifier.state_dict(),
                )
                logger.info(f"Saved PCA model to {pca_path}")

        # Try load SPCA
        spca_path = model_path(args.models_dir, args.dataset, "spca", n_comp, num_classes, input_dim, spca_mode=args.spca_mode)
        loaded = load_classifier(spca_path, device)
        if loaded and args.save_models:
            classifiers_spca[n_comp], _ = loaded
            logger.info(f"Loaded SPCA model from {spca_path}")
        else:
            clf, comps, mean_v, scaler = _build_pipeline(spca, X_train, y_train, device, num_classes, args.epochs * (2 if args.dataset.startswith("cifar") else 1), input_dim, image_input_shape=image_shape)
            classifiers_spca[n_comp] = clf
            if args.save_models:
                save_classifier(
                    spca_path,
                    algo="spca",
                    n_components=n_comp,
                    input_dim=input_dim,
                    num_classes=num_classes,
                    components=comps,
                    mean_vec=mean_v,
                    scaler_mean=scaler.mean_,
                    scaler_scale=scaler.scale_,
                    classifier_state=clf.model.classifier.state_dict(),
                    spca_mode=args.spca_mode,
                )
                logger.info(f"Saved SPCA model to {spca_path}")

    # Optionally subsample test set for faster attacks
    if getattr(args, "attack_n_test", None):
        n_use = min(len(X_test), int(args.attack_n_test))
        X_test_use = X_test[:n_use]
        y_test_use = y_test[:n_use]
    else:
        X_test_use = X_test
        y_test_use = y_test

    # Prepare clean inputs in the shape expected by ART (image-shaped)
    X_clean = X_test_use.reshape((-1,) + image_shape)

    # eps sweep
    eps_list = np.arange(args.eps_start, args.eps_end + 1e-9, args.eps_step, dtype=np.float32)

    # clean accuracy baseline
    def evaluate_dict(classifiers_dict):
        accs = {}
        for n_comp, clf in classifiers_dict.items():
            clean_acc, _ = evaluate_robustness(X_clean, X_clean, y_test_use, clf)
            accs[n_comp] = [clean_acc]
        return accs

    pca_accs = evaluate_dict(classifiers_pca)
    spca_accs = evaluate_dict(classifiers_spca)

    from tqdm import tqdm
    for attack in attacks:
        # fresh copies of robust accuracy dicts (keep same clean baseline)
        pca_accs_att = {k: v[:] for k, v in pca_accs.items()}
        spca_accs_att = {k: v[:] for k, v in spca_accs.items()}

        logger.info(f"Running attack sweep for {attack}...")
        for n_comp in classifiers_pca.keys():
            X_adv_pcas_store = []
            X_adv_spcas_store = []
            for eps in tqdm(eps_list, desc=f"[{attack}] n_comp={n_comp}"):
                attack_params = {"norm": args.norm, "batch_size": getattr(args, "attack_batch_size", 256)}
                if attack.lower() in {"pgd", "mim"}:
                    attack_params["eps_step_ratio"] = 0.25 if attack.lower() == "pgd" else 0.10
                    attack_params["max_iter"] = 40 if attack.lower() == "pgd" else 10
                if attack.lower() in {"square", "squareattack"}:
                    attack_params["max_iter"] = int(getattr(args, "square_max_iter", 250))
                    attack_params["p_init"] = 0.8
                    attack_params["nb_restarts"] = int(getattr(args, "square_restarts", 1))

                attack_pca = make_attack(classifiers_pca[n_comp], attack, float(eps), attack_params)
                attack_spca = make_attack(classifiers_spca[n_comp], attack, float(eps), attack_params)

                X_adv_pca = attack_pca.generate(x=X_clean)
                X_adv_spca = attack_spca.generate(x=X_clean)
                X_adv_pca = np.clip(X_adv_pca, 0, 1)
                X_adv_spca = np.clip(X_adv_spca, 0, 1)

                _, pca_adv_acc = evaluate_robustness(X_clean, X_adv_pca, y_test_use, classifiers_pca[n_comp])
                _, spca_adv_acc = evaluate_robustness(X_clean, X_adv_spca, y_test_use, classifiers_spca[n_comp])

                pca_accs_att[n_comp].append(pca_adv_acc)
                spca_accs_att[n_comp].append(spca_adv_acc)
                if args.save_samples:
                    X_adv_pcas_store.append(X_adv_pca[0])
                    X_adv_spcas_store.append(X_adv_spca[0])

            if args.save_samples:
                norm_label = "linf" if args.norm == np.inf else ("l2" if args.norm == 2 else ("l1" if args.norm == 1 else "custom"))
                if args.dataset == "mnist":
                    directory = (
                        f"adv_samples_{attack.lower()}_norm_{norm_label}_"
                        f"eps_{eps_list[0]}_to_{eps_list[-1]}_"
                        f"ncomp_{n_comp}_nsamples_{n_total}"
                    )
                    show_adversarial_samples_mnist(
                        X_test[0], X_adv_pcas_store, X_adv_spcas_store, eps_list.tolist(), n_comp, directory, attack, norm_label
                    )
                else:
                    directory = (
                        f"cifar10_binary_airplane_frog_adv_{attack.lower()}_norm_{norm_label}_"
                        f"eps_{eps_list[0]:.3f}_to_{eps_list[-1]:.3f}_"
                        f"ncomp_{n_comp}_nsamples_{n_total}"
                    )
                    show_adversarial_samples_cifar_binary(
                        X_test[0], X_adv_pcas_store, X_adv_spcas_store, eps_list.tolist(), n_comp, directory, attack, norm_label
                    )

        norm_label = "linf" if args.norm == np.inf else ("l2" if args.norm == 2 else ("l1" if args.norm == 1 else "custom"))
        plot_benchmark(
            np.concatenate(([0], eps_list)),
            pca_accs_att,
            spca_accs_att,
            args.n_components,
            n_total,
            attack,
            norm_label,
            prefix=prefix,
        )


def main():
    p = argparse.ArgumentParser(description="Unified SPCA vs PCA robustness experiments")
    p.add_argument("--dataset", choices=["mnist", "cifar-binary"], default="mnist")
    p.add_argument("--attack", choices=["FGSM", "PGD", "MIM", "SQUARE"], default="FGSM", help="Single attack if --attacks is not provided")
    p.add_argument("--attacks", nargs="*", help='Multiple attacks to run in one go, e.g. --attacks FGSM PGD MIM SQUARE or --attacks ALL')
    p.add_argument("--norm", type=float, default=2, help="Attack norm: inf for Linf, 2 for L2, 1 for L1")
    p.add_argument("--n-components", dest="n_components", type=int, nargs="+", default=[100, 150, 200])
    p.add_argument("--eps-start", dest="eps_start", type=float, default=0.01)
    p.add_argument("--eps-end", dest="eps_end", type=float, default=0.2)
    p.add_argument("--eps-step", dest="eps_step", type=float, default=0.01)
    p.add_argument("--n-samples", dest="n_samples", type=int, default=None)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--spca-mode", choices=["minibatch", "sparse"], default="minibatch", help="SPCA solver mode")
    p.add_argument("--save-samples", action="store_true", help="Save adversarial sample panels for the first test image")
    p.add_argument("--save-models", action="store_true", help="Cache trained classifiers to disk and reuse if available")
    p.add_argument("--models-dir", type=str, default="models", help="Directory to store/load cached models")
    # Performance knobs for attacks
    p.add_argument("--attack-batch-size", dest="attack_batch_size", type=int, default=256, help="Batch size for attack generation")
    p.add_argument("--attack-n-test", dest="attack_n_test", type=int, default=None, help="Limit number of test examples used for attacks for speed")
    p.add_argument("--square-max-iter", dest="square_max_iter", type=int, default=250, help="Max iterations for SquareAttack (default reduced for speed)")
    p.add_argument("--square-restarts", dest="square_restarts", type=int, default=1, help="Number of restarts for SquareAttack (increase for stronger attacks)")
    args = p.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
