import re
from collections import defaultdict
from types import SimpleNamespace
import pandas as pd


class MetricsFinder:
    """[summary]

    Returns
    -------
    [type]
        [description]
    """

    def __init__(self, metrics=None, metrics_df=None):
        if metrics is None:
            self.metrics = {
                "date": [],
                "model": [],
                "loss_func": [],
                "act_func": [],
                "opt_func": [],
                "depth": [],
                "batch_size": [],
                "nb_filters0": [],
                "nb_classes": [],
                "init_lr": [],
                "kernel_init": [],
                "class_weights": [],
                "pretrained_layers": [],
                "sigma_noise": [],
                "best_loss": [],
                "best_acc": [],
                "best_jaccard": [],
            }
        self.metrics_df = metrics_df
        self.defaults = defaultdict(lambda: None)
        self.__set_defaults()
        self.__compile_regex()

    def __set_defaults(self):
        for k, val in {
            "date": None,
            "model": "Unet",
            "loss_func": "cat_CE",
            "batch_size": "12",
            "maxpool": True,
            "depth": "4",
            "class_weights": False,
            "nb_filters0": "32",
            "kernel_init": "glorot_uniform",
            "bn": False,
            "sigma_noise": 0,
            "pretrained_layers": 4,
        }.items():
            self.defaults[k] = val

    def __compile_regex(self):
        self.regex = SimpleNamespace()
        self.regex.date = re.compile(r"([0-9]{4}-[0-9]{2}-[0-9]{2})")
        self.regex.arch = re.compile(r"(?<=arch_)[a-zA-Z]+")
        self.regex.loss = re.compile(r"(?<=-[0-9]{2}/)(?!models)([a-zA-Z_]+)(?=-| |/)")
        self.regex.init = re.compile(
            r"((?<=init_)|(?<=initialization_))[a-zA-Z_]+(?!=-)"
        )
        self.regex.act = re.compile(
            r"((?<=act_)|(?<=activation_)|(?<=act_[<a-zA-Z._]{35}))"
            + r"[a-zA-Z_]+((?= object at 0x[0-9abcdef]{12}>)|(?=-)|(?=/))"
        )
        self.regex.cw = re.compile(r"(?<=weights_)(True|False)")
        self.regex.filt = re.compile(r"(?<=nb_filters_)[0-9]+")
        self.regex.bn = re.compile(r"(?<=batchnorm_)(True|False)")
        self.regex.opt = re.compile(
            r"(?<=-[0-9]{2}/)(?:[a-zA-Z_]+ )(?P<opt>[a-zA-Z]+)(?=/)"
        )
        self.regex.bs = re.compile(r"(?<=bs_)(?P<bs>[0-9]+)")
        self.regex.depth = re.compile(r"(?<=block)[0-9]+")
        self.regex.pt = re.compile(r"(?<=pretrain_)([0-9]|False)")
        self.regex.sigma = re.compile(r"((?<=sigma_)|(?<=sigma_noise_))[0-9\.]+")

    def __call__(self, model, path):
        path = self.talos_replacer(path)
        model_params = {
            "date": self.regex.date,
            "model": self.regex.arch,
            "loss_func": model.loss_functions[0].__name__,
            "act_func": self.regex.act,
            "opt_func": model.optimizer.__class__.__name__,
            "depth": self.regex.depth.search(self.bottom_locator(model)).group(),
            "batch_size": self.regex.bs,
            "nb_filters0": model.layers[1].output.shape[-1].value,
            "nb_classes": model.layers[-1].output.shape[-1].value,
            "init_lr": None,
            # "init_lr": model.optimizer.get_config()["learning_rate"]
            # if model.optimizer.__class__.__name__ == "RAdam"
            # else model.optimizer.get_config()["lr"],
            "kernel_init": self.regex.init,
            "class_weights": self.regex.cw,
            "pretrained_layers": self.regex.pt,
            "sigma_noise": self.regex.sigma,
        }
        for name, param in model_params.items():
            if isinstance(param, re._pattern_type):
                res = self.search_or_default(param.search(path), self.defaults[name])
                self.metrics[name].append(res)
            else:
                self.metrics[name].append(param)

    def append_scores(self, model_scores: list):
        scores = ["best_loss", "best_acc", "best_jaccard"]
        for metric, score in zip(scores, model_scores):
            self.metrics[metric].append(score)

    def bottom_locator(self, model) -> str:
        for layer in model.layers:
            if "bottom" in layer.name:
                return layer.name

    def search_or_default(self, pattern, default):
        if pattern is None:
            return default
        return pattern.group()

    def talos_replacer(self, string):
        """replace unsuited wording from file path"""

        return re.sub(r"talos_U-net_model|arch-U-Net", "arch_Unet", string)

    def export_to_csv(self, fname):
        """write metrics to CSV
        
        Parameters
        ----------
        fname : str or file object
            path to save location
        """
        self.metrics_df = pd.DataFrame(data=self.metrics)
        self.metrics_df.to_csv(fname)

    def get_lr(self, events, index):
        if self.metrics["init_lr"][index] == None:
            for (date, opt, act, nb_filters, lr) in events:
                if date == self.metrics["date"][index]:
                    if (
                        act == self.metrics["act_func"][index]
                        and nb_filters == self.metrics["nb_filters0"][index]
                        and opt == self.metrics["opt_func"][index]
                    ):
                        self.metrics["init_lr"][index] = lr
