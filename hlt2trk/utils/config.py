import fire


class Configuration:
    def __init__(self, model: str = 'LDA', experiment: int = 0,
                 normalize: bool = True, data: str = 'lhcb') -> None:
        """Configure everything.

        Args:
            model (str, optional): Which model to train one of:
                'LDA', 'QDA', 'GNB', 'NN', 'INN', 'BDT'. Defaults to 'LDA'.
            experiment (int, optional): 0, 1, or 2.
                Exp1: uses "fdchi2", "sumpt"
                Exp2: uses "minipchi2", "vchi2"
                Exp3: uses "fdchi2", "sumpt", "minipchi2", "vchi2"
                Defaults to 0.
            normalize (bool, optional): Whether to normalize the data
                between 0 and 1.
                Defaults to True.
            data (str, optional): 'lhcb' or 'sim' data. Defaults to 'lhcb'.
        """
        self.model = model
        self.normalize = normalize
        self.data = data
        self.experiment = experiment


if __name__ == '__main__':
    fire.Fire(Configuration)
