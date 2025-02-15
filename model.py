"""
.. module:: model
    :synopsis: core SynSetMine model

.. moduleauthor:: Jiaming Shen
"""
from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import zoo
import math
from sklearn.metrics import confusion_matrix
from gensim.models.keyedvectors import Word2VecKeyedVectors


def initialize_weights(moduleList, itype="xavier"):
    """Initialize a list of modules

    :param moduleList: a list of nn.modules
    :type moduleList: list
    :param itype: name of initialization method
    :type itype: str
    :return: None
    :rtype: None
    """
    assert itype == "xavier", "Only Xavier initialization supported"

    for moduleId, module in enumerate(moduleList):
        if hasattr(module, "_modules") and len(module._modules) > 0:
            # Iterate again
            initialize_weights(module, itype)
        else:
            # Initialize weights
            name = type(module).__name__
            # If linear or embedding
            if name == "Embedding" or name == "Linear":
                fanIn = module.weight.data.size(0)
                fanOut = module.weight.data.size(1)

                factor = math.sqrt(2.0 / (fanIn + fanOut))
                weight = torch.randn(fanIn, fanOut) * factor
                module.weight.data.copy_(weight)

            # Check for bias and reset
            if hasattr(module, "bias") and hasattr(module.bias, "data"):
                module.bias.data.fill_(0.0)


class SSPM(nn.Module):
    """Synonym Set Prediction Model (SSPM), namely SynSetMine

    :param params: a dictionary containing all model specifications
    :type params: dict
    """

    def __init__(self, params):
        super(SSPM, self).__init__()
        self.initialize(params)

        if params["loss_fn"] == "cross_entropy":
            self.criterion = nn.NLLLoss()
        elif params["loss_fn"] == "max_margin":
            self.criterion = nn.MultiMarginLoss(margin=params["margin"])
        elif params["loss_fn"] in ["margin_rank", "self_margin_rank"]:
            self.criterion = nn.MarginRankingLoss(margin=params["margin"])
        elif params["loss_fn"] == "self_margin_rank_bce":
            self.criterion = nn.BCEWithLogitsLoss()

        # TODO: avoid the following self.params = params
        self.params = params
        # transfer parameters to self, therefore we have self.modelName
        for key, val in self.params.items():
            setattr(self, key, val)

        self.temperature = params["T"]  # use for temperature scaling

    def initialize(self, params):
        """Initialize model components

        :param params: a dictionary containing all model specifications
        :type params: dict
        :return: None
        :rtype: None
        """
        modelParts = zoo.select_model(
            params["modelName"],
            params["vocabSize"],
            params["embedSize"],
            params["node_hiddenSize"],
            params["combine_hiddenSize"],
            params["dropout"],
        )
        flags = [
            "node_embedder",
            "node_post_embedder",
            "node_pooler",
            "combiner",
            "scorer",
        ]

        # refine flags
        for flag in flags:
            if flag not in modelParts:
                print("Missing: %s" % flag)
            else:
                setattr(self, flag, modelParts[flag])

        # define node transform as composition
        self.node_transform = lambda x: self.node_post_embedder(self.node_embedder(x))

        # Initialize the parameters with xavier method
        modules = [
            "node_embedder",
            "node_postEmbedder",
            "combiner",
            "scorer",
        ]
        modules = [getattr(self, mod) for mod in modules if hasattr(self, mod)]
        initialize_weights(modules, "xavier")

        if params["pretrained_embedding"] != "none":
            pretrained_embedding = params["embedding"].vectors
            padding_embedding = np.zeros([1, pretrained_embedding.shape[1]])
            pretrained_embedding = np.row_stack(
                [padding_embedding, pretrained_embedding]
            )
            self.node_embedder.weight.data.copy_(torch.from_numpy(pretrained_embedding))
            if params["embed_fine_tune"] == 0:  # fix embedding without fine-tune
                self.node_embedder.weight.requires_grad = False

    def _set_scorer(self, set_tensor):
        """Return the quality score of a batch of sets

        :param set_tensor: sets to be scored, size: (batch_size, max_set_size)
        :type set_tensor: tensor
        :return: scores of all sets, size: (batch_size, 1)
        :rtype: tensor
        """
        # Element encoding
        mask = (set_tensor != 0).float().unsqueeze_(-1)  # (batch_size, max_set_size, 1)
        setEmbed = self.nodeTransform(set_tensor) * mask

        # Set encoding by pooling element representations
        setEmbed = self.node_pooler(setEmbed, dim=1)  # (batch_size, node_hiddenSize)

        # Set scoring based on set encoding, possibly conditioned on current set size
        setScores = self.scorer(setEmbed)  # (batch_size, 1)

        return setScores

    def train_step(self, train_batch):
        """Train the model on the given train_batch

        :param train_batch: a dictionary containing training batch in <set, instance> pair format
        :type train_batch: dict
        :return: batch_loss, true_positive_num, false_positive_num, false_negative_num, true_positive_num
        :rtype: tuple
        """
        # obtain set quality scores
        mask = (train_batch["set"] != 0).float().unsqueeze_(-1)
        setEmbed = self.node_transform(train_batch["set"])
        setEmbed = setEmbed * mask
        setEmbed = self.node_pooler(setEmbed, dim=1)  # (batch_size, node_hiddenSize)
        setScores = self.scorer(setEmbed)  # (batch_size， 1)

        # obtain set union instance quality scores
        instEmbed = self.node_transform(train_batch["inst"]).squeeze(
            1
        )  # (batch_size, node_hiddenSize)
        setInstSumScores = self.scorer(setEmbed + instEmbed)  # (batch_size, 1)

        # calculate score differences for model update
        score_diff = setInstSumScores - setScores  # (batch_size, 1)
        score_diff = score_diff.squeeze(-1)  # (batch_size, )
        score_diff /= self.temperature  # temperature scaling

        target = train_batch["label"].squeeze(-1).float()  # (batch_size, )
        loss = self.criterion(score_diff, target)
        loss.backward()

        # return additional target information of current batch, this may slow down model training
        y_true = target.cpu().numpy()
        y_pred = (score_diff > 0.0).squeeze().cpu().numpy()
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return loss.item(), tn, fp, fn, tp

    def predict(self, batch_set_tensor, batch_inst_tensor):
        """Make set instance pair prediction

        :param batch_set_tensor: packed sets in a collection of <set, instance> pairs, size: (batch_size, max_set_size)
        :type batch_set_tensor: tensor
        :param batch_inst_tensor: packed instances in a collection of <set, instance> pairs, size: (batch_size, 1)
        :type batch_inst_tensor: tensor
        :return:

            - scores of packed sets, (batch_size, 1)
            - scores of packed sets union with corresponding instances, (batch_size, 1)
            - the probability of adding the instance into the corresponding set, (batch_size, 1)

        :rtype: tuple
        """
        setScores = self._set_scorer(batch_set_tensor)
        setInstSumScores = self._set_scorer(
            torch.cat([batch_inst_tensor, batch_set_tensor], dim=1)
        )

        setInstSumScores /= self.temperature
        setScores /= self.temperature
        prediction = F.sigmoid(setInstSumScores - setScores)

        return setScores, setInstSumScores, prediction

    def _get_test_sip_batch_size(self, x):
        if len(x) <= 1000:
            return len(x)
        elif len(x) > 1000 and (len(x) <= 1000 * 1000):
            return len(x) / 1000
        else:
            return 1000


class BaseSSPM(nn.Module):
    module_names = [
        "node_embedder",
        "node_post_embedder",
        "node_pooler",
        "combiner",
        "scorer",
    ]

    node_embedder: nn.Embedding
    node_post_embedder: nn.Module
    node_pooler: Callable
    combiner: nn.Module
    scorer: nn.Module
    temperature: float

    def __init__(
        self,
        model_name: str,
        vocab_size: int,
        embed_size: int,
        embedding: Word2VecKeyedVectors,
        node_hidden_size: int,
        combin_hidden_size: int,
        dropout_ratio: float,
        temperature: float,
    ) -> None:
        super().__init__()

        self.temperature = temperature

        modules = zoo.select_model(
            model_name=model_name,
            vocab_size=vocab_size,
            embed_size=embed_size,
            node_hidden_size=node_hidden_size,
            combine_hidden_size=combin_hidden_size,
            dropout_ratio=dropout_ratio,
        )

        # refine flags
        for module_name in self.module_names:
            if module_name not in modules:
                print(f"Missing: {module_name}")
            else:
                setattr(self, module_name, modules[module_name])

        # define node transform as composition
        self.node_transform = lambda x: self.node_post_embedder(self.node_embedder(x))

        # Initialize the parameters with xavier method
        modules = [
            "node_embedder",
            "node_post_embedder",
            "combiner",
            "scorer",
        ]
        modules = [getattr(self, mod) for mod in modules if hasattr(self, mod)]
        initialize_weights(modules, "xavier")

        # load pretrained embeding
        pretrained_embedding = embedding.vectors
        padding_embedding = np.zeros([1, pretrained_embedding.shape[1]])
        pretrained_embedding = np.row_stack([padding_embedding, pretrained_embedding])
        self.node_embedder.weight.data.copy_(torch.from_numpy(pretrained_embedding))
        self.node_embedder.weight.requires_grad = False

    def set_scorer(self, set_tensor: torch.Tensor) -> torch.Tensor:
        """Return the quality score of a batch of sets

        :param set_tensor: sets to be scored, size: (batch_size, max_set_size)
        :type set_tensor: tensor
        :return: scores of all sets, size: (batch_size, 1)
        :rtype: tensor
        """
        # Element encoding
        mask = (set_tensor != 0).float().unsqueeze_(-1)  # (batch_size, max_set_size, 1)
        set_embed = self.node_transform(set_tensor) * mask

        # Set encoding by pooling element representations
        set_embed = self.node_pooler(set_embed, dim=1)  # (batch_size, node_hiddenSize)

        # Set scoring based on set encoding, possibly conditioned on current set size
        set_scores = self.scorer(set_embed)  # (batch_size, 1)

        return set_scores / self.temperature

    def forward(self, train_batch) -> torch.Tensor:
        """Train the model on the given train_batch

        :param train_batch: a dictionary containing training batch in <set, instance> pair format
        :type train_batch: dict
        :return: batch_loss, true_positive_num, false_positive_num, false_negative_num, true_positive_num
        :rtype: tuple
        """
        # obtain set quality scores
        mask = (train_batch["set"] != 0).float().unsqueeze_(-1)
        set_embed = self.node_transform(train_batch["set"])
        set_embed = set_embed * mask
        set_embed = self.node_pooler(set_embed, dim=1)  # (batch_size, node_hiddenSize)
        set_scores = self.scorer(set_embed)  # (batch_size， 1)

        # obtain set union instance quality scores
        inst_embed = self.node_transform(train_batch["inst"]).squeeze(
            1
        )  # (batch_size, node_hiddenSize)
        set_inst_sum_scores = self.scorer(set_embed + inst_embed)  # (batch_size, 1)

        # calculate score differences for model update
        score_diff = set_inst_sum_scores - set_scores  # (batch_size, 1)
        score_diff = score_diff.squeeze(-1)  # (batch_size, )
        score_diff /= self.temperature  # temperature scaling

        return score_diff


class JointLoss(nn.Module):
    def __init__(self, loss_fn: str, w: float = 0.7, margin=0.5) -> None:
        super().__init__()
        if loss_fn == "cross_entropy":
            self.visn_criterion = nn.NLLLoss()
            self.lang_criterion = nn.NLLLoss()
        elif loss_fn == "max_margin":
            self.visn_criterion = nn.MultiMarginLoss(margin=margin)
            self.lang_criterion = nn.MultiMarginLoss(margin=margin)
        elif loss_fn in ["margin_rank", "self_margin_rank"]:
            self.visn_criterion = nn.MarginRankingLoss(margin=margin)
            self.lang_criterion = nn.MarginRankingLoss(margin=margin)
        elif loss_fn == "self_margin_rank_bce":
            self.visn_criterion = nn.BCEWithLogitsLoss()
            self.lang_criterion = nn.BCEWithLogitsLoss()

        self.w = w

    def forward(
        self, visn_pred: torch.Tensor, lang_pred: torch.Tensor, target: torch.Tensor
    ):
        visn_loss = self.visn_criterion(visn_pred, target)
        lang_loss = self.lang_criterion(lang_pred, target)
        return self.w * visn_loss + (1 - self.w) * lang_loss


class JointSSPM(nn.Module):
    def __init__(self, params) -> None:
        super().__init__()

        self.field_sspm = BaseSSPM(
            model_name=params["modelName"],
            vocab_size=params["vocabSize"],
            embed_size=params["fieldSize"] * params["embedSize"],
            embedding=params["field_embedding"],
            node_hidden_size=params["node_hiddenSize"],
            combin_hidden_size=params["combine_hiddenSize"],
            dropout_ratio=params["dropout"],
            temperature=params["T"],
        )
        self.lang_sspm = BaseSSPM(
            model_name=params["modelName"],
            vocab_size=params["vocabSize"],
            embed_size=params["embedSize"],
            embedding=params["lang_embedding"],
            node_hidden_size=params["node_hiddenSize"],
            combin_hidden_size=params["combine_hiddenSize"],
            dropout_ratio=params["dropout"],
            temperature=params["T"],
        )

        self.criterion = JointLoss(
            params["loss_fn"], w=params["loss_w"], margin=params["margin"]
        )

        # TODO: avoid the following self.params = params
        self.params = params
        # transfer parameters to self, therefore we have self.modelName
        for key, val in self.params.items():
            setattr(self, key, val)

    def train_step(self, train_batch):
        target = train_batch["label"].squeeze(-1).float()

        field_output = self.field_sspm(train_batch)
        lang_output = self.lang_sspm(train_batch)

        loss = self.criterion(field_output, lang_output, target)
        loss.backward()

        y_true = target.cpu().numpy()
        y_pred = (lang_output > 0.0).squeeze().cpu().numpy()
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return loss.item(), tn, fp, fn, tp

    def predict(
        self, batch_set_tensor: torch.Tensor, batch_inst_tensor: torch.Tensor
    ) -> Tuple:
        """Make set instance pair prediction

        :param batch_set_tensor: packed sets in a collection of <set, instance> pairs, size: (batch_size, max_set_size)
        :type batch_set_tensor: tensor
        :param batch_inst_tensor: packed instances in a collection of <set, instance> pairs, size: (batch_size, 1)
        :type batch_inst_tensor: tensor
        :return:

            - scores of packed sets, (batch_size, 1)
            - scores of packed sets union with corresponding instances, (batch_size, 1)
            - the probability of adding the instance into the corresponding set, (batch_size, 1)

        :rtype: tuple
        """
        lang_sspm_set_scores = self.lang_sspm.set_scorer(batch_set_tensor)
        field_sspm_set_scores = self.field_sspm.set_scorer(batch_set_tensor)

        lang_set_inst_sum_scores = self.lang_sspm.set_scorer(
            torch.cat([batch_inst_tensor, batch_set_tensor], dim=1)
        )
        field_set_inst_sum_scores = self.lang_sspm.set_scorer(
            torch.cat([batch_inst_tensor, batch_set_tensor], dim=1)
        )

        lang_prediction = torch.sigmoid(
            field_sspm_set_scores - field_set_inst_sum_scores
        )
        # visn_prediction = torch.sigmoid(visn_set_inst_sum_scores - visn_sspm_set_scores)

        # prediction = visn_prediction * self.params["loss_w"] + lang_prediction * (
        #     1 - self.params["loss_w"]
        # )

        return field_sspm_set_scores, field_set_inst_sum_scores, lang_prediction
