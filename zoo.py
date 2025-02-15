import torch
import torch.nn as nn
from math import floor


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, in_tensor):
        return in_tensor


# module to split at a given point and sum
class SplitSum(nn.Module):
    def __init__(self, splitSize):
        super(SplitSum, self).__init__()
        self.splitSize = splitSize

    def forward(self, inTensor):
        # Split along particular dimension
        # If only two dims
        if inTensor.dim() == 2:
            firstHalf = inTensor[:, : self.splitSize]
            secondHalf = inTensor[:, self.splitSize :]
        else:
            firstHalf = inTensor[:, :, : self.splitSize]
            secondHalf = inTensor[:, :, self.splitSize :]
        return firstHalf + secondHalf


# module to split at a given point and max
class SplitMax(nn.Module):
    def __init__(self, splitSize):
        super(SplitMax, self).__init__()
        self.splitSize = splitSize  # where to split

    def forward(self, inTensor):
        # Split along particular dimension
        # If only two dims
        if inTensor.dim() == 2:
            firstHalf = inTensor[:, : self.splitSize]
            secondHalf = inTensor[:, self.splitSize :]
        else:
            firstHalf = inTensor[:, :, : self.splitSize]
            secondHalf = inTensor[:, :, self.splitSize :]
        numDims = firstHalf.dim()
        concat = torch.cat(
            (firstHalf.unsqueeze(numDims), secondHalf.unsqueeze(numDims)), numDims
        )
        maxPool = torch.max(concat, numDims)[0]
        return maxPool


# module to split at two given points, sum the first and the third parts, and then concat with second part
class SplitSumConcat(nn.Module):
    def __init__(self, splitSize1, splitSize2):
        super(SplitSumConcat, self).__init__()
        self.splitSize1 = splitSize1
        self.splitSize2 = splitSize2

    def forward(self, inTensor):
        # Split along particular dimension
        # If only two dims
        if inTensor.dim() == 2:
            firstPart = inTensor[:, : self.splitSize1]
            secondPart = inTensor[
                :, self.splitSize1 : (self.splitSize1 + self.splitSize2)
            ]
            thirdPart = inTensor[:, (self.splitSize1 + self.splitSize2) :]
        else:
            firstPart = inTensor[:, :, : self.splitSize1]
            secondPart = inTensor[
                :, :, self.splitSize1 : (self.splitSize1 + self.splitSize2)
            ]
            thirdPart = inTensor[:, :, (self.splitSize1 + self.splitSize2) :]

        return torch.cat([firstPart + thirdPart, secondPart], dim=-1)


# module to split at two given points, max the first and the third parts, and then concat with second part
class SplitMaxConcat(nn.Module):
    def __init__(self, splitSize1, splitSize2):
        super(SplitMaxConcat, self).__init__()
        self.splitSize1 = splitSize1
        self.splitSize2 = splitSize2

    def forward(self, inTensor):
        # Split along particular dimension
        # If only two dims
        if inTensor.dim() == 2:
            firstPart = inTensor[:, : self.splitSize1]
            secondPart = inTensor[
                :, self.splitSize1 : (self.splitSize1 + self.splitSize2)
            ]
            thirdPart = inTensor[:, (self.splitSize1 + self.splitSize2) :]
        else:
            firstPart = inTensor[:, :, : self.splitSize1]
            secondPart = inTensor[
                :, :, self.splitSize1 : (self.splitSize1 + self.splitSize2)
            ]
            thirdPart = inTensor[:, :, (self.splitSize1 + self.splitSize2) :]
        numDims = firstPart.dim()
        concat = torch.cat(
            (firstPart.unsqueeze(numDims), thirdPart.unsqueeze(numDims)), numDims
        )
        maxPool = torch.max(concat, numDims)[0]

        return torch.cat([maxPool, secondPart], dim=-1)


def select_model(
    model_name: str,
    vocab_size: int,
    embed_size: int,
    node_hidden_size: int,
    combine_hidden_size: int,
    dropout_ratio: float,
):
    """Select model architecture based on params

    :param params: a dictionary contains model architecture name and model sizes
    :type params: dict
    :return: a dictionary contains all model components
    :rtype: dict
    """
    node_embedder = nn.Embedding(
        vocab_size + 1, embed_size
    )  # in paper, referred as "Embedding Layer"
    node_post_embedder = Identity()  # in the paper, referred as "Embedding Transformer"
    node_pooler = (
        torch.sum
    )  # in the paper, this is fixed to be "sum", but you can replace it with mean/max/min function
    scorer = Identity()  # in the paper, this is referred as "Post Transformer"

    # following four modules are not discussed and used in paper
    combiner = Identity()

    if model_name == "np_lrlr_concat_lrldrl":
        node_post_embedder = nn.Sequential(
            nn.Linear(embed_size, embed_size, bias=False),
            nn.ReLU(),
            nn.Linear(embed_size, node_hidden_size),
            nn.ReLU(),
        )

        combiner = nn.Sequential(
            nn.Linear(2 * node_hidden_size, combine_hidden_size),
            nn.ReLU(),
            nn.Linear(combine_hidden_size, floor(combine_hidden_size / 2)),
            nn.Dropout(dropout_ratio),
            nn.ReLU(),
            nn.Linear(floor(combine_hidden_size / 2), 2),
        )
    elif model_name == "np_lrlr_sum_lrldrl":
        node_post_embedder = nn.Sequential(
            nn.Linear(embed_size, embed_size, bias=False),
            nn.ReLU(),
            nn.Linear(embed_size, node_hidden_size),
            nn.ReLU(),
        )

        combiner = nn.Sequential(
            SplitSum(node_hidden_size),
            nn.Linear(node_hidden_size, combine_hidden_size),
            nn.ReLU(),
            nn.Linear(combine_hidden_size, floor(combine_hidden_size / 2)),
            nn.Dropout(dropout_ratio),
            nn.ReLU(),
            nn.Linear(floor(combine_hidden_size / 2), 2),
        )
    elif model_name == "np_lrlr_max_lrldrl":
        node_post_embedder = nn.Sequential(
            nn.Linear(embed_size, embed_size, bias=False),
            nn.ReLU(),
            nn.Linear(embed_size, node_hidden_size),
            nn.ReLU(),
        )

        combiner = nn.Sequential(
            SplitMax(node_hidden_size),
            nn.Linear(node_hidden_size, combine_hidden_size),
            nn.ReLU(),
            nn.Linear(combine_hidden_size, floor(combine_hidden_size / 2)),
            nn.Dropout(dropout_ratio),
            nn.ReLU(),
            nn.Linear(floor(combine_hidden_size / 2), 2),
        )
    elif model_name == "np_lrlr_sum_lrldrls":
        node_post_embedder = nn.Sequential(
            nn.Linear(embed_size, embed_size, bias=False),
            nn.ReLU(),
            nn.Linear(embed_size, node_hidden_size),
            nn.ReLU(),
        )

        combiner = nn.Sequential(
            SplitSum(node_hidden_size),
            nn.Linear(node_hidden_size, combine_hidden_size),
            nn.ReLU(),
            nn.Linear(combine_hidden_size, floor(combine_hidden_size / 2)),
            nn.Dropout(dropout_ratio),
            nn.ReLU(),
            nn.Linear(floor(combine_hidden_size / 2), 1, bias=False),
            nn.Sigmoid(),
        )
    elif model_name == "np_lrlr_max_lrldrls":
        node_post_embedder = nn.Sequential(
            nn.Linear(embed_size, embed_size, bias=False),
            nn.ReLU(),
            nn.Linear(embed_size, node_hidden_size),
            nn.ReLU(),
        )

        combiner = nn.Sequential(
            SplitMax(node_hidden_size),
            nn.Linear(node_hidden_size, combine_hidden_size),
            nn.ReLU(),
            nn.Linear(combine_hidden_size, floor(combine_hidden_size / 2)),
            nn.Dropout(dropout_ratio),
            nn.ReLU(),
            nn.Linear(floor(combine_hidden_size / 2), 1),
            nn.Sigmoid(),
        )
    elif model_name == "np_lrlr_sd_lrlrdl":
        node_post_embedder = nn.Sequential(
            nn.Linear(embed_size, embed_size, bias=False),
            nn.ReLU(),
            nn.Linear(embed_size, node_hidden_size),
            nn.ReLU(),
        )

        scorer = nn.Sequential(
            nn.Linear(node_hidden_size, combine_hidden_size),
            nn.ReLU(),
            nn.Linear(combine_hidden_size, floor(combine_hidden_size / 2)),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(floor(combine_hidden_size / 2), 1),
        )
    elif model_name == "np_lrlr_msd_lrlrdl":
        node_pooler = torch.mean

        node_post_embedder = nn.Sequential(
            nn.Linear(embed_size, embed_size, bias=False),
            nn.ReLU(),
            nn.Linear(embed_size, node_hidden_size),
            nn.ReLU(),
        )

        scorer = nn.Sequential(
            nn.Linear(node_hidden_size, combine_hidden_size),
            nn.ReLU(),
            nn.Linear(combine_hidden_size, floor(combine_hidden_size / 2)),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(floor(combine_hidden_size / 2), 1),
        )
    elif model_name == "np_szlrlr_sd_lrlrdl":
        node_post_embedder = nn.Sequential(
            nn.Linear(embed_size + 1, embed_size + 1, bias=False),
            nn.ReLU(),
            nn.Linear(embed_size + 1, node_hidden_size),
            nn.ReLU(),
        )

        scorer = nn.Sequential(
            nn.Linear(node_hidden_size, combine_hidden_size),
            nn.ReLU(),
            nn.Linear(combine_hidden_size, floor(combine_hidden_size / 2)),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(floor(combine_hidden_size / 2), 1),
        )
    elif model_name == "np_lrlr_sd_szlrlrdl":
        node_post_embedder = nn.Sequential(
            nn.Linear(embed_size, embed_size, bias=False),
            nn.ReLU(),
            nn.Linear(embed_size, node_hidden_size),
            nn.ReLU(),
        )

        scorer = nn.Sequential(
            nn.Linear(node_hidden_size + 1, combine_hidden_size),
            nn.ReLU(),
            nn.Linear(combine_hidden_size, floor(combine_hidden_size / 2)),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(floor(combine_hidden_size / 2), 1),
        )
    elif model_name == "np_id_sd_lrlrdl":  # no embedding transformation
        node_post_embedder = Identity()
        scorer = nn.Sequential(
            nn.Linear(embed_size, combine_hidden_size),
            nn.ReLU(),
            nn.Linear(combine_hidden_size, floor(combine_hidden_size / 2)),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(floor(combine_hidden_size / 2), 1),
        )
    elif model_name == "np_id_sd_lrdl":  # no embedding transformation
        node_post_embedder = Identity()
        scorer = nn.Sequential(
            nn.Linear(embed_size, combine_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(combine_hidden_size, 1),
        )
    elif model_name == "np_lrlr_sd_dl":  # no combining layer
        node_post_embedder = nn.Sequential(
            nn.Linear(embed_size, embed_size, bias=False),
            nn.ReLU(),
            nn.Linear(embed_size, node_hidden_size),
            nn.ReLU(),
        )

        scorer = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Linear(node_hidden_size, 1),
        )
    elif model_name == "np_lrlr_sd_lrlrdls":
        node_post_embedder = nn.Sequential(
            nn.Linear(embed_size, embed_size, bias=False),
            nn.ReLU(),
            nn.Linear(embed_size, node_hidden_size),
            nn.ReLU(),
        )

        scorer = nn.Sequential(
            nn.Linear(node_hidden_size, combine_hidden_size),
            nn.ReLU(),
            nn.Linear(combine_hidden_size, floor(combine_hidden_size / 2)),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(floor(combine_hidden_size / 2), 1),
            nn.Sigmoid(),
        )
    elif model_name == "np_lrlr_sd_lrlrdlt":
        node_post_embedder = nn.Sequential(
            nn.Linear(embed_size, embed_size, bias=False),
            nn.ReLU(),
            nn.Linear(embed_size, node_hidden_size),
            nn.ReLU(),
        )

        scorer = nn.Sequential(
            nn.Linear(node_hidden_size, combine_hidden_size),
            nn.ReLU(),
            nn.Linear(combine_hidden_size, floor(combine_hidden_size / 2)),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(floor(combine_hidden_size / 2), 1, bias=False),
            nn.Tanh(),
        )
    elif model_name == "np_ltlt_sd_lrlrdlt":
        print("Using model: {}".format(model_name))
        node_post_embedder = nn.Sequential(
            nn.Linear(embed_size, embed_size, bias=False),
            nn.Tanh(),
            nn.Linear(embed_size, node_hidden_size),
            nn.Tanh(),
        )

        scorer = nn.Sequential(
            nn.Linear(node_hidden_size, combine_hidden_size),
            nn.ReLU(),
            nn.Linear(combine_hidden_size, floor(combine_hidden_size / 2)),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(floor(combine_hidden_size / 2), 1, bias=False),
            nn.Tanh(),
        )
    elif model_name == "np_lsls_sd_lrlrdlt":
        print("Using model: {}".format(model_name))
        node_post_embedder = nn.Sequential(
            nn.Linear(embed_size, embed_size, bias=False),
            nn.Sigmoid(),
            nn.Linear(embed_size, node_hidden_size),
            nn.Sigmoid(),
        )

        scorer = nn.Sequential(
            nn.Linear(node_hidden_size, combine_hidden_size),
            nn.ReLU(),
            nn.Linear(combine_hidden_size, floor(combine_hidden_size / 2)),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(floor(combine_hidden_size / 2), 1, bias=False),
            nn.Tanh(),
        )
    elif model_name == "np_none_sd_lrlrdlt":
        scorer = nn.Sequential(
            nn.Linear(embed_size, combine_hidden_size),
            nn.ReLU(),
            nn.Linear(combine_hidden_size, floor(combine_hidden_size / 2)),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(floor(combine_hidden_size / 2), 1, bias=False),
            nn.Tanh(),
        )
    elif model_name == "np_lrlr_sd_lrlrlrdrl":
        node_post_embedder = nn.Sequential(
            nn.Linear(embed_size, embed_size, bias=False),
            nn.ReLU(),
            nn.Linear(embed_size, node_hidden_size),
            nn.ReLU(),
        )

        scorer = nn.Sequential(
            nn.Linear(node_hidden_size, combine_hidden_size),
            nn.ReLU(),
            nn.Linear(combine_hidden_size, floor(combine_hidden_size / 2)),
            nn.ReLU(),
            nn.Linear(floor(combine_hidden_size / 2), floor(combine_hidden_size / 4)),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(floor(combine_hidden_size / 4), 1),
        )
    content = {
        "node_embedder": node_embedder,
        "node_post_embedder": node_post_embedder,
        "node_pooler": node_pooler,
        "combiner": combiner,
        "scorer": scorer,
    }

    return content
