import logging
from torch import nn

logger = logging.getLogger(__name__)

class ReFusionModel(nn.Module):
    def __init__(self, model, retriever, retriever_args):
        super().__init__()
        self.model = model
        self.retriever = retriever
        self.retriever_args = retriever_args
        self.retriever.replace_modules(self.model, self.retriever_args)

    def forward(
        self,
        neighbors=None,
        neighbor_texts=None,
        **kwargs,
    ):
        if neighbors != None:
            self.retriever.save_in_cache(neighbors)

        return self.model(**kwargs)

    def generate(
        self,
        neighbors=None,
        neighbor_texts=None,
        **kwargs,
    ):
        if neighbors != None:
            self.retriever.save_in_cache(neighbors)

        return self.model.generate(**kwargs)
