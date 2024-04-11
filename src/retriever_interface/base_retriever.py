class AutoRetriever:
    def __init__(self, args):
        self.args = args
        self.__load(args)

    def __load(self, args):
        if args.retriever_type == "dense":
            from src.retriever_interface.dense_retriever import DenseRetriever
            self.retriever = DenseRetriever(args)
        elif args.retriever_type == "sparse":
            from src.retriever_interface.sparse_retriever import SparseRetriever
            self.retriever = SparseRetriever(args)
        else:
            raise ValueError("Invalid retriever type")
        if args.encoder_path is not None:
            self.retriever.load_encoder(args.encoder_path, args.retriever_device)

    def retrieve(self, query_texts):
        return self.retriever.retrieve(query_texts)

    @staticmethod
    def from_pretrained(self, args):
        self.__load(args)
        return self

    def replace_modules(self, model, config):
        return self.retriever.replace_modules(model, config)
    
class BaseRetriever:
    def __init__(self, args):
        self.args = args

    def retrieve(self, query_texts):
        raise NotImplementedError

    def load_encoder(self, encoder_path):
        raise NotImplementedError