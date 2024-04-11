def load_retriever(args):
    if args.retriever_type == "dense":
        from retriever_interface.dense_retriever import DenseRetriever
        retriever = DenseRetriever(args)
    elif args.retriever_type == "sparse":
        from retriever_interface.sparse_retriever import SparseRetriever
        retriever = SparseRetriever(args)
    else:
        raise ValueError("Invalid retriever type")
    if args.encoder_path is not None:
        device = 'cpu' if args.retriever_device == -1 else 'cuda:{}'.format(args.retriever_device)
        retriever.load_encoder(args.encoder_path, device)
    return retriever
    
class BaseRetriever:
    def __init__(self, args):
        self.args = args

    def retrieve(self, query_texts):
        raise NotImplementedError

    def load_encoder(self, encoder_path):
        raise NotImplementedError