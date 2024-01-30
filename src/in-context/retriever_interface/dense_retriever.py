import sys
import numpy as np
from glob import glob
from retriever_interface.base_retriever import BaseRetriever
sys.path.append('../retriever-lib/src')
from faisslib.retriever import Retriever

class DenseRetriever(BaseRetriever):
    def __init__(
        self, 
        tokenizer, 
        encoder, 
        retriever_dir, 
        nprobe, 
        corpus_size=None,
        device_id=-1, 
        index_path=None
    ):
        super(DenseRetriever, self).__init__(tokenizer=tokenizer)
        self.encoder = encoder
        self.index = Retriever(retriever_dir, nprobe, 1, corpus_size, device_id, index_path)

    def retrieve(self, sequence_input_ids, dataset, k=1):
        queries = np.concatenate([self.tokenizer.decode(sequence_input_ids[:, d["begin_location"]:d["end_location"]]) for d in dataset], axis=0)
        assert len(queries) == len(dataset)

        self.index.topk = k
        query_embs = self.encoder.encode(queries)
        results = self.index.search(query_embs)
        
        for qid, res in results.items():
            qid = int(qid)
            d = dataset[qid]
            d["query"] = queries[qid]
            d["retrievals"] = res["text"]
        return dataset
