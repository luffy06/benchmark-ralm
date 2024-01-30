import json
import multiprocessing

from retriever_interface.base_retriever import BaseRetriever
from pyserini.search.lucene import LuceneSearcher

class SparseRetriever(BaseRetriever):
    def __init__(self, tokenizer, index_name):
        super(SparseRetriever, self).__init__(tokenizer=tokenizer)
        self.index = self._get_index(index_name)

    def _get_index(self, index_name):
        try:
            print(f"Attempting to download the index as if prebuilt by pyserini")
            return LuceneSearcher.from_prebuilt_index(index_name)
        except ValueError:
            print(f"Index does not exist in pyserini.")
            print("Attempting to treat the index as a directory (not prebuilt by pyserini)")
            return LuceneSearcher(index_name)

    def retrieve(self, sequence_input_ids, dataset, k=1):
        queries = [self.tokenizer.decode(sequence_input_ids[:, d["begin_location"]:d["end_location"]]) for d in dataset]
        assert len(queries) == len(dataset)
        all_res = self.searcher.batch_search(
            queries,
            qids=[str(i) for i in range(len(queries))],
            k=k,
            threads=multiprocessing.cpu_count()
        )

        for qid, res in all_res.items():
            qid = int(qid)
            d = dataset[qid]
            allowed_docs = []
            for hit in res:
                res_dict = json.loads(hit.raw)
                context_str = res_dict["contents"]
                allowed_docs.append({"text": context_str, "score": hit.score})
                if len(allowed_docs) >= k:
                    break
            d["query"] = queries[qid]
            d["retrievals"] = allowed_docs
        return dataset
