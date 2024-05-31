from retriever_interface.base_retriever import BaseRetriever


class ExactRetriever(BaseRetriever):
    def __init__(self, tokenizer):
        super(ExactRetriever, self).__init__(tokenizer=tokenizer)

    def retrieve(self, sequence_input_ids, dataset, k=1):
        for i, d in enumerate(dataset):
            query_ids = sequence_input_ids[:, d["begin_location"]:d["end_location"]][0]
            query_text = self.tokenizer.decode(query_ids)
            dataset[i]["query"] = query_text
            dataset[i]["retrievals"] = [{"text": query_text, "score": 1}]
        return dataset
