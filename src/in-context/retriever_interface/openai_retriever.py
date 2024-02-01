from retriever_interface.base_retriever import BaseRetriever
from openai import OpenAI

class OpenAIRetriever(BaseRetriever):
    def __init__(self, tokenizer, system_prompt="", model_name="gpt-4"):
        super(OpenAIRetriever, self).__init__(tokenizer=tokenizer)
        self.client = OpenAI()
        self.system_prompt = system_prompt
        self.model_name = model_name

    def retrieve(self, sequence_input_ids, dataset, k=1):
        for i, d in enumerate(dataset):
            query_ids = sequence_input_ids[:, d["begin_location"]:d["end_location"]-d["target_length"]][0]
            query_text = self.tokenizer.decode(query_ids)
            retrievals = []
            for _ in k:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": query_text}
                    ]
                )
                retrievals.append(completion.choices[0].message.content)
            dataset[i]["query"] = query_text
            dataset[i]["retrievals"] = [{"text": retrieval, "score": 1} for retrieval in retrievals]
        return dataset
