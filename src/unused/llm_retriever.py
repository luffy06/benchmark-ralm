import time
import logging
from retriever_interface.base_retriever import BaseRetriever
from openai import OpenAI

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

tpm_limits = {
    "gpt-3.5-turbo": 60000,
    "gpt-4": 10000,
}

rpm_limits = {
    "gpt-3.5-turbo": 500,
    "gpt-4": 500,
}

class LLMRetriever(BaseRetriever):
    def __init__(self, tokenizer, system_prompt="", model_name="gpt-4"):
        super(LLMRetriever, self).__init__(tokenizer=tokenizer)
        self.client = OpenAI()
        self.system_prompt = system_prompt
        self.model_name = model_name
        logger.info(f"System prompt: {self.system_prompt}")

    def retrieve(self, sequence_input_ids, dataset, k=1):
        num_tokens = 0
        num_requests = 0
        for i, d in enumerate(dataset):
            query_ids = sequence_input_ids[:, d["begin_location"]:d["end_location"]-d["target_length"]][0]
            query_text = self.tokenizer.decode(query_ids)
            retrievals = []
            for _ in range(k):
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": query_text}
                    ]
                )
                retrievals.append(query_text + " " + completion.choices[0].message.content)
                num_tokens += d["end_location"] - d["target_length"] - d["begin_location"] + len(self.system_prompt.split(" "))
                num_requests += 1
                if num_tokens >= tpm_limits[self.model_name] or num_requests >= rpm_limits[self.model_name]:
                    time.sleep(60)

            dataset[i]["query"] = query_text
            dataset[i]["retrievals"] = [{"text": retrieval, "score": 1} for retrieval in retrievals]
        return dataset
