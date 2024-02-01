from sentence_transformers import SentenceTransformer

def add_retriever_args(parser, retrieval_type):
    if retrieval_type == "sparse":
        parser.add_argument("--index_name", type=str, default="wikipedia-dpr")
    elif retrieval_type == "exact":
      pass
    elif retrieval_type == "dense":
        parser.add_argument("--encoder_name", type=str, default="bert-base-uncased")
        parser.add_argument("--retriever_dir", type=str, default="metadata/wikitext103-bert-base")
        parser.add_argument("--corpus_size", type=str, default='100K')
        parser.add_argument("--nprobe", type=int, default=512)
        parser.add_argument("--device_id", type=int, default=-1)
        parser.add_argument("--index_path", type=str, default=None)
    elif retrieval_type == "openai":
        parser.add_argument("--system_prompt", type=str, default="", nargs="+")
        parser.add_argument("--model_name", type=str, default="gpt-4")
    elif retrieval_type == "llm":
        parser.add_argument("--system_prompt", type=str, default="", nargs="+")
        parser.add_argument("--model_name", type=str, default="gpt-4")
    else:
        raise ValueError


def get_retriever(args, tokenizer):
    if args.retrieval_type == "sparse":
        from retriever_interface.sparse_retriever import SparseRetriever
        return SparseRetriever(
            tokenizer=tokenizer,
            index_name=args.index_name,
        )
    elif args.retrieval_type == "exact":
        from retriever_interface.exact_retriever import ExactRetriever
        return ExactRetriever(
            tokenizer=tokenizer,
        )
    elif args.retrieval_type == "dense":
        from retriever_interface.dense_retriever import DenseRetriever
        encoder = SentenceTransformer(args.encoder_name).to("cpu" if args.device_id == -1 else f"cuda:{args.device_id}")
        assert args.corpus_size.startswith("dense"), f"Wrong corpus_size pattern {args.corpus_size}"
        corpus_size = args.corpus_size.split("-")
        assert len(corpus_size) < 2, f"Wrong corpus_size pattern length {args.corpus_size}"
        corpus_size = None if len(corpus_size) < 2 else corpus_size[1]
        return DenseRetriever(
            tokenizer=tokenizer,
            encoder=encoder,
            retriever_dir=args.retriever_dir,
            nprobe=args.nprobe,
            corpus_size=args.corpus_size,
            device_id=args.device_id,
            index_path=args.index_path,
        )
    elif args.retrieval_type == "openai":
        from retriever_interface.openai_retriever import OpenAIRetriever
        system_prompt = " ".join(args.system_prompt)
        system_prompt = system_prompt.replace("%d", f"{args.stride}")
        return OpenAIRetriever(
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            model_name=args.model_name,
        )
    raise ValueError
