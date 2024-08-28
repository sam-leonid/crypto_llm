import abc
import os
import pickle
from tqdm import tqdm
from typing import List
from langchain.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings


class BaseVectorizer(abc.ABC):
    @abc.abstractmethod
    def calc_and_save_embedding(self):
        pass

    @abc.abstractmethod
    def calc_and_save_embedding_batch(self, name: str):
        # check if whitepaper exists and cmc description exists
        pass

    @abc.abstractmethod
    def get_retreiver(self, name: str):
        pass


class FAISSVectorizer(BaseVectorizer):
    def __init__(self, model_name: str = "cointegrated/LaBSE-en-ru"):
        self.wp_path = os.getenv("DATA_PATH") + "sources/whitepapers/"
        self.embedding_path = os.getenv("DATA_PATH") + "embeddings/"
        self.embedder = NVIDIAEmbeddings(
            model="nvidia/nv-embed-v1", api_key=os.getenv("NVIDIA_API_KEY")
        )

    def calc_and_save_embedding(self, name: str) -> None:
        if not os.path.exists(self.embedding_path + name):
            if os.path.exists(self.wp_path + name):
                with open(self.wp_path + name, "rb") as fp:
                    whitepaper_raw = pickle.load(fp)
                db = FAISS.from_documents(whitepaper_raw, self.embedder)
                db.save_local(self.embedding_path + name)

    def calc_and_save_embedding_batch(self, names: List[str]) -> None:
        for name in tqdm(names):
            self.calc_and_save_embedding(name)

    def get_retreiver(self, name: str, search_type: str = "similarity", k: int = 8):
        db = FAISS.load_local(
            self.embedding_path + name,
            self.hf_embeddings_model,
            allow_dangerous_deserialization=True,
        )
        return db.as_retriever(search_type="similarity", search_kwargs={"k": 8})
