from crypto_llm.chainer import LlmChainer
from dotenv import load_dotenv

load_dotenv("../.env")

# get info from cmc
# loader = CMCLoader()
# loader.get_cmc_list(limit=7)
# loader.get_info_batch()
# loader.save_info()

# get info from whitepapers
# whitepapers_list = loader.get_all_pdf_whitepapers()
# wp_loader = WhitePaperLoader()
# wp_loader.get_info_batch(list_of_links=whitepapers_list)

# vectorize
# vectorizer = FAISSVectorizer()
# vectorizer.calc_and_save_embedding_batch(["Solana", "USDC", "XRP"])
# vectorizer.get_retriever('Solana')

# chainer
chainer = LlmChainer()
# chainer.run_chain('Solana', "Какой алгоритм консенсуса в SOLANA?")
chainer.run_chain("USDC", is_summary=True)
chainer.run_chain("XRP", is_summary=True)
