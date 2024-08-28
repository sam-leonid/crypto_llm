from loader import CMCLoader
from vectorizer import FAISSVectorizer

# get info from cmc
loader = CMCLoader()
loader.get_cmc_list(limit=7)
loader.get_info_batch()
loader.save_info()

# get info from whitepapers
# whitepapers_list = loader.get_all_pdf_whitepapers()
# wp_loader = WhitePaperLoader()
# wp_loader.get_info_batch(list_of_links=whitepapers_list)

# vectorize
vectorizer = FAISSVectorizer()
vectorizer.calc_and_save_embedding_batch(["Solana", "USDC", "XRP"])
