from dotenv import load_dotenv

load_dotenv("./.env")

# test FileStorage
# storage = FileStorage()
# storage.get_symbol_by_name("ChatCoin")
# storage.get_all_cmc_list()
# storage.get_pdf_whitepaper_link("ChatCoin")
# storage.save_cmc_info()


# test CMCLoader
# loader = CMCLoader()
# loader.get_all_cmc_list()
# loader.save_info()
# loader.get_info_batch()
# loader.get_info_by_name("Shill Guard Token")
# loader.save_info()

# test WhitePaperLoader
# whitepapers_list = loader.get_all_pdf_whitepapers()
# wp_loader = WhitePaperLoader()
# wp_loader.get_info_batch(list_of_links=whitepapers_list)

# test FAISSVectorizer
# vectorizer = FAISSVectorizer()
# vectorizer.calc_and_save_embedding('USDC')
# vectorizer.calc_and_save_embedding_batch(["Solana", "USDC", "XRP"])
# vectorizer.get_retriever('Blocknet')

# test LlmChainer
# chainer = LlmChainer()
# chainer.run_chain('Solana', "Какой алгоритм консенсуса в SOLANA?")
# chainer.run_chain("ChatCoin", is_summary=True)
# chainer.run_chain("Lition", is_summary=True)
# chainer.run_chain("Lition", "Какой алгоритм консенсуса?")
