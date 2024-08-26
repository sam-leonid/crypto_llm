from loader.loader import CMCLoader

loader = CMCLoader()
loader.get_cmc_list(limit=7)
loader.get_info_batch()
loader.save_info()
