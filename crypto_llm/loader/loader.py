import abc
import coinmarketcapapi
import pandas as pd
import os
from dotenv import load_dotenv
from tqdm import tqdm
from time import sleep
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()


class BaseLoader(abc.ABC):
    @abc.abstractmethod
    def get_info_batch(self):
        pass

    @abc.abstractmethod
    def save_info(self):
        pass


class CMCLoader(BaseLoader):
    def __init__(self):
        self.cmc_client = coinmarketcapapi.CoinMarketCapAPI(os.getenv("CMC_API_KEY"))
        self.path = os.getenv("DATA_PATH") + "cmc/"
        self.cmc_list = None
        self.cmc_detailed_info = None
        logger.info("CMCLoader initialized")

    def get_cmc_list(self, limit: int = 5) -> pd.DataFrame:
        logger.info(f"Fetching CMC list with limit {limit}")
        response = self.cmc_client.cryptocurrency_listings_latest(limit=limit)
        self.cmc_list = pd.DataFrame(response.data)
        logger.info(f"Fetched {len(self.cmc_list)} items from CMC")
        return self.cmc_list

    def get_info_batch(self, sleep_time: int = 35, iters_wait: int = 2):
        logger.info("Starting to fetch detailed info")
        errors = {}

        for sym in tqdm(self.cmc_list["symbol"].str.upper().unique()):
            exc_rate = 0
            while True:
                try:
                    data = self.cmc_client.cryptocurrency_info(symbol=sym).data[sym]
                    for i in data:
                        if "urls" in i:
                            i.update(i["urls"])
                            del i["urls"]
                    self.cmc_detailed_info = pd.concat(
                        [self.cmc_detailed_info, pd.DataFrame(data)]
                    )
                    logger.debug(f"Successfully fetched info for {sym}")
                    break
                except Exception as e:
                    exc_rate += 1
                    logger.warning(
                        f"Error fetching info for {sym}. "
                        + f"Attempt {str(exc_rate)}. Error: {str(e)}"
                    )
                    if exc_rate == iters_wait:
                        errors[sym] = e
                        logger.error(
                            f"Failed to fetch info for {sym}"
                            + f"after {iters_wait} attempts"
                        )
                        break
                    sleep(sleep_time)

        logger.info("Finished fetching detailed info")
        if errors:
            logger.warning(f"Encountered errors for {len(errors)} symbols")
        return self.cmc_detailed_info

    def save_info(self):
        if self.cmc_list is not None:
            self.cmc_list.to_csv(self.path + "cmc_list.csv", index=False)
            logger.info(f"Saved CMC list to {self.path}cmc_list.csv")
        else:
            logger.warning("CMC list is None, not saving")

        if self.cmc_detailed_info is not None:
            self.cmc_detailed_info.to_csv(self.path + "cmc_info.csv", index=False)
            logger.info(f"Saved detailed CMC info to {self.path}cmc_info.csv")
        else:
            logger.warning("Detailed CMC info is None, not saving")
