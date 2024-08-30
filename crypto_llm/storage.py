import abc
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BaseStorage(abc.ABC):
    @abc.abstractmethod
    def get_wp_info(self):
        # self.wp_loader.get_info(name=currency_name, link=pdf_link):
        pass

    @abc.abstractmethod
    def get_symbol_by_name(self):
        # self.cmc_loader.get_info_by_name(currency_name)
        pass

    @abc.abstractmethod
    def save_cmc_info(self):
        # self.cmc_loader.get_info_by_name(currency_name)
        pass

    @abc.abstractmethod
    def get_pdf_whitepaper_link(self):
        # _, pdf_link = self.cmc_loader.get_pdf_whitepaper(currency_name)
        pass


class FileStorage(BaseStorage):
    def __init__(self):
        pass
